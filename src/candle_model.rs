//! Phonon-2 (Parakeet-TDT-0.6B-v3 graph) in candle: FastConformer encoder + LSTM prediction network + TDT joint,
//! with batched greedy TDT decoding. Mirrors HF `ParakeetForTDT`.

use crate::cpu_ops;
use crate::fermion::{self, ModelFiles};
use crate::mel::{Features, N_MELS};
use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{LayerNorm, Linear};
use std::collections::HashMap;

const D_MODEL: usize = 1024;
const N_HEADS: usize = 8;
const HEAD_DIM: usize = D_MODEL / N_HEADS;
const N_LAYERS: usize = 24;
const CONV_K: usize = 9;
const PRED: usize = 640;
const VOCAB: usize = 8192; // blank is VOCAB
const DURATIONS: [usize; 5] = [0, 1, 2, 3, 4];
const MAX_SYMBOLS: usize = 10;
const BN_EPS: f64 = 1e-5;
const LN_EPS: f64 = 1e-5;

/// Hypothesis for one utterance: token ids and the encoder frame each was emitted at (80 ms per frame).
#[derive(Clone, Default, Debug)]
pub struct Hyp {
    pub tokens: Vec<u32>,
    pub frames: Vec<usize>,
    /// softmax probability of each emitted token (over the vocabulary + blank)
    pub probs: Vec<f32>,
}

struct Weights {
    map: HashMap<String, fermion::Tensor>,
    device: Device,
}

impl Weights {
    fn raw(&mut self, name: &str) -> Result<fermion::Tensor> {
        self.map.remove(name).with_context(|| format!("missing tensor {name}"))
    }
    fn get(&mut self, name: &str, dtype: DType) -> Result<Tensor> {
        let t = self.raw(name)?;
        Ok(Tensor::from_vec(t.data, t.shape, &self.device)?.to_dtype(dtype)?)
    }
    fn linear(&mut self, name: &str, bias: bool, dtype: DType) -> Result<Linear> {
        let w = self.get(&format!("{name}.weight"), dtype)?;
        let b = if bias { Some(self.get(&format!("{name}.bias"), dtype)?) } else { None };
        Ok(Linear::new(w, b))
    }
    fn layer_norm(&mut self, name: &str, dtype: DType) -> Result<LayerNorm> {
        Ok(LayerNorm::new(self.get(&format!("{name}.weight"), dtype)?, self.get(&format!("{name}.bias"), dtype)?, LN_EPS))
    }
}

/// Padding information for one batch: additive key mask and time mask for the composed (CUDA) path, plain
/// lengths for the fused CPU kernels.
struct Masks {
    key: Tensor,  // [B,1,1,T], 0 or -inf
    time: Tensor, // [B,T,1], 1 or 0
    lens: Vec<usize>,
}

/// The fused kernels in `cpu_ops` handle contiguous f32 on the CPU.
fn fused(x: &Tensor) -> bool {
    x.device().is_cpu() && x.dtype() == DType::F32
}

struct FeedForward {
    l1: Linear,
    l2: Linear,
}

impl FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.l1.forward(x)?;
        let h = if fused(&h) { cpu_ops::silu(&h)? } else { h.silu()? };
        Ok(self.l2.forward(&h)?)
    }
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    pos: Linear,
    bias_u: Tensor, // [1, H, 1, Dh]
    bias_v: Tensor,
}

impl Attention {
    /// x [B,T,D], pe [1,2T-1,D]
    fn forward(&self, x: &Tensor, pe: &Tensor, masks: &Masks) -> Result<Tensor> {
        if fused(x) {
            return self.forward_cpu(x, pe, masks);
        }
        let key_mask = &masks.key;
        let (b, t, _) = x.dims3()?;
        let heads = |y: Tensor| -> Result<Tensor> { Ok(y.reshape((b, t, N_HEADS, HEAD_DIM))?.transpose(1, 2)?) };
        let q = heads(self.q.forward(x)?)?;
        let k = heads(self.k.forward(x)?)?;
        let v = heads(self.v.forward(x)?)?.contiguous()?;
        let qu = q.broadcast_add(&self.bias_u)?.contiguous()?;
        let qv = q.broadcast_add(&self.bias_v)?.contiguous()?;
        let p = self.pos.forward(pe)?; // [1, 2T-1, D]
        let p = p.reshape((1, 2 * t - 1, N_HEADS, HEAD_DIM))?.permute((0, 2, 3, 1))?.contiguous()?; // [1,H,Dh,2T-1]
        let bd = qv.broadcast_matmul(&p)?; // [B,H,T,2T-1]
        let bd = rel_shift(&bd, t)?;
        let ac = qu.matmul(&k.t()?.contiguous()?)?; // [B,H,T,T]
        let scale = (HEAD_DIM as f64).powf(-0.5);
        let scores = ((ac + bd)? * scale)?.broadcast_add(key_mask)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, t, D_MODEL))?;
        Ok(self.o.forward(&out)?)
    }

    fn forward_cpu(&self, x: &Tensor, pe: &Tensor, masks: &Masks) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let split = |y: Tensor| -> Result<Tensor> { Ok(y.reshape((b, t, N_HEADS, HEAD_DIM))?) };
        let bias = |bb: &Tensor| -> Result<Tensor> { Ok(bb.reshape((N_HEADS, HEAD_DIM))?) };
        let q = split(self.q.forward(x)?)?;
        let qu = cpu_ops::permute4(&q, [0, 2, 1, 3], Some(&bias(&self.bias_u)?))?; // [B,H,T,Dh]
        let qv = cpu_ops::permute4(&q, [2, 0, 1, 3], Some(&bias(&self.bias_v)?))?; // [H,B,T,Dh]
        let k = cpu_ops::permute4(&split(self.k.forward(x)?)?, [0, 2, 1, 3], None)?;
        let v = cpu_ops::permute4(&split(self.v.forward(x)?)?, [0, 2, 1, 3], None)?;
        let p = self.pos.forward(pe)?.reshape((1, 2 * t - 1, N_HEADS, HEAD_DIM))?;
        let p = cpu_ops::permute4(&p, [0, 2, 1, 3], None)?.squeeze(0)?; // [H,2T-1,Dh]
        let bd = qv.reshape((N_HEADS, b * t, HEAD_DIM))?.matmul(&p.t()?)?; // [H,B*T,2T-1]
        let ac = qu.matmul(&k.t()?)?; // [B,H,T,T]
        let scale = (HEAD_DIM as f32).powf(-0.5);
        let attn = cpu_ops::rel_attn_softmax(&ac, &bd, scale, &masks.lens)?;
        let out = attn.matmul(&v)?; // [B,H,T,Dh]
        let out = cpu_ops::permute4(&out, [0, 2, 1, 3], None)?.reshape((b, t, D_MODEL))?;
        Ok(self.o.forward(&out)?)
    }
}

/// out[.., i, j] = x[.., i, j + T - 1 - i] via the Transformer-XL pad/reshape trick.
fn rel_shift(x: &Tensor, t: usize) -> Result<Tensor> {
    let (b, h, _, p) = x.dims4()?;
    let x = x.pad_with_zeros(3, 1, 0)?; // [B,H,T,2T]
    let x = x.reshape((b, h, p + 1, t))?.narrow(2, 1, p)?;
    Ok(x.reshape((b, h, t, p))?.narrow(3, 0, t)?)
}

struct ConvModule {
    pw1: Linear,
    dw: Tensor,      // [K, 1, D]: depthwise taps with BatchNorm folded in
    dw_bias: Tensor, // [1, 1, D]
    pw2: Linear,
}

impl ConvModule {
    /// x [B,T,D]
    fn forward(&self, x: &Tensor, masks: &Masks) -> Result<Tensor> {
        let (_, t, _) = x.dims3()?;
        let h = self.pw1.forward(x)?;
        if fused(&h) {
            let w = self.dw.squeeze(1)?;
            let y = cpu_ops::glu_dwconv_silu(&h, &w, &self.dw_bias.flatten_all()?, &masks.lens)?;
            return Ok(self.pw2.forward(&y)?);
        }
        let time_mask = &masks.time;
        let a = h.narrow(2, 0, D_MODEL)?;
        let g = h.narrow(2, D_MODEL, D_MODEL)?;
        let h = (a * candle_nn::ops::sigmoid(&g)?)?.broadcast_mul(time_mask)?;
        let pad = (CONV_K - 1) / 2;
        let hp = h.pad_with_zeros(1, pad, pad)?;
        let mut acc = self.dw_bias.clone();
        for k in 0..CONV_K {
            acc = hp.narrow(1, k, t)?.broadcast_mul(&self.dw.get(k)?)?.broadcast_add(&acc)?;
        }
        Ok(self.pw2.forward(&acc.silu()?)?)
    }
}

struct Block {
    ff1: FeedForward,
    attn: Attention,
    conv: ConvModule,
    ff2: FeedForward,
    n_ff1: LayerNorm,
    n_att: LayerNorm,
    n_conv: LayerNorm,
    n_ff2: LayerNorm,
    n_out: LayerNorm,
}

impl Block {
    fn forward(&self, x: &Tensor, pe: &Tensor, masks: &Masks) -> Result<Tensor> {
        let add = |a: &Tensor, b: Tensor, s: f64| -> Result<Tensor> {
            Ok(if fused(a) { cpu_ops::add_scaled(a, &b, s as f32)? } else { (a + (b * s)?)? })
        };
        let x = add(x, self.ff1.forward(&self.n_ff1.forward(x)?)?, 0.5)?;
        let x = add(&x, self.attn.forward(&self.n_att.forward(&x)?, pe, masks)?, 1.0)?;
        let x = add(&x, self.conv.forward(&self.n_conv.forward(&x)?, masks)?, 1.0)?;
        let x = add(&x, self.ff2.forward(&self.n_ff2.forward(&x)?)?, 0.5)?;
        Ok(self.n_out.forward(&x)?)
    }
}

struct Conv2d {
    w: Tensor,
    b: Tensor, // [1,C,1,1]
}

struct Subsampling {
    conv0: Conv2d, // 1 -> 256, 3x3 stride 2
    dw1: Conv2d,   // depthwise 3x3 stride 2
    pw1: Conv2d,   // 1x1, w as [Co, Ci]
    dw2: Conv2d,
    pw2: Conv2d,
    linear: Linear,
}

/// Depthwise 3x3 / stride 2 / pad 1 conv over [B,C,H,W], as nine strided taps.
fn dw_conv_s2(x: &Tensor, c: &Conv2d) -> Result<Tensor> {
    let (b, ch, h, w) = x.dims4()?;
    let (ho, wo) = ((h - 1) / 2 + 1, (w - 1) / 2 + 1);
    let x = x.pad_with_zeros(2, 1, 2 * ho + 1 - h)?.pad_with_zeros(3, 1, 2 * wo + 1 - w)?;
    let x = x.reshape((b, ch, ho + 1, 2, wo + 1, 2))?;
    let mut acc = c.b.clone();
    for ky in 0..3 {
        for kx in 0..3 {
            let tap = x
                .narrow(2, ky / 2, ho)?
                .narrow(3, ky % 2, 1)?
                .narrow(4, kx / 2, wo)?
                .narrow(5, kx % 2, 1)?
                .reshape((b, ch, ho, wo))?;
            let k = c.w.narrow(1, ky * 3 + kx, 1)?.reshape((1, ch, 1, 1))?;
            acc = tap.broadcast_mul(&k)?.broadcast_add(&acc)?;
        }
    }
    Ok(acc)
}

fn pw_conv(x: &Tensor, c: &Conv2d) -> Result<Tensor> {
    let (b, ch, h, w) = x.dims4()?;
    let y = c.w.broadcast_matmul(&x.reshape((b, ch, h * w))?)?;
    Ok(y.reshape((b, (), h, w))?.broadcast_add(&c.b)?)
}

fn sub_len(l: usize) -> usize {
    if l == 0 { 0 } else { (l - 1) / 2 + 1 }
}

fn time_mask_4d(lens: &[usize], t: usize, dtype: DType, dev: &Device) -> Result<Tensor> {
    let m: Vec<f32> = lens.iter().flat_map(|&l| (0..t).map(move |i| if i < l { 1.0 } else { 0.0 })).collect();
    Ok(Tensor::from_vec(m, (lens.len(), 1, t, 1), dev)?.to_dtype(dtype)?)
}

impl Subsampling {
    /// feats [B,T,128] -> ([B,T/8,D], lengths)
    fn forward(&self, feats: &Tensor, lens: &[usize]) -> Result<(Tensor, Vec<usize>)> {
        if fused(feats) {
            return self.forward_cpu(feats, lens);
        }
        let (dt, dev) = (feats.dtype(), feats.device().clone());
        let x = feats.unsqueeze(1)?;
        let mut lens: Vec<usize> = lens.iter().map(|&l| sub_len(l)).collect();
        let x = x.conv2d(&self.conv0.w, 1, 2, 1, 1)?.broadcast_add(&self.conv0.b)?;
        let x = x.broadcast_mul(&time_mask_4d(&lens, x.dim(2)?, dt, &dev)?)?.relu()?;

        lens.iter_mut().for_each(|l| *l = sub_len(*l));
        let x = dw_conv_s2(&x, &self.dw1)?;
        let m = time_mask_4d(&lens, x.dim(2)?, dt, &dev)?;
        let x = x.broadcast_mul(&m)?;
        let x = pw_conv(&x, &self.pw1)?.broadcast_mul(&m)?.relu()?;

        lens.iter_mut().for_each(|l| *l = sub_len(*l));
        let x = dw_conv_s2(&x, &self.dw2)?;
        let m = time_mask_4d(&lens, x.dim(2)?, dt, &dev)?;
        let x = x.broadcast_mul(&m)?;
        let x = pw_conv(&x, &self.pw2)?.broadcast_mul(&m)?.relu()?;

        let (b, c, t, f) = x.dims4()?;
        let x = x.transpose(1, 2)?.reshape((b, t, c * f))?;
        Ok((self.linear.forward(&x)?, lens))
    }

    fn forward_cpu(&self, feats: &Tensor, lens: &[usize]) -> Result<(Tensor, Vec<usize>)> {
        let mut lens: Vec<usize> = lens.iter().map(|&l| sub_len(l)).collect();
        let x = feats.unsqueeze(1)?.conv2d(&self.conv0.w, 1, 2, 1, 1)?;
        let mut x = cpu_ops::bias_mask_relu(&x, &self.conv0.b, &lens)?;
        for (dw, pw) in [(&self.dw1, &self.pw1), (&self.dw2, &self.pw2)] {
            lens.iter_mut().for_each(|l| *l = sub_len(*l));
            let y = cpu_ops::dwconv2d_s2(&x, &dw.w, &dw.b.flatten_all()?, &lens)?;
            let (b, c, h, w) = y.dims4()?;
            let y = pw.w.broadcast_matmul(&y.reshape((b, c, h * w))?)?.reshape((b, c, h, w))?;
            x = cpu_ops::bias_mask_relu(&y, &pw.b, &lens)?;
        }
        let (b, c, t, f) = x.dims4()?;
        let x = cpu_ops::permute4(&x, [0, 2, 1, 3], None)?.reshape((b, t, c * f))?;
        Ok((self.linear.forward(&x)?, lens))
    }
}

/// Relative positional table for positions T-1 .. -(T-1), sin/cos interleaved: [1, 2T-1, D].
fn rel_pos_emb(t: usize, dtype: DType, dev: &Device) -> Result<Tensor> {
    let n = 2 * t - 1;
    let mut pe = vec![0f32; n * D_MODEL];
    for k in 0..n {
        let pos = (t as f32 - 1.0) - k as f32;
        for i in 0..D_MODEL / 2 {
            let inv = 1.0 / 10000f32.powf((2 * i) as f32 / D_MODEL as f32);
            let a = pos * inv;
            pe[k * D_MODEL + 2 * i] = a.sin();
            pe[k * D_MODEL + 2 * i + 1] = a.cos();
        }
    }
    Ok(Tensor::from_vec(pe, (1, n, D_MODEL), dev)?.to_dtype(dtype)?)
}

struct Lstm {
    w: Vec<Tensor>, // per layer [2*PRED, 4*PRED] = [W_ih; W_hh]^T
    b: Vec<Tensor>, // per layer [4*PRED] = b_ih + b_hh
}

struct Decoder {
    embedding: Tensor, // [VOCAB+1, PRED], blank row is zero
    lstm: Lstm,
    proj: Linear,
    head: Linear,
}

type State = (Tensor, Vec<Tensor>, Vec<Tensor>); // (projected output g, h per layer, c per layer)

impl Decoder {
    fn step(&self, x: &Tensor, h: &[Tensor], c: &[Tensor]) -> Result<State> {
        let mut inp = x.clone();
        let (mut hs, mut cs) = (Vec::new(), Vec::new());
        for l in 0..self.lstm.w.len() {
            let gates = Tensor::cat(&[&inp, &h[l]], 1)?.matmul(&self.lstm.w[l])?.broadcast_add(&self.lstm.b[l])?;
            let i = candle_nn::ops::sigmoid(&gates.narrow(1, 0, PRED)?)?;
            let f = candle_nn::ops::sigmoid(&gates.narrow(1, PRED, PRED)?)?;
            let g = gates.narrow(1, 2 * PRED, PRED)?.tanh()?;
            let o = candle_nn::ops::sigmoid(&gates.narrow(1, 3 * PRED, PRED)?)?;
            let c2 = ((f * &c[l])? + (i * g)?)?;
            let h2 = (o * c2.tanh()?)?;
            inp = h2.clone();
            hs.push(h2);
            cs.push(c2);
        }
        Ok((self.proj.forward(&inp)?, hs, cs))
    }

    fn joint(&self, f: &Tensor, g: &Tensor) -> Result<Tensor> {
        Ok(self.head.forward(&(f + g)?.relu()?)?)
    }
}

pub struct Phonon {
    sub: Subsampling,
    blocks: Vec<Block>,
    enc_proj: Linear,
    dec: Decoder,
    device: Device,
    dtype: DType,
}

impl Phonon {
    pub fn load(files: &ModelFiles, device: &Device, dtype: DType) -> Result<Self> {
        let map = fermion::read_container(&files.container)?;
        let mut w = Weights { map, device: device.clone() };
        let dt = dtype;

        let conv = |w: &mut Weights, name: &str, pointwise: bool| -> Result<Conv2d> {
            let k = w.get(&format!("encoder.subsampling.layers.{name}.weight"), dt)?;
            let k = if pointwise { k.squeeze(3)?.squeeze(2)? } else { k };
            let b = w.get(&format!("encoder.subsampling.layers.{name}.bias"), dt)?;
            let c = b.dim(0)?;
            Ok(Conv2d { w: k, b: b.reshape((1, c, 1, 1))? })
        };
        let conv0 = conv(&mut w, "0", false)?;
        let mut dw1 = conv(&mut w, "2", false)?;
        dw1.w = dw1.w.reshape((256, 9))?;
        let pw1 = conv(&mut w, "3", true)?;
        let mut dw2 = conv(&mut w, "5", false)?;
        dw2.w = dw2.w.reshape((256, 9))?;
        let pw2 = conv(&mut w, "6", true)?;
        let sub = Subsampling { conv0, dw1, pw1, dw2, pw2, linear: w.linear("encoder.subsampling.linear", true, dt)? };

        let mut blocks = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            let p = format!("encoder.layers.{i}");
            let hb = |w: &mut Weights, n: &str| -> Result<Tensor> {
                Ok(w.get(&format!("{p}.self_attn.{n}"), dt)?.reshape((1, N_HEADS, 1, HEAD_DIM))?)
            };
            let attn = Attention {
                q: w.linear(&format!("{p}.self_attn.q_proj"), false, dt)?,
                k: w.linear(&format!("{p}.self_attn.k_proj"), false, dt)?,
                v: w.linear(&format!("{p}.self_attn.v_proj"), false, dt)?,
                o: w.linear(&format!("{p}.self_attn.o_proj"), false, dt)?,
                pos: w.linear(&format!("{p}.self_attn.relative_k_proj"), false, dt)?,
                bias_u: hb(&mut w, "bias_u")?,
                bias_v: hb(&mut w, "bias_v")?,
            };
            // fold eval-mode BatchNorm into the depthwise conv
            let dwk = w.raw(&format!("{p}.conv.depthwise_conv.weight"))?; // [D,1,K]
            let gamma = w.raw(&format!("{p}.conv.norm.weight"))?.data;
            let beta = w.raw(&format!("{p}.conv.norm.bias"))?.data;
            let mean = w.raw(&format!("{p}.conv.norm.running_mean"))?.data;
            let var = w.raw(&format!("{p}.conv.norm.running_var"))?.data;
            w.map.remove(&format!("{p}.conv.norm.num_batches_tracked"));
            let mut taps = vec![0f32; CONV_K * D_MODEL];
            let mut bias = vec![0f32; D_MODEL];
            for ch in 0..D_MODEL {
                let s = gamma[ch] / (var[ch] + BN_EPS as f32).sqrt();
                for k in 0..CONV_K {
                    taps[k * D_MODEL + ch] = dwk.data[ch * CONV_K + k] * s;
                }
                bias[ch] = beta[ch] - mean[ch] * s;
            }
            let conv = ConvModule {
                pw1: w.linear(&format!("{p}.conv.pointwise_conv1"), false, dt)?,
                dw: Tensor::from_vec(taps, (CONV_K, 1, D_MODEL), device)?.to_dtype(dt)?,
                dw_bias: Tensor::from_vec(bias, (1, 1, D_MODEL), device)?.to_dtype(dt)?,
                pw2: w.linear(&format!("{p}.conv.pointwise_conv2"), false, dt)?,
            };
            blocks.push(Block {
                ff1: FeedForward {
                    l1: w.linear(&format!("{p}.feed_forward1.linear1"), false, dt)?,
                    l2: w.linear(&format!("{p}.feed_forward1.linear2"), false, dt)?,
                },
                attn,
                conv,
                ff2: FeedForward {
                    l1: w.linear(&format!("{p}.feed_forward2.linear1"), false, dt)?,
                    l2: w.linear(&format!("{p}.feed_forward2.linear2"), false, dt)?,
                },
                n_ff1: w.layer_norm(&format!("{p}.norm_feed_forward1"), dt)?,
                n_att: w.layer_norm(&format!("{p}.norm_self_att"), dt)?,
                n_conv: w.layer_norm(&format!("{p}.norm_conv"), dt)?,
                n_ff2: w.layer_norm(&format!("{p}.norm_feed_forward2"), dt)?,
                n_out: w.layer_norm(&format!("{p}.norm_out"), dt)?,
            });
        }

        // the prediction network and joint are tiny and run step by step: keep them in f32
        let f32 = DType::F32;
        let enc_proj = w.linear("encoder_projector", true, dt)?;
        let mut lw = Vec::new();
        let mut lb = Vec::new();
        for l in 0..2 {
            let wih = w.get(&format!("decoder.lstm.weight_ih_l{l}"), f32)?;
            let whh = w.get(&format!("decoder.lstm.weight_hh_l{l}"), f32)?;
            lw.push(Tensor::cat(&[&wih, &whh], 1)?.t()?.contiguous()?);
            let bih = w.get(&format!("decoder.lstm.bias_ih_l{l}"), f32)?;
            let bhh = w.get(&format!("decoder.lstm.bias_hh_l{l}"), f32)?;
            lb.push((bih + bhh)?);
        }
        let dec = Decoder {
            embedding: w.get("decoder.embedding.weight", f32)?,
            lstm: Lstm { w: lw, b: lb },
            proj: w.linear("decoder.decoder_projector", true, f32)?,
            head: w.linear("joint.head", true, f32)?,
        };
        if !w.map.is_empty() {
            let mut left: Vec<_> = w.map.keys().cloned().collect();
            left.sort();
            bail!("unused tensors in container: {left:?}");
        }
        let _ = &files.config;
        Ok(Self { sub, blocks, enc_proj, dec, device: device.clone(), dtype })
    }

    /// Encoder + joint encoder projection: ([B,T',640] f32, valid lengths).
    pub fn encode(&self, feats: &[&Features]) -> Result<(Tensor, Vec<usize>)> {
        let b = feats.len();
        let tmax = feats.iter().map(|f| f.frames).max().unwrap_or(0);
        let mut data = vec![0f32; b * tmax * N_MELS];
        for (i, f) in feats.iter().enumerate() {
            data[i * tmax * N_MELS..i * tmax * N_MELS + f.data.len()].copy_from_slice(&f.data);
        }
        let x = Tensor::from_vec(data, (b, tmax, N_MELS), &self.device)?.to_dtype(self.dtype)?;
        let lens: Vec<usize> = feats.iter().map(|f| f.valid).collect();
        let (mut x, lens) = self.sub.forward(&x, &lens)?;
        let t = x.dim(1)?;
        let pe = rel_pos_emb(t, self.dtype, &self.device)?;
        let km: Vec<f32> =
            lens.iter().flat_map(|&l| (0..t).map(move |i| if i < l { 0.0 } else { f32::NEG_INFINITY })).collect();
        let tm: Vec<f32> = lens.iter().flat_map(|&l| (0..t).map(move |i| if i < l { 1.0 } else { 0.0 })).collect();
        let masks = Masks {
            key: Tensor::from_vec(km, (b, 1, 1, t), &self.device)?.to_dtype(self.dtype)?,
            time: Tensor::from_vec(tm, (b, t, 1), &self.device)?.to_dtype(self.dtype)?,
            lens: lens.clone(),
        };
        for blk in &self.blocks {
            x = blk.forward(&x, &pe, &masks)?;
        }
        let enc = self.enc_proj.forward(&x)?.to_dtype(DType::F32)?;
        Ok((enc, lens))
    }

    /// Batched greedy TDT decoding over encoder outputs [B,T,640].
    pub fn greedy(&self, enc: &Tensor, lens: &[usize]) -> Result<Vec<Hyp>> {
        let dev = &self.device;
        let (b, tmax, _) = enc.dims3()?;
        let flat = enc.reshape((b * tmax, PRED))?;
        let mut hyps = vec![Hyp::default(); b];
        let mut rows: Vec<usize> = (0..b).filter(|&i| lens[i] > 0).collect();
        if rows.is_empty() {
            return Ok(hyps);
        }
        let mut t = vec![0usize; b];
        let mut sym = vec![0usize; b];
        let z = Tensor::zeros((rows.len(), PRED), DType::F32, dev)?;
        let zs = vec![z.clone(), z.clone()];
        let (mut g, mut h, mut c) = self.dec.step(&z, &zs, &zs)?;

        while !rows.is_empty() {
            let n = rows.len();
            let idx: Vec<u32> = rows.iter().map(|&r| (r * tmax + t[r]) as u32).collect();
            let f = flat.index_select(&Tensor::new(idx, dev)?, 0)?;
            let logits = self.dec.joint(&f, &g)?;
            let vocab = logits.narrow(1, 0, VOCAB + 1)?;
            let tok = vocab.argmax(1)?;
            let dur = logits.narrow(1, VOCAB + 1, DURATIONS.len())?.argmax(1)?;
            // probability of the argmax token: 1 / sum(exp(logit - max))
            let prob = vocab.broadcast_sub(&vocab.max_keepdim(1)?)?.exp()?.sum(1)?.recip()?;
            // one host sync per step: ids (< 2^24) travel as f32 next to the probabilities
            let all = Tensor::cat(&[&tok.to_dtype(DType::F32)?, &dur.to_dtype(DType::F32)?, &prob], 0)?.to_vec1::<f32>()?;
            let toks: Vec<u32> = all[..n].iter().map(|&x| x as u32).collect();
            let durs: Vec<u32> = all[n..2 * n].iter().map(|&x| x as u32).collect();
            let probs = &all[2 * n..];

            let mut emit = vec![0u8; n];
            for (j, &r) in rows.iter().enumerate() {
                let mut d = DURATIONS[durs[j] as usize];
                if toks[j] as usize != VOCAB {
                    hyps[r].tokens.push(toks[j]);
                    hyps[r].frames.push(t[r]);
                    hyps[r].probs.push(probs[j]);
                    emit[j] = 1;
                    sym[r] += 1;
                    if d == 0 && sym[r] >= MAX_SYMBOLS {
                        d = 1;
                    }
                } else if d == 0 {
                    d = 1;
                }
                if d > 0 {
                    sym[r] = 0;
                }
                t[r] += d;
            }
            if emit.iter().any(|&e| e == 1) {
                let x = self.dec.embedding.index_select(&tok, 0)?;
                let (g2, h2, c2) = self.dec.step(&x, &h, &c)?;
                let m = Tensor::from_vec(emit, (n, 1), dev)?.broadcast_as((n, PRED))?;
                g = m.where_cond(&g2, &g)?;
                for l in 0..h.len() {
                    h[l] = m.where_cond(&h2[l], &h[l])?;
                    c[l] = m.where_cond(&c2[l], &c[l])?;
                }
            }
            let keep: Vec<u32> = (0..n).filter(|&j| t[rows[j]] < lens[rows[j]]).map(|j| j as u32).collect();
            if keep.len() < n {
                rows = keep.iter().map(|&j| rows[j as usize]).collect();
                if rows.is_empty() {
                    break;
                }
                let ki = Tensor::new(keep, dev)?;
                g = g.index_select(&ki, 0)?;
                for l in 0..h.len() {
                    h[l] = h[l].index_select(&ki, 0)?;
                    c[l] = c[l].index_select(&ki, 0)?;
                }
            }
        }
        Ok(hyps)
    }
}
