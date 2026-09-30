//! Fused, rayon-parallel f32 kernels for the CPU path. candle's CPU elementwise and strided-copy ops run on one
//! thread, which left the encoder spending most of its time outside the (multi-threaded) matmuls. Each op here
//! replaces a chain of those passes; the CUDA path keeps the composed candle ops.

use candle_core::{CpuStorage, CustomOp1, CustomOp2, CustomOp3, Layout, Result, Shape, Tensor, bail};
use rayon::prelude::*;

fn f32s<'a>(s: &'a CpuStorage, l: &Layout) -> Result<&'a [f32]> {
    let d = s.as_slice::<f32>()?;
    match l.contiguous_offsets() {
        Some((a, b)) => Ok(&d[a..b]),
        None => bail!("cpu_ops: input must be contiguous"),
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

struct Silu;

impl CustomOp1 for Silu {
    fn name(&self) -> &'static str {
        "par-silu"
    }
    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let x = f32s(s, l)?;
        let mut out = vec![0f32; x.len()];
        out.par_chunks_mut(1 << 14).zip(x.par_chunks(1 << 14)).for_each(|(o, x)| {
            for (o, &v) in o.iter_mut().zip(x) {
                *o = v * sigmoid(v);
            }
        });
        Ok((CpuStorage::F32(out), l.shape().clone()))
    }
}

pub fn silu(x: &Tensor) -> Result<Tensor> {
    x.contiguous()?.apply_op1_no_bwd(&Silu)
}

/// a + s * b
struct AddScaled(f32);

impl CustomOp2 for AddScaled {
    fn name(&self) -> &'static str {
        "par-add-scaled"
    }
    fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout) -> Result<(CpuStorage, Shape)> {
        let (a, b) = (f32s(s1, l1)?, f32s(s2, l2)?);
        if a.len() != b.len() {
            bail!("add_scaled: shape mismatch");
        }
        let s = self.0;
        let mut out = vec![0f32; a.len()];
        out.par_chunks_mut(1 << 14).zip(a.par_chunks(1 << 14).zip(b.par_chunks(1 << 14))).for_each(|(o, (a, b))| {
            for i in 0..o.len() {
                o[i] = a[i] + s * b[i];
            }
        });
        Ok((CpuStorage::F32(out), l1.shape().clone()))
    }
}

pub fn add_scaled(a: &Tensor, b: &Tensor, s: f32) -> Result<Tensor> {
    a.contiguous()?.apply_op2_no_bwd(&b.contiguous()?, &AddScaled(s))
}

/// 4-D permutation that keeps the last dim in place, optionally adding a bias indexed by the input's last two
/// dims (the per-head `bias_u` / `bias_v`).
struct Permute4([usize; 3]);

impl Permute4 {
    fn run(&self, x: &[f32], dims: &[usize], bias: Option<&[f32]>) -> (Vec<f32>, Vec<usize>) {
        let p = self.0;
        let d3 = dims[3];
        let in_stride = [dims[1] * dims[2] * d3, dims[2] * d3, d3];
        let od = [dims[p[0]], dims[p[1]], dims[p[2]]];
        let mut out = vec![0f32; x.len()];
        out.par_chunks_mut(d3).enumerate().for_each(|(row, o)| {
            let o2 = row % od[2];
            let o1 = (row / od[2]) % od[1];
            let o0 = row / (od[2] * od[1]);
            let mut ic = [0usize; 3];
            ic[p[0]] = o0;
            ic[p[1]] = o1;
            ic[p[2]] = o2;
            let off = ic[0] * in_stride[0] + ic[1] * in_stride[1] + ic[2] * in_stride[2];
            o.copy_from_slice(&x[off..off + d3]);
            if let Some(b) = bias {
                let bo = ic[2] * d3;
                for (v, bb) in o.iter_mut().zip(&b[bo..bo + d3]) {
                    *v += bb;
                }
            }
        });
        (out, vec![od[0], od[1], od[2], d3])
    }
}

impl CustomOp1 for Permute4 {
    fn name(&self) -> &'static str {
        "par-permute4"
    }
    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let (out, shape) = self.run(f32s(s, l)?, l.dims(), None);
        Ok((CpuStorage::F32(out), Shape::from(shape)))
    }
}

impl CustomOp2 for Permute4 {
    fn name(&self) -> &'static str {
        "par-permute4-bias"
    }
    fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout) -> Result<(CpuStorage, Shape)> {
        let (out, shape) = self.run(f32s(s1, l1)?, l1.dims(), Some(f32s(s2, l2)?));
        Ok((CpuStorage::F32(out), Shape::from(shape)))
    }
}

/// Permute a contiguous 4-D tensor by `perm` (with perm[3] == 3), adding `bias` [dims[2], dims[3]] if given.
pub fn permute4(x: &Tensor, perm: [usize; 4], bias: Option<&Tensor>) -> Result<Tensor> {
    if perm[3] != 3 {
        bail!("permute4 keeps the last dim");
    }
    let op = Permute4([perm[0], perm[1], perm[2]]);
    let x = x.contiguous()?;
    match bias {
        Some(b) => x.apply_op2_no_bwd(&b.contiguous()?, &op),
        None => x.apply_op1_no_bwd(&op),
    }
}

/// softmax((ac + rel_shift(bd)) * scale) over valid keys. ac [B,H,T,T]; bd [H,B,T,2T-1] (positions T-1..-(T-1)).
struct RelAttnSoftmax {
    scale: f32,
    lens: Vec<usize>,
}

impl CustomOp2 for RelAttnSoftmax {
    fn name(&self) -> &'static str {
        "rel-attn-softmax"
    }
    fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout) -> Result<(CpuStorage, Shape)> {
        let (ac, bd) = (f32s(s1, l1)?, f32s(s2, l2)?);
        let &[b, h, t, _] = l1.dims() else { bail!("ac must be 4-D") };
        let p = 2 * t - 1;
        let mut out = vec![0f32; ac.len()];
        out.par_chunks_mut(t).enumerate().for_each(|(row, o)| {
            let i = row % t;
            let hh = (row / t) % h;
            let bb = row / (t * h);
            let len = self.lens[bb].clamp(1, t);
            let a = &ac[row * t..(row + 1) * t];
            let boff = ((hh * b + bb) * t + i) * p + (t - 1 - i);
            let d = &bd[boff..boff + t];
            let mut m = f32::NEG_INFINITY;
            for j in 0..len {
                let v = (a[j] + d[j]) * self.scale;
                o[j] = v;
                m = m.max(v);
            }
            let mut sum = 0f32;
            for v in &mut o[..len] {
                *v = (*v - m).exp();
                sum += *v;
            }
            let inv = 1.0 / sum;
            for v in &mut o[..len] {
                *v *= inv;
            }
        });
        Ok((CpuStorage::F32(out), l1.shape().clone()))
    }
}

pub fn rel_attn_softmax(ac: &Tensor, bd: &Tensor, scale: f32, lens: &[usize]) -> Result<Tensor> {
    ac.contiguous()?.apply_op2_no_bwd(&bd.contiguous()?, &RelAttnSoftmax { scale, lens: lens.to_vec() })
}

/// Conformer conv module middle: silu(depthwise_conv(glu(h) * mask) + bias), BatchNorm already folded.
/// h [B,T,2C], w [K,C], bias [C] -> [B,T,C].
struct GluDwConv {
    lens: Vec<usize>,
}

impl CustomOp3 for GluDwConv {
    fn name(&self) -> &'static str {
        "glu-dwconv-silu"
    }
    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (h, w, bias) = (f32s(s1, l1)?, f32s(s2, l2)?, f32s(s3, l3)?);
        let &[b, t, c2] = l1.dims() else { bail!("h must be 3-D") };
        let c = c2 / 2;
        let k = w.len() / c;
        let pad = (k - 1) / 2;
        let mut g = vec![0f32; b * t * c];
        g.par_chunks_mut(c).enumerate().for_each(|(row, o)| {
            if row % t < self.lens[row / t] {
                let src = &h[row * c2..(row + 1) * c2];
                for ch in 0..c {
                    o[ch] = src[ch] * sigmoid(src[c + ch]);
                }
            }
        });
        let mut out = vec![0f32; b * t * c];
        out.par_chunks_mut(c).enumerate().for_each(|(row, o)| {
            let (bb, tt) = (row / t, row % t);
            o.copy_from_slice(bias);
            for kk in 0..k {
                let src_t = tt as isize + kk as isize - pad as isize;
                if src_t < 0 || src_t >= t as isize {
                    continue;
                }
                let src = &g[(bb * t + src_t as usize) * c..][..c];
                let wk = &w[kk * c..(kk + 1) * c];
                for ch in 0..c {
                    o[ch] += wk[ch] * src[ch];
                }
            }
            for v in o.iter_mut() {
                *v *= sigmoid(*v);
            }
        });
        Ok((CpuStorage::F32(out), Shape::from((b, t, c))))
    }
}

pub fn glu_dwconv_silu(h: &Tensor, w: &Tensor, bias: &Tensor, lens: &[usize]) -> Result<Tensor> {
    h.contiguous()?.apply_op3_no_bwd(&w.contiguous()?, &bias.contiguous()?, &GluDwConv { lens: lens.to_vec() })
}

/// Depthwise 3x3 / stride 2 / pad 1 conv + bias over [B,C,H,W], rows at or past `lens[b]` (along H) zeroed.
/// w [C,9], bias [C].
struct DwConv2dS2 {
    lens: Vec<usize>,
}

impl CustomOp3 for DwConv2dS2 {
    fn name(&self) -> &'static str {
        "dwconv2d-s2"
    }
    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (x, w, bias) = (f32s(s1, l1)?, f32s(s2, l2)?, f32s(s3, l3)?);
        let &[b, c, h, wd] = l1.dims() else { bail!("x must be 4-D") };
        let (ho, wo) = ((h - 1) / 2 + 1, (wd - 1) / 2 + 1);
        let mut out = vec![0f32; b * c * ho * wo];
        out.par_chunks_mut(ho * wo).enumerate().for_each(|(bc, o)| {
            let (bb, ch) = (bc / c, bc % c);
            let src = &x[bc * h * wd..(bc + 1) * h * wd];
            let k = &w[ch * 9..ch * 9 + 9];
            for oy in 0..ho.min(self.lens[bb]) {
                for ox in 0..wo {
                    let mut acc = bias[ch];
                    for ky in 0..3 {
                        let y = (2 * oy + ky) as isize - 1;
                        if y < 0 || y >= h as isize {
                            continue;
                        }
                        for kx in 0..3 {
                            let xx = (2 * ox + kx) as isize - 1;
                            if xx < 0 || xx >= wd as isize {
                                continue;
                            }
                            acc += k[ky * 3 + kx] * src[y as usize * wd + xx as usize];
                        }
                    }
                    o[oy * wo + ox] = acc;
                }
            }
        });
        Ok((CpuStorage::F32(out), Shape::from((b, c, ho, wo))))
    }
}

pub fn dwconv2d_s2(x: &Tensor, w: &Tensor, bias: &Tensor, lens: &[usize]) -> Result<Tensor> {
    x.contiguous()?.apply_op3_no_bwd(&w.contiguous()?, &bias.contiguous()?, &DwConv2dS2 { lens: lens.to_vec() })
}

/// relu((x + bias[c]) * mask) over [B,C,H,W], rows at or past `lens[b]` (along H) zeroed.
struct BiasMaskRelu {
    lens: Vec<usize>,
}

impl CustomOp2 for BiasMaskRelu {
    fn name(&self) -> &'static str {
        "bias-mask-relu"
    }
    fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout) -> Result<(CpuStorage, Shape)> {
        let (x, bias) = (f32s(s1, l1)?, f32s(s2, l2)?);
        let &[_, c, h, w] = l1.dims() else { bail!("x must be 4-D") };
        let mut out = vec![0f32; x.len()];
        out.par_chunks_mut(h * w).zip(x.par_chunks(h * w)).enumerate().for_each(|(bc, (o, x))| {
            let (bb, ch) = (bc / c, bc % c);
            let n = self.lens[bb].min(h) * w;
            for i in 0..n {
                o[i] = (x[i] + bias[ch]).max(0.0);
            }
        });
        Ok((CpuStorage::F32(out), l1.shape().clone()))
    }
}

pub fn bias_mask_relu(x: &Tensor, bias: &Tensor, lens: &[usize]) -> Result<Tensor> {
    x.contiguous()?.apply_op2_no_bwd(&bias.flatten_all()?, &BiasMaskRelu { lens: lens.to_vec() })
}
