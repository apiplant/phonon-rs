//! Reader for `fermion-five-value-parakeet-v1` containers (port of `fermion_container.py`).
//!
//! Layout: u64 LE header length, JSON header `{format, index: [{n, k, shape, b}]}`, then each record's blob in
//! index order. Record kinds:
//! - `five_value` [O, I]: base-3 trits (5 per byte, code-1 = sign), then one bit per non-zero choosing the row's
//!   `hi` over its `lo` magnitude, then fp16 `lo[O]`, fp16 `hi[O]`. Emitted as `<name>.weight`.
//! - `int6` / `int8`: per-row symmetric integers (6-bit packed four to three bytes, offset 32), fp16 row scales.
//! - `fp16`: raw.

use anyhow::{Context, Result, bail};
use half::f16;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

pub const FORMAT: &str = "fermion-five-value-parakeet-v1";

#[derive(Deserialize)]
struct Header {
    format: String,
    index: Vec<Entry>,
}

#[derive(Deserialize)]
struct Entry {
    n: String,
    k: String,
    shape: Vec<usize>,
    b: usize,
}

pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Everything the runtime needs from a model directory or release archive.
pub struct ModelFiles {
    pub container: Vec<u8>,
    pub config: serde_json::Value,
}

/// Load from a directory holding `model.fermion` + `config.json`, a `model.fermion` path, or the release
/// `phonon-2.bps.tar.zst` archive.
pub fn load_files(path: &Path) -> Result<ModelFiles> {
    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    if path.is_file() && (name.ends_with(".tar.zst") || name.ends_with(".tzst")) {
        let dec = zstd::stream::Decoder::new(std::fs::File::open(path)?)?;
        let mut ar = tar::Archive::new(dec);
        let (mut container, mut config) = (None, None);
        for e in ar.entries()? {
            let mut e = e?;
            let p = e.path()?.to_string_lossy().into_owned();
            let mut buf = Vec::with_capacity(e.size() as usize);
            match p.rsplit('/').next().unwrap_or("") {
                "model.fermion" => {
                    e.read_to_end(&mut buf)?;
                    container = Some(buf);
                }
                "config.json" => {
                    e.read_to_end(&mut buf)?;
                    config = Some(serde_json::from_slice(&buf)?);
                }
                _ => {}
            }
        }
        return Ok(ModelFiles {
            container: container.context("archive has no model.fermion")?,
            config: config.context("archive has no config.json")?,
        });
    }
    let (container_path, dir) = if path.is_dir() {
        (path.join("model.fermion"), path.to_path_buf())
    } else {
        (path.to_path_buf(), path.parent().unwrap_or(Path::new(".")).to_path_buf())
    };
    let container =
        std::fs::read(&container_path).with_context(|| format!("reading {}", container_path.display()))?;
    let cfg_path = dir.join("config.json");
    let config = serde_json::from_slice(&std::fs::read(&cfg_path).with_context(|| format!("reading {}", cfg_path.display()))?)?;
    Ok(ModelFiles { container, config })
}

/// Expand every record to f32, keyed by its HF Transformers (ParakeetForTDT) name.
pub fn read_container(buf: &[u8]) -> Result<HashMap<String, Tensor>> {
    if buf.len() < 8 {
        bail!("container too short");
    }
    let n = u64::from_le_bytes(buf[..8].try_into()?) as usize;
    let header: Header = serde_json::from_slice(&buf[8..8 + n])?;
    if header.format != FORMAT {
        bail!("unexpected container format {:?}", header.format);
    }
    let mut records = Vec::with_capacity(header.index.len());
    let mut off = 8 + n;
    for e in &header.index {
        let blob = buf.get(off..off + e.b).with_context(|| format!("truncated record {}", e.n))?;
        records.push((e, blob));
        off += e.b;
    }
    if off != buf.len() {
        bail!("trailing bytes in container");
    }
    records
        .into_par_iter()
        .map(|(e, blob)| {
            let t = match e.k.as_str() {
                "five_value" => (format!("{}.weight", e.n), five_value(blob, &e.shape)?),
                "int6" => (e.n.clone(), intn(blob, &e.shape, 6)?),
                "int8" => (e.n.clone(), intn(blob, &e.shape, 8)?),
                "fp16" => (e.n.clone(), Tensor { shape: e.shape.clone(), data: f16s(blob).collect() }),
                k => bail!("unknown record kind {k} for {}", e.n),
            };
            Ok(t)
        })
        .collect()
}

fn f16s(b: &[u8]) -> impl Iterator<Item = f32> + '_ {
    b.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
}

fn five_value(blob: &[u8], shape: &[usize]) -> Result<Tensor> {
    let (o, i) = (shape[0], shape[1]);
    let rb = i.div_ceil(5);
    let trits = &blob[..o * rb];
    let nnz: usize = trits
        .iter()
        .enumerate()
        .map(|(idx, &byte)| {
            let valid = (i - (idx % rb) * 5).min(5);
            let mut x = byte;
            (0..valid)
                .filter(|_| {
                    let d = x % 3;
                    x /= 3;
                    d != 1
                })
                .count()
        })
        .sum();
    let rbytes = nnz.div_ceil(8);
    let bits = &blob[o * rb..o * rb + rbytes];
    let scales = &blob[o * rb + rbytes..];
    if scales.len() != 4 * o {
        bail!("five_value record size mismatch");
    }
    let lo: Vec<f32> = f16s(&scales[..2 * o]).collect();
    let hi: Vec<f32> = f16s(&scales[2 * o..]).collect();

    let mut data = vec![0f32; o * i];
    let mut bit = 0usize;
    for r in 0..o {
        let row = &mut data[r * i..(r + 1) * i];
        for (c, &byte) in trits[r * rb..(r + 1) * rb].iter().enumerate() {
            let mut x = byte;
            for k in 0..5 {
                let col = c * 5 + k;
                if col >= i {
                    break;
                }
                let d = x % 3;
                x /= 3;
                if d != 1 {
                    let is_hi = (bits[bit / 8] >> (bit % 8)) & 1 == 1;
                    bit += 1;
                    let mag = if is_hi { hi[r] } else { lo[r] };
                    row[col] = if d == 2 { mag } else { -mag };
                }
            }
        }
    }
    Ok(Tensor { shape: shape.to_vec(), data })
}

fn intn(blob: &[u8], shape: &[usize], bits: u32) -> Result<Tensor> {
    let o = shape[0];
    let total: usize = shape.iter().product();
    let ncol = total / o;
    let (body, sc) = blob.split_at(blob.len() - 2 * o);
    let scales: Vec<f32> = f16s(sc).collect();
    let q: Vec<i32> = match bits {
        8 => body.iter().take(total).map(|&b| b as i8 as i32).collect(),
        6 => body
            .chunks_exact(3)
            .flat_map(|b| {
                let p = b[0] as u32 | (b[1] as u32) << 8 | (b[2] as u32) << 16;
                [0, 6, 12, 18].map(|s| ((p >> s) & 0x3f) as i32 - 32)
            })
            .take(total)
            .collect(),
        _ => bail!("unsupported int{bits}"),
    };
    if q.len() != total {
        bail!("int{bits} record size mismatch");
    }
    let data = q.iter().enumerate().map(|(j, &v)| v as f32 * scales[j / ncol]).collect();
    Ok(Tensor { shape: shape.to_vec(), data })
}
