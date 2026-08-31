//! Model loading and batched transcription shared by the `phonon` and `phonon-dictate` binaries.

use crate::text::{self, Word};
use crate::{audio, candle_model, fermion, mel};
use anyhow::{Context, Result};
use candle_core::{DType, Device};
#[cfg(feature = "cli")]
use clap::ValueEnum;
use rayon::prelude::*;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "cli", derive(ValueEnum))]
pub enum DeviceArg {
    Auto,
    Cpu,
    Cuda,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "cli", derive(ValueEnum))]
pub enum Precision {
    F32,
    F16,
    Bf16,
}

pub struct Engine {
    model: candle_model::Phonon,
    vocab: text::Vocab,
    mel: mel::Mel,
    pub cuda: bool,
    pub precision: Precision,
}

/// One rayon worker per physical core: SMT siblings slow the f32 GEMMs down rather than up.
pub fn init_threads(threads: Option<usize>) -> usize {
    let n = threads.unwrap_or_else(num_cpus::get_physical);
    rayon::ThreadPoolBuilder::new().num_threads(n).build_global().ok();
    // candle's CPU matmuls size their pool from this
    unsafe { std::env::set_var("RAYON_NUM_THREADS", n.to_string()) };
    n
}

/// `$PHONON_MODEL`-less default: the unpacked release next to the working dir or the binary, or the archive.
pub fn default_model() -> Result<PathBuf> {
    let mut roots = vec![PathBuf::from("."), PathBuf::from("..")];
    if let Some(dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(|p| p.to_path_buf())) {
        // target/release/phonon -> the repo root is three levels up
        roots.extend(dir.ancestors().take(4).map(|p| p.to_path_buf()));
    }
    let names = ["model_phonon2_c4c_int6", "phonon-2.bps.tar.zst"];
    roots
        .iter()
        .flat_map(|r| names.iter().map(move |n| r.join(n)))
        .find(|p| p.exists())
        .context("no model found; pass --model or set PHONON_MODEL")
}

impl Engine {
    pub fn load(model: Option<PathBuf>, device: DeviceArg, dtype: Option<Precision>) -> Result<Self> {
        let cuda = match device {
            DeviceArg::Cpu => false,
            DeviceArg::Cuda => true,
            DeviceArg::Auto => candle_core::utils::cuda_is_available(),
        };
        let precision = dtype.unwrap_or(if cuda { Precision::F16 } else { Precision::F32 });
        let dev = if cuda { Device::new_cuda(0).context("CUDA device (build with --features cuda)")? } else { Device::Cpu };
        let dt = match precision {
            Precision::F32 => DType::F32,
            Precision::F16 => DType::F16,
            Precision::Bf16 => DType::BF16,
        };
        let path = match model {
            Some(p) => p,
            None => default_model()?,
        };
        let files = fermion::load_files(&path)?;
        Ok(Self {
            vocab: text::Vocab::from_config(&files.config)?,
            model: candle_model::Phonon::load(&files, &dev, dt)?,
            mel: mel::Mel::new(),
            cuda,
            precision,
        })
    }

    /// Words per chunk, in input order.
    pub fn run(&self, chunks: &[audio::Chunk], batch_size: usize) -> Result<Vec<Vec<Word>>> {
        let feats: Vec<mel::Features> = chunks.par_iter().map(|c| self.mel.compute(&c.samples)).collect();
        // longest first so each batch holds similar lengths
        let mut order: Vec<usize> = (0..chunks.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(feats[i].frames));
        let mut out = vec![Vec::new(); chunks.len()];
        for batch in order.chunks(batch_size.max(1)) {
            let fs: Vec<&mel::Features> = batch.iter().map(|&i| &feats[i]).collect();
            let t = Instant::now();
            let (enc, lens) = self.model.encode(&fs)?;
            enc.device().synchronize()?;
            let te = t.elapsed();
            let hyps = self.model.greedy(&enc, &lens)?;
            log::debug!("batch of {}: encoder {te:.2?}, decoder {:.2?}", fs.len(), t.elapsed() - te);
            for (&i, h) in batch.iter().zip(hyps) {
                let offset = chunks[i].offset as f32 / audio::SAMPLE_RATE as f32;
                out[i] = self.vocab.words(&h.tokens, &h.frames, &h.probs, offset);
            }
        }
        Ok(out)
    }

    /// Transcribe one stretch of 16 kHz mono audio of any length (cut into `chunk_secs` pieces at quiet points).
    pub fn transcribe(&self, samples: &[f32], chunk_secs: f32) -> Result<Vec<Word>> {
        let chunks = audio::chunk(0, samples, chunk_secs);
        Ok(self.run(&chunks, 16)?.into_iter().flatten().collect())
    }
}
