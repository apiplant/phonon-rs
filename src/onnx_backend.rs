//! ONNX Runtime backend through transcribe-rs's Parakeet engine. Needs the directory written by
//! `scripts/export_onnx.py` (encoder-model.onnx, decoder_joint-model.onnx, nemo128.onnx, vocab.txt).

use crate::text::Word;
use anyhow::{Result, anyhow};
use std::path::Path;
use transcribe_rs::onnx::Quantization;
use transcribe_rs::onnx::parakeet::{ParakeetModel, ParakeetParams, TimestampGranularity};
use transcribe_rs::{OrtAccelerator, set_ort_accelerator};

pub struct OnnxPhonon {
    model: ParakeetModel,
}

impl OnnxPhonon {
    pub fn load(dir: &Path, cuda: bool, fp16: bool) -> Result<Self> {
        set_ort_accelerator(if cuda { OrtAccelerator::Cuda } else { OrtAccelerator::CpuOnly });
        let q = if fp16 { Quantization::FP16 } else { Quantization::FP32 };
        let model = ParakeetModel::load(dir, &q).map_err(|e| anyhow!("loading ONNX model from {}: {e}", dir.display()))?;
        Ok(Self { model })
    }

    pub fn transcribe(&mut self, samples: &[f32], offset: f32) -> Result<Vec<Word>> {
        let params = ParakeetParams { timestamp_granularity: Some(TimestampGranularity::Word), ..Default::default() };
        let r = self.model.transcribe_with(samples, &params).map_err(|e| anyhow!("{e}"))?;
        Ok(r.segments
            .unwrap_or_default()
            .into_iter()
            .map(|s| Word { word: s.text, start: s.start + offset, end: s.end + offset })
            .collect())
    }
}
