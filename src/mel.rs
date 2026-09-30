//! Log-mel frontend matching HF `ParakeetFeatureExtractor` (NeMo `AudioToMelSpectrogramPreprocessor` without
//! dither): preemphasis 0.97, centred 512-point STFT with a symmetric 400-sample Hann window and hop 160, power
//! spectrum, 128 Slaney mel bands, log(x + 2^-24), then per-band normalisation over the valid frames.

use rayon::prelude::*;
use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;

pub const N_MELS: usize = 128;
const N_FFT: usize = 512;
const HOP: usize = 160;
const WIN: usize = 400;
const N_BINS: usize = N_FFT / 2 + 1;
const PREEMPH: f32 = 0.97;
const LOG_GUARD: f32 = 5.960_464_5e-8; // 2^-24
const EPS: f32 = 1e-5;

pub struct Mel {
    filters: Vec<f32>, // [N_MELS, N_BINS]
    window: Vec<f32>,  // [N_FFT], the 400-point window centred in 512
    fft: Arc<dyn RealToComplex<f32>>,
}

/// Normalised features `[frames, N_MELS]` and the number of valid frames (the last frame is padding).
pub struct Features {
    pub data: Vec<f32>,
    pub frames: usize,
    pub valid: usize,
}

impl Default for Mel {
    fn default() -> Self {
        Self::new()
    }
}

impl Mel {
    pub fn new() -> Self {
        let mut window = vec![0f32; N_FFT];
        let off = (N_FFT - WIN) / 2;
        for n in 0..WIN {
            window[off + n] = (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / (WIN - 1) as f64).cos()) as f32;
        }
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(N_FFT);
        Self { filters: slaney_mel(16_000.0, N_FFT, N_MELS), window, fft }
    }

    pub fn compute(&self, samples: &[f32]) -> Features {
        let len = samples.len();
        // preemphasis, then centre padding of n_fft/2 zeros on each side
        let mut padded = vec![0f32; len + N_FFT];
        for i in 0..len {
            padded[N_FFT / 2 + i] = if i == 0 { samples[0] } else { samples[i] - PREEMPH * samples[i - 1] };
        }
        let frames = 1 + len / HOP;
        let valid = len / HOP;

        let mut data = vec![0f32; frames * N_MELS];
        data.par_chunks_mut(N_MELS * 64).enumerate().for_each(|(blk, out)| {
            let mut input = self.fft.make_input_vec();
            let mut spec = self.fft.make_output_vec();
            let mut scratch = self.fft.make_scratch_vec();
            let mut power = [0f32; N_BINS];
            for (j, row) in out.chunks_mut(N_MELS).enumerate() {
                let start = (blk * 64 + j) * HOP;
                for (k, x) in input.iter_mut().enumerate() {
                    *x = padded[start + k] * self.window[k];
                }
                self.fft.process_with_scratch(&mut input, &mut spec, &mut scratch).unwrap();
                for (p, c) in power.iter_mut().zip(&spec) {
                    *p = c.re * c.re + c.im * c.im;
                }
                for (m, v) in row.iter_mut().enumerate() {
                    let f = &self.filters[m * N_BINS..(m + 1) * N_BINS];
                    let s: f32 = f.iter().zip(&power).map(|(a, b)| a * b).sum();
                    *v = (s + LOG_GUARD).ln();
                }
            }
        });

        // per-band mean / unbiased std over the valid frames; padding frames are zeroed
        let n = valid.max(1) as f32;
        let mut mean = [0f32; N_MELS];
        let mut var = [0f32; N_MELS];
        for row in data.chunks(N_MELS).take(valid) {
            for (m, v) in row.iter().enumerate() {
                mean[m] += v;
            }
        }
        mean.iter_mut().for_each(|m| *m /= n);
        for row in data.chunks(N_MELS).take(valid) {
            for (m, v) in row.iter().enumerate() {
                var[m] += (v - mean[m]).powi(2);
            }
        }
        let denom = (valid.max(2) - 1) as f32;
        let inv: Vec<f32> = var.iter().map(|v| 1.0 / ((v / denom).sqrt() + EPS)).collect();
        for (t, row) in data.chunks_mut(N_MELS).enumerate() {
            for (m, v) in row.iter_mut().enumerate() {
                *v = if t < valid { (*v - mean[m]) * inv[m] } else { 0.0 };
            }
        }
        Features { data, frames, valid }
    }
}

/// librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr/2, htk=False, norm="slaney")
fn slaney_mel(sr: f64, n_fft: usize, n_mels: usize) -> Vec<f32> {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    let min_log_mel = MIN_LOG_HZ / F_SP;
    let logstep = 6.4f64.ln() / 27.0;
    let hz_to_mel = |f: f64| if f >= MIN_LOG_HZ { min_log_mel + (f / MIN_LOG_HZ).ln() / logstep } else { f / F_SP };
    let mel_to_hz = |m: f64| if m >= min_log_mel { MIN_LOG_HZ * ((m - min_log_mel) * logstep).exp() } else { m * F_SP };

    let n_bins = n_fft / 2 + 1;
    let fftfreqs: Vec<f64> = (0..n_bins).map(|k| k as f64 * sr / n_fft as f64).collect();
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(sr / 2.0));
    let mel_f: Vec<f64> =
        (0..n_mels + 2).map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (n_mels + 1) as f64)).collect();

    let mut w = vec![0f32; n_mels * n_bins];
    for i in 0..n_mels {
        let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
        let (d0, d1) = (mel_f[i + 1] - mel_f[i], mel_f[i + 2] - mel_f[i + 1]);
        for (k, &f) in fftfreqs.iter().enumerate() {
            let lower = -(mel_f[i] - f) / d0;
            let upper = (mel_f[i + 2] - f) / d1;
            let v = lower.min(upper).max(0.0) as f32;
            w[i * n_bins + k] = v * enorm as f32;
        }
    }
    w
}
