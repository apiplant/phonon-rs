mod audio;
mod candle_model;
mod cpu_ops;
mod fermion;
mod mel;
#[cfg(feature = "onnx")]
mod onnx_backend;
mod text;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device};
use clap::{Parser, ValueEnum};
use rayon::prelude::*;
use serde::Serialize;
use std::path::PathBuf;
use std::time::Instant;
use text::Word;

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum, Debug)]
enum Backend {
    /// Native Rust graph on candle; reads model.fermion (or the .tar.zst release) directly
    Candle,
    /// ONNX Runtime via transcribe-rs; reads a directory made by scripts/export_onnx.py
    Onnx,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum, Debug)]
enum DeviceArg {
    Auto,
    Cpu,
    Cuda,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum, Debug)]
enum Precision {
    F32,
    F16,
    Bf16,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum, Debug)]
enum Format {
    Text,
    Json,
    Srt,
}

/// Transcribe English speech with Phonon-2 (five-value Parakeet-TDT-0.6B-v3) on CPU or CUDA.
#[derive(Parser, Debug)]
#[command(name = "phonon", version)]
struct Args {
    /// Audio files (wav, flac, mp3, ogg/vorbis, m4a/aac, ...); any sample rate, mixed to mono
    #[arg(required = true)]
    files: Vec<PathBuf>,

    /// Model: for candle a dir with model.fermion + config.json, a model.fermion file, or phonon-2.bps.tar.zst;
    /// for onnx the exported ONNX dir
    #[arg(short, long, env = "PHONON_MODEL")]
    model: Option<PathBuf>,

    #[arg(short, long, value_enum, default_value_t = Backend::Candle)]
    backend: Backend,

    #[arg(short, long, value_enum, default_value_t = DeviceArg::Auto)]
    device: DeviceArg,

    /// Encoder precision (candle default: f32 on CPU, f16 on CUDA; onnx default f32, f16 loads encoder-model.fp16.onnx)
    #[arg(long, value_enum)]
    dtype: Option<Precision>,

    /// Longest piece of audio run through the encoder at once; longer files are cut at quiet points
    #[arg(long, default_value_t = 30.0)]
    chunk_secs: f32,

    /// Chunks per encoder batch (candle)
    #[arg(long, default_value_t = 16)]
    batch_size: usize,

    #[arg(short, long, value_enum, default_value_t = Format::Text)]
    format: Format,

    /// CPU threads (default: number of physical cores)
    #[arg(long)]
    threads: Option<usize>,

    /// Print load / decode timings and real-time factor to stderr
    #[arg(long)]
    bench: bool,

    /// Run one untimed pass over the first chunk first (CUDA kernel/cuBLAS warm-up)
    #[arg(long)]
    warmup: bool,
}

#[derive(Serialize)]
struct FileResult {
    file: String,
    duration: f32,
    text: String,
    words: Vec<Word>,
}

enum Engine {
    Candle { model: candle_model::Phonon, vocab: text::Vocab, mel: mel::Mel },
    #[cfg(feature = "onnx")]
    Onnx(onnx_backend::OnnxPhonon),
}

impl Engine {
    /// Words per chunk, in input order.
    fn run(&mut self, chunks: &[audio::Chunk], batch_size: usize) -> Result<Vec<Vec<Word>>> {
        let offset = |c: &audio::Chunk| c.offset as f32 / audio::SAMPLE_RATE as f32;
        match self {
            Engine::Candle { model, vocab, mel } => {
                let feats: Vec<mel::Features> = chunks.par_iter().map(|c| mel.compute(&c.samples)).collect();
                // longest first so each batch holds similar lengths
                let mut order: Vec<usize> = (0..chunks.len()).collect();
                order.sort_by_key(|&i| std::cmp::Reverse(feats[i].frames));
                let mut out = vec![Vec::new(); chunks.len()];
                for batch in order.chunks(batch_size.max(1)) {
                    let fs: Vec<&mel::Features> = batch.iter().map(|&i| &feats[i]).collect();
                    let t = Instant::now();
                    let (enc, lens) = model.encode(&fs)?;
                    enc.device().synchronize()?;
                    let te = t.elapsed();
                    let hyps = model.greedy(&enc, &lens)?;
                    log::debug!("batch of {}: encoder {te:.2?}, decoder {:.2?}", fs.len(), t.elapsed() - te);
                    for (&i, h) in batch.iter().zip(hyps) {
                        out[i] = vocab.words(&h.tokens, &h.frames, offset(&chunks[i]));
                    }
                }
                Ok(out)
            }
            #[cfg(feature = "onnx")]
            Engine::Onnx(m) => chunks.iter().map(|c| m.transcribe(&c.samples, offset(c))).collect(),
        }
    }
}

fn default_model(backend: Backend) -> Result<PathBuf> {
    let candidates: &[&str] = match backend {
        Backend::Candle => &[
            "model_phonon2_c4c_int6",
            "../model_phonon2_c4c_int6",
            "phonon-2.bps.tar.zst",
            "../phonon-2.bps.tar.zst",
        ],
        Backend::Onnx => &["phonon-2-onnx", "../phonon-2-onnx"],
    };
    candidates
        .iter()
        .map(PathBuf::from)
        .find(|p| p.exists())
        .with_context(|| format!("no model found (tried {candidates:?}); pass --model or set PHONON_MODEL"))
}

fn srt_time(t: f32) -> String {
    let ms = (t.max(0.0) * 1000.0).round() as u64;
    format!("{:02}:{:02}:{:02},{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
}

fn srt(words: &[Word]) -> String {
    // cues end at sentence punctuation, a pause over 1 s, or ~7 s / 16 words
    let mut out = String::new();
    let mut cue: Vec<&Word> = Vec::new();
    let mut n = 0;
    let mut flush = |cue: &mut Vec<&Word>, out: &mut String| {
        if let (Some(a), Some(b)) = (cue.first(), cue.last()) {
            n += 1;
            let text = cue.iter().map(|w| w.word.as_str()).collect::<Vec<_>>().join(" ");
            out.push_str(&format!("{n}\n{} --> {}\n{text}\n\n", srt_time(a.start), srt_time(b.end)));
        }
        cue.clear();
    };
    for (i, w) in words.iter().enumerate() {
        cue.push(w);
        let next_gap = words.get(i + 1).map(|nx| nx.start - w.end).unwrap_or(0.0);
        let span = w.end - cue[0].start;
        if w.word.ends_with(['.', '?', '!']) || next_gap > 1.0 || span > 7.0 || cue.len() >= 16 {
            flush(&mut cue, &mut out);
        }
    }
    flush(&mut cue, &mut out);
    out
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let a = Args::parse();
    // One thread per physical core: SMT siblings slow the f32 GEMMs down rather than up.
    let threads = a.threads.unwrap_or_else(num_cpus::get_physical);
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().ok();
    // candle's CPU matmuls size their pool from this
    unsafe { std::env::set_var("RAYON_NUM_THREADS", threads.to_string()) };

    let cuda = match a.device {
        DeviceArg::Cpu => false,
        DeviceArg::Cuda => true,
        DeviceArg::Auto => candle_core::utils::cuda_is_available(),
    };
    let model_path = match &a.model {
        Some(p) => p.clone(),
        None => default_model(a.backend)?,
    };

    let precision = a.dtype.unwrap_or(if cuda && a.backend == Backend::Candle { Precision::F16 } else { Precision::F32 });
    let t0 = Instant::now();
    let mut engine = match a.backend {
        Backend::Candle => {
            let device = if cuda { Device::new_cuda(0).context("CUDA device (build with --features cuda)")? } else { Device::Cpu };
            let dtype = match precision {
                Precision::F32 => DType::F32,
                Precision::F16 => DType::F16,
                Precision::Bf16 => DType::BF16,
            };
            let files = fermion::load_files(&model_path)?;
            let vocab = text::Vocab::from_config(&files.config)?;
            let model = candle_model::Phonon::load(&files, &device, dtype)?;
            Engine::Candle { model, vocab, mel: mel::Mel::new() }
        }
        #[cfg(feature = "onnx")]
        Backend::Onnx => {
            if precision == Precision::Bf16 {
                bail!("the onnx backend supports --dtype f32 or f16");
            }
            Engine::Onnx(onnx_backend::OnnxPhonon::load(&model_path, cuda, precision == Precision::F16)?)
        }
        #[cfg(not(feature = "onnx"))]
        Backend::Onnx => bail!("this build has no onnx backend (build with --features onnx)"),
    };
    let load_s = t0.elapsed().as_secs_f32();

    let t1 = Instant::now();
    let audios: Vec<Vec<f32>> = a.files.par_iter().map(|p| audio::load(p)).collect::<Result<_>>()?;
    let chunks: Vec<audio::Chunk> =
        audios.iter().enumerate().flat_map(|(i, s)| audio::chunk(i, s, a.chunk_secs)).collect();
    let audio_s = t1.elapsed().as_secs_f32();

    if a.warmup {
        if let Some(c) = chunks.first() {
            let n = c.samples.len().min(10 * audio::SAMPLE_RATE);
            let w = audio::Chunk { file: 0, offset: 0, samples: c.samples[..n].to_vec() };
            engine.run(std::slice::from_ref(&w), 1)?;
        }
    }

    let t2 = Instant::now();
    let words = engine.run(&chunks, a.batch_size)?;
    let infer_s = t2.elapsed().as_secs_f32();

    let mut results: Vec<FileResult> = a
        .files
        .iter()
        .zip(&audios)
        .map(|(p, s)| FileResult {
            file: p.display().to_string(),
            duration: s.len() as f32 / audio::SAMPLE_RATE as f32,
            text: String::new(),
            words: Vec::new(),
        })
        .collect();
    for (c, w) in chunks.iter().zip(words) {
        results[c.file].words.extend(w);
    }
    for r in &mut results {
        r.text = text::join(&r.words);
    }

    match a.format {
        Format::Json => println!("{}", serde_json::to_string_pretty(&results)?),
        Format::Text => {
            for r in &results {
                if results.len() > 1 {
                    println!("{}: {}", r.file, r.text);
                } else {
                    println!("{}", r.text);
                }
            }
        }
        Format::Srt => {
            for r in &results {
                if results.len() > 1 {
                    println!("# {}", r.file);
                }
                print!("{}", srt(&r.words));
            }
        }
    }

    if a.bench {
        let total: f32 = results.iter().map(|r| r.duration).sum();
        let dev = if cuda { "cuda" } else { "cpu" };
        eprintln!(
            "backend={:?} device={dev} dtype={:?} threads={threads} | load {load_s:.2}s | audio decode {audio_s:.2}s | {total:.1}s of audio in {} chunks transcribed in {infer_s:.3}s = {:.1}x realtime",
            a.backend,
            precision,
            chunks.len(),
            total / infer_s.max(1e-6)
        );
    }
    Ok(())
}
