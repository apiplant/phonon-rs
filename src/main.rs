use anyhow::Result;
use clap::{Parser, ValueEnum};
use phonon::engine::{DeviceArg, Engine, Precision};
use phonon::text::{self, Word};
use phonon::{audio, mic};
use rayon::prelude::*;
use serde::Serialize;
use std::path::PathBuf;
use std::time::Instant;

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
    #[arg(required_unless_present_any = ["mic", "list_mics"])]
    files: Vec<PathBuf>,

    /// Transcribe the microphone live: the utterance in progress shows on stderr, finished ones print to stdout
    #[arg(long, conflicts_with = "files")]
    mic: bool,

    /// Input device for --mic: any part of a name from --list-mics (default: the system default input)
    #[arg(long)]
    mic_device: Option<String>,

    /// List audio input devices (* marks the default) and exit
    #[arg(long)]
    list_mics: bool,

    /// --mic: pause (ms) that ends an utterance
    #[arg(long, default_value_t = 700)]
    silence_ms: u32,

    /// --mic: fixed RMS level (of the 200 Hz high-passed signal) counted as speech (default: adaptive, 4x the
    /// measured noise floor)
    #[arg(long)]
    vad_threshold: Option<f32>,

    /// --mic: drop one- or two-word utterances below this confidence (noise often decodes as "Yeah." / "Okay.");
    /// 0 keeps everything. RUST_LOG=phonon=debug shows what gets dropped
    #[arg(long, default_value_t = 0.9)]
    min_confidence: f32,

    /// Model: a dir with model.fermion + config.json, a model.fermion file, or phonon-2.bps.tar.zst
    #[arg(short, long, env = "PHONON_MODEL")]
    model: Option<PathBuf>,

    #[arg(short, long, value_enum, default_value_t = DeviceArg::Auto)]
    device: DeviceArg,

    /// Encoder precision (default: f32 on CPU, f16 on CUDA)
    #[arg(long, value_enum)]
    dtype: Option<Precision>,

    /// Longest piece of audio run through the encoder at once; longer files are cut at quiet points
    #[arg(long, default_value_t = 30.0)]
    chunk_secs: f32,

    /// Chunks per encoder batch
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

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,pulseaudio=off")).init();
    let a = Args::parse();
    if a.list_mics {
        return mic::list();
    }
    let threads = phonon::engine::init_threads(a.threads);
    let t0 = Instant::now();
    let engine = Engine::load(a.model.clone(), a.device, a.dtype)?;
    let (cuda, precision) = (engine.cuda, engine.precision);
    let load_s = t0.elapsed().as_secs_f32();

    if a.mic {
        let opts = mic::MicOptions {
            device: a.mic_device.clone(),
            vad_threshold: a.vad_threshold,
            silence_ms: a.silence_ms,
            max_secs: a.chunk_secs,
            json: a.format == Format::Json,
            srt: a.format == Format::Srt,
            min_confidence: a.min_confidence,
        };
        let transcribe = |samples: &[f32]| -> Result<Vec<Word>> {
            let chunk = audio::Chunk { file: 0, offset: 0, samples: samples.to_vec() };
            Ok(engine.run(std::slice::from_ref(&chunk), 1)?.pop().unwrap_or_default())
        };
        return mic::run(&opts, &transcribe);
    }

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
                print!("{}", text::srt(&r.words));
            }
        }
    }

    if a.bench {
        let total: f32 = results.iter().map(|r| r.duration).sum();
        let dev = if cuda { "cuda" } else { "cpu" };
        eprintln!(
            "device={dev} dtype={:?} threads={threads} | load {load_s:.2}s | audio decode {audio_s:.2}s | {total:.1}s of audio in {} chunks transcribed in {infer_s:.3}s = {:.1}x realtime",
            precision,
            chunks.len(),
            total / infer_s.max(1e-6)
        );
    }
    Ok(())
}
