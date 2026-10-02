/** Everything the home page says about the project, in one place. */

export interface Card {
  name: string;
  href: string;
  tagline: string;
  body: string;
}

export interface Feature {
  title: string;
  body: string;
}

export const SITE = {
  pkg: "phonon-rs",
  hasCuda: true,
  bins: ["phonon", "phonon-dictate"],
  badges: ["Rust", "candle", "WebAssembly", "Apache-2.0"],
  hero: { pre: "Phonon-2 speech-to-text, ", accent: "in Rust", post: "." },
  lead: "Transcribe files or the microphone, dictate into any window on Linux or macOS, or run the whole model in a browser tab. A 164 MB quantised Parakeet that matches its 2.5 GB teacher. No Python, no server.",
  heroNote: "Runs Phonon-2 from FermionResearch: the five-value quantised Parakeet-TDT-0.6B-v3.",
  demo: { href: "/demo", label: "▶ Transcribe in your browser" },
  cargo: `cargo install phonon-rs
cargo install phonon-rs --features cuda   # with CUDA support`,
  terminal: {
    title: "phonon",
    command: "phonon talk.wav --format json",
    outputLang: "json",
    output: `[
  {
    "file": "talk.wav",
    "duration": 8.0,
    "text": "It followed from the special theory of relativity that mass and ...",
    "words": [
      { "word": "It", "start": 0.88, "end": 1.12, "confidence": 0.9997 },
      { "word": "followed", "start": 1.12, "end": 1.68, "confidence": 0.9999 },
      { "word": "from", "start": 1.68, "end": 2.0, "confidence": 0.9999 },
      { "word": "the", "start": 2.0, "end": 2.32, "confidence": 0.9983 },
      ...
    ]
  }
]`,
  },
  cardsTitle: "Two binaries, one crate",
  cardsLead: "A transcriber for files and microphones, and a push-to-talk dictation daemon, on the same engine.",
  cards: [
    {
      name: "phonon",
      href: "/docs/cli",
      tagline: "Files and microphone to text",
      body: "Any format symphonia decodes (wav, flac, mp3, ogg, m4a). Plain text, word-timed JSON or SRT subtitles. Live microphone mode with an energy VAD.",
    },
    {
      name: "phonon-dictate",
      href: "/docs/dictate",
      tagline: "Press a key, speak, text appears",
      body: "A daemon for Linux (Wayland or X11) and macOS that keeps the model loaded and types the transcript into the focused window. Toggle or push-to-talk hotkeys.",
    },
  ] as Card[],
  featuresTitle: "Small model, full speed",
  featuresLead: "Phonon-2 holds each encoder weight at one of five learned levels, about 2.1 bits. The runtime reads that container directly.",
  features: [
    {
      title: "Runs in your browser",
      body: "The whole model compiles to WebAssembly. Pick Phonon-2 from Hugging Face or a local folder; audio never leaves the tab.",
    },
    {
      title: "Accurate",
      body: "5.21 % average word error across the Open ASR Leaderboard's seven English sets, within a hair of the 2.5 GB full-precision teacher.",
    },
    {
      title: "Matches the reference",
      body: "Text identical to the PyTorch ParakeetForTDT reference on all 73 test utterances, on CPU f32, CUDA f32 and CUDA f16.",
    },
    {
      title: "Fast on CPU and GPU",
      body: "48x real time on 16 CPU threads, 1,300x on an RTX 4090. Fused rayon kernels replace candle's single-threaded elementwise ops.",
    },
    {
      title: "Word timestamps",
      body: "Every word carries start and end times and a confidence. Export as JSON or ready-to-use SRT cues.",
    },
    {
      title: "Live microphone",
      body: "An energy VAD with an adaptive noise floor splits speech into utterances; the one in progress updates twice a second.",
    },
    {
      title: "Reads the release archive",
      body: "Point it at the unpacked directory, model.fermion, or phonon-2.bps.tar.zst itself. Startup is 0.3 to 0.7 seconds.",
    },
    {
      title: "Long audio",
      body: "Cut into chunks at the quietest 20 ms inside the last quarter of each window, so the cuts land in pauses and not inside words.",
    },
    {
      title: "CPU or CUDA",
      body: "f32 on CPU, f16 on CUDA by default; the prediction network and joint always run in f32.",
    },
  ] as Feature[],
  lib: {
    lead: "The inference engine is a library: load the model, hand it 16 kHz mono samples, get timed words back. Everything beyond inference (microphone, CLI, dictation) sits behind features.",
    add: `cargo add phonon-rs --no-default-features`,
    caption: "src/main.rs",
    snippet: `use phonon::engine::{DeviceArg, Engine};
use phonon::{audio, text};

// None: downloads Phonon-2 into ~/.cache/phonon-rs on first use
let engine = Engine::load(None, DeviceArg::Auto, None)?;
let samples = audio::load("talk.wav".as_ref())?;     // 16 kHz mono f32
let words = engine.transcribe(&samples, 30.0)?;      // 30 s chunks, cut at pauses
println!("{}", text::join(&words));
for w in &words {
    println!("{:6.2}-{:6.2}  {:.2}  {}", w.start, w.end, w.confidence, w.word);
}`,
  },
};
