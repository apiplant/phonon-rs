# phonon — Phonon-2 speech-to-text in Rust

A command-line transcriber for [Phonon-2](../README.md), the five-value quantised Parakeet-TDT-0.6B-v3. It is a
native Rust port of the HF `ParakeetForTDT` graph on [candle](https://github.com/huggingface/candle) and runs on
CPU or CUDA. It reads `model.fermion` directly: the unpacked directory, the file itself, or the
`phonon-2.bps.tar.zst` release archive. It transcribes files or the microphone.

## Speed vs the Python version

Measured on a Ryzen 9 7950X (16 cores) and an RTX 4090. The audio is one hour of LibriSpeech (the 73 utterances of
`hf-internal-testing/librispeech_asr_dummy`, looped), cut into 30 s chunks and run in batches of 16 by both.
"PyTorch" is the same HF `ParakeetForTDT` model that `reference_transformers.py` loads, driven the same way
(torch 2.12, transformers 5.17).

| | device | 1 h of audio | speed |
|---|---|--:|--:|
| **phonon (candle)** | CUDA f16 | 2.8 s | 1,304x |
| PyTorch | CUDA f16 | 2.6 s | 1,372x |
| PyTorch | CUDA f32 | 4.0 s | 896x |
| **phonon (candle)** | CPU f32, 16 threads | 75 s | 48x |
| PyTorch | CPU f32, 16 threads | 59 s | 61x |
| `reference_transformers.py` as shipped | CPU f32, one utterance at a time | ≈ 167 s (481 s of audio in 22.3 s) | 22x |

Startup is 0.3–0.7 s, against about 30 s for the Python reference to build and load the model.

On the 73 utterances, candle produces text identical to `reference_transformers.py` (PyTorch fp32) on CPU f32
(73/73), CUDA f32 (73/73) and CUDA f16 (73/73), for a WER of 4.26 % in each case. bf16 differs on one utterance.

## Build

```bash
cargo build --release                    # CPU
cargo build --release --features cuda    # + CUDA (needs the CUDA toolkit; nvcc on PATH)
```

### Using it as a library

Everything beyond inference is behind features: `mic` (cpal, ctrlc; `phonon::mic`), `cli` (the `phonon` binary; clap, env_logger) and `dictate` (the `phonon-dictate` binary; evdev). `dictate` implies `cli`, which implies `mic`; the defaults enable `dictate`. For inference only:

```toml
phonon = { path = "...", default-features = false }
```

On Linux, the microphone is opened through PulseAudio's protocol, which PipeWire also serves via pipewire-pulse.
That way `--mic-device` picks the sound server's named sources and never opens raw ALSA hardware the server already
holds. Plain ALSA is the fallback when no sound server is running. cpal still links ALSA, so the build needs the
ALSA dev files (`alsa-lib` / `libasound2-dev`).

## Run

```bash
# the model: unpack the release archive next to this directory (or pass the .tar.zst itself to --model)
mkdir -p ../model_phonon2_c4c_int6 && tar --zstd -xf ../phonon-2.bps.tar.zst -C ../model_phonon2_c4c_int6

./target/release/phonon recording.wav                         # CUDA if built with it and a GPU is present, else CPU
./target/release/phonon talk.mp3 --device cpu --format srt > talk.srt
./target/release/phonon a.flac b.m4a --format json            # words with start/end times
./target/release/phonon hour.flac --bench                     # timings + real-time factor on stderr

./target/release/phonon --mic                                 # live from the default input; Ctrl-C to stop
./target/release/phonon --list-mics
./target/release/phonon --mic --mic-device wave --format json # one JSON line per utterance
```

Files can be in any format symphonia decodes (wav, flac, mp3, ogg/vorbis, aac/m4a, ...). They are mixed to mono and
resampled to 16 kHz.

Options that matter:

- `--device auto|cpu|cuda`
- `--dtype f32|f16|bf16`: encoder precision. Defaults to f32 on CPU and f16 on CUDA. The prediction network and joint always run in f32.
- `--chunk-secs 30`: long audio is cut at the quietest 20 ms inside the last quarter of each window. In `--mic` mode this is also the longest single utterance.
- `--batch-size 16`: chunks per encoder batch.
- `--threads N`: defaults to the physical core count, because SMT siblings slow the GEMMs down.
- `--model PATH`, or the `PHONON_MODEL` environment variable. Defaults to `../model_phonon2_c4c_int6`, then `../phonon-2.bps.tar.zst`, then the archive in `~/.cache/phonon-rs/Phonon-2/` (`$XDG_CACHE_HOME`), downloaded from Hugging Face on first use.

### Microphone mode

`--mic` captures the chosen input and splits it into utterances with an energy VAD. The speech threshold is 4x a
running noise-floor estimate, or `--vad-threshold` if you set one.

- While you speak, the current utterance is re-transcribed about every half second and shown dimmed on stderr. On a slow CPU the refresh backs off.
- After a pause of `--silence-ms` (700 ms by default), the final text goes to stdout, as plain text, one JSON line per utterance (`--format json`, with word times from the session start) or SRT cues.
- Ctrl-C flushes the utterance in progress and exits.

## Dictation: `phonon-dictate`

A second binary for push-button dictation on Linux (Wayland or X11) and macOS. Press a hotkey, speak, and press it again.
The transcript is typed into whatever window has focus. The model stays loaded, so a sentence comes back in about
50 ms on a GPU.

```bash
cargo install --path . --features cuda            # puts phonon and phonon-dictate in ~/.cargo/bin
phonon-dictate setup                              # press your combo twice; saved to ~/.config/phonon/dictate.json
phonon-dictate setup --hold                       # same, but push-to-talk (record while held)
phonon-dictate                                    # run the daemon (keep it running: autostart / systemd user unit)
```

Instead of a saved hotkey, you can:

- pass one on the command line: `phonon-dictate --key f9`, `--key ctrl+alt+d`, `--key super+space`, `--hold`
- skip the evdev hotkey and bind `phonon-dictate toggle` in *System Settings → Shortcuts* (KDE) or your compositor's config. `start` / `stop` also exist, and the daemon listens on `$XDG_RUNTIME_DIR/phonon-dictate.sock`. Run the daemon with `--no-hotkey` in that case.

On macOS the hotkey comes from a Quartz event tap and the text is typed with posted key events (any character, whatever the layout), so macOS asks for **Input Monitoring** (to see the hotkey) and **Accessibility** (to type) for the app that runs it, in System Settings > Privacy & Security. `cmd` means Command, `alt` Option, `KEY_FN` the Fn/globe key; `--paste-keys` defaults to `cmd+v`; notifications use Notification Center.

How it works on Linux, and what it needs:

- **Hotkeys** are read from `/dev/input/event*` (evdev), so they work under any compositor. Keyboards plugged in later are picked up too. Left and right Ctrl/Shift/Super count as the same key. The hotkey is not grabbed, so the focused app also sees it. Pick a combo that does nothing else (F13–F24, Pause, an unused Super+letter, a mouse side button).
- **Text** is typed through a virtual keyboard on `/dev/uinput`, using the US layout. Characters it cannot type are pasted: `wl-copy`, then Ctrl+V. `--output paste` always pastes (use it with a non-US layout), `--output clipboard` only copies, and `--output stdout` prints.
- **Both need the `input` group**: `sudo usermod -aG input $USER`, then log in again.
- **Recordings without enough speech** (under 0.25 s voiced, measured above 200 Hz), or decoding to one or two low-confidence words, are discarded. Knocks and breaths don't type "Yeah.".
- **Other options:** `--mic-device`, `--quiet` (no notifications), `--no-space` (no trailing space), `--max-secs 300`.

Example systemd user unit (`~/.config/systemd/user/phonon-dictate.service`):

```ini
[Unit]
Description=Phonon dictation
After=graphical-session.target

[Service]
ExecStart=%h/.cargo/bin/phonon-dictate --model /path/to/model_phonon2_c4c_int6
Restart=on-failure

[Install]
WantedBy=graphical-session.target
```

## How it works

- `src/fermion.rs` reads the `fermion-five-value-parakeet-v1` container, a port of `fermion_container.py`. It expands five-value rows ({0, ±lo, ±hi} per row) and int6 tables to f32. It can read straight out of the `.tar.zst`.
- `src/mel.rs` is the HF `ParakeetFeatureExtractor`: preemphasis 0.97, centred 512-point STFT with a 400-sample symmetric Hann window, 128 Slaney mel bands, log, and per-band normalisation.
- `src/candle_model.rs` holds the 24-layer FastConformer, the 2-layer LSTM prediction network and the TDT joint. Eval BatchNorm is folded into the depthwise conv. Greedy TDT decoding (token and duration heads) is batched across chunks, with one host sync per step.
- `src/cpu_ops.rs` has fused, rayon-parallel f32 kernels (rel-pos attention softmax, GLU + depthwise conv + SiLU, head permutes, residual adds, subsampling convs). candle's CPU elementwise and copy ops are single-threaded, and before these kernels they took more time than the matmuls. CUDA uses the composed candle ops.
- `src/mic.rs` does cpal capture (PulseAudio/PipeWire first), a 200 Hz high-passed energy VAD with an onset rule, and utterance segmentation. One- or two-word utterances below `--min-confidence` (0.9) are dropped.
- `src/bin/phonon-dictate.rs` is the dictation daemon: evdev hotkeys, uinput typing, a control socket and notifications.

## Use as a library

The crate is published as `phonon-rs`; the library is `phonon`.

```toml
[dependencies]
phonon-rs = { version = "0.1", default-features = false }                          # CPU, inference only
# phonon-rs = { version = "0.1", default-features = false, features = ["cuda"] }   # + CUDA (opt-in)
```

Always set `default-features = false` for library use: the defaults enable the `phonon` and
`phonon-dictate` binaries (clap, cpal, evdev). CUDA is never on by default; the `cuda` feature is passed
down to candle and needs the CUDA toolkit to build. `mic` adds `phonon::mic` (cpal) if you want
microphone capture.

```rust
use phonon::engine::{DeviceArg, Engine};
use phonon::{audio, text};

// `None` finds the model next to the binary or in ~/.cache/phonon-rs (downloaded on first use, 164 MB);
// or pass `Some(path)` to a model directory, model.fermion or phonon-2.bps.tar.zst.
let engine = Engine::load(None, DeviceArg::Auto, None)?;
let samples = audio::load("talk.wav".as_ref())?;      // any symphonia format -> 16 kHz mono f32
let words = engine.transcribe(&samples, 30.0)?;       // chunks of up to 30 s, cut at pauses
println!("{}", text::join(&words));
```

## Install

Prebuilt packages (phonon, phonon-dictate) for macOS (Apple Silicon), Linux x86_64 and Linux arm64:

```bash
brew tap apiplant/tap && brew install apiplant/tap/phonon-rs      # macOS, Linux
sudo apt install phonon-rs      # Debian/Ubuntu, after adding apt.apiplant.com
sudo pacman -S phonon-rs        # Arch, after adding apiplant.github.io/pacman
```

CUDA builds (Linux x86_64, NVIDIA GPU) are separate packages: `phonon-rs-cuda` (`brew install apiplant/tap/phonon-rs-cuda`, `sudo apt install phonon-rs-cuda`, `sudo pacman -S phonon-rs-cuda`). They conflict with `phonon-rs`.

Setup commands for the apt and pacman repositories, the plain archives and the release process are in [`packaging/README.md`](packaging/README.md). Release archives are on the [releases page](https://github.com/apiplant/phonon-rs/releases).

Website and in-browser demo: <https://phonon-rs.apiplant.com>.
