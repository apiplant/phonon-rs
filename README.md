# phonon — Phonon-2 speech-to-text in Rust

A command-line transcriber for [Phonon-2](../README.md) (the five-value quantised Parakeet-TDT-0.6B-v3) with two
backends, each on CPU or CUDA:

| backend | what it runs | model input |
|---|---|---|
| `candle` (default) | a native Rust port of the HF `ParakeetForTDT` graph on [candle](https://github.com/huggingface/candle), batched greedy TDT decoding | `model.fermion` directly (the unpacked dir, the file, or the `phonon-2.bps.tar.zst` release archive) |
| `onnx` | ONNX Runtime through [transcribe-rs](https://github.com/cjpais/transcribe-rs)'s Parakeet engine | an ONNX export made by `scripts/export_onnx.py` |

## Results

Measured on a Ryzen 9 7950X (16 cores) and an RTX 4090. Speed is one hour of LibriSpeech audio (the 73 utterances of
`hf-internal-testing/librispeech_asr_dummy`, looped). Accuracy is on those same 73 utterances, compared with
`reference_transformers.py` (PyTorch fp32).

| backend | device | 1 h of audio | speed | same text as the PyTorch reference | WER |
|---|---|--:|--:|--:|--:|
| **candle** | CUDA (f16) | **2.8 s** | **1,304x** | 73/73 | 4.26 % |
| **candle** | CPU (f32, 16 threads) | **75 s** | **48x** | 73/73 | 4.26 % |
| onnx (transcribe-rs) | CUDA (f32) | 20.8 s | 173x | 54/73 | 5.83 % |
| onnx (transcribe-rs) | CPU (f32) | 106 s | 34x | 54/73 | 5.83 % |
| PyTorch reference | — | — | — | — | 4.26 % |

**Use `candle`.** It is the fastest on both devices and the only one that decodes TDT correctly.
transcribe-rs's Parakeet decoder ignores the TDT duration head: it treats the model as plain RNN-T and never
advances past a frame after emitting a token, so it stutters on some words (`Sir Fre Fre Fre … Frederick`) and
walks every encoder frame one by one.

## Build

```bash
cargo build --release                    # CPU: candle + ONNX Runtime
cargo build --release --features cuda    # + CUDA for both (needs the CUDA toolkit; nvcc on PATH)
cargo build --release --no-default-features [--features cuda]   # candle only, no ONNX Runtime download
```

The `onnx` feature downloads a prebuilt ONNX Runtime (CPU or CUDA build) through the `ort` crate.

## Run

```bash
# the model: unpack the release archive next to this directory (or pass the .tar.zst itself to --model)
mkdir -p ../model_phonon2_c4c_int6 && tar --zstd -xf ../phonon-2.bps.tar.zst -C ../model_phonon2_c4c_int6

./target/release/phonon recording.wav                         # CUDA if built with it and a GPU is present, else CPU
./target/release/phonon talk.mp3 --device cpu --format srt > talk.srt
./target/release/phonon a.flac b.m4a --format json            # words with start/end times
./target/release/phonon hour.flac --bench                     # timings + real-time factor on stderr
```

Options that matter:

- `--device auto|cpu|cuda`
- `--dtype f32|f16|bf16`: encoder precision. Defaults to f32 on CPU and f16 on CUDA. The prediction network and joint always run in f32.
- `--chunk-secs 30`: long audio is cut at the quietest 20 ms inside the last quarter of each window.
- `--batch-size 16`: chunks per encoder batch.
- `--threads N`: defaults to the physical core count, because SMT siblings slow the GEMMs down.
- `--model PATH`, or the `PHONON_MODEL` environment variable. Defaults to `../model_phonon2_c4c_int6`, then `../phonon-2.bps.tar.zst`.

Any format symphonia decodes works (wav, flac, mp3, ogg/vorbis, aac/m4a, ...). Input is mixed to mono and
resampled to 16 kHz.

### ONNX backend

```bash
pip install torch "transformers>=5.17" onnx huggingface_hub
python scripts/export_onnx.py ../model_phonon2_c4c_int6/model.fermion ../phonon-2-onnx          # fp32, 2.4 GB
python scripts/export_onnx.py ../model_phonon2_c4c_int6/model.fermion ../phonon-2-onnx --fp16   # adds encoder-model.fp16.onnx
./target/release/phonon recording.wav --backend onnx [--dtype f16]
```

The export follows the `istupakov/parakeet-tdt-0.6b-v3-onnx` layout that transcribe-rs expects, and copies its
weight-free `nemo128.onnx` preprocessor and `vocab.txt`. The joint's 1024→640 encoder projection is moved into
the encoder graph, so it runs once per frame rather than once per decode step.

## How the candle backend works

- `src/fermion.rs` reads the `fermion-five-value-parakeet-v1` container, a port of `fermion_container.py`. It expands five-value rows ({0, ±lo, ±hi} per row) and int6 tables to f32. It can read straight out of the `.tar.zst`.
- `src/mel.rs` is the HF `ParakeetFeatureExtractor`: preemphasis 0.97, centred 512-point STFT with a 400-sample symmetric Hann window, 128 Slaney mel bands, log, and per-band normalisation.
- `src/candle_model.rs` holds the 24-layer FastConformer, the 2-layer LSTM prediction network and the TDT joint. Eval BatchNorm is folded into the depthwise conv. Greedy TDT decoding is batched across chunks, with one host sync per step.
- `src/cpu_ops.rs` has fused, rayon-parallel f32 kernels (rel-pos attention softmax, GLU + depthwise conv + SiLU, head permutes, residual adds, subsampling convs). candle's CPU elementwise and copy ops are single-threaded, and before these kernels they took more time than the matmuls. CUDA uses the composed candle ops.

Checked against `reference_transformers.py` (PyTorch, fp32) on the 73 LibriSpeech utterances of
`hf-internal-testing/librispeech_asr_dummy`: candle gives identical text on CPU f32 (73/73), CUDA f32 (73/73) and
CUDA f16 (73/73). bf16 differs on one utterance.
