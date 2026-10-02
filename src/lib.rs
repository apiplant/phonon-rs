//! Phonon-2 speech recognition on candle: model, frontend, audio I/O and (with the `mic` feature) microphone capture shared by the
//! `phonon` and `phonon-dictate` binaries.

pub mod audio;
pub mod candle_model;
#[cfg(feature = "dictate")]
pub mod dictate;
mod cpu_ops;
#[cfg(not(target_arch = "wasm32"))]
pub mod download;
pub mod engine;
pub mod fermion;
pub mod mel;
#[cfg(feature = "mic")]
pub mod mic;
pub mod text;
#[cfg(target_arch = "wasm32")]
pub mod wasm;
