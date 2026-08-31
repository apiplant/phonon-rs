//! Phonon-2 speech recognition on candle: model, frontend, audio I/O and (with the `mic` feature) microphone capture shared by the
//! `phonon` and `phonon-dictate` binaries.

pub mod audio;
pub mod candle_model;
mod cpu_ops;
pub mod engine;
pub mod fermion;
pub mod mel;
#[cfg(feature = "mic")]
pub mod mic;
pub mod text;
