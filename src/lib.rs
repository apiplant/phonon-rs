//! Phonon-2 speech recognition on candle: model, frontend, audio I/O and microphone capture shared by the
//! `phonon` and `phonon-dictate` binaries.

pub mod audio;
pub mod candle_model;
mod cpu_ops;
pub mod engine;
pub mod fermion;
pub mod mel;
pub mod mic;
pub mod text;
