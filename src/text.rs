//! SentencePiece detokenisation and word timestamps.

use anyhow::{Context, Result};
use serde::Serialize;

pub const FRAME_SECS: f32 = 0.08; // 10 ms hop x 8 subsampling

#[derive(Serialize, Clone, Debug)]
pub struct Word {
    pub word: String,
    pub start: f32,
    pub end: f32,
}

pub struct Vocab {
    pieces: Vec<String>,
}

impl Vocab {
    /// The model's `labels` list from config.json.
    pub fn from_config(cfg: &serde_json::Value) -> Result<Self> {
        let labels = cfg["labels"].as_array().context("config.json has no labels")?;
        let pieces = labels.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect();
        Ok(Self { pieces })
    }

    fn is_special(p: &str) -> bool {
        (p.starts_with("<|") && p.ends_with("|>")) || p == "<unk>" || p == "<pad>"
    }

    /// Words with start/end times, `offset` seconds added. A word ends where the next begins (or one frame after
    /// its last token).
    pub fn words(&self, tokens: &[u32], frames: &[usize], offset: f32) -> Vec<Word> {
        let mut words: Vec<Word> = Vec::new();
        let mut last_frame = 0usize;
        for (&tok, &fr) in tokens.iter().zip(frames) {
            let Some(p) = self.pieces.get(tok as usize) else { continue };
            if Self::is_special(p) {
                continue;
            }
            let t = offset + fr as f32 * FRAME_SECS;
            let (new_word, body) = match p.strip_prefix('\u{2581}') {
                Some(rest) => (true, rest),
                None => (words.is_empty(), p.as_str()),
            };
            if new_word {
                if let Some(w) = words.last_mut() {
                    w.end = t.max(w.start);
                }
                words.push(Word { word: body.to_string(), start: t, end: t });
            } else if let Some(w) = words.last_mut() {
                w.word.push_str(body);
            }
            last_frame = fr;
        }
        if let Some(w) = words.last_mut() {
            w.end = offset + (last_frame + 1) as f32 * FRAME_SECS;
        }
        words.retain(|w| !w.word.is_empty());
        words
    }
}

pub fn join(words: &[Word]) -> String {
    words.iter().map(|w| w.word.as_str()).collect::<Vec<_>>().join(" ")
}
