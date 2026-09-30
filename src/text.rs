//! SentencePiece detokenisation and word timestamps.

use anyhow::{Context, Result};
use serde::Serialize;

pub const FRAME_SECS: f32 = 0.08; // 10 ms hop x 8 subsampling

#[derive(Serialize, Clone, Debug)]
pub struct Word {
    pub word: String,
    pub start: f32,
    pub end: f32,
    /// geometric mean of the word's token probabilities
    pub confidence: f32,
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
    pub fn words(&self, tokens: &[u32], frames: &[usize], probs: &[f32], offset: f32) -> Vec<Word> {
        let mut words: Vec<Word> = Vec::new();
        let mut logp: Vec<(f32, u32)> = Vec::new(); // per word: sum of ln p, token count
        let mut last_frame = 0usize;
        for ((&tok, &fr), &pr) in tokens.iter().zip(frames).zip(probs) {
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
                words.push(Word { word: body.to_string(), start: t, end: t, confidence: 0.0 });
                logp.push((0.0, 0));
            } else if let Some(w) = words.last_mut() {
                w.word.push_str(body);
            }
            let l = logp.last_mut().unwrap();
            l.0 += pr.max(1e-9).ln();
            l.1 += 1;
            last_frame = fr;
        }
        if let Some(w) = words.last_mut() {
            w.end = offset + (last_frame + 1) as f32 * FRAME_SECS;
        }
        for (w, (sum, n)) in words.iter_mut().zip(&logp) {
            w.confidence = (sum / (*n).max(1) as f32).exp();
        }
        words.retain(|w| !w.word.is_empty());
        words
    }
}

/// Geometric mean of the word confidences (0 for no words).
pub fn confidence(words: &[Word]) -> f32 {
    if words.is_empty() {
        return 0.0;
    }
    (words.iter().map(|w| w.confidence.max(1e-9).ln()).sum::<f32>() / words.len() as f32).exp()
}

pub fn join(words: &[Word]) -> String {
    words.iter().map(|w| w.word.as_str()).collect::<Vec<_>>().join(" ")
}

pub fn srt_time(t: f32) -> String {
    let ms = (t.max(0.0) * 1000.0).round() as u64;
    format!("{:02}:{:02}:{:02},{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
}

pub fn srt(words: &[Word]) -> String {
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
