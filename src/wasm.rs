//! wasm-bindgen bindings for running Phonon-2 in the browser.
//!
//! The model bytes (the `phonon-2.bps.tar.zst` release archive, or an unpacked `model.fermion` plus
//! `config.json`) are fetched by JS and handed to [`WasmPhonon::load_archive`] / [`WasmPhonon::load_model`];
//! there is no filesystem here. Audio is decoded by the browser (Web Audio) and arrives as 16 kHz mono f32.
//! Results go back as JSON strings so the JS side never needs a Rust struct layout.

use crate::audio;
use crate::engine::{Engine, Precision};
use crate::{fermion, text};
use serde::Serialize;
use wasm_bindgen::prelude::*;

fn js_err(e: anyhow::Error) -> JsValue {
    // `{:#}` keeps the whole context chain; the outermost message alone rarely says what went wrong.
    JsValue::from_str(&format!("{e:#}"))
}

/// Readable panic messages (shape mismatches inside candle, ...) in the browser console.
#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct WasmPhonon {
    engine: Engine,
}

#[derive(Serialize)]
struct Transcript {
    text: String,
    words: Vec<text::Word>,
    srt: String,
    /// Mean word confidence (geometric).
    confidence: f32,
}

#[wasm_bindgen]
impl WasmPhonon {
    /// Loads the model from the bytes of `phonon-2.bps.tar.zst`. Float32 on CPU is the only combination that
    /// makes sense in a browser tab.
    #[wasm_bindgen(js_name = loadArchive)]
    pub fn load_archive(archive: &[u8]) -> Result<WasmPhonon, JsValue> {
        let files = fermion::load_archive(archive).map_err(js_err)?;
        Self::from_files(files)
    }

    /// Loads the model from an unpacked `model.fermion` (as bytes) and `config.json` (as text).
    #[wasm_bindgen(js_name = loadModel)]
    pub fn load_model(container: Vec<u8>, config_json: &str) -> Result<WasmPhonon, JsValue> {
        let files = fermion::from_parts(container, config_json.as_bytes()).map_err(js_err)?;
        Self::from_files(files)
    }

    fn from_files(files: fermion::ModelFiles) -> Result<WasmPhonon, JsValue> {
        let engine = Engine::from_files(files, false, Precision::F32).map_err(js_err)?;
        Ok(WasmPhonon { engine })
    }

    /// Transcribes mono audio at `sample_rate` Hz (resampled to 16 kHz first when it differs). Audio longer than
    /// `chunk_secs` is cut at quiet points. `on_progress(done, total)` runs after every chunk.
    pub fn transcribe(
        &self,
        samples: Vec<f32>,
        sample_rate: u32,
        chunk_secs: f32,
        on_progress: &js_sys::Function,
    ) -> Result<String, JsValue> {
        let samples = audio::resample(samples, sample_rate as usize).map_err(js_err)?;
        let chunks = audio::chunk(0, &samples, chunk_secs);
        let total = chunks.len();
        let mut words = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            let out = self.engine.run(std::slice::from_ref(chunk), 1).map_err(js_err)?;
            words.extend(out.into_iter().flatten());
            let _ = on_progress.call2(&JsValue::NULL, &JsValue::from((i + 1) as u32), &JsValue::from(total as u32));
        }
        let transcript = Transcript {
            text: text::join(&words),
            srt: text::srt(&words),
            confidence: text::confidence(&words),
            words,
        };
        serde_json::to_string(&transcript).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
