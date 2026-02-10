//! SenseVoice ASR - WASM module for speech recognition.

mod audio;
mod tokenizer;

#[path = "gen/sensevoice.rs"]
mod sensevoice;

use lele::features::{Cmvn, FeatureConfig, SenseVoiceFrontend};
use lele::tensor::{IntoLogits, TensorView};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct SenseVoiceEngine {
    model: sensevoice::SenseVoice<'static>,
    tokenizer: tokenizer::Tokenizer,
}

#[wasm_bindgen]
impl SenseVoiceEngine {
    #[wasm_bindgen(constructor)]
    pub fn new(weights: &[u8], tokens: &str) -> Result<SenseVoiceEngine, JsError> {
        let leaked: &'static [u8] = Box::leak(weights.to_vec().into_boxed_slice());
        let model = sensevoice::SenseVoice::new(leaked);
        let tokenizer = tokenizer::Tokenizer::from_string(tokens).map_err(|e| JsError::new(&e))?;
        Ok(SenseVoiceEngine { model, tokenizer })
    }

    pub fn recognize(&self, audio_data: &[f32], sample_rate: u32) -> Result<String, JsError> {
        let audio = if sample_rate != 16000 {
            audio::resample(audio_data, sample_rate as usize, 16000)
        } else {
            audio_data.to_vec()
        };

        let config = FeatureConfig {
            sample_rate: 16000,
            n_mels: 80,
            frame_length_ms: 25.0,
            frame_shift_ms: 10.0,
            lfr_m: 7,
            lfr_n: 6,
        };

        let frontend = SenseVoiceFrontend::new(config);
        let features = frontend.compute(&audio);

        let cmvn = Cmvn::default();
        let normalized = cmvn.compute(&features);

        let (t, d) = (normalized.size(0), normalized.size(1));
        let speech = if normalized.dim() == 2 {
            TensorView::from_owned(normalized.data.to_vec(), vec![1, t, d])
        } else {
            normalized
        };

        let speech_lengths = TensorView::from_owned(vec![t as i64], vec![1]);
        let language = TensorView::from_owned(vec![0i64], vec![1]);
        let text_norm = TensorView::from_owned(vec![0i64], vec![1]);

        let output = self
            .model
            .forward(speech, speech_lengths, language, text_norm)
            .into_logits();

        let batch_size = output.size(0);
        let time_steps = output.size(1);
        let vocab_size = output.size(2);

        let texts = self
            .tokenizer
            .decode_greedy(&output.data, batch_size, time_steps, vocab_size);
        Ok(texts.into_iter().next().unwrap_or_default())
    }
}

/// Parse WAV file bytes and return JSON with samples + sample rate.
#[wasm_bindgen]
pub fn decode_wav(wav_bytes: &[u8]) -> Result<JsValue, JsError> {
    let (samples, sample_rate) = audio::decode_wav_bytes(wav_bytes)
        .map_err(|e| JsError::new(&format!("WAV decode error: {}", e)))?;
    let result = serde_json::json!({ "samples": samples, "sampleRate": sample_rate });
    Ok(JsValue::from_str(&serde_json::to_string(&result).unwrap()))
}

/// Encode f32 samples to WAV bytes.
#[wasm_bindgen]
pub fn encode_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    audio::encode_wav_bytes(samples, sample_rate)
}
