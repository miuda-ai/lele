//! Supertonic TTS - WASM module for text-to-speech synthesis.

mod audio;
mod config;
mod processor;

#[path = "gen/durationpredictor.rs"]
mod durationpredictor;
#[path = "gen/textencoder.rs"]
mod textencoder;
#[path = "gen/vectorestimator.rs"]
mod vectorestimator;
#[path = "gen/vocoder.rs"]
mod vocoder;

use lele::tensor::TensorView;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct SupertonicEngine {
    config: config::Config,
    text_processor: processor::UnicodeProcessor,
    text_encoder: textencoder::TextEncoder<'static>,
    duration_predictor: durationpredictor::DurationPredictor<'static>,
    vector_estimator: vectorestimator::VectorEstimator<'static>,
    vocoder_model: vocoder::Vocoder<'static>,
    style_cache: std::collections::HashMap<String, Style>,
}

struct Style {
    ttl_data: Vec<f32>,
    ttl_shape: Vec<usize>,
    dp_data: Vec<f32>,
    dp_shape: Vec<usize>,
}

#[wasm_bindgen]
impl SupertonicEngine {
    #[wasm_bindgen(constructor)]
    pub fn new(
        te_weights: &[u8],
        dp_weights: &[u8],
        ve_weights: &[u8],
        vo_weights: &[u8],
        config_json: &str,
        unicode_indexer_json: &str,
    ) -> Result<SupertonicEngine, JsError> {
        let te_leaked: &'static [u8] = Box::leak(te_weights.to_vec().into_boxed_slice());
        let dp_leaked: &'static [u8] = Box::leak(dp_weights.to_vec().into_boxed_slice());
        let ve_leaked: &'static [u8] = Box::leak(ve_weights.to_vec().into_boxed_slice());
        let vo_leaked: &'static [u8] = Box::leak(vo_weights.to_vec().into_boxed_slice());

        let mut config: config::Config = serde_json::from_str(config_json)
            .map_err(|e| JsError::new(&format!("Config parse error: {}", e)))?;
        config.fix();

        let text_processor = processor::UnicodeProcessor::from_json(unicode_indexer_json)
            .map_err(|e| JsError::new(&format!("Unicode indexer error: {}", e)))?;

        Ok(SupertonicEngine {
            config,
            text_processor,
            text_encoder: textencoder::TextEncoder::new(te_leaked),
            duration_predictor: durationpredictor::DurationPredictor::new(dp_leaked),
            vector_estimator: vectorestimator::VectorEstimator::new(ve_leaked),
            vocoder_model: vocoder::Vocoder::new(vo_leaked),
            style_cache: std::collections::HashMap::new(),
        })
    }

    pub fn load_style(&mut self, name: &str, style_json: &str) -> Result<(), JsError> {
        if self.style_cache.contains_key(name) {
            return Ok(());
        }

        let data: config::VoiceStyleData = serde_json::from_str(style_json)
            .map_err(|e| JsError::new(&format!("Style parse error: {}", e)))?;

        let bsz = 1;
        let ttl_dim1 = data.style_ttl.dims[1];
        let ttl_dim2 = data.style_ttl.dims[2];
        let mut ttl_flat = vec![0.0; bsz * ttl_dim1 * ttl_dim2];
        let mut idx = 0;
        for batch in &data.style_ttl.data {
            for row in batch {
                for &val in row {
                    if idx < ttl_flat.len() {
                        ttl_flat[idx] = val;
                        idx += 1;
                    }
                }
            }
        }

        let dp_dim1 = data.style_dp.dims[1];
        let dp_dim2 = data.style_dp.dims[2];
        let mut dp_flat = vec![0.0; bsz * dp_dim1 * dp_dim2];
        idx = 0;
        for batch in &data.style_dp.data {
            for row in batch {
                for &val in row {
                    if idx < dp_flat.len() {
                        dp_flat[idx] = val;
                        idx += 1;
                    }
                }
            }
        }

        self.style_cache.insert(
            name.to_string(),
            Style {
                ttl_data: ttl_flat,
                ttl_shape: vec![bsz, ttl_dim1, ttl_dim2],
                dp_data: dp_flat,
                dp_shape: vec![bsz, dp_dim1, dp_dim2],
            },
        );

        Ok(())
    }

    pub fn synthesize(
        &mut self,
        text: &str,
        lang: &str,
        style_name: &str,
        speed: f32,
        steps: u32,
    ) -> Result<Vec<f32>, JsError> {
        if !self.style_cache.contains_key(style_name) {
            return Err(JsError::new(&format!(
                "Style '{}' not loaded. Call load_style() first.",
                style_name
            )));
        }

        let chunks = processor::chunk_text(text, None);
        let mut full_audio = Vec::new();

        for chunk in chunks {
            if chunk.trim().is_empty() {
                continue;
            }

            let (text_ids_vec, mask_data, mask_shape) = self
                .text_processor
                .call(&[chunk], &[lang.to_string()])
                .map_err(|e| JsError::new(&format!("Text processing error: {}", e)))?;

            let bsz = 1;
            let max_len = text_ids_vec[0].len();
            let mut text_ids_i64 = vec![0i64; max_len];
            for (i, &id) in text_ids_vec[0].iter().enumerate() {
                text_ids_i64[i] = id as i64;
            }

            let text_ids_shape = [bsz, max_len];
            let text_ids_tv = TensorView::new(&text_ids_i64, &text_ids_shape);
            let text_mask_tv = TensorView::new(&mask_data, &mask_shape);

            let style = self.style_cache.get(style_name).unwrap();
            let style_dp_tv = TensorView::new(&style.dp_data, &style.dp_shape);
            let style_ttl_tv = TensorView::new(&style.ttl_data, &style.ttl_shape);

            // 1. Duration Predictor
            let duration_tv = self.duration_predictor.forward(
                text_ids_tv.clone(),
                style_dp_tv,
                text_mask_tv.clone(),
            );
            let mut duration = duration_tv.data.to_vec();
            for d in duration.iter_mut() {
                *d /= speed;
            }

            let total_duration_seconds: f32 = duration.iter().sum();
            let duration_batch = vec![total_duration_seconds];

            // 2. Text Encoder
            let text_emb_tv =
                self.text_encoder
                    .forward(text_ids_tv, style_ttl_tv.clone(), text_mask_tv.clone());

            // 3. Vector Estimator (Diffusion loop)
            let (mut xt_data, xt_shape, latent_mask_data, latent_mask_shape) =
                processor::sample_noisy_latent(
                    &duration_batch,
                    self.config.ae.sample_rate,
                    self.config.ae.base_chunk_size,
                    self.config.ttl.chunk_compress_factor,
                    self.config.ttl.latent_dim,
                );

            let total_step_data = vec![steps as f32; bsz];
            let total_step_shape = [bsz];
            let total_step_tv = TensorView::new(&total_step_data, &total_step_shape);

            for step in 0..steps {
                let current_step_data = vec![step as f32; bsz];
                let current_step_shape = [bsz];
                let current_step_tv = TensorView::new(&current_step_data, &current_step_shape);

                let xt_tv = TensorView::new(&xt_data, &xt_shape);
                let latent_mask_tv = TensorView::new(&latent_mask_data, &latent_mask_shape);

                let denoised_tv = self.vector_estimator.forward(
                    xt_tv,
                    text_emb_tv.clone(),
                    style_ttl_tv.clone(),
                    latent_mask_tv,
                    text_mask_tv.clone(),
                    current_step_tv,
                    total_step_tv.clone(),
                );

                xt_data = denoised_tv.data.to_vec();
            }

            // 4. Vocoder
            let xt_tv = TensorView::new(&xt_data, &xt_shape);
            let audio_tv = self.vocoder_model.forward(xt_tv);

            let audio_data = audio_tv.data.to_vec();
            let expected_len =
                (total_duration_seconds * self.config.ae.sample_rate as f32) as usize;
            let actual_len = audio_data.len().min(expected_len);

            full_audio.extend_from_slice(&audio_data[..actual_len]);
        }

        Ok(full_audio)
    }

    pub fn sample_rate(&self) -> i32 {
        self.config.ae.sample_rate
    }
}

/// Encode f32 samples to WAV bytes.
#[wasm_bindgen]
pub fn encode_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    audio::encode_wav_bytes(samples, sample_rate)
}
