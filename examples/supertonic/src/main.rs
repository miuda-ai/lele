mod audio;
mod config;
mod durationpredictor;
mod processor;
mod textencoder;
mod vectorestimator;
mod vocoder;

use anyhow::{Context, Result};
use audio::WavWriter;
use config::{Config, VoiceStyleData};
use lele::tensor::TensorView;
use processor::{UnicodeProcessor, chunk_text, sample_noisy_latent};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::durationpredictor::DurationPredictor;
use crate::textencoder::TextEncoder;
use crate::vectorestimator::VectorEstimator;
use crate::vocoder::Vocoder;

pub struct Style {
    pub ttl_data: Vec<f32>,
    pub ttl_shape: Vec<usize>,
    pub dp_data: Vec<f32>,
    pub dp_shape: Vec<usize>,
}

pub struct SupertonicTts<'a> {
    config: Config,
    text_processor: UnicodeProcessor,
    text_encoder: TextEncoder<'a>,
    duration_predictor: DurationPredictor<'a>,
    vector_estimator: VectorEstimator<'a>,
    vocoder: Vocoder<'a>,
    voice_styles_dir: PathBuf,
    style_cache: HashMap<String, Style>,
}

impl<'a> SupertonicTts<'a> {
    pub fn new(
        weights_dir: &Path,
        config_path: &Path,
        voice_styles_dir: &Path,
        text_encoder_weights: &'a [u8],
        duration_predictor_weights: &'a [u8],
        vector_estimator_weights: &'a [u8],
        vocoder_weights: &'a [u8],
    ) -> Result<Self> {
        let mut config: Config = serde_json::from_reader(fs::File::open(config_path)?)?;
        config.fix();

        let text_processor = UnicodeProcessor::new(weights_dir.join("unicode_indexer.json"))?;

        Ok(Self {
            config,
            text_processor,
            text_encoder: TextEncoder::new(text_encoder_weights),
            duration_predictor: DurationPredictor::new(duration_predictor_weights),
            vector_estimator: VectorEstimator::new(vector_estimator_weights),
            vocoder: Vocoder::new(vocoder_weights),
            voice_styles_dir: voice_styles_dir.to_path_buf(),
            style_cache: HashMap::new(),
        })
    }

    pub fn load_style(&mut self, name: &str) -> Result<()> {
        if self.style_cache.contains_key(name) {
            return Ok(());
        }

        let path = self.voice_styles_dir.join(format!("{}.json", name));
        let file = fs::File::open(&path)
            .with_context(|| format!("Failed to open voice style: {:?}", path))?;
        let data: VoiceStyleData = serde_json::from_reader(file)?;

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
        steps: usize,
    ) -> Result<Vec<f32>> {
        let t_overhead = std::time::Instant::now();
        self.load_style(style_name)?;
        eprintln!("[OVERHEAD] load_style: {:.2}ms", t_overhead.elapsed().as_secs_f64() * 1000.0);
        let style = self.style_cache.get(style_name).unwrap();

        let t_chunk = std::time::Instant::now();
        let chunks = chunk_text(text, None);
        eprintln!("[OVERHEAD] chunk_text: {:.2}ms", t_chunk.elapsed().as_secs_f64() * 1000.0);
        let mut full_audio = Vec::new();

        for chunk in chunks {
            if chunk.trim().is_empty() {
                continue;
            }

            let t_tok = std::time::Instant::now();
            let (text_ids_vec, mask_data, mask_shape) =
                self.text_processor.call(&[chunk], &[lang.to_string()])?;
            eprintln!("[OVERHEAD] tokenization: {:.2}ms", t_tok.elapsed().as_secs_f64() * 1000.0);
            let bsz = 1;
            let max_len = text_ids_vec[0].len();
            let mut text_ids_i64 = vec![0i64; max_len];
            for (i, &id) in text_ids_vec[0].iter().enumerate() {
                text_ids_i64[i] = id as i64;
            }

            let text_ids_shape = [bsz, max_len];
            let text_ids_tv = TensorView::new(&text_ids_i64, &text_ids_shape);
            let text_mask_tv = TensorView::new(&mask_data, &mask_shape);
            let style_dp_tv = TensorView::new(&style.dp_data, &style.dp_shape);
            let style_ttl_tv = TensorView::new(&style.ttl_data, &style.ttl_shape);

            // 1. Duration Predictor
            let t_dp = std::time::Instant::now();
            let duration_tv = self.duration_predictor.forward(
                text_ids_tv.clone(),
                style_dp_tv,
                text_mask_tv.clone(),
            );
            let mut duration = duration_tv.data.to_vec();
            for d in duration.iter_mut() {
                *d /= speed;
            }
            eprintln!("[STAGE] DurationPredictor: {:.2}ms", t_dp.elapsed().as_secs_f64() * 1000.0);

            let total_duration_seconds: f32 = duration.iter().sum();
            let duration_batch = vec![total_duration_seconds];

            // 2. Text Encoder
            let t_te = std::time::Instant::now();
            let text_emb_tv =
                self.text_encoder
                    .forward(text_ids_tv, style_ttl_tv.clone(), text_mask_tv.clone());
            eprintln!("[STAGE] TextEncoder: {:.2}ms", t_te.elapsed().as_secs_f64() * 1000.0);

            // 3. Vector Estimator (Loop)
            let (mut xt_data, xt_shape, latent_mask_data, latent_mask_shape) = sample_noisy_latent(
                &duration_batch,
                self.config.ae.sample_rate,
                self.config.ae.base_chunk_size,
                self.config.ttl.chunk_compress_factor,
                self.config.ttl.latent_dim,
            );

            let total_step_data = vec![steps as f32; bsz];
            let total_step_shape = [bsz];
            let total_step_tv = TensorView::new(&total_step_data, &total_step_shape);

            let t_ve = std::time::Instant::now();
            for step in 0..steps {
                let current_step_data = vec![step as f32; bsz];
                let current_step_shape = [bsz];
                let current_step_tv = TensorView::new(&current_step_data, &current_step_shape);

                let xt_tv = TensorView::new(&xt_data, &xt_shape);
                let latent_mask_tv = TensorView::new(&latent_mask_data, &latent_mask_shape);

                let t_step = std::time::Instant::now();
                let denoised_tv = self.vector_estimator.forward(
                    xt_tv,
                    text_emb_tv.clone(),
                    style_ttl_tv.clone(),
                    latent_mask_tv,
                    text_mask_tv.clone(),
                    current_step_tv,
                    total_step_tv.clone(),
                );
                eprintln!("[STAGE]   VE step {}: {:.2}ms", step, t_step.elapsed().as_secs_f64() * 1000.0);

                xt_data = denoised_tv.data.to_vec();
            }
            eprintln!("[STAGE] VectorEstimator total: {:.2}ms", t_ve.elapsed().as_secs_f64() * 1000.0);

            // Apply latent mask (zero out positions beyond sequence length)
            for d in 0..xt_shape[1] {
                for t in 0..xt_shape[2] {
                    let mask_idx = t; // mask shape is [1, latent_len]
                    let latent_idx = d * xt_shape[2] + t;
                    xt_data[latent_idx] *= latent_mask_data[mask_idx];
                }
            }

            // Denormalize for vocoder: the flow-matching model outputs latents
            // in normalized space (std≈scale). Divide by scale to recover raw
            // latent space (std≈1.0) that the vocoder/AE-decoder expects.
            let normalizer_scale = self.config.ttl.normalizer.scale;
            for v in xt_data.iter_mut() {
                *v /= normalizer_scale;
            }

            // 4. Vocoder
            let t_voc = std::time::Instant::now();
            let xt_tv = TensorView::new(&xt_data, &xt_shape);
            let audio_tv = self.vocoder.forward(xt_tv);
            eprintln!("[STAGE] Vocoder: {:.2}ms", t_voc.elapsed().as_secs_f64() * 1000.0);

            let audio_data = audio_tv.data.to_vec();
            let expected_len =
                (total_duration_seconds * self.config.ae.sample_rate as f32) as usize;
            let actual_len = audio_data.len().min(expected_len);

            // Clip to [-1, 1] to prevent distortion from slight overflows
            for &sample in &audio_data[..actual_len] {
                full_audio.push(sample.clamp(-1.0, 1.0));
            }
        }

        Ok(full_audio)
    }
}

fn main() -> Result<()> {
    let text = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Hello, this is a test of supertonic TTS in pure Rust.".to_string());
    let output_path = "output.wav";
    let lang = "en";
    let style_name = "M1";

    println!("=== Supertonic TTS Pure Rust Inference ===");
    println!("Text: {}", text);

    let weights_dir = Path::new("examples/supertonic/onnx");
    let gen_dir = Path::new("examples/supertonic/src");
    let voice_styles_dir = Path::new("examples/supertonic/voice_styles");
    let config_path = weights_dir.join("tts.json");

    // Warmup lele GEMM to avoid ~25ms first-call overhead for various matrix sizes
    {
        // Various sizes used in supertonic models
        let warmup_sizes = vec![
            (256, 64, 63),   // DurationPredictor
            (1024, 256, 62), // TextEncoder/VE
            (512, 1024, 62), // VE
            (1024, 512, 62), // VE
            (2048, 512, 372), // Vocoder
            (512, 2048, 372), // Vocoder
        ];
        for &(m, k, n) in warmup_sizes.iter() {
            let warmup_a_data = vec![0.0f32; m * k];
            let warmup_b_data = vec![0.0f32; k * n];
            let mut warmup_out = Vec::new();
            let shape_a = [1usize, m, k];
            let shape_b = [1usize, k, n];
            let a_tv = lele::tensor::TensorView::new(&warmup_a_data, &shape_a);
            let b_tv = lele::tensor::TensorView::new(&warmup_b_data, &shape_b);
            let _ = lele::kernels::matmul(&a_tv, &b_tv, &mut warmup_out);
        }
    }

    println!("Loading weights... {}", gen_dir.display());
    let te_weights = fs::read(gen_dir.join("textencoder_weights.bin"))
        .context(format!("{}/textencoder_weights.bin", gen_dir.display()))?;
    let dp_weights = fs::read(gen_dir.join("durationpredictor_weights.bin"))?;
    let ve_weights = fs::read(gen_dir.join("vectorestimator_weights.bin"))?;
    let vo_weights = fs::read(gen_dir.join("vocoder_weights.bin"))?;

    let mut tts = SupertonicTts::new(
        weights_dir,
        &config_path,
        voice_styles_dir,
        &te_weights,
        &dp_weights,
        &ve_weights,
        &vo_weights,
    )?;

    println!("Synthesizing...");
    // Pre-load style before timing (matches ORT benchmark which loads before timing starts)
    tts.load_style(style_name)?;
    let start = Instant::now();
    let audio = tts.synthesize(&text, lang, style_name, 1.0, 5)?;
    let elapsed = start.elapsed().as_secs_f64();
    let audio_duration = audio.len() as f64 / tts.config.ae.sample_rate as f64;
    let rtf = elapsed / audio_duration;

    println!("✓ Synthesized in {:.2}s", elapsed);
    println!("✓ Audio duration: {:.2}s", audio_duration);
    println!("✓ Real-time factor (RTF): {:.4}x", rtf);

    lele::profiling::print_report();

    WavWriter::save(output_path, &audio, tts.config.ae.sample_rate as u32)?;
    println!("✓ Saved to {}", output_path);

    Ok(())
}
