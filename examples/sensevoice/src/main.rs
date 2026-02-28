mod audio;
mod sensevoice;
mod tokenizer;

use audio::WavReader;
use lele::features::{Cmvn, FeatureConfig, SenseVoiceFrontend};
use lele::tensor::{IntoLogits, TensorView};
use std::env;
use std::time::Instant;
use tokenizer::Tokenizer;

fn main() {
    println!("=== SenseVoice Pure Rust Inference ===\n");

    // Load model weights
    println!("Loading SenseVoice weights...");
    let bin = std::fs::read("examples/sensevoice/src/sensevoice_weights.bin")
        .or_else(|_| std::fs::read("examples/sensevoice/sensevoice_weights.bin"))
        .or_else(|_| std::fs::read("sensevoice_weights.bin"))
        .expect("Failed to load weights. Make sure sensevoice_weights.bin is in examples/sensevoice/src/");

    let model = sensevoice::SenseVoice::new(&bin);
    println!(
        "✓ Model loaded ({:.2} MB)\n",
        bin.len() as f64 / 1024.0 / 1024.0
    );

    // Load audio (from command line arg or use dummy data)
    let (audio, sample_rate) = if let Some(wav_path) = env::args().nth(1) {
        println!("Loading audio from: {}", wav_path);
        match WavReader::load(&wav_path) {
            Ok((samples, sr)) => {
                println!("✓ Loaded {} samples at {} Hz\n", samples.len(), sr);
                (samples, sr as usize)
            }
            Err(e) => {
                eprintln!("Warning: Failed to load audio file: {}", e);
                eprintln!("Using synthetic audio instead\n");
                let sr = 16000;
                let samples = vec![0.01; sr * 3];
                (samples, sr)
            }
        }
    } else {
        println!("No audio file specified, using synthetic audio");
        let sr = 16000;
        let duration_secs = 3.0;
        let num_samples = (sr as f32 * duration_secs) as usize;
        println!(
            "Creating test audio: {} samples at {} Hz\n",
            num_samples, sr
        );
        (vec![0.01; num_samples], sr)
    };

    // Create frontend pipeline
    println!("--- Feature Extraction ---");
    let config = FeatureConfig {
        sample_rate,
        n_mels: 80,
        frame_length_ms: 25.0,
        frame_shift_ms: 10.0,
        lfr_m: 7,
        lfr_n: 6,
    };

    let frontend = SenseVoiceFrontend::new(config);

    let e2e_start = Instant::now();

    // Extract features (Mel + LFR combined)
    let start = Instant::now();
    let features = frontend.compute(&audio);
    println!(
        "✓ Features extracted: shape {:?}, took {:.2}ms",
        features.shape,
        start.elapsed().as_secs_f64() * 1000.0
    );

    // Apply CMVN normalization
    // SenseVoice usually uses Global CMVN, and the working log shows mean=0.0
    // let normalized_features = features;
    let cmvn = Cmvn::default();

    let start = Instant::now();
    let normalized_features = cmvn.compute(&features);
    let cmvn_time = start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "✓ CMVN normalized: shape {:?}, took {:.2}ms",
        normalized_features.shape, cmvn_time
    );
    // Debug: Print feature statistics
    let feat_min = normalized_features
        .data
        .iter()
        .cloned()
        .fold(f32::INFINITY, f32::min);
    let feat_max = normalized_features
        .data
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let feat_mean =
        normalized_features.data.iter().sum::<f32>() / normalized_features.data.len() as f32;
    println!(
        "Features stats: min={:.3}, max={:.3}, mean={:.3}",
        feat_min, feat_max, feat_mean
    );
    // Prepare model inputs
    println!("\n--- Model Inference ---");

    // Reshape to [1, T, D] if needed
    let (t, d) = (normalized_features.size(0), normalized_features.size(1));
    let speech = if normalized_features.dim() == 2 {
        TensorView::from_owned(normalized_features.data.to_vec(), vec![1, t, d])
    } else {
        normalized_features
    };

    let speech_lengths = TensorView::from_owned(vec![t as i64], vec![1]);
    let language = TensorView::from_owned(vec![3i64], vec![1]); // Shape [1]
    let text_norm = TensorView::from_owned(vec![0i64], vec![1]); // Shape [1], 1=with itn

    println!("Input shapes:");
    println!("  speech: {:?}", speech.shape);
    println!("  speech_lengths: {:?}", speech_lengths.shape);
    println!("  language: {:?} (3=Chinese)", language.shape);
    println!("  text_norm: {:?}", text_norm.shape);

    // Clone inputs for benchmark (need to do this before first forward which moves them)
    let speech_bench = speech.to_owned();
    let speech_lengths_bench = speech_lengths.to_owned();
    let language_bench = language.to_owned();
    let text_norm_bench = text_norm.to_owned();

    // Run inference - single pass for accurate RTF measurement
    let start = Instant::now();

    let output = model
        .forward(speech, speech_lengths, language, text_norm)
        .into_logits();

    let inference_elapsed = start.elapsed();
    let e2e_elapsed = e2e_start.elapsed();

    let audio_duration_sec = audio.len() as f64 / sample_rate as f64;
    let model_rtf = inference_elapsed.as_secs_f64() / audio_duration_sec;
    let e2e_rtf = e2e_elapsed.as_secs_f64() / audio_duration_sec;

    println!(
        "✓ Inference completed in {:.2}ms",
        inference_elapsed.as_secs_f64() * 1000.0
    );
    println!("✓ Model RTF: {:.4}", model_rtf);
    println!(
        "✓ Total pipeline RTF: {:.4} (Audio: {:.2}s)",
        e2e_rtf, audio_duration_sec
    );
    println!(
        "✓ Pipeline overhead: {:.1}%",
        (e2e_rtf - model_rtf) / e2e_rtf * 100.0
    );
    println!("✓ Output shape: {:?}", output.shape);

    // Load tokenizer
    println!("\n--- Tokenization ---");
    let tokenizer = Tokenizer::from_file("examples/sensevoice/sensevoice.int8.tokens.txt")
        .or_else(|_| Tokenizer::from_file("sensevoice.int8.tokens.txt"))
        .expect("Failed to load tokenizer file");

    println!("✓ Tokenizer loaded: {} tokens", tokenizer.vocab_size());

    // Decode logits to text - handle both 2D [T, V] and 3D [B, T, V] outputs
    let (time_steps, vocab_size) = if output.dim() == 3 {
        (output.size(1), output.size(2))
    } else {
        (output.size(0), output.size(1))
    };

    println!(
        "Decoding logits [time={}, vocab={}]...",
        time_steps, vocab_size
    );

    // Decode the output
    let start = Instant::now();
    let texts = tokenizer.decode_greedy(&output.data, 1, time_steps, vocab_size);
    println!(
        "✓ Decoding took {:.2}ms",
        start.elapsed().as_secs_f64() * 1000.0
    );

    println!("\n=== Transcription Result ===");
    for (i, text) in texts.iter().enumerate() {
        println!("Batch {}: \"{}\"", i, text);
    }

    // Benchmark: Run multiple iterations to show steady-state performance
    println!("\n=== Performance Benchmark ===");
    println!("Running 10 warm iterations...");

    let mut times = Vec::new();
    for _ in 0..10 {
        let bench_start = Instant::now();
        let _ = model
            .forward(
                speech_bench.clone(),
                speech_lengths_bench.clone(),
                language_bench.clone(),
                text_norm_bench.clone(),
            )
            .into_logits();
        times.push(bench_start.elapsed().as_secs_f64() * 1000.0);
    }

    let avg_time: f64 = times.iter().sum::<f64>() / times.len() as f64;
    let min_time = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_time = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let avg_rtf = avg_time / 1000.0 / audio_duration_sec;
    let min_rtf = min_time / 1000.0 / audio_duration_sec;

    println!(
        "✓ Average inference: {:.2}ms (RTF: {:.4})",
        avg_time, avg_rtf
    );
    println!(
        "✓ Min/Max: {:.2}ms / {:.2}ms (RTF: {:.4} / {:.4})",
        min_time,
        max_time,
        min_rtf,
        max_time / 1000.0 / audio_duration_sec
    );

    println!("\n=== Success! ===");
    println!("SenseVoice full pipeline completed successfully.");
}
