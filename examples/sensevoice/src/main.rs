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
        .or_else(|_| std::fs::read("src/sensevoice_weights.bin"))
        .expect("Failed to load weights. Make sure sensevoice_weights.bin is in the current directory or examples/sensevoice/");

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
    println!(
        "✓ CMVN normalized: shape {:?}, took {:.2}ms",
        normalized_features.shape,
        start.elapsed().as_secs_f64() * 1000.0
    );
    // Dump features for Python verification
    use std::io::Write;
    let mut file = std::fs::File::create("features.bin").unwrap();
    for x in normalized_features.data.iter() {
        file.write_all(&x.to_le_bytes()).unwrap();
    }
    println!("✓ Dumped features to features.bin");

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

    // Run inference
    println!("\nRunning forward pass...");
    let start = Instant::now();

    let output = model
        .forward(speech, speech_lengths, language, text_norm)
        .into_logits();

    let elapsed = start.elapsed();
    let e2e_elapsed = e2e_start.elapsed();

    let audio_duration_sec = audio.len() as f64 / sample_rate as f64;
    let rtf = elapsed.as_secs_f64() / audio_duration_sec;
    let e2e_rtf = e2e_elapsed.as_secs_f64() / audio_duration_sec;

    println!(
        "✓ Inference completed in {:.2}ms",
        elapsed.as_secs_f64() * 1000.0
    );
    println!("✓ Model RTF: {:.4}", rtf);
    println!(
        "✓ Total RTF: {:.4} (Audio: {:.2}s)",
        e2e_rtf, audio_duration_sec
    );
    println!("✓ Output shape: {:?}", output.shape);

    // Load tokenizer
    println!("\n--- Tokenization ---");
    let tokenizer = Tokenizer::from_file("examples/sensevoice/sensevoice.int8.tokens.txt")
        .or_else(|_| Tokenizer::from_file("sensevoice.int8.tokens.txt"))
        .expect("Failed to load tokenizer file");

    println!("✓ Tokenizer loaded: {} tokens", tokenizer.vocab_size());

    // Decode logits to text
    let batch_size = output.size(0);
    let time_steps = output.size(1);
    let vocab_size = output.size(2);

    println!(
        "Decoding logits [batch={}, time={}, vocab={}]...",
        batch_size, time_steps, vocab_size
    );

    // Show some sample predictions for debugging
    println!("\nSample token predictions:");
    for t in 0..time_steps {
        let offset = t * vocab_size;
        let logit_slice = &output.data[offset..offset + vocab_size];

        // Get top 1 prediction
        let (max_idx, max_val) = logit_slice
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        if max_idx != 0 {
            println!("  t={}: {} ({:.2})", t, max_idx, max_val);
        }
    }

    // Also show logits statistics
    let logits_min = output.data.iter().cloned().fold(f32::INFINITY, f32::min);
    let logits_max = output
        .data
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let logits_mean = output.data.iter().sum::<f32>() / output.data.len() as f32;
    println!(
        "\nLogits stats: min={:.2}, max={:.2}, mean={:.2}",
        logits_min, logits_max, logits_mean
    );

    // Check if one token dominates across all time steps
    let mut token_counts = vec![0; vocab_size];
    for t in 0..time_steps {
        let offset = t * vocab_size;
        let logit_slice = &output.data[offset..offset + vocab_size];
        let token_id = logit_slice
            .iter()
            .enumerate()
            .max_by(|(_, a): &(usize, &f32), (_, b): &(usize, &f32)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        token_counts[token_id] += 1;
    }
    let (most_common_token, count) = token_counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, count)| count)
        .unwrap();
    println!(
        "Most common predicted token: {} ({}/{} time steps = {:.1}%)",
        most_common_token,
        count,
        time_steps,
        100.0 * *count as f32 / time_steps as f32
    );

    let texts = tokenizer.decode_greedy(&output.data, batch_size, time_steps, vocab_size);

    println!("\n=== Transcription Result ===");
    for (i, text) in texts.iter().enumerate() {
        println!("Batch {}: \"{}\"", i, text);
    }

    println!("\n=== Success! ===");
    println!("SenseVoice full pipeline completed successfully.");
}
