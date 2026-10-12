mod audio;
mod silerovad;

use audio::WavReader;
use lele::tensor::TensorView;
use std::env;
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
struct VadConfig {
    pub threshold: f32,
    pub min_silence_ms: f32,
    pub min_speech_ms: f32,
    pub speech_pad_ms: f32,
    pub merge_gap_ms: f32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.3,
            min_silence_ms: 200.0,
            min_speech_ms: 400.0,
            speech_pad_ms: 120.0,
            merge_gap_ms: 200.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VadSegment {
    pub start: usize,
    pub end: usize,
}

fn main() {
    println!("=== Silero VAD Pure Rust Inference ===\n");
    println!("CWD: {:?}", std::env::current_dir());

    // 1. Load Weights
    println!("Loading Silero weights...");
    let bin = std::fs::read("examples/silero/src/silerovad_weights.bin")
        .or_else(|_| std::fs::read("silerovad_weights.bin"))
        .or_else(|_| std::fs::read("src/silerovad_weights.bin"))
        .expect("Failed to load weights.");
    let model = silerovad::SileroVad::new(&bin);
    println!(
        "✓ Model loaded ({:.2} MB)\n",
        bin.len() as f64 / 1024.0 / 1024.0
    );

    // 2. Load Audio
    let (audio, sample_rate) = if let Some(wav_path) = env::args().nth(1) {
        println!("Loading audio from: {}", wav_path);
        let (samples, sr) = WavReader::load(&wav_path).expect("Failed to load wav");
        (samples, sr as u32)
    } else {
        let default_wav = "fixtures/zh.wav";
        if std::path::Path::new(default_wav).exists() {
            println!("No audio file specified, using default: {}", default_wav);
            let (samples, sr) = WavReader::load(default_wav).expect("Failed to load default wav");
            (samples, sr as u32)
        } else {
            println!("No audio file specified and default not found, using synthetic audio");
            (vec![0.0; 16000 * 3], 16000)
        }
    };

    // 3. Inference Loop
    let chunk_size = 512;
    // Pad to multiple of chunk_size
    let remainder = audio.len() % chunk_size;
    let padded_len = if remainder > 0 {
        audio.len() + chunk_size - remainder
    } else {
        audio.len()
    };
    let mut padded_audio = audio.clone();
    padded_audio.resize(padded_len, 0.0);

    let num_chunks = padded_len / chunk_size;
    println!(
        "Processing {} chunks ({} samples at {} Hz)...",
        num_chunks, padded_len, sample_rate
    );

    // Initial state: [2, 1, 128] zeros
    let mut state_data = vec![0.0f32; 2 * 1 * 128];
    // SR: [1] - needs to be i64
    let sr_data = vec![sample_rate as i64];

    let start_total = Instant::now();
    let mut all_outputs = Vec::with_capacity(num_chunks);
    let mut ws = silerovad::SileroVadWorkspace::new();

    // Warmup run
    {
        let input_data: Vec<f32> = padded_audio[..chunk_size]
            .iter()
            .map(|&x| x * 32768.0)
            .collect();
        let input = TensorView::from_owned(input_data, vec![1, chunk_size]);
        let state = TensorView::from_owned(state_data.clone(), vec![2, 1, 128]);
        let sr = TensorView::from_slice(&sr_data, vec![1]);
        let (output, new_state) = model.forward_with_workspace(&mut ws, input, state, sr);
        if let Some(&prob) = output.data.get(0) {
            all_outputs.push(prob);
        }
        state_data = new_state.data.to_vec();
    }

    let start_infer = Instant::now();
    for i in 1..num_chunks {
        let chunk_start = i * chunk_size;
        let chunk_end = chunk_start + chunk_size;
        let chunk_data = &padded_audio[chunk_start..chunk_end];

        let input_data: Vec<f32> = chunk_data.iter().map(|&x| x * 32768.0).collect();
        let input = TensorView::from_owned(input_data, vec![1, chunk_size]);
        let state = TensorView::from_owned(state_data.clone(), vec![2, 1, 128]);
        let sr = TensorView::from_slice(&sr_data, vec![1]);

        // Forward
        let (output, new_state) = model.forward_with_workspace(&mut ws, input, state, sr);

        // Output is [1, 1] probability
        if let Some(&prob) = output.data.get(0) {
            all_outputs.push(prob);
        }

        // Update state
        state_data = new_state.data.to_vec();
    }
    let infer_elapsed = start_infer.elapsed();

    let total_elapsed = start_total.elapsed();
    let avg_chunk_us = infer_elapsed.as_micros() as f64 / (num_chunks - 1) as f64;
    let audio_duration = padded_len as f64 / sample_rate as f64;
    let rtf = total_elapsed.as_secs_f64() / audio_duration;

    println!("✓ Inference completed.");
    println!(
        "✓ Total Time: {:.2}ms (steady-state {:.2}ms for {} chunks)",
        total_elapsed.as_secs_f64() * 1000.0,
        infer_elapsed.as_secs_f64() * 1000.0,
        num_chunks - 1,
    );
    println!(
        "✓ Avg per chunk: {:.2}µs ({:.4}ms)",
        avg_chunk_us,
        avg_chunk_us / 1000.0
    );
    println!("✓ RTF: {:.4} (Audio: {:.2}s)", rtf, audio_duration);

    let max_prob = all_outputs.iter().cloned().fold(0.0f32, f32::max);
    println!("✓ Max Probability: {:.4}", max_prob);

    // 4. Collect Segments (Using Reference Logic)
    let config = VadConfig::default();
    let ms_to_samples = |ms: f32, sr: u32| ((sr as f32) * (ms / 1000.0)).round() as usize;

    let min_silence_samples = ms_to_samples(config.min_silence_ms, sample_rate).max(1);
    let min_speech_samples = ms_to_samples(config.min_speech_ms, sample_rate).max(1);
    let speech_pad_samples = ms_to_samples(config.speech_pad_ms, sample_rate);
    let merge_gap_samples = ms_to_samples(config.merge_gap_ms, sample_rate);

    let mut segments = Vec::new();
    let mut triggered = false;
    let mut curr_speech_start = 0usize;
    let mut silence_acc = 0usize;

    for (i, &prob) in all_outputs.iter().enumerate() {
        let offset = i * chunk_size;
        let frame_end = (offset + chunk_size).min(padded_len);

        if prob >= config.threshold {
            if !triggered {
                triggered = true;
                curr_speech_start = offset.saturating_sub(speech_pad_samples);
                silence_acc = 0;
            } else {
                silence_acc = 0;
            }
        } else if triggered {
            silence_acc += frame_end - offset;
            if silence_acc >= min_silence_samples {
                let mut end = frame_end + speech_pad_samples;
                if end > audio.len() {
                    end = audio.len();
                }
                if end > curr_speech_start && (end - curr_speech_start) >= min_speech_samples {
                    segments.push(VadSegment {
                        start: curr_speech_start,
                        end,
                    });
                }
                triggered = false;
                silence_acc = 0;
            }
        }
    }

    if triggered {
        let end = audio.len();
        if end > curr_speech_start && (end - curr_speech_start) >= min_speech_samples {
            segments.push(VadSegment {
                start: curr_speech_start,
                end,
            });
        }
    }

    // Post-process: Merge segments
    let mut merged: Vec<VadSegment> = Vec::new();
    if !segments.is_empty() {
        segments.sort_by_key(|seg| seg.start);
        for seg in segments {
            if let Some(last) = merged.last_mut() {
                if seg.start <= last.end {
                    if seg.end > last.end {
                        last.end = seg.end;
                    }
                    continue;
                }
                let gap = seg.start.saturating_sub(last.end);
                if gap <= merge_gap_samples {
                    if seg.end > last.end {
                        last.end = seg.end;
                    }
                    continue;
                }
            }
            merged.push(seg);
        }
    }

    println!("=== Detected Segments ===");
    if merged.is_empty() {
        println!("No speech detected.");
    } else {
        for (i, seg) in merged.iter().enumerate() {
            let start_sec = seg.start as f32 / sample_rate as f32;
            let end_sec = seg.end as f32 / sample_rate as f32;
            println!(
                "Segment {}: {:.2}s - {:.2}s ({:.2}s)",
                i + 1,
                start_sec,
                end_sec,
                end_sec - start_sec
            );
        }
    }
}
