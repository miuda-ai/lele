//! Hojo-TTS-Light pure-Rust inference (Token-LM TTS).
//!
//! Pipeline mirrors `infer_onnx.py` from the Hojo-TTS-Light repo:
//!   prompt wav -> [Encoder] -> audio codes
//!   (prompt tokens) -> [LLM autoregressive] -> speech token codes
//!   codes -> [Decoder] -> 24kHz waveform
//!
//! The three ONNX sub-models are AOT-compiled to Rust by `build.rs`; if any of
//! them fails to compile the crate still builds (a stub panics with a clear
//! message at runtime), so `cargo build` always succeeds.

mod audio;
mod decoder;
mod encoder;
mod llm;
mod llm_cache;
mod tokenizer;

use anyhow::{Context, Result, bail};
use lele::tensor::TensorView;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use std::time::Instant;

use crate::decoder::Decoder;
use crate::encoder::Encoder;
use crate::llm::Llm;
use crate::tokenizer::HojoTokenizer;

const CODEC_SAMPLE_RATE: u32 = 16000;
const OUTPUT_SAMPLE_RATE: u32 = 24000;

// Special tokens (see infer_onnx.py).
const SPEECH_TOKEN_PATTERN_OPEN: &str = "[";
const SPEECH_TOKEN_PATTERN_CLOSE: &str = "]";
const SPEECH_START_TOKEN: &str = "[speech_start]";
const SPEECH_END_TOKEN: &str = "[speech_end]";
const REF_TEXT_START_TOKEN: &str = "[ref_text_start]";
const REF_TEXT_END_TOKEN: &str = "[ref_text_end]";
const TARGET_TEXT_START_TOKEN: &str = "[target_text_start]";
const TARGET_TEXT_END_TOKEN: &str = "[target_text_end]";
const REF_SPEECH_START_TOKEN: &str = "[ref_speech_start]";
const REF_SPEECH_END_TOKEN: &str = "[ref_speech_end]";
const TARGET_SPEECH_START_TOKEN: &str = "[target_speech_start]";
const TARGET_SPEECH_END_TOKEN: &str = "[target_speech_end]";

struct GenArgs {
    text: String,
    prompt_wav: String,
    prompt_text: String,
    output: String,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    max_new_tokens: usize,
    min_new_tokens: usize,
    repetition_penalty: f32,
    seed: u64,
}

fn parse_args() -> GenArgs {
    let mut a = GenArgs {
        text: "今天天气怎么样。".to_string(),
        prompt_wav: "examples/hojo-tts/assets/zh1.wav".to_string(),
        prompt_text: "现在的外卖确实坑多，要不咱换家稍微贵点的？可能品质好点。".to_string(),
        output: "output_hojo.wav".to_string(),
        temperature: 0.8,
        top_p: 0.95,
        top_k: 0,
        max_new_tokens: 2048,
        min_new_tokens: 10,
        repetition_penalty: 1.1,
        seed: 42,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--prompt-wav" => a.prompt_wav = args.next().unwrap_or(a.prompt_wav.clone()),
            "--prompt-text" => a.prompt_text = args.next().unwrap_or(a.prompt_text.clone()),
            "--output" => a.output = args.next().unwrap_or(a.output.clone()),
            "--temperature" => {
                a.temperature = args.next().and_then(|s| s.parse().ok()).unwrap_or(a.temperature)
            }
            "--top-p" => a.top_p = args.next().and_then(|s| s.parse().ok()).unwrap_or(a.top_p),
            "--top-k" => a.top_k = args.next().and_then(|s| s.parse().ok()).unwrap_or(a.top_k),
            "--max-new-tokens" => {
                a.max_new_tokens = args.next().and_then(|s| s.parse().ok()).unwrap_or(a.max_new_tokens)
            }
            "--min-new-tokens" => {
                a.min_new_tokens = args.next().and_then(|s| s.parse().ok()).unwrap_or(a.min_new_tokens)
            }
            "--repetition-penalty" => {
                a.repetition_penalty =
                    args.next().and_then(|s| s.parse().ok()).unwrap_or(a.repetition_penalty)
            }
            "--seed" => a.seed = args.next().and_then(|s| s.parse().ok()).unwrap_or(a.seed),
            _ => a.text = arg,
        }
    }
    a
}

/// Resolve the generation control tokens, falling back like the reference
/// script when the "target_*" variants are absent from the vocab.
struct GenTokens {
    ref_text_start: &'static str,
    ref_text_end: &'static str,
    target_text_start: &'static str,
    target_text_end: &'static str,
    target_speech_end: &'static str,
    ref_speech_start: &'static str,
    ref_speech_end: &'static str,
    target_speech_start: &'static str,
}

fn token_present(tok: &str, tk: &HojoTokenizer, unk_id: i64) -> bool {
    match tk.convert_token_to_id(tok) {
        Some(id) => id as i64 != unk_id,
        None => false,
    }
}

fn resolve_gen_tokens(tk: &HojoTokenizer, unk_id: i64) -> GenTokens {
    let target_speech_end = if token_present(TARGET_SPEECH_END_TOKEN, tk, unk_id) {
        TARGET_SPEECH_END_TOKEN
    } else {
        SPEECH_END_TOKEN
    };
    let (ref_speech_start, ref_speech_end, target_speech_start) =
        if token_present(TARGET_SPEECH_START_TOKEN, tk, unk_id) {
            (
                REF_SPEECH_START_TOKEN,
                REF_SPEECH_END_TOKEN,
                TARGET_SPEECH_START_TOKEN,
            )
        } else {
            (SPEECH_START_TOKEN, SPEECH_END_TOKEN, SPEECH_START_TOKEN)
        };
    GenTokens {
        ref_text_start: REF_TEXT_START_TOKEN,
        ref_text_end: REF_TEXT_END_TOKEN,
        target_text_start: TARGET_TEXT_START_TOKEN,
        target_text_end: TARGET_TEXT_END_TOKEN,
        target_speech_end,
        ref_speech_start,
        ref_speech_end,
        target_speech_start,
    }
}

fn build_prompt(g: &GenTokens, ref_text: &str, target_text: &str, speech_tokens_str: &str) -> String {
    format!(
        "{rs}{ref_text}{re} {tts}{target_text}{tte}{rss}{speech}{rse}{tss}",
        rs = g.ref_text_start,
        ref_text = ref_text,
        re = g.ref_text_end,
        tts = g.target_text_start,
        target_text = target_text,
        tte = g.target_text_end,
        rss = g.ref_speech_start,
        speech = speech_tokens_str,
        rse = g.ref_speech_end,
        tss = g.target_speech_start,
    )
}

/// Build {token_id -> audio code} for speech tokens of the form `[<int>]`.
fn build_id_to_code(tk: &HojoTokenizer) -> HashMap<i64, i64> {
    let mut map = HashMap::new();
    for id in 0..(tk.vocab_size() as u32) {
        if let Some(s) = tk.id_to_token_str(id) {
            if let Some(code) = parse_speech_token(s) {
                map.insert(id as i64, code);
            }
        }
    }
    map
}

fn parse_speech_token(s: &str) -> Option<i64> {
    let inner = s.strip_prefix(SPEECH_TOKEN_PATTERN_OPEN)?;
    let inner = inner.strip_suffix(SPEECH_TOKEN_PATTERN_CLOSE)?;
    if inner.is_empty() {
        return None;
    }
    // Allow a leading '-'.
    let mut chars = inner.chars();
    let first = chars.next().unwrap();
    if !(first == '-' || first.is_ascii_digit()) {
        return None;
    }
    if !chars.all(|c| c.is_ascii_digit()) {
        return None;
    }
    inner.parse::<i64>().ok()
}

/// HF-style sampling: repetition penalty -> temperature -> top-k -> top-p.
fn sample_next(
    logits_row: &[f32],
    cur_ids: &[i64],
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repetition_penalty: f32,
    rng: &mut ChaCha8Rng,
) -> usize {
    use rand::Rng;
    let n = logits_row.len();
    let mut logits: Vec<f64> = logits_row.iter().map(|&v| v as f64).collect();

    if repetition_penalty != 1.0 {
        let mut seen = std::collections::HashSet::new();
        for &tid in cur_ids {
            let tid = tid as usize;
            if tid < n && seen.insert(tid) {
                if logits[tid] < 0.0 {
                    logits[tid] *= repetition_penalty as f64;
                } else {
                    logits[tid] /= repetition_penalty as f64;
                }
            }
        }
    }

    if temperature <= 0.0 {
        return argmax(&logits);
    }

    for l in logits.iter_mut() {
        *l /= temperature as f64;
    }

    if top_k > 0 && top_k < n {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
        let keep: std::collections::HashSet<usize> = idx.iter().take(top_k).copied().collect();
        for (i, l) in logits.iter_mut().enumerate() {
            if !keep.contains(&i) {
                *l = f64::NEG_INFINITY;
            }
        }
    }

    // softmax
    let maxv = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut probs: Vec<f64> = logits.iter().map(|&l| (l - maxv).exp()).collect();
    let sum: f64 = probs.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return argmax(&logits);
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }

    if (0.0..1.0).contains(&top_p) {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));
        let mut cum = 0.0;
        let mut keep = vec![false; n];
        for (rank, &i) in idx.iter().enumerate() {
            keep[i] = true;
            cum += probs[i];
            if cum >= top_p as f64 && rank > 0 {
                break;
            }
        }
        let mut new_sum = 0.0;
        for (i, p) in probs.iter_mut().enumerate() {
            if !keep[i] {
                *p = 0.0;
            } else {
                new_sum += *p;
            }
        }
        if new_sum <= 0.0 {
            return idx[0];
        }
        for p in probs.iter_mut() {
            *p /= new_sum;
        }
    }

    let r: f64 = rng.gen_range(0.0f64..1.0f64);
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r <= acc {
            return i;
        }
    }
    n - 1
}

fn argmax(xs: &[f64]) -> usize {
    xs.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn main() -> Result<()> {
    let args = parse_args();

    println!("=== Hojo-TTS-Light Pure Rust Inference ===");
    println!("Target text : {}", args.text);
    println!("Prompt wav  : {}", args.prompt_wav);
    println!("Prompt text : {}", args.prompt_text);

    let gen_dir = std::path::Path::new("examples/hojo-tts/src");
    let tok_path = std::path::Path::new("examples/hojo-tts/tokenizer/tokenizer.json");

    println!("Loading tokenizer...");
    let tokenizer = HojoTokenizer::from_path(tok_path)
        .with_context(|| format!("load tokenizer {:?}", tok_path))?;
    println!("Tokenizer vocab size: {}", tokenizer.vocab_size());

    let unk_id = tokenizer.convert_token_to_id("[UNK]").unwrap_or(0) as i64;
    let gen_tokens = resolve_gen_tokens(&tokenizer, unk_id);
    let speech_end_id = tokenizer
        .convert_token_to_id(gen_tokens.target_speech_end)
        .context("speech_end token missing from vocab")? as i64;

    println!("Loading compiled weights...");
    let enc_weights = std::fs::read(gen_dir.join("encoder_weights.bin"))
        .context("encoder_weights.bin")?;
    let llm_weights = std::fs::read(gen_dir.join("llm_weights.bin")).context("llm_weights.bin")?;
    let dec_weights = std::fs::read(gen_dir.join("decoder_weights.bin"))
        .context("decoder_weights.bin")?;

    let encoder = Encoder::new(&enc_weights);
    let llm = Llm::new(&llm_weights);
    let decoder = Decoder::new(&dec_weights);

    // 1. Load + resample prompt audio to 16kHz.
    let (wav, sr) = audio::WavReader::load(&args.prompt_wav)
        .with_context(|| format!("read prompt wav {}", args.prompt_wav))?;
    let prompt_wav = audio::resample_sinc(&wav, sr, CODEC_SAMPLE_RATE);
    if prompt_wav.is_empty() {
        bail!("Prompt audio is empty");
    }
    println!(
        "Prompt audio: {} samples @ {}Hz -> {} @ {}Hz",
        wav.len(),
        sr,
        prompt_wav.len(),
        CODEC_SAMPLE_RATE
    );

    // 2. Encoder: wav [1,1,T] (f32) -> codes (i64).
    let t_enc = Instant::now();
    let enc_shape = [1usize, 1, prompt_wav.len()];
    let wav_tv = TensorView::new(&prompt_wav, &enc_shape);
    let enc_out = encoder.forward(wav_tv);
    let prompt_codes: Vec<i64> = enc_out.data.iter().map(|&v| v).collect();
    let enc_time = t_enc.elapsed().as_secs_f64();
    eprintln!(
        "[STAGE] Encoder: {:.2}ms ({} codes)",
        enc_time * 1000.0,
        prompt_codes.len()
    );
    if let Some(dir) = std::env::var("HOJO_DUMP").ok() {
        let path = format!("{}/enc_codes.bin", dir);
        let bytes: Vec<u8> = prompt_codes.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, &bytes).ok();
        eprintln!("[DUMP] enc_codes -> {} ({} vals)", path, prompt_codes.len());
    }
    if prompt_codes.is_empty() {
        bail!("Encoder produced no codes");
    }

    // 3. Build prompt token string + encode.
    let speech_tokens_str: String = prompt_codes
        .iter()
        .map(|c| format!("[{}]", c))
        .collect();
    let prompt_str = build_prompt(
        &gen_tokens,
        &args.prompt_text,
        &args.text,
        &speech_tokens_str,
    );
    let input_ids = tokenizer.encode(&prompt_str, true);
    println!("Prompt token ids: {}", input_ids.len());
    if let Some(dir) = std::env::var("HOJO_DUMP").ok() {
        let path = format!("{}/input_ids.bin", dir);
        let bytes: Vec<u8> = input_ids.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, &bytes).ok();
        eprintln!("[DUMP] input_ids -> {} ({} vals)", path, input_ids.len());
    }

    let id_to_code = build_id_to_code(&tokenizer);

    // 4. Autoregressive generation.
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    let mut cur = input_ids.clone();
    let mut generated: Vec<i64> = Vec::new();
    let gen_start = Instant::now();

    let mut cache = llm_cache::LlmCache::new(&llm);
    let logits = llm.forward_prompt(&mut cache, &cur);

    let vocab = logits.len();
    let row = &logits[..];

    let mut next_id = sample_next(
        row,
        &cur,
        args.temperature,
        args.top_p,
        args.top_k,
        args.repetition_penalty,
        &mut rng,
    ) as i64;

    generated.push(next_id);
    cur.push(next_id);

    if !(next_id == speech_end_id && 1 >= args.min_new_tokens) {
        for step in 1..args.max_new_tokens {
            let logits = llm.forward_token(&mut cache, next_id);

            let row = &logits[..];

            let next_id_new = sample_next(
                row,
                &cur,
                args.temperature,
                args.top_p,
                args.top_k,
                args.repetition_penalty,
                &mut rng,
            ) as i64;

            generated.push(next_id_new);
            cur.push(next_id_new);

            if next_id_new == speech_end_id && step + 1 >= args.min_new_tokens {
                break;
            }
            next_id = next_id_new;
        }
    }
    let llm_time = gen_start.elapsed().as_secs_f64();
    eprintln!(
        "[STAGE] LLM generation: {:.2}s ({} tokens)",
        llm_time,
        generated.len()
    );

    // 5. Extract audio codes from the generated tail (up to speech_end).
    let eos_pos = generated.iter().position(|&id| id == speech_end_id);
    let tail_end = eos_pos.unwrap_or(generated.len());
    let audio_codes: Vec<i64> = generated[..tail_end]
        .iter()
        .filter_map(|id| id_to_code.get(id).copied())
        .collect();

    if let Some(dir) = std::env::var("HOJO_DUMP").ok() {
        let path = format!("{}/generated.bin", dir);
        let bytes: Vec<u8> = generated.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, &bytes).ok();
        let path2 = format!("{}/audio_codes.bin", dir);
        let bytes2: Vec<u8> = audio_codes.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path2, &bytes2).ok();
        eprintln!("[DUMP] generated+audio_codes -> {} vals + {} vals", generated.len(), audio_codes.len());
    }

    if audio_codes.is_empty() {
        bail!("No audio codes were produced by the LLM");
    }
    println!("Generated audio codes: {}", audio_codes.len());

    // 6. Decoder: codes [1,1,N] (i64) -> wav (f32).
    let t_dec = Instant::now();
    let dec_shape = [1usize, 1, audio_codes.len()];
    let codes_tv = TensorView::new(&audio_codes, &dec_shape);
    let wav_out = decoder.forward(codes_tv);
    let samples: Vec<f32> = wav_out.data.iter().copied().collect();
    if let Some(dir) = std::env::var("HOJO_DUMP").ok() {
        let path = format!("{}/dec_output.bin", dir);
        let bytes: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, &bytes).ok();
        eprintln!("[DUMP] dec_output -> {} ({} vals)", path, samples.len());
    }
    let dec_time = t_dec.elapsed().as_secs_f64();
    eprintln!(
        "[STAGE] Decoder: {:.2}ms ({} samples)",
        dec_time * 1000.0,
        samples.len()
    );

    let elapsed = enc_time + llm_time + dec_time;
    let audio_duration = samples.len() as f64 / OUTPUT_SAMPLE_RATE as f64;
    let rtf = if audio_duration > 0.0 {
        elapsed / audio_duration
    } else {
        0.0
    };
    println!("Synthesized {:.2}s of audio in {:.2}s (RTF {:.4})", audio_duration, elapsed, rtf);

    audio::WavWriter::save(&args.output, &samples, OUTPUT_SAMPLE_RATE)
        .with_context(|| format!("save output {}", args.output))?;
    println!("Saved to {}", args.output);
    Ok(())
}
