mod audio;
mod sentencepiece;
mod codecdecodefull;
mod decodestep;
mod localfixedsampledframe;
mod prefill;

use anyhow::{Context, Result};
use audio::WavWriter;
use lele::tensor::TensorView;
use rand::Rng;
use sentencepiece::SentencePiece;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;

use codecdecodefull::CodecDecodeFull;
use decodestep::DecodeStep;
use localfixedsampledframe::LocalFixedSampledFrame;
use prefill::Prefill;

const N_VQ: usize = 16;
const ROW_WIDTH: usize = 17;
const AUDIO_PAD_TOKEN_ID: i64 = 1024;
const AUDIO_START_TOKEN_ID: i64 = 6;
const AUDIO_END_TOKEN_ID: i64 = 7;
const AUDIO_USER_SLOT_TOKEN_ID: i64 = 8;
const AUDIO_ASSISTANT_SLOT_TOKEN_ID: i64 = 9;

#[derive(Debug, Deserialize, Serialize)]
struct Manifest {
    tts_config: TtsConfig,
    prompt_templates: PromptTemplates,
    generation_defaults: GenerationDefaults,
    builtin_voices: Vec<BuiltinVoice>,
}

#[derive(Debug, Deserialize, Serialize)]
struct TtsConfig {
    n_vq: usize,
    #[allow(dead_code)]
    audio_pad_token_id: i32,
    pad_token_id: i32,
    im_start_token_id: i32,
    im_end_token_id: i32,
    audio_start_token_id: i32,
    audio_end_token_id: i32,
    audio_user_slot_token_id: i32,
    audio_assistant_slot_token_id: i32,
}

#[derive(Debug, Deserialize, Serialize)]
struct PromptTemplates {
    user_prompt_prefix_token_ids: Vec<i32>,
    user_prompt_after_reference_token_ids: Vec<i32>,
    assistant_prompt_prefix_token_ids: Vec<i32>,
}

#[derive(Debug, Deserialize, Serialize)]
struct GenerationDefaults {
    max_new_frames: usize,
    #[allow(dead_code)]
    sample_mode: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct BuiltinVoice {
    voice: String,
    prompt_audio_codes: Vec<Vec<i32>>,
}

fn build_text_rows(token_ids: &[i64]) -> Vec<Vec<i64>> {
    token_ids
        .iter()
        .map(|tid| {
            let mut row = vec![AUDIO_PAD_TOKEN_ID; ROW_WIDTH];
            row[0] = *tid;
            row
        })
        .collect()
}

fn build_audio_prefix_rows(prompt_audio_codes: &[Vec<i32>], slot_token_id: i64) -> Vec<Vec<i64>> {
    prompt_audio_codes
        .iter()
        .map(|codes| {
            let mut row = vec![AUDIO_PAD_TOKEN_ID; ROW_WIDTH];
            row[0] = slot_token_id;
            for (i, &c) in codes.iter().take(N_VQ).enumerate() {
                row[i + 1] = c as i64;
            }
            row
        })
        .collect()
}

fn build_request_rows(
    manifest: &Manifest,
    prompt_audio_codes: &[Vec<i32>],
    text_token_ids: &[i32],
) -> Vec<Vec<i64>> {
    let prefix_ids: Vec<i64> = manifest
        .prompt_templates
        .user_prompt_prefix_token_ids
        .iter()
        .copied()
        .map(|v| v as i64)
        .chain(std::iter::once(AUDIO_START_TOKEN_ID))
        .collect();

    let suffix_ids: Vec<i64> = std::iter::once(AUDIO_END_TOKEN_ID)
        .chain(manifest.prompt_templates.user_prompt_after_reference_token_ids.iter().map(|&v| v as i64))
        .chain(text_token_ids.iter().map(|&v| v as i64))
        .chain(manifest.prompt_templates.assistant_prompt_prefix_token_ids.iter().map(|&v| v as i64))
        .chain(std::iter::once(AUDIO_START_TOKEN_ID))
        .collect();

    let mut rows = Vec::new();
    rows.extend(build_text_rows(&prefix_ids));
    rows.extend(build_audio_prefix_rows(prompt_audio_codes, AUDIO_USER_SLOT_TOKEN_ID));
    rows.extend(build_text_rows(&suffix_ids));
    rows
}

fn flatten_rows_i64(rows: &[Vec<i64>]) -> (Vec<i64>, Vec<usize>) {
    let seq_len = rows.len();
    let mut data = Vec::with_capacity(seq_len * ROW_WIDTH);
    for row in rows {
        for &v in row {
            data.push(v);
        }
    }
    (data, vec![1, seq_len, ROW_WIDTH])
}

pub struct MossTtsNano {
    manifest: Manifest,
    tokenizer: SentencePiece,
    prefill: Prefill<'static>,
    decode_step: DecodeStep<'static>,
    local_frame: LocalFixedSampledFrame<'static>,
    codec: CodecDecodeFull<'static>,
    _prefill_weights: Vec<u8>,
    _decode_weights: Vec<u8>,
    _frame_weights: Vec<u8>,
    _codec_weights: Vec<u8>,
    prefill_ws: prefill::PrefillWorkspace,
    decode_ws: decodestep::DecodeStepWorkspace,
    frame_ws: localfixedsampledframe::LocalFixedSampledFrameWorkspace,
    codec_ws: codecdecodefull::CodecDecodeFullWorkspace,
    rng: rand::rngs::StdRng,
}

impl MossTtsNano {
    pub fn new(model_dir: &Path) -> Result<Self> {
        let tts_dir = model_dir.join("tts");
        let _codec_dir = model_dir.join("codec");

        let manifest: Manifest = serde_json::from_reader(std::fs::File::open(
            tts_dir.join("browser_poc_manifest.json"),
        )?)?;

        let tokenizer = SentencePiece::load(tts_dir.join("tokenizer.model"))
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let gen_dir = Path::new("examples/moss-tts-nano/src");

        let prefill_weights = std::fs::read(gen_dir.join("prefill_weights.bin"))?;
        let decode_weights = std::fs::read(gen_dir.join("decodestep_weights.bin"))?;
        let frame_weights = std::fs::read(gen_dir.join("localfixedsampledframe_weights.bin"))?;
        let codec_weights = std::fs::read(gen_dir.join("codecdecodefull_weights.bin"))?;

        let prefill = Prefill::new(unsafe { std::mem::transmute::<&[u8], &'static [u8]>(&prefill_weights) });
        let decode_step = DecodeStep::new(unsafe { std::mem::transmute::<&[u8], &'static [u8]>(&decode_weights) });
        let local_frame = LocalFixedSampledFrame::new(unsafe { std::mem::transmute::<&[u8], &'static [u8]>(&frame_weights) });
        let codec = CodecDecodeFull::new(unsafe { std::mem::transmute::<&[u8], &'static [u8]>(&codec_weights) });

        Ok(Self {
            manifest,
            tokenizer,
            prefill,
            decode_step,
            local_frame,
            codec,
            _prefill_weights: prefill_weights,
            _decode_weights: decode_weights,
            _frame_weights: frame_weights,
            _codec_weights: codec_weights,
            prefill_ws: prefill::PrefillWorkspace::new(),
            decode_ws: decodestep::DecodeStepWorkspace::new(),
            frame_ws: localfixedsampledframe::LocalFixedSampledFrameWorkspace::new(),
            codec_ws: codecdecodefull::CodecDecodeFullWorkspace::new(),
            rng: rand::SeedableRng::seed_from_u64(1234),
        })
    }

    pub fn encode_text(&self, text: &str) -> Vec<i32> {
        self.tokenizer.encode(text).into_iter().map(|id| id as i32).collect()
    }

    pub fn get_voice_codes(&self, voice: &str) -> Result<&Vec<Vec<i32>>> {
        self.manifest
            .builtin_voices
            .iter()
            .find(|v| v.voice == voice)
            .map(|v| &v.prompt_audio_codes)
            .with_context(|| format!("Voice '{}' not found", voice))
    }

    pub fn synthesize(&mut self, text: &str, voice: &str) -> Result<Vec<f32>> {
        let text_token_ids = self.encode_text(text);
        let prompt_audio_codes = self.get_voice_codes(voice)?.clone();

        let request_rows = build_request_rows(&self.manifest, &prompt_audio_codes, &text_token_ids);
        let seq_len = request_rows.len();

        eprintln!("[SETUP] seq_len={}, voice={}", seq_len, voice);

        let (input_data, input_shape) = flatten_rows_i64(&request_rows);
        let input_ids = TensorView::<i64>::from_owned(input_data, input_shape);

        let mask_data = vec![1i64; seq_len];
        let attention_mask = TensorView::<i64>::from_owned(mask_data, vec![1usize, seq_len]);

        // 1. Prefill
        let t0 = Instant::now();
        let prefill_out = self.prefill.forward_with_workspace(
            &mut self.prefill_ws,
            input_ids,
            attention_mask,
        );
        eprintln!("[STAGE] Prefill: {:.2}ms", t0.elapsed().as_secs_f64() * 1000.0);

        let (global_hidden, pk0, pv0, pk1, pv1, pk2, pv2, pk3, pv3, pk4, pv4,
             pk5, pv5, pk6, pv6, pk7, pv7, pk8, pv8, pk9, pv9, pk10, pv10, pk11, pv11) = prefill_out;

        let mut past_kv: Vec<TensorView<'static>> = vec![
            pk0.to_owned(), pv0.to_owned(), pk1.to_owned(), pv1.to_owned(),
            pk2.to_owned(), pv2.to_owned(), pk3.to_owned(), pv3.to_owned(),
            pk4.to_owned(), pv4.to_owned(), pk5.to_owned(), pv5.to_owned(),
            pk6.to_owned(), pv6.to_owned(), pk7.to_owned(), pv7.to_owned(),
            pk8.to_owned(), pv8.to_owned(), pk9.to_owned(), pv9.to_owned(),
            pk10.to_owned(), pv10.to_owned(), pk11.to_owned(), pv11.to_owned(),
        ];

        // Extract last hidden state: [1, seq, 768] -> [1, 768]
        let mut current_hidden = extract_last_hidden(&global_hidden);

        let mut past_valid_length = seq_len;
        let max_frames = self.manifest.generation_defaults.max_new_frames;

        let mut generated_frames: Vec<Vec<i64>> = Vec::new();
        let mut repetition_seen: Vec<std::collections::HashSet<i64>> =
            (0..N_VQ).map(|_| std::collections::HashSet::new()).collect();

        // 2. Autoregressive frame generation
        let t_gen = Instant::now();
        for step in 0..max_frames {
            // Build repetition mask [1, 16, 1024]
            let mut rep_mask = vec![0i64; N_VQ * 1024];
            for (ch, tokens) in repetition_seen.iter().enumerate() {
                for &t in tokens {
                    if (t as usize) < 1024 {
                        rep_mask[ch * 1024 + t as usize] = 1;
                    }
                }
            }
            let rep_mask_tv = TensorView::<i64>::from_owned(rep_mask, vec![1, N_VQ, 1024]);

            // Random numbers for sampling
            let rand_u: f32 = self.rng.r#gen::<f32>().min(0.99999994);
            let audio_rand: Vec<f32> = (0..N_VQ).map(|_| self.rng.r#gen::<f32>().min(0.99999994)).collect();
            let rand_tv = TensorView::from_owned(vec![rand_u], vec![1]);
            let audio_rand_tv = TensorView::from_owned(audio_rand, vec![1, N_VQ]);

            let t_frame = Instant::now();
            let (should_continue_tv, frame_ids_tv) = self.local_frame.forward_with_workspace(
                &mut self.frame_ws,
                current_hidden.clone(),
                rep_mask_tv,
                rand_tv,
                audio_rand_tv,
            );
            let frame_ms = t_frame.elapsed().as_secs_f64() * 1000.0;

            let should_continue = should_continue_tv.data.first().map(|&v| v).unwrap_or(0) != 0;
            if !should_continue {
                eprintln!("[GEN] Stop at frame {}", step);
                break;
            }

            let frame: Vec<i64> = frame_ids_tv.data.iter().copied().collect();
            for (ch, &t) in frame.iter().enumerate() {
                repetition_seen[ch].insert(t);
            }
            generated_frames.push(frame.clone());

            // Build next row for decode_step
            let mut next_row = vec![AUDIO_PAD_TOKEN_ID; ROW_WIDTH];
            next_row[0] = AUDIO_ASSISTANT_SLOT_TOKEN_ID;
            for (i, &t) in frame.iter().enumerate() {
                next_row[i + 1] = t;
            }
            let next_ids = TensorView::<i64>::from_owned(next_row, vec![1, 1, ROW_WIDTH]);
            let pvl_tv = TensorView::<i64>::from_owned(vec![past_valid_length as i64], vec![1]);

            // Run decode_step with all 24 KV caches
            let t_decode = Instant::now();
            let decode_out = self.decode_step.forward_with_workspace(
                &mut self.decode_ws,
                next_ids,
                pvl_tv,
                past_kv[0].clone(), past_kv[1].clone(), past_kv[2].clone(), past_kv[3].clone(),
                past_kv[4].clone(), past_kv[5].clone(), past_kv[6].clone(), past_kv[7].clone(),
                past_kv[8].clone(), past_kv[9].clone(), past_kv[10].clone(), past_kv[11].clone(),
                past_kv[12].clone(), past_kv[13].clone(), past_kv[14].clone(), past_kv[15].clone(),
                past_kv[16].clone(), past_kv[17].clone(), past_kv[18].clone(), past_kv[19].clone(),
                past_kv[20].clone(), past_kv[21].clone(), past_kv[22].clone(), past_kv[23].clone(),
            );

            let (gh, npk0, npv0, npk1, npv1, npk2, npv2, npk3, npv3, npk4, npv4,
                 npk5, npv5, npk6, npv6, npk7, npv7, npk8, npv8, npk9, npv9, npk10, npv10, npk11, npv11) = decode_out;

            current_hidden = extract_last_hidden(&gh);

            let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
            eprintln!("[FRAME] {}: local={:.1}ms decode={:.1}ms", step, frame_ms, decode_ms);

            past_kv = vec![
                npk0.to_owned(), npv0.to_owned(), npk1.to_owned(), npv1.to_owned(),
                npk2.to_owned(), npv2.to_owned(), npk3.to_owned(), npv3.to_owned(),
                npk4.to_owned(), npv4.to_owned(), npk5.to_owned(), npv5.to_owned(),
                npk6.to_owned(), npv6.to_owned(), npk7.to_owned(), npv7.to_owned(),
                npk8.to_owned(), npv8.to_owned(), npk9.to_owned(), npv9.to_owned(),
                npk10.to_owned(), npv10.to_owned(), npk11.to_owned(), npv11.to_owned(),
            ];
            past_valid_length += 1;
        }
        eprintln!("[STAGE] Generation: {:.2}s ({} frames)", t_gen.elapsed().as_secs_f64(), generated_frames.len());

        if generated_frames.is_empty() {
            return Ok(Vec::new());
        }

        // 3. Codec decode
        let t_codec = Instant::now();
        let num_frames = generated_frames.len();
        let mut audio_codes = vec![0i64; num_frames * N_VQ];
        for (fi, frame) in generated_frames.iter().enumerate() {
            for (ci, &code) in frame.iter().take(N_VQ).enumerate() {
                audio_codes[fi * N_VQ + ci] = code;
            }
        }

        let codes_tv = TensorView::<i64>::from_owned(audio_codes, vec![1, num_frames, N_VQ]);
        let lengths_tv = TensorView::<i64>::from_owned(vec![num_frames as i64], vec![1]);

        let t_codec_fwd = Instant::now();
        let (audio_tv, audio_lens_tv) = self.codec.forward_with_workspace(
            &mut self.codec_ws,
            codes_tv,
            lengths_tv,
        );
        eprintln!("[CODEC] forward: {:.0}ms", t_codec_fwd.elapsed().as_secs_f64() * 1000.0);

        let audio_len = audio_lens_tv.data.first().map(|&v| v as usize).unwrap_or(0);

        let channels = audio_tv.shape.get(1).copied().unwrap_or(2);
        let total_samples = audio_len * channels;

        let mut interleaved = Vec::with_capacity(total_samples);
        for t in 0..audio_len {
            for ch in 0..channels {
                let idx = ch * audio_len + t;
                if idx < audio_tv.data.len() {
                    interleaved.push(audio_tv.data[idx].clamp(-1.0, 1.0));
                }
            }
        }
        let max_abs = interleaved.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        eprintln!("[STAGE] Codec decode: {:.2}ms ({} samples, max_abs={:.6})", t_codec.elapsed().as_secs_f64() * 1000.0, audio_len, max_abs);

        Ok(interleaved)
    }
}

fn extract_last_hidden(hidden: &TensorView<'_, f32>) -> TensorView<'static, f32> {
    if hidden.shape.len() == 2 {
        return hidden.to_owned();
    }
    // [1, seq, 768] -> [1, 768] (last position)
    let dims = hidden.shape.as_ref();
    let seq = dims[1];
    let dim = dims[2];
    let offset = (seq - 1) * dim;
    let data = &hidden.data[offset..offset + dim];
    TensorView::from_owned(data.to_vec(), vec![1, dim])
}

fn main() -> Result<()> {
    let text = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Hello, this is a test of MOSS TTS Nano in pure Rust.".to_string());
    let output_path = "output_moss.wav";
    let voice = std::env::args().nth(2).unwrap_or_else(|| "Junhao".to_string());

    println!("=== MOSS-TTS-Nano Pure Rust Inference ===");
    println!("Text: {}", text);
    println!("Voice: {}", voice);

    let model_dir = Path::new("examples/moss-tts-nano/models");
    let mut tts = MossTtsNano::new(model_dir)?;
    println!("Model loaded. Vocab size: {}", tts.tokenizer.vocab_size());

    println!("Synthesizing...");
    let start = Instant::now();
    let audio = tts.synthesize(&text, &voice)?;
    let elapsed = start.elapsed().as_secs_f64();

    let sample_rate = 48000u32;
    let channels = 2usize;
    let audio_duration = audio.len() as f64 / (sample_rate as f64 * channels as f64);
    let rtf = elapsed / audio_duration;

    println!("Synthesized in {:.2}s", elapsed);
    println!("Audio duration: {:.2}s", audio_duration);
    println!("Real-time factor (RTF): {:.4}x", rtf);

    WavWriter::save_stereo(output_path, &audio, channels, sample_rate)?;
    println!("Saved to {}", output_path);

    Ok(())
}
