use anyhow::{Result, bail};
use rand_distr::{Distribution, Normal};
use regex::Regex;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use unicode_normalization::UnicodeNormalization;

pub const AVAILABLE_LANGS: &[&str] = &["en", "ko", "es", "pt", "fr", "zh"];

pub fn is_valid_lang(lang: &str) -> bool {
    AVAILABLE_LANGS.contains(&lang)
}

pub struct UnicodeProcessor {
    indexer: Vec<i64>,
}

impl UnicodeProcessor {
    pub fn new<P: AsRef<Path>>(unicode_indexer_json_path: P) -> Result<Self> {
        let file = File::open(unicode_indexer_json_path)?;
        let reader = BufReader::new(file);
        let indexer: Vec<i64> = serde_json::from_reader(reader)?;
        Ok(UnicodeProcessor { indexer })
    }

    pub fn call(
        &self,
        text_list: &[String],
        lang_list: &[String],
    ) -> Result<(Vec<Vec<i64>>, Vec<f32>, Vec<usize>)> {
        let mut processed_texts: Vec<String> = Vec::new();
        for (text, lang) in text_list.iter().zip(lang_list.iter()) {
            let processed = preprocess_text(text, lang)?;
            processed_texts.push(processed);
        }

        let mut text_ids: Vec<Vec<i64>> = Vec::new();
        for text in processed_texts {
            let mut ids = Vec::new();
            for char_val in text.chars() {
                let idx = char_val as usize;
                let mut token_id = 0;
                if idx < self.indexer.len() {
                    token_id = self.indexer[idx];
                    if token_id == -1 {
                        token_id = 0;
                    }
                }
                ids.push(token_id);
            }
            text_ids.push(ids);
        }

        let text_ids_lengths: Vec<usize> = text_ids.iter().map(|ids| ids.len()).collect();
        let (mask_data, mask_shape) = get_text_mask(&text_ids_lengths);
        Ok((text_ids, mask_data, mask_shape))
    }
}

pub fn preprocess_text(text: &str, lang: &str) -> Result<String> {
    let mut text: String = text.nfkd().collect();

    let emoji_pattern = Regex::new(r"[\x{1F600}-\x{1F64F}\x{1F300}-\x{1F5FF}\x{1F680}-\x{1F6FF}\x{1F700}-\x{1F77F}\x{1F780}-\x{1F7FF}\x{1F800}-\x{1F8FF}\x{1F900}-\x{1F9FF}\x{1FA00}-\x{1FA6F}\x{1FA70}-\x{1FAFF}\x{2600}-\x{26FF}\x{2700}-\x{27BF}\x{1F1E6}-\x{1F1FF}]+").unwrap();
    text = emoji_pattern.replace_all(&text, "").to_string();

    let replacements = [
        ("–", "-"),
        ("‑", "-"),
        ("—", "-"),
        ("_", " "),
        ("\u{201C}", "\""),
        ("\u{201D}", "\""),
        ("\u{2018}", "'"),
        ("\u{2019}", "'"),
        ("´", "'"),
        ("`", "'"),
        ("[", " "),
        ("]", " "),
        ("|", " "),
        ("/", " "),
        ("#", " "),
        ("→", " "),
        ("←", " "),
    ];

    for (from, to) in &replacements {
        text = text.replace(from, to);
    }

    let special_symbols = ["♥", "☆", "♡", "©", "\\"];
    for symbol in &special_symbols {
        text = text.replace(symbol, "");
    }

    text = Regex::new(r"\s+")
        .unwrap()
        .replace_all(&text, " ")
        .to_string();
    text = text.trim().to_string();

    if !text.is_empty() {
        let ends_with_punct =
            Regex::new(r#"[.!?;:,'"\u{201C}\u{201D}\u{2018}\u{2019})\]}…。」』】〉》›»]$"#)
                .unwrap();
        if !ends_with_punct.is_match(&text) {
            text.push('.');
        }
    }

    if !is_valid_lang(lang) {
        bail!(
            "Invalid language: {}. Available: {:?}",
            lang,
            AVAILABLE_LANGS
        );
    }

    text = format!("<{}>{}</{}>", lang, text, lang);
    Ok(text)
}

pub fn length_to_mask(lengths: &[usize], max_len: Option<usize>) -> (Vec<f32>, Vec<usize>) {
    let bsz = lengths.len();
    let max_len = max_len.unwrap_or_else(|| *lengths.iter().max().unwrap_or(&0));

    let mut data = vec![0.0; bsz * 1 * max_len];
    for (i, &len) in lengths.iter().enumerate() {
        for j in 0..len.min(max_len) {
            data[i * max_len + j] = 1.0;
        }
    }
    (data, vec![bsz, 1, max_len])
}

pub fn get_text_mask(text_ids_lengths: &[usize]) -> (Vec<f32>, Vec<usize>) {
    let max_len = *text_ids_lengths.iter().max().unwrap_or(&0);
    length_to_mask(text_ids_lengths, Some(max_len))
}

pub fn sample_noisy_latent(
    duration: &[f32],
    sample_rate: i32,
    base_chunk_size: i32,
    chunk_compress: i32,
    latent_dim: i32,
) -> (Vec<f32>, Vec<usize>, Vec<f32>, Vec<usize>) {
    let bsz = duration.len();
    let chunk_size = (base_chunk_size * chunk_compress) as usize;

    let wav_lengths: Vec<usize> = duration
        .iter()
        .map(|&d| (d * sample_rate as f32) as usize)
        .collect();

    let wav_len_max = wav_lengths.iter().max().copied().unwrap_or(0);

    let latent_len = (wav_len_max + chunk_size - 1) / chunk_size;
    let latent_dim_val = (latent_dim * chunk_compress) as usize;

    let mut noisy_latent = vec![0.0f32; bsz * latent_dim_val * latent_len];
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::thread_rng();

    for val in noisy_latent.iter_mut() {
        *val = normal.sample(&mut rng);
    }

    let latent_lengths: Vec<usize> = wav_lengths
        .iter()
        .map(|&len| (len + chunk_size - 1) / chunk_size)
        .collect();
    let (latent_mask, mask_shape) = length_to_mask(&latent_lengths, Some(latent_len));

    for b in 0..bsz {
        for d in 0..latent_dim_val {
            for t in 0..latent_len {
                let mask_idx = b * latent_len + t;
                let latent_idx = (b * latent_dim_val + d) * latent_len + t;
                noisy_latent[latent_idx] *= latent_mask[mask_idx];
            }
        }
    }

    (
        noisy_latent,
        vec![bsz, latent_dim_val, latent_len],
        latent_mask,
        mask_shape,
    )
}

pub fn chunk_text(text: &str, max_len: Option<usize>) -> Vec<String> {
    let max_len = max_len.unwrap_or(300);
    let text = text.trim();
    if text.is_empty() {
        return vec![String::new()];
    }

    let para_re = Regex::new(r"\n\s*\n").unwrap();
    let paragraphs: Vec<&str> = para_re.split(text).collect();
    let mut chunks = Vec::new();

    for para in paragraphs {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }
        if para.len() <= max_len {
            chunks.push(para.to_string());
        } else {
            // Very simple split for now
            chunks.push(para[..para.len().min(max_len)].to_string());
        }
    }
    chunks
}
