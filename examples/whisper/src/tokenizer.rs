//! Whisper's tokens: text out of token ids, and the special tokens a prompt
//! and a decoding step need. Only decoding needs the vocabulary; the prompt is
//! built from special ids, so no merge rules are involved.

use serde_json::Value;
use std::collections::HashMap;

pub struct Tokens {
    /// The bytes of each ordinary token, by id.
    bytes: Vec<Vec<u8>>,
    /// `<|endoftext|>`; every id above it is a special token.
    pub eot: i64,
    /// `<|startoftranscript|>`.
    pub start: i64,
    pub transcribe: i64,
    pub no_timestamps: i64,
    /// Language codes and their tokens.
    pub languages: Vec<(String, i64)>,
    /// Tokens never produced.
    pub suppress: Vec<i64>,
    /// Tokens not produced first.
    pub begin_suppress: Vec<i64>,
}

/// GPT-2's map from the printable characters a vocabulary is written in back
/// to the bytes they stand for.
fn byte_decoder() -> HashMap<char, u8> {
    let mut printable: Vec<u32> = (b'!' as u32..=b'~' as u32)
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut chars = printable.clone();
    let mut next = 0;
    for b in 0..256u32 {
        if !printable.contains(&b) {
            printable.push(b);
            chars.push(256 + next);
            next += 1;
        }
    }
    printable
        .iter()
        .zip(&chars)
        .map(|(&b, &c)| (char::from_u32(c).unwrap(), b as u8))
        .collect()
}

impl Tokens {
    /// From `vocab.json` and `generation_config.json`.
    pub fn load(vocab: &str, config: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let vocab: HashMap<String, i64> = serde_json::from_str(vocab)?;
        let config: Value = serde_json::from_str(config)?;
        let id = |v: &Value, what: &str| -> Result<i64, String> {
            v.as_i64()
                .ok_or_else(|| format!("generation_config.json: no {what}"))
        };
        let ids = |name: &str| -> Vec<i64> {
            config[name]
                .as_array()
                .map(|a| a.iter().filter_map(Value::as_i64).collect())
                .unwrap_or_default()
        };

        let decoder = byte_decoder();
        let mut bytes = vec![Vec::new(); vocab.values().max().map_or(0, |&m| m as usize + 1)];
        for (token, &i) in &vocab {
            bytes[i as usize] = token
                .chars()
                .filter_map(|c| decoder.get(&c).copied())
                .collect();
        }

        let mut languages: Vec<(String, i64)> = config["lang_to_id"]
            .as_object()
            .ok_or("generation_config.json: no lang_to_id")?
            .iter()
            .filter_map(|(k, v)| {
                Some((
                    k.trim_start_matches("<|")
                        .trim_end_matches("|>")
                        .to_string(),
                    v.as_i64()?,
                ))
            })
            .collect();
        languages.sort_by_key(|&(_, i)| i);
        Ok(Self {
            bytes,
            eot: id(&config["eos_token_id"], "eos_token_id")?,
            start: id(&config["decoder_start_token_id"], "decoder_start_token_id")?,
            transcribe: id(&config["task_to_id"]["transcribe"], "task_to_id.transcribe")?,
            no_timestamps: id(&config["no_timestamps_token_id"], "no_timestamps_token_id")?,
            languages,
            suppress: ids("suppress_tokens"),
            begin_suppress: ids("begin_suppress_tokens"),
        })
    }

    pub fn language(&self, code: &str) -> Option<i64> {
        self.languages
            .iter()
            .find(|(c, _)| c == code)
            .map(|&(_, i)| i)
    }

    pub fn language_code(&self, token: i64) -> Option<&str> {
        self.languages
            .iter()
            .find(|&&(_, i)| i == token)
            .map(|(c, _)| c.as_str())
    }

    /// The text of `ids`, without the special tokens.
    pub fn text(&self, ids: &[i64]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .filter(|&&i| i < self.eot)
            .flat_map(|&i| self.bytes[i as usize].iter().copied())
            .collect();
        String::from_utf8_lossy(&bytes).trim().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Byte-level tokens are decoded to bytes, so a character split across
    /// tokens comes out whole, and the specials are left out.
    #[test]
    fn text_joins_bytes_across_tokens() {
        // "Ġ" is the space; "Ð" "Ð¿" are the two bytes of "п" (0xD0 0xBF) split in two.
        let vocab = r#"{"Ġhi": 0, "Ð": 1, "¿": 2, "<|endoftext|>": 3, "<|ru|>": 4}"#;
        let config = r#"{"eos_token_id": 3, "decoder_start_token_id": 4,
            "task_to_id": {"transcribe": 5}, "no_timestamps_token_id": 6,
            "lang_to_id": {"<|ru|>": 4}, "suppress_tokens": [1]}"#;
        let tokens = Tokens::load(vocab, config).unwrap();
        assert_eq!(tokens.text(&[0, 1, 2, 3, 4]), "hiп");
        assert_eq!(tokens.language("ru"), Some(4));
        assert_eq!(tokens.language_code(4), Some("ru"));
        assert_eq!(tokens.suppress, [1]);
    }
}
