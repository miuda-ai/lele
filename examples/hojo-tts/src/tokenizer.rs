//! Self-contained reader for a HuggingFace `tokenizers` `tokenizer.json`.
//!
//! Faithfully implements the pipeline used by Hojo-TTS-Light's tokenizer
//! (a BPE model): Lowercase + whitespace-collapse normalization, a
//! `Digits(individual) + Split(\w+|[^\w\s]+|\s+)` pre-tokenizer, BPE merge
//! scoring, `added_tokens` atomic matching, and a `TemplateProcessing`
//! post-processor. Validated bit-for-bit against HF `tokenizers` (Python).

use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use unicode_normalization::UnicodeNormalization;

struct TemplateItem {
    is_sequence: bool,
    token: String,
}

pub struct HojoTokenizer {
    /// token string -> id
    vocab: HashMap<String, u32>,
    /// id -> token string
    id_to_token: HashMap<u32, String>,
    /// atomic (special + bracket-int) tokens matched literally in input
    atomic: HashSet<String>,
    /// atomic tokens sorted by descending length (longest-match scan)
    atomic_sorted: Vec<String>,
    /// BPE merge ranks: (a, b) -> rank (lower rank applied first)
    bpe_ranks: HashMap<(String, String), u32>,
    unk_id: u32,
    do_lower_case: bool,
    do_nfkd: bool,
    do_nfkc: bool,
    collapse_ws: bool,
    single_template: Vec<TemplateItem>,
}

impl HojoTokenizer {
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let raw = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("read tokenizer {}", path.as_ref().display()))?;
        Self::from_str(&raw)
    }

    pub fn from_str(raw: &str) -> Result<Self> {
        let root: Value = serde_json::from_str(raw).context("parse tokenizer.json")?;
        let model = root.get("model").cloned().unwrap_or(Value::Null);

        let mut vocab: HashMap<String, u32> = HashMap::new();
        if let Some(v) = model.get("vocab").and_then(|v| v.as_object()) {
            for (tok, id) in v {
                if let Some(id) = id.as_u64().map(|i| i as u32) {
                    vocab.insert(tok.clone(), id);
                }
            }
        }

        let unk_id = model
            .get("unk_token")
            .and_then(|v| v.as_str())
            .and_then(|t| vocab.get(t).copied())
            .unwrap_or(0);

        // BPE merges may be stored as ["a","b"] arrays or "a b" strings.
        let mut bpe_ranks: HashMap<(String, String), u32> = HashMap::new();
        if let Some(merges) = model.get("merges").and_then(|v| v.as_array()) {
            for (rank, m) in merges.iter().enumerate() {
                let pair = if let Some(arr) = m.as_array() {
                    match (arr.get(0).and_then(|x| x.as_str()), arr.get(1).and_then(|x| x.as_str())) {
                        (Some(a), Some(b)) => Some((a.to_string(), b.to_string())),
                        _ => None,
                    }
                } else {
                    m.as_str().and_then(|s| {
                        let mut it = s.splitn(2, ' ');
                        match (it.next(), it.next()) {
                            (Some(a), Some(b)) => Some((a.to_string(), b.to_string())),
                            _ => None,
                        }
                    })
                };
                if let Some(p) = pair {
                    bpe_ranks.insert(p, rank as u32);
                }
            }
        }

        // added_tokens (special tokens + speech tokens [0]..[N]).
        let mut atomic: HashSet<String> = HashSet::new();
        if let Some(arr) = root.get("added_tokens").and_then(|v| v.as_array()) {
            for tok in arr {
                if let (Some(id), Some(content)) = (
                    tok.get("id").and_then(|v| v.as_u64()).map(|i| i as u32),
                    tok.get("content").and_then(|v| v.as_str()).map(|s| s.to_string()),
                ) {
                    vocab.insert(content.clone(), id);
                    atomic.insert(content);
                }
            }
        }
        // Bracket-int tokens (e.g. "[0]") are atomic even if only in model.vocab.
        for tok in vocab.keys() {
            if !atomic.contains(tok) && is_bracket_int_token(tok) {
                atomic.insert(tok.clone());
            }
        }

        let mut id_to_token: HashMap<u32, String> = HashMap::new();
        for (tok, id) in &vocab {
            id_to_token.entry(*id).or_insert_with(|| tok.clone());
        }

        let mut atomic_sorted: Vec<String> = atomic.iter().cloned().collect();
        atomic_sorted.sort_by(|a, b| b.len().cmp(&a.len()));

        // Normalizer detection (supports Sequence and single normalizers).
        let mut do_lower_case = false;
        let mut do_nfkd = false;
        let mut do_nfkc = false;
        let mut collapse_ws = false;
        let norm = root.get("normalizer");
        let norm_items: Vec<&Value> = match norm.and_then(|n| n.get("type")).and_then(|v| v.as_str()) {
            Some("Sequence") => norm
                .and_then(|n| n.get("normalizers"))
                .and_then(|v| v.as_array())
                .map(|a| a.iter().collect())
                .unwrap_or_default(),
            Some(_) => norm.map(|n| vec![n]).unwrap_or_default(),
            None => Vec::new(),
        };
        for n in &norm_items {
            match n.get("type").and_then(|v| v.as_str()) {
                Some("Lowercase") => do_lower_case = true,
                Some("NFKD") => do_nfkd = true,
                Some("NFKC") => do_nfkc = true,
                Some("NFD") => do_nfkd = true,
                Some("NFC") => do_nfkc = true,
                Some("Replace") => {
                    // Collapse \s+ -> " "
                    let is_ws = n
                        .get("pattern")
                        .and_then(|p| p.get("Regex"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.contains("\\s"))
                        .unwrap_or(false);
                    if is_ws {
                        collapse_ws = true;
                    }
                }
                Some("BertNormalizer") => {
                    if n.get("lowercase").and_then(|v| v.as_bool()).unwrap_or(false) {
                        do_lower_case = true;
                    }
                }
                _ => {}
            }
        }

        // Post-processor template (single-sequence encode).
        let mut single_template: Vec<TemplateItem> = Vec::new();
        if let Some(pp) = root.get("post_processor") {
            if pp.get("type").and_then(|v| v.as_str()) == Some("TemplateProcessing") {
                if let Some(single) = pp.get("single").and_then(|v| v.as_array()) {
                    for item in single {
                        if let Some(obj) = item.as_object() {
                            if obj.contains_key("Sequence") {
                                single_template.push(TemplateItem {
                                    is_sequence: true,
                                    token: "A".to_string(),
                                });
                            } else if let Some(st) = obj.get("SpecialToken").and_then(|v| v.as_str())
                            {
                                single_template.push(TemplateItem {
                                    is_sequence: false,
                                    token: st.to_string(),
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(Self {
            vocab,
            id_to_token,
            atomic,
            atomic_sorted,
            bpe_ranks,
            unk_id,
            do_lower_case,
            do_nfkd,
            do_nfkc,
            collapse_ws,
            single_template,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn convert_token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    pub fn id_to_token_str(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }

    /// Encode text to token ids. When `add_special_tokens` the post-processor
    /// template is applied (a no-op for Hojo, whose template is just `Seq A`).
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<i64> {
        let body = self.tokenize_to_ids(text);
        if !add_special_tokens || self.single_template.is_empty() {
            return body;
        }
        let mut out = Vec::new();
        for item in &self.single_template {
            if item.is_sequence {
                out.extend_from_slice(&body);
            } else if let Some(id) = self.convert_token_to_id(&item.token) {
                out.push(id as i64);
            }
        }
        out
    }

    fn tokenize_to_ids(&self, text: &str) -> Vec<i64> {
        self.tokenize(text)
            .iter()
            .map(|t| self.convert_token_to_id(t).unwrap_or(self.unk_id) as i64)
            .collect()
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let normalized = self.normalize(text);
        let mut tokens: Vec<String> = Vec::new();
        for piece in self.split_atomic(&normalized) {
            if self.atomic.contains(&piece) {
                tokens.push(piece);
            } else {
                for word in self.pre_tokenize(&piece) {
                    for sub in self.bpe(&word) {
                        tokens.push(sub);
                    }
                }
            }
        }
        tokens
    }

    fn normalize(&self, text: &str) -> String {
        let mut s: String = if self.do_lower_case {
            text.to_lowercase()
        } else {
            text.to_string()
        };
        if self.do_nfkd {
            s = s.nfkd().collect();
        } else if self.do_nfkc {
            s = s.nfkc().collect();
        }
        if self.collapse_ws {
            let mut out = String::with_capacity(s.len());
            let mut prev_ws = false;
            for ch in s.chars() {
                if ch.is_whitespace() {
                    if !prev_ws {
                        out.push(' ');
                        prev_ws = true;
                    }
                } else {
                    out.push(ch);
                    prev_ws = false;
                }
            }
            s = out;
        }
        s
    }

    /// Left-to-right longest-match split over atomic tokens.
    fn split_atomic(&self, text: &str) -> Vec<String> {
        if self.atomic_sorted.is_empty() {
            return vec![text.to_string()];
        }
        let chars: Vec<char> = text.chars().collect();
        let mut pieces: Vec<String> = Vec::new();
        let mut i = 0;
        let mut buf = String::new();
        while i < chars.len() {
            let tail: String = chars[i..].iter().collect();
            let matched = self
                .atomic_sorted
                .iter()
                .find(|c| tail.starts_with(c.as_str()));
            match matched {
                Some(m) => {
                    if !buf.is_empty() {
                        pieces.push(std::mem::take(&mut buf));
                    }
                    pieces.push(m.to_string());
                    i += m.chars().count();
                }
                None => {
                    buf.push(chars[i]);
                    i += 1;
                }
            }
        }
        if !buf.is_empty() {
            pieces.push(buf);
        }
        pieces
    }

    /// `Digits(individual) + Split(\w+|[^\w\s]+|\s+)` pre-tokenization.
    /// Word runs are kept, digits split individually, whitespace -> " ".
    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut tokens: Vec<String> = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            if c.is_whitespace() {
                while i < chars.len() && chars[i].is_whitespace() {
                    i += 1;
                }
                tokens.push(" ".to_string());
            } else if is_word_char(c) {
                let mut cur = String::new();
                while i < chars.len() && is_word_char(chars[i]) {
                    if chars[i].is_ascii_digit() {
                        if !cur.is_empty() {
                            tokens.push(std::mem::take(&mut cur));
                        }
                        tokens.push(chars[i].to_string());
                    } else {
                        cur.push(chars[i]);
                    }
                    i += 1;
                }
                if !cur.is_empty() {
                    tokens.push(cur);
                }
            } else {
                let mut cur = String::new();
                while i < chars.len() && !is_word_char(chars[i]) && !chars[i].is_whitespace() {
                    cur.push(chars[i]);
                    i += 1;
                }
                tokens.push(cur);
            }
        }
        tokens
    }

    /// BPE merge scoring over a single pre-token's characters.
    fn bpe(&self, word: &str) -> Vec<String> {
        if word.is_empty() {
            return Vec::new();
        }
        let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        loop {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..symbols.len().saturating_sub(1) {
                if let Some(&rank) = self.bpe_ranks.get(&(symbols[i].clone(), symbols[i + 1].clone()))
                {
                    match best {
                        Some((_, r)) if r <= rank => {}
                        _ => best = Some((i, rank)),
                    }
                }
            }
            match best {
                None => break,
                Some((idx, _)) => {
                    let merged = format!("{}{}", symbols[idx], symbols[idx + 1]);
                    symbols.splice(idx..=idx + 1, std::iter::once(merged));
                }
            }
        }
        let unk = self
            .id_to_token
            .get(&self.unk_id)
            .cloned()
            .unwrap_or_else(|| "[UNK]".to_string());
        let mut out = Vec::new();
        for s in &symbols {
            if self.vocab.contains_key(s) {
                out.push(s.clone());
            } else {
                return vec![unk];
            }
        }
        out
    }
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn is_bracket_int_token(s: &str) -> bool {
    let inner = match s.strip_prefix('[').and_then(|i| i.strip_suffix(']')) {
        Some(i) => i,
        None => return false,
    };
    if inner.is_empty() {
        return false;
    }
    let mut chars = inner.chars();
    let first = chars.next().unwrap();
    (first == '-' || first.is_ascii_digit()) && chars.all(|c| c.is_ascii_digit())
}
