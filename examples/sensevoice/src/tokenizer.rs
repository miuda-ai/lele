use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub struct Tokenizer {
    id_to_token: Vec<String>,
}

impl Tokenizer {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut id_to_token = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let parts: Vec<&str> = line.rsplitn(2, ' ').collect();
            if parts.len() != 2 {
                continue;
            }

            let id: usize = parts[0].parse().unwrap_or(0);
            let token = parts[1].to_string();

            // Ensure vector is large enough
            if id >= id_to_token.len() {
                id_to_token.resize(id + 1, String::new());
            }

            id_to_token[id] = token;
        }

        Ok(Tokenizer { id_to_token })
    }

    /// Greedy decode matching the ort reference (scripts/sensevoice_bench.py):
    /// skip blank (id 0), collapse CTC repeats, and keep tags like <|zh|>.
    pub fn decode_greedy(
        &self,
        logits: &[f32],
        batch_size: usize,
        time_steps: usize,
        vocab_size: usize,
    ) -> Vec<String> {
        assert_eq!(logits.len(), batch_size * time_steps * vocab_size);

        let mut results = Vec::new();

        for b in 0..batch_size {
            let mut tokens = Vec::new();
            let mut prev: usize = usize::MAX;

            for t in 0..time_steps {
                let offset = (b * time_steps + t) * vocab_size;
                let logit_slice = &logits[offset..offset + vocab_size];

                // Find argmax
                let token_id = logit_slice
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                // ort: `if idx != 0 and idx != prev`, tags are kept
                if token_id != 0 && token_id != prev {
                    let token = self
                        .id_to_token
                        .get(token_id)
                        .filter(|t| !t.is_empty())
                        .cloned()
                        .unwrap_or_else(|| format!("<{}>", token_id));
                    tokens.push(token);
                }
                prev = token_id;
            }

            // Join tokens and clean up
            let text = tokens.join("");
            // Replace sentencepiece underscore with space
            let text = text.replace("▁", " ");
            // Trim and clean up
            let text = text.trim().to_string();

            results.push(text);
        }

        results
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }
}
