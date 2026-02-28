/// Tokenizer for SenseVoice (WASM-compatible, no filesystem).

pub struct Tokenizer {
    id_to_token: Vec<String>,
}

impl Tokenizer {
    /// Parse tokenizer from string content (instead of file).
    pub fn from_string(content: &str) -> Result<Self, String> {
        let mut id_to_token = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.rsplitn(2, ' ').collect();
            if parts.len() != 2 {
                continue;
            }

            let id: usize = parts[0].parse().unwrap_or(0);
            let token = parts[1].to_string();

            if id >= id_to_token.len() {
                id_to_token.resize(id + 1, String::new());
            }
            id_to_token[id] = token;
        }

        Ok(Tokenizer { id_to_token })
    }

    /// Greedy decode logits to text.
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

            for t in 0..time_steps {
                let offset = (b * time_steps + t) * vocab_size;
                let logit_slice = &logits[offset..offset + vocab_size];

                let token_id = logit_slice
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                if token_id < self.id_to_token.len() {
                    let token = &self.id_to_token[token_id];
                    if token_id == 0 || (token.starts_with("<|") && token.ends_with("|>")) {
                        continue;
                    }
                    tokens.push(token.clone());
                }
            }

            let text = tokens.join("");
            let text = text.replace("▁", " ");
            let text = text.trim().to_string();
            results.push(text);
        }

        results
    }
}
