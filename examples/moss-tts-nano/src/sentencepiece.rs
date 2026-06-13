use std::collections::HashMap;
use std::path::Path;
use unicode_normalization::UnicodeNormalization;

struct Piece {
    piece: String,
    score: f32,
}

pub struct SentencePiece {
    pieces: Vec<Piece>,
    piece_to_id: HashMap<String, u32>,
    unk_id: u32,
    _unk_score: f32,
}

fn read_varint(data: &[u8], pos: &mut usize) -> u64 {
    let mut result: u64 = 0;
    let mut shift = 0;
    while *pos < data.len() {
        let byte = data[*pos];
        *pos += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    result
}

fn parse_sentencepiece(data: &[u8]) -> Option<Piece> {
    let mut pos = 0;
    let mut piece = String::new();
    let mut score: f32 = 0.0;
    while pos < data.len() {
        let tag = data[pos] as u64;
        pos += 1;
        let field_num = tag >> 3;
        let wire_type = tag & 0x07;
        match (field_num, wire_type) {
            (1, 2) => {
                let len = read_varint(data, &mut pos) as usize;
                let bytes = &data[pos..pos + len];
                pos += len;
                piece = String::from_utf8_lossy(bytes).into_owned();
            }
            (2, 5) => {
                let bytes: [u8; 4] = data[pos..pos + 4].try_into().ok()?;
                pos += 4;
                score = f32::from_le_bytes(bytes);
            }
            (3, 0) => {
                let _ptype = read_varint(data, &mut pos);
            }
            (_, 0) => {
                read_varint(data, &mut pos);
            }
            (_, 2) => {
                let len = read_varint(data, &mut pos) as usize;
                pos += len;
            }
            (_, 5) => {
                pos += 4;
            }
            (_, 1) => {
                pos += 8;
            }
            _ => {}
        }
    }
    if piece.is_empty() {
        None
    } else {
        Some(Piece { piece, score })
    }
}

impl SentencePiece {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read(path)?;
        Self::from_bytes(&data)
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let mut pos = 0;
        let mut pieces: Vec<Piece> = Vec::new();
        let mut piece_to_id: HashMap<String, u32> = HashMap::new();

        while pos < data.len() {
            let tag = data[pos] as u64;
            pos += 1;
            let field_num = tag >> 3;
            let wire_type = tag & 0x07;
            match (field_num, wire_type) {
                (1, 2) => {
                    let len = read_varint(data, &mut pos) as usize;
                    let sub_data = &data[pos..pos + len];
                    pos += len;
                    if let Some(piece) = parse_sentencepiece(sub_data) {
                        let id = pieces.len() as u32;
                        piece_to_id.insert(piece.piece.clone(), id);
                        pieces.push(piece);
                    }
                }
                (_, 2) => {
                    let len = read_varint(data, &mut pos) as usize;
                    pos += len;
                }
                (_, 0) => {
                    read_varint(data, &mut pos);
                }
                (_, 5) => {
                    pos += 4;
                }
                (_, 1) => {
                    pos += 8;
                }
                _ => {}
            }
        }

        let unk_score = if pieces.is_empty() {
            -1000.0
        } else {
            pieces
                .iter()
                .filter(|p| !p.piece.starts_with('<'))
                .map(|p| p.score)
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .map(|s| s - 10.0)
                .unwrap_or(-1000.0)
        };

        Ok(Self {
            pieces,
            piece_to_id,
            unk_id: 0,
            _unk_score: unk_score,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    pub fn debug_lookup(&self, s: &str) {
        if let Some(&id) = self.piece_to_id.get(s) {
            eprintln!("[SP] '{}' -> id={}, score={:.2}", s, id, self.pieces[id as usize].score);
        } else {
            eprintln!("[SP] '{}' -> NOT FOUND", s);
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let nfkc: String = text.nfkc().collect();
        let escaped = nfkc.replace(' ', "\u{2581}");

        let mut normalized = String::with_capacity(escaped.len() + 1);
        normalized.push('\u{2581}');
        let mut last_was_meta = true;
        for ch in escaped.chars() {
            if ch == '\u{2581}' {
                if !last_was_meta {
                    normalized.push(ch);
                    last_was_meta = true;
                }
            } else {
                normalized.push(ch);
                last_was_meta = false;
            }
        }

        let chars: Vec<char> = normalized.chars().collect();
        let n = chars.len();

        let mut best_score = vec![f32::NEG_INFINITY; n + 1];
        let mut best_prev: Vec<i32> = vec![-1; n + 1];
        best_score[0] = 0.0;

        for end in 1..=n {
            let max_len = end.min(64);
            for length in 1..=max_len {
                let start = end - length;
                let substring: String = chars[start..end].iter().collect();
                if let Some(&id) = self.piece_to_id.get(&substring) {
                    let s = self.pieces[id as usize].score;
                    if best_score[start] != f32::NEG_INFINITY {
                        let candidate = best_score[start] + s;
                        if candidate > best_score[end] {
                            best_score[end] = candidate;
                            best_prev[end] = start as i32;
                        }
                    }
                }
            }

            if best_prev[end] == -1 && best_score[end] == f32::NEG_INFINITY {
                best_score[end] = best_score[end - 1] + self._unk_score;
                best_prev[end] = (end - 1) as i32;
            }
        }

        let mut tokens: Vec<u32> = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let prev = best_prev[pos] as usize;
            let substring: String = chars[prev..pos].iter().collect();
            let id = self.piece_to_id.get(&substring).copied().unwrap_or(self.unk_id);
            tokens.push(id);
            pos = prev;
        }
        tokens.reverse();
        tokens
    }
}
