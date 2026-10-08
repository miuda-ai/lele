pub mod tone;

use lele::tensor::{TensorView, f16};

pub const SAMPLE_RATE: u32 = 8000;
/// The model consumes 300 ms of 8 kHz audio per call...
pub const CHUNK: usize = 2400;
/// ...and emits one frame of log-probabilities per 30 ms of it.
pub const FRAMES_PER_CHUNK: usize = 10;
const STATE_SIZE: usize = 219729;
/// Indices 0..=33 of the CTC alphabet; 34 is the blank.
const LABELS: [char; 34] = [
    'а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с',
    'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', ' ',
];
pub const BLANK: usize = 34;
const VOCAB: usize = LABELS.len() + 1;

/// The token's character; `None` for the blank.
pub fn label(token: usize) -> Option<char> {
    LABELS.get(token).copied()
}

/// Whether a frame's best token says someone is talking: anything but the
/// blank and the word separator.
pub fn is_speech(token: usize) -> bool {
    token < LABELS.len() - 1
}

pub fn load_weights() -> std::io::Result<Vec<u8>> {
    std::fs::read("examples/tone/src/tone_weights.bin")
        .or_else(|_| std::fs::read("src/tone_weights.bin"))
}

/// Streaming acoustic model: audio in 300 ms chunks, best token per frame out,
/// with the hidden state carried from one chunk to the next.
pub struct Recognizer<'a> {
    model: tone::TOne<'a>,
    ws: tone::TOneWorkspace,
    /// Kept as fp16 between chunks, as the reference pipeline does; it halves
    /// the per-stream memory and matches its numerics.
    state: Vec<f16>,
    state_f32: Vec<f32>,
}

impl<'a> Recognizer<'a> {
    pub fn new(weights: &'a [u8]) -> Self {
        Self {
            model: tone::TOne::new(weights),
            ws: tone::TOneWorkspace::new(),
            state: vec![f16::ZERO; STATE_SIZE],
            state_f32: vec![0.0; STATE_SIZE],
        }
    }

    /// Forgets everything heard so far.
    pub fn reset(&mut self) {
        self.state.fill(f16::ZERO);
    }

    /// Feeds `CHUNK` samples in the int16 range; returns the most likely
    /// token of each of the chunk's frames.
    pub fn feed(&mut self, chunk: &[i32]) -> [usize; FRAMES_PER_CHUNK] {
        assert_eq!(chunk.len(), CHUNK);
        let x: Vec<i64> = chunk.iter().map(|&s| s as i64).collect();
        for (d, s) in self.state_f32.iter_mut().zip(&self.state) {
            *d = s.to_f32();
        }
        let (logprobs, next) = self.model.forward_with_workspace(
            &mut self.ws,
            TensorView::from_owned(x, vec![1, CHUNK, 1]),
            TensorView::from_slice(&self.state_f32, vec![1, STATE_SIZE]),
        );
        for (d, &s) in self.state.iter_mut().zip(next.data.iter()) {
            *d = f16::from_f32(s);
        }
        let mut best = [BLANK; FRAMES_PER_CHUNK];
        for (b, frame) in best.iter_mut().zip(logprobs.data.chunks_exact(VOCAB)) {
            *b = frame
                .iter()
                .enumerate()
                .fold((BLANK, f32::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m })
                .0;
        }
        best
    }
}

/// Greedy CTC: collapse repeated tokens, then drop blanks.
#[derive(Default)]
pub struct GreedyCtc {
    prev: Option<usize>,
    text: String,
}

impl GreedyCtc {
    pub fn push(&mut self, token: usize) {
        if self.prev != Some(token)
            && let Some(c) = label(token)
        {
            self.text.push(c);
        }
        self.prev = Some(token);
    }

    /// The text so far, with runs of separators squeezed.
    pub fn text(&self) -> String {
        self.text.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }
}
