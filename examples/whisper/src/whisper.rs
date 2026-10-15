//! Speech to text with Whisper: the encoder reads 30 seconds of log-mel
//! features; the decoder then writes the text one token at a time, the first
//! step with the full decoder and the later ones with the decoder that reads
//! the key/value caches of the steps before.

use crate::mel::{LogMel, N_MELS, SAMPLE_RATE, Window};
use crate::tokenizer::Tokens;
use crate::whisperdecoder::{WhisperDecoder, WhisperDecoderWorkspace};
use crate::whisperdecoderpast::{WhisperDecoderPast, WhisperDecoderPastWorkspace};
use crate::whisperencoder::{WhisperEncoder, WhisperEncoderWorkspace};
use lele::tensor::TensorView;

// `first_step` and `next_step`, which call the decoders for the model's layer
// count, written by build.rs.
include!("glue.rs");

/// Most tokens one window may produce.
const MAX_TOKENS: usize = 224;

/// The generated weights, and the vocabulary.
pub struct Weights {
    pub encoder: Vec<u8>,
    pub decoder: Vec<u8>,
    pub decoder_past: Vec<u8>,
    pub vocab: String,
    pub config: String,
}

impl Weights {
    /// Reads them from this crate's source tree, wherever the program is run.
    pub fn load() -> std::io::Result<Self> {
        let dir = env!("CARGO_MANIFEST_DIR");
        let read = |name: &str| std::fs::read(format!("{dir}/{name}"));
        let text = |name: &str| std::fs::read_to_string(format!("{dir}/{name}"));
        Ok(Self {
            encoder: read("src/whisperencoder_weights.bin")?,
            decoder: read("src/whisperdecoder_weights.bin")?,
            decoder_past: read("src/whisperdecoderpast_weights.bin")?,
            vocab: text("vocab.json")?,
            config: text("generation_config.json")?,
        })
    }
}

/// A tensor kept between steps.
#[derive(Clone, Default)]
pub(crate) struct Kv {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Kv {
    pub(crate) fn of(t: &TensorView<'_>) -> Self {
        Self {
            data: t.data.to_vec(),
            shape: t.shape.to_vec(),
        }
    }

    pub(crate) fn view(&self) -> TensorView<'_> {
        TensorView::from_slice(&self.data, self.shape.clone())
    }
}

/// What the decoder remembers of the steps so far, per layer.
#[derive(Default)]
pub(crate) struct Caches {
    /// Keys and values of the tokens decoded so far; they grow by one a step.
    self_key: Vec<Kv>,
    self_value: Vec<Kv>,
    /// Keys and values of the audio, which do not change.
    audio_key: Vec<Kv>,
    audio_value: Vec<Kv>,
}

#[derive(Clone, Debug)]
pub struct Transcript {
    pub text: String,
    /// The language code, as given or as detected, of the last window.
    pub language: String,
}

pub struct Whisper<'a> {
    encoder: WhisperEncoder<'a>,
    encoder_ws: WhisperEncoderWorkspace,
    decoder: WhisperDecoder<'a>,
    decoder_ws: WhisperDecoderWorkspace,
    decoder_past: WhisperDecoderPast<'a>,
    decoder_past_ws: WhisperDecoderPastWorkspace,
    mel: LogMel,
    tokens: Tokens,
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .fold(
            (0, f32::NEG_INFINITY),
            |m, (i, &v)| if v > m.1 { (i, v) } else { m },
        )
        .0
}

impl<'a> Whisper<'a> {
    pub fn new(weights: &'a Weights) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            encoder: WhisperEncoder::new(&weights.encoder),
            encoder_ws: WhisperEncoderWorkspace::new(),
            decoder: WhisperDecoder::new(&weights.decoder),
            decoder_ws: WhisperDecoderWorkspace::new(),
            decoder_past: WhisperDecoderPast::new(&weights.decoder_past),
            decoder_past_ws: WhisperDecoderPastWorkspace::new(),
            mel: LogMel::new(),
            tokens: Tokens::load(&weights.vocab, &weights.config)?,
        })
    }

    pub fn tokens(&self) -> &Tokens {
        &self.tokens
    }

    /// Transcribes 16 kHz mono audio in [-1, 1], 30 seconds at a time.
    /// `language` is a code such as "ru"; without one, it is detected.
    pub fn transcribe(
        &mut self,
        audio: &[f32],
        language: Option<&str>,
    ) -> Result<Transcript, String> {
        let language = language
            .map(|code| {
                self.tokens
                    .language(code)
                    .ok_or_else(|| format!("unknown language {code:?}"))
            })
            .transpose()?;
        let mut text = Vec::new();
        let mut detected = String::new();
        for window in audio.chunks(Window::ASR.samples) {
            let (piece, code) = self.transcribe_window(window, language);
            if !piece.is_empty() {
                text.push(piece);
            }
            detected = code;
        }
        Ok(Transcript {
            text: text.join(" "),
            language: detected,
        })
    }

    fn transcribe_window(&mut self, audio: &[f32], language: Option<i64>) -> (String, String) {
        // The encoder.
        let features = self.mel.features(audio, Window::ASR);
        let frames = Window::ASR.frames();
        let encoded = self.encoder.forward_with_workspace(
            &mut self.encoder_ws,
            TensorView::from_owned(features, vec![1, N_MELS, frames]),
        );
        let hidden = Kv::of(&encoded);

        // The decoder's first step reads only the start token. It gives the
        // audio's keys and values for all the steps after, and the logits that
        // say which language is spoken.
        let (first_logits, mut caches) = first_step(
            &self.decoder,
            &mut self.decoder_ws,
            self.tokens.start,
            &hidden,
        );

        let language = language.unwrap_or_else(|| {
            self.tokens
                .languages
                .iter()
                .map(|&(_, id)| id)
                .max_by(|&a, &b| first_logits[a as usize].total_cmp(&first_logits[b as usize]))
                .unwrap_or(self.tokens.start)
        });
        let code = self
            .tokens
            .language_code(language)
            .unwrap_or("?")
            .to_string();

        // The prompt, then the text: each token read, one step, to get the next.
        let seconds = audio.len() as f64 / SAMPLE_RATE as f64;
        let limit = ((seconds * 20.0) as usize + 20).min(MAX_TOKENS);
        let prompt = [language, self.tokens.transcribe, self.tokens.no_timestamps];
        let mut logits = Vec::new();
        for token in prompt {
            logits = self.step(token, &mut caches);
        }
        let mut generated: Vec<i64> = Vec::new();
        while generated.len() < limit {
            // Only ordinary tokens: no timestamps, languages or tasks.
            for l in &mut logits[self.tokens.eot as usize + 1..] {
                *l = f32::NEG_INFINITY;
            }
            for &s in &self.tokens.suppress {
                logits[s as usize] = f32::NEG_INFINITY;
            }
            if generated.is_empty() {
                for &s in &self.tokens.begin_suppress {
                    logits[s as usize] = f32::NEG_INFINITY;
                }
            }
            let next = argmax(&logits) as i64;
            if next == self.tokens.eot {
                break;
            }
            generated.push(next);
            // A decoder stuck repeating one token will not get out of it.
            if generated.len() >= 6 && generated[generated.len() - 6..].iter().all(|&t| t == next) {
                generated.truncate(generated.len() - 5);
                break;
            }
            logits = self.step(next, &mut caches);
        }
        (self.tokens.text(&generated), code)
    }

    /// Reads one token; returns the logits for the next.
    fn step(&mut self, token: i64, caches: &mut Caches) -> Vec<f32> {
        next_step(&self.decoder_past, &mut self.decoder_past_ws, token, caches)
    }
}
