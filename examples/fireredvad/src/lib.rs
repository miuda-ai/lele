pub mod fbank;
pub mod fireredvad;

use fbank::{FRAME_LEN, FRAME_SHIFT, Fbank, NUM_BINS};
use lele::tensor::TensorView;

pub const SAMPLE_RATE: u32 = 16000;
/// Frames per second: one every `FRAME_SHIFT` samples.
pub const FRAME_RATE: usize = SAMPLE_RATE as usize / FRAME_SHIFT;
const BLOCKS: usize = 8;
const CACHE: usize = 128 * 19;

fn read(file: &str) -> std::io::Result<Vec<u8>> {
    std::fs::read(format!("examples/fireredvad/{file}")).or_else(|_| std::fs::read(file))
}

pub fn load_weights() -> std::io::Result<Vec<u8>> {
    read("src/fireredvad_weights.bin")
}

/// Reads a 16-bit PCM WAV, walking the RIFF chunks rather than assuming a
/// 44-byte header, and averages the channels down to mono. Samples stay in
/// the int16 range, which the model's features are computed from.
pub fn read_wav(path: &str) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }
    let (mut channels, mut rate, mut bits) = (0usize, 0u32, 0u16);
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into()?) as usize;
        let body = &bytes[pos + 8..(pos + 8 + len).min(bytes.len())];
        if id == b"fmt " {
            channels = u16::from_le_bytes([body[2], body[3]]) as usize;
            rate = u32::from_le_bytes(body[4..8].try_into()?);
            bits = u16::from_le_bytes([body[14], body[15]]);
        } else if id == b"data" {
            if bits != 16 || channels == 0 {
                return Err(format!("need 16-bit PCM, got {bits}-bit x {channels}").into());
            }
            let samples = body
                .chunks_exact(2 * channels)
                .map(|frame| {
                    let sum: i32 = frame
                        .chunks_exact(2)
                        .map(|s| i16::from_le_bytes([s[0], s[1]]) as i32)
                        .sum();
                    (sum / channels as i32) as f32
                })
                .collect();
            return Ok((samples, rate));
        }
        pos += 8 + len + (len & 1);
    }
    Err("no data chunk".into())
}

/// Global mean and variance normalization of the features, with the
/// statistics the model was trained with.
pub struct Cmvn {
    means: Vec<f64>,
    inv_stddevs: Vec<f64>,
}

impl Cmvn {
    /// Reads `cmvn.json`, which build.rs downloads with the model.
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let json: serde_json::Value = serde_json::from_slice(&read("cmvn.json")?)?;
        let field = |name: &str| -> Result<Vec<f64>, Box<dyn std::error::Error>> {
            let values: Option<Vec<f64>> = json[name]
                .as_array()
                .and_then(|a| a.iter().map(|v| v.as_f64()).collect());
            match values {
                Some(v) if v.len() == NUM_BINS => Ok(v),
                _ => Err(format!("cmvn.json: `{name}` is not {NUM_BINS} numbers").into()),
            }
        };
        Ok(Self {
            means: field("means")?,
            inv_stddevs: field("inv_stddevs")?,
        })
    }

    /// Normalizes one frame. The statistics are doubles, as in the reference
    /// pipeline, which rounds to float only at the end.
    pub fn apply(&self, frame: &mut [f32]) {
        for ((x, &m), &s) in frame.iter_mut().zip(&self.means).zip(&self.inv_stddevs) {
            *x = ((*x as f64 - m) * s) as f32;
        }
    }
}

/// Streaming speech probabilities: audio in, one probability per 10 ms frame
/// out, as soon as the frame's 25 ms of audio are in. Each of the model's
/// eight blocks carries its last 19 frames from one call to the next, so
/// feeding the audio in any pieces gives the same probabilities, up to
/// rounding.
pub struct Detector<'a> {
    model: fireredvad::FireRedVad<'a>,
    ws: fireredvad::FireRedVadWorkspace,
    fbank: Fbank,
    cmvn: Cmvn,
    caches: Vec<f32>,
    next_caches: Vec<f32>,
    /// Audio not yet in a frame, starting at the next frame.
    pending: Vec<f32>,
    feats: Vec<f32>,
}

impl<'a> Detector<'a> {
    pub fn new(weights: &'a [u8], cmvn: Cmvn) -> Self {
        Self {
            model: fireredvad::FireRedVad::new(weights),
            ws: fireredvad::FireRedVadWorkspace::new(),
            fbank: Fbank::new(),
            cmvn,
            caches: vec![0.0; BLOCKS * CACHE],
            next_caches: vec![0.0; BLOCKS * CACHE],
            pending: Vec::new(),
            feats: Vec::new(),
        }
    }

    /// Forgets everything heard so far.
    pub fn reset(&mut self) {
        self.caches.fill(0.0);
        self.pending.clear();
    }

    /// Feeds 16 kHz samples in the int16 range; appends the speech
    /// probability of each frame they complete to `probs`.
    pub fn feed(&mut self, samples: &[f32], probs: &mut Vec<f32>) {
        self.pending.extend_from_slice(samples);
        if self.pending.len() < FRAME_LEN {
            return;
        }
        let frames = (self.pending.len() - FRAME_LEN) / FRAME_SHIFT + 1;
        self.feats.resize(frames * NUM_BINS, 0.0);
        for (i, feat) in self.feats.chunks_exact_mut(NUM_BINS).enumerate() {
            let start = i * FRAME_SHIFT;
            self.fbank
                .compute(&self.pending[start..start + FRAME_LEN], feat);
            self.cmvn.apply(feat);
        }
        self.pending.drain(..frames * FRAME_SHIFT);

        let cache =
            |i: usize| TensorView::from_slice(&self.caches[i * CACHE..][..CACHE], vec![1, 128, 19]);
        let out = self.model.forward_with_workspace(
            &mut self.ws,
            TensorView::from_slice(&self.feats, vec![1, frames, NUM_BINS]),
            cache(0),
            cache(1),
            cache(2),
            cache(3),
            cache(4),
            cache(5),
            cache(6),
            cache(7),
        );
        probs.extend_from_slice(&out.0.data);
        let next = [
            &out.1, &out.2, &out.3, &out.4, &out.5, &out.6, &out.7, &out.8,
        ];
        for (dst, src) in self.next_caches.chunks_exact_mut(CACHE).zip(next) {
            dst.copy_from_slice(&src.data);
        }
        std::mem::swap(&mut self.caches, &mut self.next_caches);
    }
}

/// How `Segmenter` turns probabilities into segments; the defaults are the
/// reference pipeline's.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Frames the probability is averaged over.
    pub smooth_window: usize,
    pub threshold: f64,
    /// Frames of audio kept before the detected start of speech.
    pub pad_start: usize,
    /// Frames of speech in a row that start a segment.
    pub min_speech: usize,
    /// A segment is cut after this many frames, and the next one starts.
    pub max_speech: usize,
    /// Frames of silence in a row that end a segment.
    pub min_silence: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            smooth_window: 5,
            threshold: 0.5,
            pad_start: 5,
            min_speech: 8,
            max_speech: 2000,
            min_silence: 20,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    /// Speech started at this frame (0-based).
    Start(usize),
    /// The speech that started at `.0` ended at frame `.1` (0-based,
    /// inclusive).
    End(usize, usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Silence,
    PossibleSpeech,
    Speech,
    PossibleSilence,
}

/// When speech starts and ends: the reference pipeline's
/// `StreamVadPostprocessor`, which smooths the probabilities over a few
/// frames, thresholds them, and only switches state after enough frames of
/// speech or silence in a row.
pub struct Segmenter {
    config: Config,
    /// Frames seen; the reference numbers frames from 1, so this is also the
    /// 1-based index of the last one.
    frames: usize,
    window: std::collections::VecDeque<f64>,
    window_sum: f64,
    state: State,
    speech: usize,
    silence: usize,
    hit_max_speech: bool,
    /// 1-based, 0 for none.
    last_start: usize,
    last_end: usize,
}

impl Segmenter {
    pub fn new(config: Config) -> Self {
        let smooth_window = config.smooth_window.max(1);
        Self {
            config: Config {
                smooth_window,
                pad_start: config.pad_start.max(smooth_window),
                ..config
            },
            frames: 0,
            window: Default::default(),
            window_sum: 0.0,
            state: State::Silence,
            speech: 0,
            silence: 0,
            hit_max_speech: false,
            last_start: 0,
            last_end: 0,
        }
    }

    /// Takes the probability of the next frame; returns the events it
    /// causes, in order.
    pub fn push(&mut self, prob: f32) -> impl Iterator<Item = Event> + use<> {
        let c = self.config;
        self.frames += 1;
        self.window.push_back(prob as f64);
        self.window_sum += prob as f64;
        if self.window.len() > c.smooth_window {
            self.window_sum -= self.window.pop_front().unwrap();
        }
        let is_speech = self.window_sum / self.window.len() as f64 >= c.threshold;

        let (mut start, mut end) = (None, None);
        if self.hit_max_speech {
            self.hit_max_speech = false;
            self.last_start = self.frames;
            start = Some(self.frames);
        }
        match self.state {
            State::Silence if is_speech => {
                self.state = State::PossibleSpeech;
                self.speech += 1;
            }
            State::Silence => {
                self.silence += 1;
                self.speech = 0;
            }
            State::PossibleSpeech if is_speech => {
                self.speech += 1;
                if self.speech >= c.min_speech {
                    self.state = State::Speech;
                    let first = (self.frames + 1).saturating_sub(self.speech + c.pad_start);
                    self.last_start = first.max(1).max(self.last_end + 1);
                    start = Some(self.last_start);
                    self.silence = 0;
                }
            }
            State::PossibleSpeech => {
                self.state = State::Silence;
                self.silence = 1;
                self.speech = 0;
            }
            State::Speech | State::PossibleSilence if is_speech => {
                self.speech += 1;
                self.state = State::Speech;
                self.silence = 0;
                if self.speech >= c.max_speech {
                    self.hit_max_speech = true;
                    self.speech = 0;
                    end = Some(self.close());
                }
            }
            State::Speech => {
                self.speech += 1;
                self.state = State::PossibleSilence;
                self.silence += 1;
            }
            State::PossibleSilence => {
                self.speech += 1;
                self.silence += 1;
                if self.silence >= c.min_silence {
                    self.state = State::Silence;
                    self.speech = 0;
                    end = Some(self.close());
                }
            }
        }
        // The reference numbers frames from 1, and its timestamps subtract
        // one; these are 0-based.
        let start = start.map(|s| Event::Start(s.saturating_sub(1)));
        let end = end.map(|(s, e)| Event::End(s.saturating_sub(1), e.saturating_sub(1)));
        start.into_iter().chain(end)
    }

    /// Ends the open segment, if any, at the last frame: the end of the
    /// audio.
    pub fn finish(&mut self) -> Option<Event> {
        let start = std::mem::take(&mut self.last_start);
        (start != 0).then(|| Event::End(start - 1, self.frames.saturating_sub(1)))
    }

    /// (start, end) of the segment ending now, 1-based.
    fn close(&mut self) -> (usize, usize) {
        let start = std::mem::take(&mut self.last_start);
        self.last_end = self.frames;
        (start, self.frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The events of the reference `StreamVadPostprocessor` with the default
    /// config on the same probabilities: a segment cut at `max_speech`, a
    /// blip too short to start one, and a padded start after silence. The
    /// probabilities are exact in f32, so both average the same doubles.
    #[test]
    fn segmenter_matches_reference() {
        let probs = [
            (0.0, 30),
            (0.875, 2100),
            (0.125, 30),
            (0.75, 5),
            (0.25, 3),
            (0.625, 40),
            (0.0, 25),
        ];
        let mut segmenter = Segmenter::new(Config::default());
        let events: Vec<Event> = probs
            .iter()
            .flat_map(|&(p, n)| std::iter::repeat_n(p, n))
            .flat_map(|p| segmenter.push(p).collect::<Vec<_>>())
            .collect();
        use Event::*;
        assert_eq!(
            events,
            [
                Start(27),
                End(27, 2031),
                Start(2032),
                End(2032, 2151),
                Start(2166),
                End(2166, 2228),
            ]
        );
        assert_eq!(segmenter.finish(), None);
    }

    #[test]
    fn finish_ends_open_segment() {
        let mut segmenter = Segmenter::new(Config::default());
        let events: Vec<Event> = (0..50)
            .flat_map(|_| segmenter.push(1.0).collect::<Vec<_>>())
            .collect();
        assert_eq!(events, [Event::Start(0)]);
        assert_eq!(segmenter.finish(), Some(Event::End(0, 49)));
    }
}
