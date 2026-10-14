//! Whisper's log-mel spectrogram, as `WhisperFeatureExtractor` computes it:
//! 80 Slaney mel bins over 16 kHz audio, 25 ms windows every 10 ms.

use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

pub const SAMPLE_RATE: usize = 16_000;
const N_FFT: usize = 400;
pub const HOP: usize = 160;
const BINS: usize = N_FFT / 2 + 1;
pub const N_MELS: usize = 80;

/// How audio is laid out in the fixed-length window a model reads.
#[derive(Clone, Copy, Debug)]
pub struct Window {
    /// Length of the window, in samples. Longer audio keeps its end.
    pub samples: usize,
    /// Whether shorter audio is zero-padded in front of it (rather than after
    /// it).
    pub pad_front: bool,
    /// Whether the window is scaled to zero mean and unit variance first,
    /// padding included.
    pub normalize: bool,
}

impl Window {
    /// Whisper's: 30 seconds, the audio first.
    pub const ASR: Window = Window {
        samples: 30 * SAMPLE_RATE,
        pad_front: false,
        normalize: false,
    };

    /// Frames of features a window gives.
    pub fn frames(&self) -> usize {
        self.samples / HOP
    }
}

/// One triangular mel filter: its weights from bin `start` on.
struct Filter {
    start: usize,
    weights: Vec<f32>,
}

pub struct LogMel {
    window: Vec<f32>,
    filters: Vec<Filter>,
    fft: Arc<dyn Fft<f32>>,
}

fn hertz_to_mel(f: f64) -> f64 {
    // Slaney: linear below 1 kHz, logarithmic above.
    let (min_log_hz, min_log_mel, logstep) = (1000.0, 15.0, 6.4f64.ln() / 27.0);
    if f >= min_log_hz {
        min_log_mel + (f / min_log_hz).ln() / logstep
    } else {
        3.0 * f / 200.0
    }
}

fn mel_to_hertz(m: f64) -> f64 {
    let (min_log_hz, min_log_mel, logstep) = (1000.0, 15.0, 6.4f64.ln() / 27.0);
    if m >= min_log_mel {
        min_log_hz * (logstep * (m - min_log_mel)).exp()
    } else {
        200.0 * m / 3.0
    }
}

/// Slaney-normalised triangular filters from 0 to 8 kHz over the FFT's bins.
fn mel_filters() -> Vec<Filter> {
    let (lo, hi) = (hertz_to_mel(0.0), hertz_to_mel(SAMPLE_RATE as f64 / 2.0));
    let edges: Vec<f64> = (0..N_MELS + 2)
        .map(|i| mel_to_hertz(lo + (hi - lo) * i as f64 / (N_MELS + 1) as f64))
        .collect();
    let bin_hz = |k: usize| k as f64 * (SAMPLE_RATE as f64 / 2.0) / (BINS - 1) as f64;
    (0..N_MELS)
        .map(|m| {
            let (left, center, right) = (edges[m], edges[m + 1], edges[m + 2]);
            let norm = 2.0 / (right - left);
            let weight = |k: usize| {
                let f = bin_hz(k);
                let up = (f - left) / (center - left);
                let down = (right - f) / (right - center);
                (up.min(down).max(0.0) * norm) as f32
            };
            let first = (0..BINS).find(|&k| weight(k) > 0.0).unwrap_or(0);
            let last = (0..BINS).rev().find(|&k| weight(k) > 0.0).unwrap_or(first);
            Filter {
                start: first,
                weights: (first..=last).map(weight).collect(),
            }
        })
        .collect()
}

impl Default for LogMel {
    fn default() -> Self {
        Self::new()
    }
}

impl LogMel {
    pub fn new() -> Self {
        // Periodic Hann window.
        let window = (0..N_FFT)
            .map(|n| {
                (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / N_FFT as f64).cos()) as f32
            })
            .collect();
        Self {
            window,
            filters: mel_filters(),
            fft: FftPlanner::new().plan_fft_forward(N_FFT),
        }
    }

    /// The model's input, `[N_MELS, window.frames()]` row-major, for `audio`
    /// (16 kHz, mono, in [-1, 1]) laid out in `window`.
    pub fn features(&self, audio: &[f32], window: Window) -> Vec<f32> {
        let (samples, frames) = (window.samples, window.frames());
        let mut x = vec![0f32; samples];
        let used = &audio[audio.len().saturating_sub(samples)..];
        if window.pad_front {
            x[samples - used.len()..].copy_from_slice(used);
        } else {
            x[..used.len()].copy_from_slice(used);
        }
        if window.normalize {
            let mean = x.iter().map(|&v| v as f64).sum::<f64>() / samples as f64;
            let var = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / samples as f64;
            let scale = 1.0 / (var + 1e-7).sqrt();
            for v in &mut x {
                *v = ((*v as f64 - mean) * scale) as f32;
            }
        }

        // Frames are centred, with the signal reflected at both ends; the last
        // of the `frames + 1` is dropped.
        let reflect = |i: isize| -> f32 {
            let n = samples as isize;
            let i = if i < 0 { -i } else { i };
            x[if i >= n { 2 * (n - 1) - i } else { i } as usize]
        };
        let mut mel = vec![0f32; N_MELS * frames];
        let mut frame = vec![Complex32::new(0.0, 0.0); N_FFT];
        let mut power = vec![0f32; BINS];
        let mut scratch = vec![Complex32::new(0.0, 0.0); self.fft.get_inplace_scratch_len()];
        for t in 0..frames {
            let first = (t * HOP) as isize - (N_FFT / 2) as isize;
            for (n, f) in frame.iter_mut().enumerate() {
                *f = Complex32::new(reflect(first + n as isize) * self.window[n], 0.0);
            }
            self.fft.process_with_scratch(&mut frame, &mut scratch);
            for (p, c) in power.iter_mut().zip(&frame) {
                *p = c.norm_sqr();
            }
            for (m, filter) in self.filters.iter().enumerate() {
                let sum: f32 = filter
                    .weights
                    .iter()
                    .zip(&power[filter.start..])
                    .map(|(w, p)| w * p)
                    .sum();
                mel[m * frames + t] = sum.max(1e-10).log10();
            }
        }
        // Compress the dynamic range to 8 decades below the loudest bin.
        let floor = mel.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) - 8.0;
        for v in &mut mel {
            *v = (v.max(floor) + 4.0) / 4.0;
        }
        mel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smart Turn's window: 8 seconds, the audio last, normalised.
    const TURN: Window = Window {
        samples: 8 * SAMPLE_RATE,
        pad_front: true,
        normalize: true,
    };

    fn tone(freq: f64, samples: usize) -> Vec<f32> {
        (0..samples)
            .map(|i| {
                (2.0 * std::f64::consts::PI * freq * i as f64 / SAMPLE_RATE as f64).sin() as f32
            })
            .collect()
    }

    /// The mel scale and its inverse agree, and 1 kHz is where the scale
    /// turns logarithmic.
    #[test]
    fn mel_scale_round_trips() {
        for f in [0.0, 100.0, 999.0, 1000.0, 4000.0, 8000.0] {
            assert!((mel_to_hertz(hertz_to_mel(f)) - f).abs() < 1e-6, "{f}");
        }
        assert!((hertz_to_mel(1000.0) - 15.0).abs() < 1e-9);
    }

    /// Slaney normalisation gives every filter the same area in hertz.
    #[test]
    fn filters_have_equal_area_in_hertz() {
        let bin_hz = SAMPLE_RATE as f64 / 2.0 / (BINS - 1) as f64;
        for (m, f) in mel_filters().iter().enumerate() {
            let area: f64 = f.weights.iter().map(|&w| w as f64).sum::<f64>() * bin_hz;
            // The lowest filters are narrower than a bin or two, so coarse.
            assert!((0.7..1.3).contains(&area), "filter {m}: area {area}");
        }
    }

    /// A pure tone lights the mel bin that contains it.
    #[test]
    fn tone_lands_in_its_mel_bin() {
        let f = LogMel::new().features(&tone(1000.0, TURN.samples), TURN);
        let frames = TURN.frames();
        assert_eq!(f.len(), N_MELS * frames);
        // Mel index of 1 kHz: 15 mel of the 80 spanning 0..8 kHz (45.2 mel).
        let want = (hertz_to_mel(1000.0) / hertz_to_mel(8000.0) * (N_MELS + 1) as f64).round()
            as usize
            - 1;
        let mid = frames / 2;
        let loudest = (0..N_MELS)
            .max_by(|&a, &b| f[a * frames + mid].total_cmp(&f[b * frames + mid]))
            .unwrap();
        assert!(loudest.abs_diff(want) <= 1, "{loudest} vs {want}");
        // Everything is within the 8-decade window, scaled to (x+4)/4.
        let (lo, hi) = f
            .iter()
            .fold((f32::MAX, f32::MIN), |(l, h), &v| (l.min(v), h.max(v)));
        assert!((hi - lo) <= 2.0 + 1e-4, "range {}", hi - lo);
    }

    /// Short audio is padded as the window says: the frames of the padding are
    /// all alike, and the frames of the audio itself are not.
    #[test]
    fn short_audio_is_padded_where_the_window_says() {
        let lm = LogMel::new();
        let audio = tone(440.0, SAMPLE_RATE);
        let column = |f: &[f32], frames: usize, t: usize| {
            (0..N_MELS).map(|m| f[m * frames + t]).collect::<Vec<_>>()
        };
        let distance =
            |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>();

        let frames = TURN.frames();
        let f = lm.features(&audio, TURN);
        assert!(distance(&column(&f, frames, 10), &column(&f, frames, 300)) < 1e-3);
        assert!(distance(&column(&f, frames, 300), &column(&f, frames, frames - 10)) > 1.0);

        let frames = Window::ASR.frames();
        let f = lm.features(&audio, Window::ASR);
        assert_eq!(f.len(), N_MELS * frames);
        assert!(
            distance(
                &column(&f, frames, frames - 10),
                &column(&f, frames, frames - 300)
            ) < 1e-3
        );
        assert!(distance(&column(&f, frames, 50), &column(&f, frames, frames - 10)) > 1.0);
    }
}
