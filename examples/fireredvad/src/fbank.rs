//! Kaldi filterbank features, as kaldi-native-fbank computes them with the
//! options FireRedVAD uses: 16 kHz, 25 ms frames every 10 ms, no dither,
//! 80 mel bins. The rest are Kaldi's defaults: DC offset removed,
//! pre-emphasis 0.97, Povey window, 512-point FFT, power spectrum, mel bins
//! from 20 Hz to Nyquist, log floored at `f32::EPSILON`.

use lele::kernels::fft::Rfft;

pub const FRAME_LEN: usize = 400;
pub const FRAME_SHIFT: usize = 160;
pub const NUM_BINS: usize = 80;
const FFT_LEN: usize = 512;
const PREEMPH: f32 = 0.97;
const LOW_FREQ: f32 = 20.0;

pub struct Fbank {
    window: Vec<f32>,
    /// Each mel bin's first FFT bin and weights.
    bins: Vec<(usize, Vec<f32>)>,
    fft: Rfft,
    frame: Vec<f32>,
    scratch: Vec<f32>,
    re: Vec<f32>,
    im: Vec<f32>,
    power: Vec<f32>,
}

impl Default for Fbank {
    fn default() -> Self {
        Self::new()
    }
}

impl Fbank {
    pub fn new() -> Self {
        let a = 2.0 * std::f64::consts::PI / (FRAME_LEN - 1) as f64;
        let window = (0..FRAME_LEN)
            .map(|i| (0.5 - 0.5 * (a * i as f64).cos()).powf(0.85) as f32)
            .collect();
        let fft = Rfft::new(FFT_LEN);
        Self {
            window,
            bins: mel_bins(16000.0),
            scratch: vec![0.0; fft.scratch_len()],
            fft,
            frame: vec![0.0; FFT_LEN],
            re: vec![0.0; FFT_LEN / 2 + 1],
            im: vec![0.0; FFT_LEN / 2 + 1],
            power: vec![0.0; FFT_LEN / 2],
        }
    }

    /// The log mel energies of one frame of `FRAME_LEN` samples in the
    /// int16 range.
    pub fn compute(&mut self, samples: &[f32], out: &mut [f32]) {
        assert_eq!(samples.len(), FRAME_LEN);
        assert_eq!(out.len(), NUM_BINS);
        let frame = &mut self.frame[..FRAME_LEN];
        let mean = samples.iter().sum::<f32>() / FRAME_LEN as f32;
        for (y, &x) in frame.iter_mut().zip(samples) {
            *y = x - mean;
        }
        for i in (1..FRAME_LEN).rev() {
            frame[i] -= PREEMPH * frame[i - 1];
        }
        frame[0] -= PREEMPH * frame[0];
        for (x, &w) in frame.iter_mut().zip(&self.window) {
            *x *= w;
        }
        // Samples past `FRAME_LEN` stay zero: the frame is zero-padded.
        self.fft
            .forward(&self.frame, &mut self.scratch, &mut self.re, &mut self.im);
        for ((p, &re), &im) in self.power.iter_mut().zip(&self.re).zip(&self.im) {
            *p = re * re + im * im;
        }
        for (y, (start, weights)) in out.iter_mut().zip(&self.bins) {
            let energy: f32 = weights
                .iter()
                .zip(&self.power[*start..])
                .map(|(&w, &p)| w * p)
                .sum();
            *y = energy.max(f32::EPSILON).ln();
        }
    }
}

fn mel(freq: f32) -> f32 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

/// Kaldi's triangular mel bins over the FFT bins below Nyquist.
fn mel_bins(sample_rate: f32) -> Vec<(usize, Vec<f32>)> {
    let bin_width = sample_rate / FFT_LEN as f32;
    let (mel_low, mel_high) = (mel(LOW_FREQ), mel(0.5 * sample_rate));
    let delta = (mel_high - mel_low) / (NUM_BINS + 1) as f32;
    (0..NUM_BINS)
        .map(|bin| {
            let left = mel_low + bin as f32 * delta;
            let center = mel_low + (bin + 1) as f32 * delta;
            let right = mel_low + (bin + 2) as f32 * delta;
            let weights: Vec<(usize, f32)> = (0..FFT_LEN / 2)
                .filter_map(|i| {
                    let m = mel(bin_width * i as f32);
                    (m > left && m < right).then(|| {
                        let w = if m <= center {
                            (m - left) / (center - left)
                        } else {
                            (right - m) / (right - center)
                        };
                        (i, w)
                    })
                })
                .collect();
            (weights[0].0, weights.into_iter().map(|(_, w)| w).collect())
        })
        .collect()
}
