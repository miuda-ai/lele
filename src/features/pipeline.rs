use crate::features::{Lfr, LfrConfig, SparseMelBank, hann_window, log_compress};
use crate::kernels::fft::Rfft;
use crate::kernels::simd::simd_call;
use crate::tensor::TensorView;
use fearless_simd::{Level, Simd, f32x8};
use fearless_simd_macros::simd;

#[derive(Debug, Clone)]
pub struct FeatureConfig {
    pub sample_rate: usize,
    pub n_mels: usize,
    pub frame_length_ms: f32,
    pub frame_shift_ms: f32,
    pub lfr_m: usize,
    pub lfr_n: usize,
}
impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_mels: 80,
            frame_length_ms: 25.0,
            frame_shift_ms: 10.0,
            lfr_m: 7,
            lfr_n: 6,
        }
    }
}
pub struct SenseVoiceFrontend {
    config: FeatureConfig,
    window: Vec<f32>,
    fft: Rfft,
    mel_bank: SparseMelBank,
    fft_len: usize,
    hop_len: usize,
    lfr: Lfr,
    level: Level,
}
impl SenseVoiceFrontend {
    pub fn new(config: FeatureConfig) -> Self {
        let frame_len = (config.sample_rate as f32 * config.frame_length_ms / 1000.0) as usize;
        let n_fft = if frame_len > 400 { 1024 } else { 512 };
        let fft_len = n_fft;
        let hop_len = (config.sample_rate as f32 * config.frame_shift_ms / 1000.0) as usize;
        let window = hann_window(frame_len);
        let fft = Rfft::new(fft_len);
        let mel_bank = SparseMelBank::new(
            config.sample_rate as f32,
            fft_len,
            config.n_mels,
            20.0,
            None,
        );
        let lfr = Lfr::new(LfrConfig {
            m: config.lfr_m,
            n: config.lfr_n,
        });
        Self {
            config,
            window,
            fft,
            mel_bank,
            fft_len,
            hop_len,
            lfr,
            level: Level::new(),
        }
    }

    pub fn compute(&self, pcm: &[f32]) -> TensorView<'static> {
        let frame_len = self.window.len();
        let hop_len = self.hop_len;
        if pcm.len() < frame_len {
            return TensorView::empty();
        }
        let num_frames = (pcm.len() - frame_len) / hop_len + 1;

        let n_bins = self.fft_len / 2 + 1;
        let mut mel_output = vec![0.0; num_frames * self.config.n_mels];
        // Samples past `frame_len` stay zero: the frame is zero-padded to the FFT length.
        let mut frame_buf = vec![0.0; self.fft_len];
        let mut fft_scratch = vec![0.0; self.fft.scratch_len()];
        let (mut fft_re, mut fft_im) = (vec![0.0; n_bins], vec![0.0; n_bins]);
        let mut power_spec = vec![0.0; n_bins];

        for i in 0..num_frames {
            let start = i * hop_len;
            let end = start + frame_len;

            // 1-4. Scale, mean subtraction, pre-emphasis and windowing
            prepare_frame(
                self.level,
                &pcm[start..end],
                &self.window,
                &mut frame_buf[..frame_len],
            );

            // 5. FFT
            self.fft
                .forward(&frame_buf, &mut fft_scratch, &mut fft_re, &mut fft_im);

            // 6. Power Spectrum
            for ((p, &re), &im) in power_spec.iter_mut().zip(&fft_re).zip(&fft_im) {
                *p = re * re + im * im;
            }

            // 7. Mel Filterbank
            let mel_frame = &mut mel_output[i * self.config.n_mels..(i + 1) * self.config.n_mels];
            self.mel_bank.apply(&power_spec, mel_frame);

            // 8. Log compression
            log_compress(mel_frame, 1e-5);
        }
        let mel_tensor = TensorView::from_owned(mel_output, vec![num_frames, self.config.n_mels]);
        self.lfr.compute(&mel_tensor)
    }
}

/// Scale from [-1, 1] samples to the 16-bit range the model was trained on.
const PCM_SCALE: f32 = 32768.0;
const PREEMPH_COEFF: f32 = 0.97;

/// Writes the windowed frame of `pcm` to `out`: samples scaled by
/// `PCM_SCALE`, minus their mean, pre-emphasized, times `window`.
///
/// Pre-emphasis subtracts the previous mean-subtracted sample, not the
/// previous pre-emphasized one, so every output depends only on two inputs and
/// the whole chain runs in one vector pass. Each step rounds as it would one
/// step at a time, so the result matches running them in sequence bit for bit.
fn prepare_frame(level: Level, pcm: &[f32], window: &[f32], out: &mut [f32]) {
    let n = pcm.len();
    assert!(window.len() >= n && out.len() >= n);
    if n == 0 {
        return;
    }
    // Summed in order: a vector sum would round differently.
    let mean = pcm.iter().map(|&x| x * PCM_SCALE).sum::<f32>() / n as f32;
    simd_call!(level, prepare_frame_simd(pcm, mean, window, out))
}

#[simd]
fn prepare_frame_simd<S: Simd>(simd: S, pcm: &[f32], mean: f32, window: &[f32], out: &mut [f32]) {
    use fearless_simd::prelude::*;
    let n = pcm.len();
    let centered = |x: f32| x * PCM_SCALE - mean;
    out[0] = centered(pcm[0]) * window[0];

    let (scale, mean_v, coeff) = (
        f32x8::splat(simd, PCM_SCALE),
        f32x8::splat(simd, mean),
        f32x8::splat(simd, PREEMPH_COEFF),
    );
    let (cur, cur_rest) = pcm[1..].as_chunks::<8>();
    let (prev, prev_rest) = pcm[..n - 1].as_chunks::<8>();
    let (win, win_rest) = window[1..n].as_chunks::<8>();
    let (dst, dst_rest) = out[1..n].as_chunks_mut::<8>();
    for (((x, p), w), y) in cur.iter().zip(prev).zip(win).zip(dst) {
        let x = f32x8::from_slice(simd, x) * scale - mean_v;
        let p = f32x8::from_slice(simd, p) * scale - mean_v;
        ((x - coeff * p) * f32x8::from_slice(simd, w)).store_slice(y);
    }
    let tail = cur_rest.iter().zip(prev_rest).zip(win_rest).zip(dst_rest);
    for (((&x, &p), &w), y) in tail {
        *y = (centered(x) - PREEMPH_COEFF * centered(p)) * w;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::RealFft;
    use crate::features::fft::Complex;
    use crate::kernels::test_util::levels;

    /// The frame preparation as separate passes over a buffer.
    fn prepare_frame_reference(pcm: &[f32], window: &[f32], out: &mut [f32]) {
        let mut raw: Vec<f32> = pcm.iter().map(|&x| x * PCM_SCALE).collect();
        let mean = raw.iter().sum::<f32>() / raw.len() as f32;
        for x in &mut raw {
            *x -= mean;
        }
        for j in (1..raw.len()).rev() {
            raw[j] -= PREEMPH_COEFF * raw[j - 1];
        }
        for ((y, &x), &w) in out.iter_mut().zip(&raw).zip(window) {
            *y = x * w;
        }
    }

    fn pcm(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / 16000.0;
                0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                    + 0.05 * ((i * 7919 % 1000) as f32 / 1000.0 - 0.5)
            })
            .collect()
    }

    #[test]
    fn test_prepare_frame_matches_separate_passes() {
        let sizes = (1..=40).chain([400, 401, 1023]);
        for n in sizes {
            let pcm = pcm(n);
            let window = hann_window(n);
            let mut want = vec![0.0; n];
            prepare_frame_reference(&pcm, &window, &mut want);
            for level in levels() {
                let mut got = vec![f32::NAN; n];
                prepare_frame(level, &pcm, &window, &mut got);
                for (j, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        w.to_bits(),
                        "n={n} j={j} {level:?}: {g} vs {w}"
                    );
                }
            }
        }
    }

    /// The frontend as it was before the frame preparation was fused and the
    /// FFT called directly.
    fn compute_reference(fe: &SenseVoiceFrontend, pcm: &[f32]) -> Vec<f32> {
        let frame_len = fe.window.len();
        let num_frames = (pcm.len() - frame_len) / fe.hop_len + 1;
        let n_mels = fe.config.n_mels;
        let fft = RealFft::new(fe.fft_len);
        let mut mel = vec![0.0; num_frames * n_mels];
        let mut frame = vec![0.0; fe.fft_len];
        let mut spectrum = vec![Complex { re: 0.0, im: 0.0 }; fe.fft_len];
        let mut scratch = vec![Complex { re: 0.0, im: 0.0 }; fft.scratch_len()];
        let mut power = vec![0.0; fe.fft_len / 2 + 1];
        for (i, mel_frame) in mel.chunks_exact_mut(n_mels).enumerate() {
            let start = i * fe.hop_len;
            prepare_frame_reference(
                &pcm[start..start + frame_len],
                &fe.window,
                &mut frame[..frame_len],
            );
            fft.process_with_scratch(&frame, &mut spectrum, &mut scratch);
            for (p, c) in power.iter_mut().zip(&spectrum) {
                *p = c.re * c.re + c.im * c.im;
            }
            fe.mel_bank.apply(&power, mel_frame);
            log_compress(mel_frame, 1e-5);
        }
        let mel = TensorView::from_owned(mel, vec![num_frames, n_mels]);
        fe.lfr.compute(&mel).data.into_owned()
    }

    #[test]
    fn test_compute_matches_reference() {
        let fe = SenseVoiceFrontend::new(FeatureConfig::default());
        for n in [400, 559, 560, 16000, 16123] {
            let pcm = pcm(n);
            let got = fe.compute(&pcm);
            let want = compute_reference(&fe, &pcm);
            assert_eq!(got.data.len(), want.len(), "n={n}");
            for (j, (g, w)) in got.data.iter().zip(&want).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(), "n={n} j={j}: {g} vs {w}");
            }
        }
    }

    #[test]
    fn test_compute_short_input_is_empty() {
        let fe = SenseVoiceFrontend::new(FeatureConfig::default());
        assert!(fe.compute(&pcm(399)).data.is_empty());
    }
}
