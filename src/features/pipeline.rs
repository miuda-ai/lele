use crate::features::{Lfr, LfrConfig, RealFft, SparseMelBank, hann_window, log_compress};
use crate::tensor::TensorView;
use rustfft::num_complex::Complex;
#[cfg(nightly_build)]
use std::simd::prelude::*;

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
    fft: RealFft,
    mel_bank: SparseMelBank,
    fft_len: usize,
    hop_len: usize,
    lfr: Lfr,
}
impl SenseVoiceFrontend {
    pub fn new(config: FeatureConfig) -> Self {
        let frame_len = (config.sample_rate as f32 * config.frame_length_ms / 1000.0) as usize;
        let n_fft = if frame_len > 400 { 1024 } else { 512 };
        let fft_len = n_fft;
        let hop_len = (config.sample_rate as f32 * config.frame_shift_ms / 1000.0) as usize;
        let window = hann_window(frame_len);
        let fft = RealFft::new(fft_len);
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
        }
    }

    pub fn compute(&self, pcm: &[f32]) -> TensorView<'static> {
        let frame_len = self.window.len();
        let hop_len = self.hop_len;
        if pcm.len() < frame_len {
            return TensorView::empty();
        }
        let num_frames = (pcm.len() - frame_len) / hop_len + 1;

        let mut mel_output = vec![0.0; num_frames * self.config.n_mels];
        let mut frame_buf = vec![0.0; self.fft_len];
        let mut raw_frame = vec![0.0; frame_len];
        let mut fft_out = vec![Complex { re: 0.0, im: 0.0 }; self.fft_len];
        let mut power_spec = vec![0.0; self.fft_len / 2 + 1];
        let mut fft_scratch = vec![Complex { re: 0.0, im: 0.0 }; self.fft.scratch_len()];

        let scale = 32768.0;
        let preemph_coeff = 0.97f32;

        for i in 0..num_frames {
            let start = i * hop_len;
            let end = start + frame_len;

            // 1. Scale and Copy
            let pcm_slice = &pcm[start..end];
            #[cfg(nightly_build)]
            {
                let (prefix, middle, _suffix) = pcm_slice.as_simd::<8>();
                let offset = prefix.len();
                for j in 0..offset {
                    raw_frame[j] = pcm_slice[j] * scale;
                }
                let scale_vec = f32x8::splat(scale);
                for j in 0..middle.len() {
                    let res = middle[j] * scale_vec;
                    res.copy_to_slice(&mut raw_frame[offset + j * 8..]);
                }
                for j in (offset + middle.len() * 8)..frame_len {
                    raw_frame[j] = pcm_slice[j] * scale;
                }
            }
            #[cfg(not(nightly_build))]
            {
                for j in 0..frame_len {
                    raw_frame[j] = pcm_slice[j] * scale;
                }
            }

            // 2. Mean Subtraction
            let sum: f32 = raw_frame.iter().sum();
            let mean = sum / frame_len as f32;
            #[cfg(nightly_build)]
            {
                let mean_vec = f32x8::splat(mean);

                let (prefix, middle, suffix) = raw_frame.as_simd_mut::<8>();
                for x in prefix {
                    *x -= mean;
                }
                for x_vec in middle {
                    *x_vec -= mean_vec;
                }
                for x in suffix {
                    *x -= mean;
                }
            }
            #[cfg(not(nightly_build))]
            {
                for x in &mut raw_frame {
                    *x -= mean;
                }
            }

            // 3. Pre-emphasis (inherently sequential)
            for j in (1..frame_len).rev() {
                raw_frame[j] -= preemph_coeff * raw_frame[j - 1];
            }

            // 4. Windowing
            #[cfg(nightly_build)]
            {
                let (prefix, middle, _suffix) = raw_frame.as_simd::<8>();
                let offset = prefix.len();
                for j in 0..offset {
                    frame_buf[j] = raw_frame[j] * self.window[j];
                }
                for j in 0..middle.len() {
                    let idx = offset + j * 8;
                    let w = f32x8::from_slice(&self.window[idx..idx + 8]);
                    let res = middle[j] * w;
                    res.copy_to_slice(&mut frame_buf[idx..idx + 8]);
                }
                for j in (offset + middle.len() * 8)..frame_len {
                    frame_buf[j] = raw_frame[j] * self.window[j];
                }
            }
            #[cfg(not(nightly_build))]
            {
                for j in 0..frame_len {
                    frame_buf[j] = raw_frame[j] * self.window[j];
                }
            }

            for j in frame_len..self.fft_len {
                frame_buf[j] = 0.0;
            }

            // 5. FFT
            self.fft
                .process_with_scratch(&frame_buf, &mut fft_out, &mut fft_scratch);

            // 6. Power Spectrum
            for j in 0..power_spec.len() {
                let re = fft_out[j].re;
                let im = fft_out[j].im;
                power_spec[j] = re * re + im * im;
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
