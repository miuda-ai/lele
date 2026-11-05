pub fn hz_to_mel_htk(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}
pub fn mel_to_hz_htk(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}
pub fn mel_filterbank(
    sample_rate: f32,
    n_fft: usize,
    n_mels: usize,
    f_min: f32,
    f_max: Option<f32>,
) -> Vec<f32> {
    let f_max = f_max.unwrap_or(sample_rate / 2.0);
    let n_freqs = n_fft / 2 + 1;
    let mel_min = hz_to_mel_htk(f_min);
    let mel_max = hz_to_mel_htk(f_max);
    let mel_points = n_mels + 2;
    let mel_step = (mel_max - mel_min) / (n_mels + 1) as f32;
    let mel_freqs: Vec<f32> = (0..mel_points)
        .map(|i| mel_min + i as f32 * mel_step)
        .collect();
    let hz_freqs: Vec<f32> = mel_freqs.iter().map(|&m| mel_to_hz_htk(m)).collect();
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * sample_rate / n_fft as f32)
        .collect();

    let mut weights = vec![0.0; n_mels * n_freqs];
    for i in 0..n_mels {
        let f_left = hz_freqs[i];
        let f_center = hz_freqs[i + 1];
        let f_right = hz_freqs[i + 2];
        for j in 0..n_freqs {
            let f = fft_freqs[j];
            let mut val = 0.0;
            if f > f_left && f < f_center {
                val = (f - f_left) / (f_center - f_left);
            } else if f >= f_center && f < f_right {
                val = (f_right - f) / (f_right - f_center);
            }
            weights[i * n_freqs + j] = val;
        }
    }
    weights
}

#[derive(Clone, Debug)]
pub struct SparseMelBank {
    pub n_mels: usize,
    pub n_freqs: usize,
    /// Each filter has (start_bin, weights)
    pub filters: Vec<(usize, Vec<f32>)>,
}

impl SparseMelBank {
    pub fn new(
        sample_rate: f32,
        n_fft: usize,
        n_mels: usize,
        f_min: f32,
        f_max: Option<f32>,
    ) -> Self {
        let weights = mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max);
        let n_freqs = n_fft / 2 + 1;
        let mut filters = Vec::with_capacity(n_mels);

        for i in 0..n_mels {
            let row = &weights[i * n_freqs..(i + 1) * n_freqs];
            let mut start = 0;
            while start < n_freqs && row[start] == 0.0 {
                start += 1;
            }
            let mut end = n_freqs;
            while end > start && row[end - 1] == 0.0 {
                end -= 1;
            }

            if start < end {
                filters.push((start, row[start..end].to_vec()));
            } else {
                filters.push((0, vec![]));
            }
        }

        Self {
            n_mels,
            n_freqs,
            filters,
        }
    }

    pub fn apply(&self, power_spectrum: &[f32], output: &mut [f32]) {
        assert_eq!(power_spectrum.len(), self.n_freqs);
        assert_eq!(output.len(), self.n_mels);

        for (i, (start, weights)) in self.filters.iter().enumerate() {
            let mut sum = 0.0;
            let spectrum_part = &power_spectrum[*start..*start + weights.len()];
            for j in 0..weights.len() {
                sum += weights[j] * spectrum_part[j];
            }
            output[i] = sum;
        }
    }
}
pub fn apply_mel_bank(
    power_spectrum: &[f32],
    mel_filters: &[f32],
    n_mels: usize,
    output: &mut [f32],
) {
    let n_freqs = power_spectrum.len();
    assert_eq!(mel_filters.len(), n_mels * n_freqs);
    assert_eq!(output.len(), n_mels);
    for (i, out) in output.iter_mut().enumerate() {
        let row = &mel_filters[i * n_freqs..][..n_freqs];
        *out = row.iter().zip(power_spectrum).map(|(&w, &p)| w * p).sum();
    }
}
pub fn log_compress(input: &mut [f32], eps: f32) {
    for x in input.iter_mut() {
        *x = (x.max(eps)).ln();
    }
}
