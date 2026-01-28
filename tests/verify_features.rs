use approx::assert_abs_diff_eq;
use lele::features::{hann_window, RealFft};
use rustfft::num_complex::Complex;

#[test]
fn test_hann_window_correctness() {
    let size = 4;
    let window = hann_window(size);
    // Standard symmetric hanning window for N=4
    // n=0..3. Formula: 0.5 * (1 - cos(2*pi*n/3))
    // n=0: 0
    // n=1: 0.75
    // n=2: 0.75
    // n=3: 0
    let expected = vec![0.0, 0.75, 0.75, 0.0];

    // Assert length
    assert_eq!(window.len(), size);

    for (a, b) in window.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-6);
    }
}

#[test]
fn test_fft_impulse() {
    // FFT of [1, 0, 0, 0] is [1, 1, 1, 1]
    let input = vec![1.0, 0.0, 0.0, 0.0];
    let fft = RealFft::new(4);
    let mut output = vec![Complex::default(); 4];

    fft.process(&input, &mut output);

    for val in output.iter() {
        assert_abs_diff_eq!(val.re, 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(val.im, 0.0, epsilon = 1e-6);
    }
}

#[test]
fn test_fft_dc() {
    // FFT of [1, 1, 1, 1] is [4, 0, 0, 0]
    let input = vec![1.0, 1.0, 1.0, 1.0];
    let fft = RealFft::new(4);
    let mut output = vec![Complex::default(); 4];

    fft.process(&input, &mut output);

    assert_abs_diff_eq!(output[0].re, 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[1].re, 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[2].re, 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[3].re, 0.0, epsilon = 1e-6);
}

#[test]
fn test_mel_conversion() {
    use lele::features::{hz_to_mel_htk, mel_to_hz_htk};
    assert_abs_diff_eq!(hz_to_mel_htk(0.0), 0.0, epsilon = 1e-4);
    // 700Hz -> 2595 * log10(2) = 2595 * 0.30103 = 781.1728
    assert_abs_diff_eq!(hz_to_mel_htk(700.0), 781.1728, epsilon = 1e-3);

    // Inverse
    assert_abs_diff_eq!(mel_to_hz_htk(hz_to_mel_htk(1000.0)), 1000.0, epsilon = 1e-3);
}

#[test]
fn test_mel_filterbank_shape() {
    use lele::features::mel_filterbank;
    let n_mels = 10;
    let n_fft = 512;
    let sr = 16000.0;
    let weights = mel_filterbank(sr, n_fft, n_mels, 0.0, None);

    let n_freqs = n_fft / 2 + 1; // 257
    assert_eq!(weights.len(), n_mels * n_freqs);

    // Check if some values are non-zero
    let sum: f32 = weights.iter().sum();
    assert!(sum > 0.0);
}
