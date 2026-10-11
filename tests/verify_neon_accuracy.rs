use lele::kernels::{math, norm};
use lele::tensor::TensorView;
use std::borrow::Cow;

#[cfg(target_arch = "aarch64")]
#[test]
fn test_layernorm_accuracy() {
    let norm_size = 10;
    let input_data: Vec<f32> = (0..norm_size).map(|x| x as f32).collect();
    let input = TensorView {
        data: Cow::Borrowed(&input_data),
        shape: Cow::Owned(vec![1, norm_size]),
    };

    let gamma = vec![1.0; norm_size];
    let beta = vec![0.0; norm_size];
    let gamma_v = TensorView {
        data: Cow::Borrowed(&gamma),
        shape: Cow::Owned(vec![norm_size]),
    };
    let beta_v = TensorView {
        data: Cow::Borrowed(&beta),
        shape: Cow::Owned(vec![norm_size]),
    };

    let mut out_scalar = Vec::new();

    // Test via the main norm::layer_norm which dispatches to NEON on aarch64
    norm::layer_norm(&input, &gamma_v, &beta_v, -1, 1e-5, &mut out_scalar);

    // Verify against manually computed expected values
    // For input [0,1,2,...,9], mean=4.5, var=8.25, inv_std=1/sqrt(8.25+1e-5)
    let mean: f32 = input_data.iter().sum::<f32>() / norm_size as f32;
    let var: f32 = input_data
        .iter()
        .map(|x| (x - mean) * (x - mean))
        .sum::<f32>()
        / norm_size as f32;
    let inv_std = 1.0 / (var + 1e-5f32).sqrt();

    for i in 0..norm_size {
        let expected = (input_data[i] - mean) * inv_std * gamma[i] + beta[i];
        let diff = (out_scalar[i] - expected).abs();
        assert!(
            diff < 1e-5,
            "LayerNorm mismatch at index {}: got={}, expected={}, diff={}",
            i,
            out_scalar[i],
            expected,
            diff
        );
    }
}
