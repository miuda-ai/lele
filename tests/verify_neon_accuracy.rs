use lele::kernels::{math, norm, quantization};
use lele::tensor::TensorView;
use std::borrow::Cow;

#[cfg(target_arch = "aarch64")]
#[test]
fn test_relu_accuracy() {
    let input_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 0.5, 1.5];
    let input = TensorView {
        data: Cow::Borrowed(&input_data),
        shape: Cow::Owned(vec![input_data.len()]),
    };

    let mut out_scalar = Vec::new();
    let mut out_neon = Vec::new();

    math::relu(&input, &mut out_scalar);
    lele::kernels::neon::math::relu(&input, &mut out_neon);

    for i in 0..input_data.len() {
        assert_eq!(out_scalar[i], out_neon[i], "ReLU mismatch at index {}", i);
    }
}

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
    let var: f32 = input_data.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / norm_size as f32;
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

#[cfg(target_arch = "aarch64")]
#[test]
fn test_quantization_accuracy() {
    let input_data = vec![-10.0, -5.0, 0.0, 5.0, 10.0, 2.0, 3.0, 4.0];
    let input = TensorView {
        data: Cow::Borrowed(&input_data),
        shape: Cow::Owned(vec![input_data.len()]),
    };

    let mut out_y_s = Vec::new();
    let mut out_s_s = Vec::new();
    let mut out_z_s = Vec::new();

    let mut out_y_n = Vec::new();
    let mut out_s_n = Vec::new();
    let mut out_z_n = Vec::new();

    quantization::dynamic_quantize_linear(&input, &mut out_y_s, &mut out_s_s, &mut out_z_s);
    lele::kernels::neon::quantization::dynamic_quantize_linear(
        &input,
        &mut out_y_n,
        &mut out_s_n,
        &mut out_z_n,
    );

    assert!((out_s_s[0] - out_s_n[0]).abs() < 1e-6, "Scale mismatch");
    assert!((out_z_s[0] - out_z_n[0]).abs() < 1e-6, "ZP mismatch");

    for i in 0..input_data.len() {
        assert_eq!(
            out_y_s[i], out_y_n[i],
            "Quantization mismatch at index {}",
            i
        );
    }
}
