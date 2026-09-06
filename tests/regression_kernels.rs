// Regression tests for optimized kernels.
use lele::kernels::*;
use lele::tensor::TensorView;

fn assert_close(a: &[f32], b: &[f32], tol: f32, name: &str) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch ({} vs {})", name, a.len(), b.len());
    let mut max_diff = 0.0f32;
    let mut max_idx = 0;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > max_diff {
            max_diff = d;
            max_idx = i;
        }
    }
    assert!(
        max_diff <= tol,
        "{}: max diff {:.6e} at idx {} (got {:.8}, expected {:.8}), tol {:.6e}",
        name, max_diff, max_idx, a[max_idx], b[max_idx], tol,
    );
}

fn ref_conv2d(
    input: &[f32], weight: &[f32], bias: Option<&[f32]>,
    n: usize, oc: usize, ic: usize,
    ih: usize, iw: usize,
    kh: usize, kw: usize,
    sy: usize, sx: usize,
    pt: usize, pl: usize, pb: usize, pr: usize,
    groups: usize, relu: bool,
) -> Vec<f32> {
    let ic_g = ic / groups;
    let oc_g = oc / groups;
    let oh = (ih + pt + pb - kh) / sy + 1;
    let ow = (iw + pl + pr - kw) / sx + 1;
    let mut out = vec![0.0f32; n * oc * oh * ow];
    for b in 0..n {
        for g in 0..groups {
            for oci in 0..oc_g {
                let out_c = g * oc_g + oci;
                for oy in 0..oh {
                    for ox in 0..ow {
                        let mut sum = 0.0f32;
                        for c in 0..ic_g {
                            let in_c = g * ic_g + c;
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let iy = oy * sy + ky;
                                    let ix = ox * sx + kx;
                                    if iy >= pt && ix >= pl && iy < ih + pt && ix < iw + pl {
                                        let ri = iy - pt;
                                        let rx = ix - pl;
                                        let iv = input[b * ic * ih * iw + in_c * ih * iw + ri * iw + rx];
                                        let wv = weight[out_c * ic_g * kh * kw + c * kh * kw + ky * kw + kx];
                                        sum += iv * wv;
                                    }
                                }
                            }
                        }
                        if let Some(bias) = bias { sum += bias[out_c]; }
                        if relu && sum < 0.0 { sum = 0.0; }
                        out[b * oc * oh * ow + out_c * oh * ow + oy * ow + ox] = sum;
                    }
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Conv2d tests
// ---------------------------------------------------------------------------

#[test]
fn test_conv2d_3x3_stride1_pad1() {
    let (n, ic, oc, ih, iw) = (1, 2, 3, 8, 8);
    let (kh, kw) = (3, 3);

    let input: Vec<f32> = (0..n * ic * ih * iw).map(|i| i as f32 * 0.1 - 2.0).collect();
    let weight: Vec<f32> = (0..oc * ic * kh * kw).map(|i| i as f32 * 0.05 - 1.0).collect();
    let bias: Vec<f32> = (0..oc).map(|i| i as f32 * 0.01).collect();

    let inp_t = TensorView::from_slice(&input, vec![n, ic, ih, iw]);
    let w_t = TensorView::from_slice(&weight, vec![oc, ic, kh, kw]);
    let b_t = TensorView::from_slice(&bias, vec![oc]);

    let mut out_buf = Vec::new();
    let result = conv2d_fused(&inp_t, &w_t, Some(&b_t), &[1, 1], 1, &[1, 1, 1, 1], &[1, 1], true, &mut out_buf);

    let expected = ref_conv2d(&input, &weight, Some(&bias), n, oc, ic, ih, iw, kh, kw, 1, 1, 1, 1, 1, 1, 1, true);
    assert_close(&result.data, &expected, 1e-3, "conv2d_3x3_s1_p1");
}

#[test]
fn test_conv2d_3x3_stride1_no_bias() {
    let (n, ic, oc, ih, iw) = (1, 1, 2, 6, 6);
    let (kh, kw) = (3, 3);

    let input: Vec<f32> = (0..n * ic * ih * iw).map(|i| i as f32 * 0.3 - 1.0).collect();
    let weight: Vec<f32> = (0..oc * ic * kh * kw).map(|i| i as f32 * 0.2 - 0.5).collect();

    let inp_t = TensorView::from_slice(&input, vec![n, ic, ih, iw]);
    let w_t = TensorView::from_slice(&weight, vec![oc, ic, kh, kw]);

    let mut out_buf = Vec::new();
    let result = conv2d_fused(&inp_t, &w_t, None, &[1, 1], 1, &[1, 1, 1, 1], &[1, 1], false, &mut out_buf);

    let expected = ref_conv2d(&input, &weight, None, n, oc, ic, ih, iw, kh, kw, 1, 1, 1, 1, 1, 1, 1, false);
    assert_close(&result.data, &expected, 1e-3, "conv2d_3x3_s1_no_bias");
}

#[test]
fn test_conv2d_3x3_stride2() {
    let (n, ic, oc, ih, iw) = (1, 3, 4, 16, 16);
    let (kh, kw) = (3, 3);

    let input: Vec<f32> = (0..n * ic * ih * iw).map(|i| i as f32 * 0.01 - 1.0).collect();
    let weight: Vec<f32> = (0..oc * ic * kh * kw).map(|i| i as f32 * 0.03).collect();
    let bias: Vec<f32> = vec![0.1; oc];

    let inp_t = TensorView::from_slice(&input, vec![n, ic, ih, iw]);
    let w_t = TensorView::from_slice(&weight, vec![oc, ic, kh, kw]);
    let b_t = TensorView::from_slice(&bias, vec![oc]);

    let mut out_buf = Vec::new();
    let result = conv2d_fused(&inp_t, &w_t, Some(&b_t), &[1, 1], 1, &[1, 1, 1, 1], &[2, 2], true, &mut out_buf);

    let expected = ref_conv2d(&input, &weight, Some(&bias), n, oc, ic, ih, iw, kh, kw, 2, 2, 1, 1, 1, 1, 1, true);
    assert_close(&result.data, &expected, 1e-3, "conv2d_3x3_s2");
}

/// Runs a 3x3 stride-1 pad-1 depthwise conv through the dedicated depthwise
/// path and through the generic im2col+GEMM path (same conv, weights expanded
/// to a dense block-diagonal kernel) and asserts the two agree.
fn assert_depthwise_matches_im2col(
    n: usize,
    groups: usize,
    ih: usize,
    iw: usize,
    input: &[f32],
    weight: &[f32],
    name: &str,
) {
    let (kh, kw) = (3, 3);
    let inp_t = TensorView::from_slice(input, vec![n, groups, ih, iw]);
    let w_dw = TensorView::from_slice(weight, vec![groups, 1, kh, kw]);

    let mut out_buf = Vec::new();
    let result = conv2d(&inp_t, &w_dw, None, &[1, 1], groups as i64, &[1, 1, 1, 1], &[1, 1], &mut out_buf);

    assert_eq!(result.shape.as_ref(), &[n, groups, ih, iw]);
    for v in result.data.iter() {
        assert!(v.is_finite(), "{}: depthwise output should be finite", name);
    }

    let mut full_w = vec![0.0f32; groups * groups * kh * kw];
    for g in 0..groups {
        for k in 0..kh * kw {
            full_w[g * groups * kh * kw + g * kh * kw + k] = weight[g * kh * kw + k];
        }
    }
    let w_full = TensorView::from_slice(&full_w, vec![groups, groups, kh, kw]);
    let mut out_buf2 = Vec::new();
    let ref_result = conv2d_fused(&inp_t, &w_full, None, &[1, 1], 1, &[1, 1, 1, 1], &[1, 1], false, &mut out_buf2);
    assert_close(&result.data, &ref_result.data, 1e-3, name);
}

#[test]
fn test_conv2d_depthwise_3x3_s1() {
    // iw = 10 leaves a 2-wide scalar tail after the 8-wide vector block, so the
    // right-edge padding column is produced by the tail path.
    let groups = 4;
    let (n, ih, iw) = (1, 10, 10);
    let input: Vec<f32> = (0..n * groups * ih * iw).map(|i| i as f32 * 0.2 - 3.0).collect();
    let weight: Vec<f32> = (0..groups * 9).map(|i| i as f32 * 0.1 - 0.5).collect();
    assert_depthwise_matches_im2col(n, groups, ih, iw, &input, &weight, "dw_3x3_s1_crosscheck");
}

#[test]
fn test_conv2d_depthwise_3x3_s1_widths() {
    // Sweep widths so the last output column is produced by each of the vector
    // block boundary cases: no vector iterations, exact multiples of 8, and
    // assorted remainders.
    let groups = 3;
    let (n, ih) = (1, 5);
    for iw in [1usize, 2, 3, 7, 8, 9, 15, 16, 17, 24] {
        let input: Vec<f32> = (0..n * groups * ih * iw)
            .map(|i| (i % 53) as f32 * 0.11 - 2.0)
            .collect();
        let weight: Vec<f32> = (0..groups * 9).map(|i| (i % 17) as f32 * 0.07 - 0.4).collect();
        assert_depthwise_matches_im2col(
            n, groups, ih, iw, &input, &weight,
            &format!("dw_3x3_s1_iw{}", iw),
        );
    }
}

#[test]
fn test_conv2d_depthwise_3x3_s1_64ch() {
    let groups = 64;
    let (n, ih, iw) = (1, 8, 8);
    let (kh, kw) = (3, 3);

    let input: Vec<f32> = (0..n * groups * ih * iw).map(|i| (i % 97) as f32 * 0.07 - 1.0).collect();
    let weight: Vec<f32> = (0..groups * kh * kw).map(|i| (i % 31) as f32 * 0.05 - 0.3).collect();
    assert_depthwise_matches_im2col(n, groups, ih, iw, &input, &weight, "dw_3x3_s1_64ch");
}

#[test]
fn test_conv2d_depthwise_3x3_s1_128ch() {
    let groups = 128;
    let (n, ih, iw) = (1, 6, 6);
    let (kh, kw) = (3, 3);

    let input: Vec<f32> = (0..n * groups * ih * iw).map(|i| (i % 73) as f32 * 0.03).collect();
    let weight: Vec<f32> = (0..groups * kh * kw).map(|i| (i % 19) as f32 * 0.1 - 0.5).collect();
    assert_depthwise_matches_im2col(n, groups, ih, iw, &input, &weight, "dw_3x3_s1_128ch");
}

#[test]
fn test_conv2d_1x1() {
    let (n, ic, oc, ih, iw) = (1, 16, 8, 4, 4);
    let (kh, kw) = (1, 1);

    let input: Vec<f32> = (0..n * ic * ih * iw).map(|i| i as f32 * 0.1).collect();
    let weight: Vec<f32> = (0..oc * ic).map(|i| i as f32 * 0.01).collect();
    let bias: Vec<f32> = (0..oc).map(|i| i as f32 * 0.001).collect();

    let inp_t = TensorView::from_slice(&input, vec![n, ic, ih, iw]);
    let w_t = TensorView::from_slice(&weight, vec![oc, ic, 1, 1]);
    let b_t = TensorView::from_slice(&bias, vec![oc]);

    let mut out_buf = Vec::new();
    let result = conv2d_fused(&inp_t, &w_t, Some(&b_t), &[1, 1], 1, &[0, 0, 0, 0], &[1, 1], true, &mut out_buf);

    let expected = ref_conv2d(&input, &weight, Some(&bias), n, oc, ic, ih, iw, kh, kw, 1, 1, 0, 0, 0, 0, 1, true);
    assert_close(&result.data, &expected, 1e-3, "conv2d_1x1");
}

#[test]
fn test_conv2d_pointwise_after_depthwise() {
    let groups = 32;
    let (n, ih, iw) = (1, 4, 4);
    let pw_out = 64;
    let (kh, kw) = (1, 1);

    let input: Vec<f32> = (0..n * groups * ih * iw).map(|i| i as f32 * 0.05).collect();
    let pw_weight: Vec<f32> = (0..pw_out * groups).map(|i| i as f32 * 0.02 - 0.5).collect();
    let bias: Vec<f32> = (0..pw_out).map(|i| i as f32 * 0.001).collect();

    let inp_t = TensorView::from_slice(&input, vec![n, groups, ih, iw]);
    let w_t = TensorView::from_slice(&pw_weight, vec![pw_out, groups, 1, 1]);
    let b_t = TensorView::from_slice(&bias, vec![pw_out]);

    let mut out_buf = Vec::new();
    let result = conv2d_fused(&inp_t, &w_t, Some(&b_t), &[1, 1], 1, &[0, 0, 0, 0], &[1, 1], true, &mut out_buf);

    let expected = ref_conv2d(&input, &pw_weight, Some(&bias), n, pw_out, groups, ih, iw, kh, kw, 1, 1, 0, 0, 0, 0, 1, true);
    assert_close(&result.data, &expected, 1e-2, "pw_after_dw");
}

// ---------------------------------------------------------------------------
// MaxPool2d tests
// ---------------------------------------------------------------------------

fn ref_maxpool2d(
    input: &[f32], n: usize, c: usize, ih: usize, iw: usize,
    kh: usize, kw: usize, sy: usize, sx: usize,
    pt: usize, pl: usize, pb: usize, pr: usize,
) -> (Vec<f32>, usize, usize) {
    let oh = (ih + pt + pb - kh) / sy + 1;
    let ow = (iw + pl + pr - kw) / sx + 1;
    let mut out = vec![f32::NEG_INFINITY; n * c * oh * ow];
    for b in 0..n {
        for ch in 0..c {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut val = f32::NEG_INFINITY;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy = oy * sy + ky;
                            let ix = ox * sx + kx;
                            if iy >= pt && ix >= pl && iy < ih + pt && ix < iw + pl {
                                let ri = iy - pt;
                                let rx = ix - pl;
                                let v = input[b * c * ih * iw + ch * ih * iw + ri * iw + rx];
                                if v > val { val = v; }
                            }
                        }
                    }
                    out[b * c * oh * ow + ch * oh * ow + oy * ow + ox] = val;
                }
            }
        }
    }
    (out, oh, ow)
}

#[test]
fn test_maxpool2d_2x2_s2() {
    let (n, c, ih, iw) = (1, 3, 8, 8);
    let input: Vec<f32> = (0..n * c * ih * iw).map(|i| i as f32 * 0.1).collect();
    let inp_t = TensorView::from_slice(&input, vec![n, c, ih, iw]);

    let mut out_buf = Vec::new();
    let result = max_pool2d(&inp_t, &[2, 2], &[2, 2], &[0, 0, 0, 0], &[1, 1], false, &mut out_buf);

    let (expected, _, _) = ref_maxpool2d(&input, n, c, ih, iw, 2, 2, 2, 2, 0, 0, 0, 0);
    assert_close(&result.data, &expected, 1e-6, "maxpool_2x2_s2");
}

#[test]
fn test_maxpool2d_8x1_s8x1() {
    let (n, c, ih, iw) = (1, 4, 32, 10);
    let input: Vec<f32> = (0..n * c * ih * iw).map(|i| ((i * 7 + 3) % 100) as f32 * 0.1).collect();
    let inp_t = TensorView::from_slice(&input, vec![n, c, ih, iw]);

    let mut out_buf = Vec::new();
    let result = max_pool2d(&inp_t, &[8, 1], &[8, 1], &[0, 0, 0, 0], &[1, 1], false, &mut out_buf);

    let (expected, _, _) = ref_maxpool2d(&input, n, c, ih, iw, 8, 1, 8, 1, 0, 0, 0, 0);
    assert_close(&result.data, &expected, 1e-6, "maxpool_8x1_s8x1");
}

#[test]
fn test_maxpool2d_3x3_s1_pad1() {
    let (n, c, ih, iw) = (1, 2, 6, 6);
    let input: Vec<f32> = (0..n * c * ih * iw).map(|i| ((i * 13 + 7) % 50) as f32 * 0.2).collect();
    let inp_t = TensorView::from_slice(&input, vec![n, c, ih, iw]);

    let mut out_buf = Vec::new();
    let result = max_pool2d(&inp_t, &[3, 3], &[1, 1], &[1, 1, 1, 1], &[1, 1], false, &mut out_buf);

    let (expected, _, _) = ref_maxpool2d(&input, n, c, ih, iw, 3, 3, 1, 1, 1, 1, 1, 1);
    assert_close(&result.data, &expected, 1e-6, "maxpool_3x3_s1_p1");
}

#[test]
fn test_maxpool2d_2x2_s2_nonsquare() {
    let (n, c, ih, iw) = (1, 2, 32, 300);
    let input: Vec<f32> = (0..n * c * ih * iw).map(|i| ((i * 7 + 3) % 97) as f32 * 0.05).collect();
    let inp_t = TensorView::from_slice(&input, vec![n, c, ih, iw]);

    let mut out_buf = Vec::new();
    let result = max_pool2d(&inp_t, &[2, 2], &[2, 2], &[0, 0, 0, 0], &[1, 1], false, &mut out_buf);

    let (expected, _, _) = ref_maxpool2d(&input, n, c, ih, iw, 2, 2, 2, 2, 0, 0, 0, 0);
    assert_close(&result.data, &expected, 1e-6, "maxpool_nonsquare");
}

#[test]
fn test_maxpool2d_neg_values() {
    let input = vec![
        -10.0, -5.0, -3.0, -1.0,
        -8.0, -2.0, -6.0, -4.0,
        -7.0, -9.0, -11.0, -12.0,
        -13.0, -14.0, -15.0, -16.0,
    ];
    let inp_t = TensorView::from_slice(&input, vec![1, 1, 4, 4]);

    let mut out_buf = Vec::new();
    let result = max_pool2d(&inp_t, &[2, 2], &[2, 2], &[0, 0, 0, 0], &[1, 1], false, &mut out_buf);

    let (expected, _, _) = ref_maxpool2d(&input, 1, 1, 4, 4, 2, 2, 2, 2, 0, 0, 0, 0);
    assert_close(&result.data, &expected, 1e-6, "maxpool_neg");
}

// ---------------------------------------------------------------------------
// Pad (reflect) tests
// ---------------------------------------------------------------------------

#[test]
fn test_pad_reflect_symmetric() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let inp_t = TensorView::from_slice(&input, vec![1, 1, 2, 3]);

    let mut out_buf = Vec::new();
    let result = pad(&inp_t, &[0, 0, 1, 1, 0, 0, 1, 1], None, "reflect", &mut out_buf);

    assert_eq!(result.shape.as_ref(), &[1, 1, 4, 5]);
    let d = &result.data;
    // Verify it's not all zeros (reflect should produce non-zero at borders)
    let sum: f32 = d.iter().sum();
    assert!(sum > 0.0, "pad reflect should produce non-zero output");
    // Verify original data is preserved in the center
    assert_eq!(d[5 + 1], 1.0);
    assert_eq!(d[5 + 2], 2.0);
    assert_eq!(d[5 + 3], 3.0);
    assert_eq!(d[5 * 2 + 1], 4.0);
    assert_eq!(d[5 * 2 + 2], 5.0);
    assert_eq!(d[5 * 2 + 3], 6.0);
}

#[test]
fn test_pad_reflect_1d_horizontal() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let inp_t = TensorView::from_slice(&input, vec![1, 1, 1, 5]);

    // Pad 2 left, 2 right on the W dimension using ONNX pad format
    // For 4D tensor [N,C,H,W]: pads = [N_begin, C_begin, H_begin, W_begin, N_end, C_end, H_end, W_end]
    let mut out_buf = Vec::new();
    let result = pad(&inp_t, &[0, 0, 0, 2, 0, 0, 0, 2], None, "constant", &mut out_buf);

    // With constant pad (default 0), should get [0,0,1,2,3,4,5,0,0]
    assert_eq!(result.shape.as_ref(), &[1, 1, 1, 9]);
    let expected = vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0];
    assert_close(&result.data, &expected, 1e-6, "pad_const_1d");
}

#[test]
fn test_pad_constant_mode() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let inp_t = TensorView::from_slice(&input, vec![1, 1, 2, 2]);
    let cv = TensorView::from_slice(&[99.0f32], vec![1]);

    let mut out_buf = Vec::new();
    let result = pad(&inp_t, &[0, 0, 1, 1, 0, 0, 1, 1], Some(&cv), "constant", &mut out_buf);

    assert_eq!(result.shape.as_ref(), &[1, 1, 4, 4]);
    let expected = vec![
        99.0, 99.0, 99.0, 99.0,
        99.0, 1.0, 2.0, 99.0,
        99.0, 3.0, 4.0, 99.0,
        99.0, 99.0, 99.0, 99.0,
    ];
    assert_close(&result.data, &expected, 1e-6, "pad_constant");
}

// ---------------------------------------------------------------------------
// STFT and power spectrum tests
// ---------------------------------------------------------------------------

#[test]
fn test_stft_power_spectrum_dc_signal() {
    let signal = vec![1.0f32; 512];
    let inp_t = TensorView::from_slice(&signal, vec![1, 512]);

    let mut out_buf = Vec::new();
    let result = stft_power_spectrum(&inp_t, 256, 64, 256, None, &mut out_buf);

    let n_freqs = 129;
    let n_frames = result.shape[result.shape.len() - 2];
    for frame in 0..n_frames {
        let base = frame * n_freqs;
        let dc_power = result.data[base];
        assert!(dc_power > 1000.0, "DC power should be large, got {}", dc_power);
        for freq in 2..n_freqs {
            assert!(result.data[base + freq] < 1.0,
                "Non-DC bin {} in frame {} should be near 0, got {}",
                freq, frame, result.data[base + freq]);
        }
    }
}

#[test]
fn test_stft_vs_stft_power_consistency() {
    let signal: Vec<f32> = (0..800).map(|i| (i as f32 * 0.01).sin()).collect();
    let inp_t = TensorView::from_slice(&signal, vec![1, 800]);

    let mut stft_buf = Vec::new();
    let stft_result = stft(&inp_t, 256, 128, 256, None, &mut stft_buf);

    let mut power_buf = Vec::new();
    let power_result = stft_power_spectrum(&inp_t, 256, 128, 256, None, &mut power_buf);

    // Both should produce non-empty output
    assert!(!stft_result.data.is_empty());
    assert!(!power_result.data.is_empty());

    // stft output has 2x elements (re, im interleaved)
    let n_freqs = 129;
    let n_stft_frames = stft_result.data.len() / (n_freqs * 2);
    let n_power_frames = power_result.data.len() / n_freqs;
    assert_eq!(n_stft_frames, n_power_frames);

    // Verify power = re^2 + im^2 for a few samples
    for frame in 0..n_stft_frames.min(3) {
        for freq in 0..n_freqs {
            let stft_idx = (frame * n_freqs + freq) * 2;
            let re = stft_result.data[stft_idx];
            let im = stft_result.data[stft_idx + 1];
            let expected = re * re + im * im;
            let actual = power_result.data[frame * n_freqs + freq];
            assert!(
                (expected - actual).abs() < 1e-3,
                "mismatch at frame={} freq={}: expected={:.6}, got={:.6}",
                frame, freq, expected, actual,
            );
        }
    }
}

#[test]
fn test_stft_power_spectrum_known_sinusoid() {
    let n_fft = 256;
    let sr = 16000.0f32;
    let freq = 1000.0f32;
    let signal: Vec<f32> = (0..1600)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
        .collect();
    let inp_t = TensorView::from_slice(&signal, vec![1, 1600]);

    let mut out_buf = Vec::new();
    let result = stft_power_spectrum(&inp_t, n_fft, 160, n_fft, None, &mut out_buf);

    let n_freqs = n_fft / 2 + 1;
    let freq_bin = (freq / sr * n_fft as f32).round() as usize;
    assert!(freq_bin < n_freqs);

    let mid_frame = result.shape[result.shape.len() - 2] / 2;
    let base = mid_frame * n_freqs;
    let peak_power = result.data[base + freq_bin];
    let max_other: f32 = result.data[base..base + n_freqs]
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != freq_bin && (*i < 5 || *i > n_freqs - 5))
        .map(|(_, &v)| v)
        .fold(0.0f32, f32::max);

    assert!(
        peak_power > max_other * 5.0,
        "Sinusoid peak at bin {} should dominate: peak={:.2}, max_other={:.2}",
        freq_bin, peak_power, max_other,
    );
}

// ---------------------------------------------------------------------------
// FFT tests
// ---------------------------------------------------------------------------

#[test]
fn test_fft_precomputed_vs_scalar() {
    let n = 64;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();

    let mut scalar_re = vec![0.0f32; n / 2 + 1];
    let mut scalar_im = vec![0.0f32; n / 2 + 1];
    lele::kernels::fft::rfft_forward_f32(&input, &mut scalar_re, &mut scalar_im);

    let (tw_re, tw_im, bit_rev) = lele::kernels::fft::precompute_twiddles(n);
    let mut re_buf = vec![0.0f32; n];
    let mut im_buf = vec![0.0f32; n];
    let mut pre_re = vec![0.0f32; n / 2 + 1];
    let mut pre_im = vec![0.0f32; n / 2 + 1];
    lele::kernels::fft::rfft_forward_f32_precomputed(
        &input, &tw_re, &tw_im, &bit_rev, &mut re_buf, &mut im_buf, &mut pre_re, &mut pre_im,
    );

    assert_close(&scalar_re, &pre_re, 1e-5, "fft_re");
    assert_close(&scalar_im, &pre_im, 1e-5, "fft_im");
}

#[test]
fn test_fft_parseval_theorem() {
    let n = 128;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin() + (i as f32 * 0.05).cos()).collect();

    let (tw_re, tw_im, bit_rev) = lele::kernels::fft::precompute_twiddles(n);
    let mut re_buf = vec![0.0; n]; let mut im_buf = vec![0.0; n];
    let mut freq_re = vec![0.0; n/2+1]; let mut freq_im = vec![0.0; n/2+1];
    lele::kernels::fft::rfft_forward_f32_precomputed(
        &input, &tw_re, &tw_im, &bit_rev, &mut re_buf, &mut im_buf, &mut freq_re, &mut freq_im,
    );

    let time_energy: f32 = input.iter().map(|x| x * x).sum();
    let half = n / 2 + 1;
    let freq_energy: f32 = (0..half).map(|i| {
        let p = freq_re[i] * freq_re[i] + freq_im[i] * freq_im[i];
        if i == 0 || i == half - 1 { p } else { 2.0 * p }
    }).sum::<f32>() / n as f32;

    assert!(
        (time_energy - freq_energy).abs() / time_energy < 1e-4,
        "Parseval: time={:.4}, freq={:.4}", time_energy, freq_energy,
    );
}

#[test]
fn test_fft_linearity() {
    let n = 32;
    let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5).sin()).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).cos()).collect();
    let scale = 2.5f32;
    let ab: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x + scale * y).collect();

    let (tw_re, tw_im, bit_rev) = lele::kernels::fft::precompute_twiddles(n);
    let mut rb = vec![0.0; n]; let mut ib = vec![0.0; n];
    let mut fr = vec![0.0; n/2+1]; let mut fi = vec![0.0; n/2+1];

    lele::kernels::fft::rfft_forward_f32_precomputed(&ab, &tw_re, &tw_im, &bit_rev, &mut rb, &mut ib, &mut fr, &mut fi);
    let ab_re = fr.clone(); let ab_im = fi.clone();

    lele::kernels::fft::rfft_forward_f32_precomputed(&a, &tw_re, &tw_im, &bit_rev, &mut rb, &mut ib, &mut fr, &mut fi);
    let a_re = fr.clone(); let a_im = fi.clone();

    lele::kernels::fft::rfft_forward_f32_precomputed(&b, &tw_re, &tw_im, &bit_rev, &mut rb, &mut ib, &mut fr, &mut fi);
    let b_re = fr.clone(); let b_im = fi.clone();

    let half = n / 2 + 1;
    for i in 0..half {
        assert!((ab_re[i] - (a_re[i] + scale * b_re[i])).abs() < 1e-4, "FFT linearity re[{}]", i);
        assert!((ab_im[i] - (a_im[i] + scale * b_im[i])).abs() < 1e-4, "FFT linearity im[{}]", i);
    }
}

// ---------------------------------------------------------------------------
// GRU tests
// ---------------------------------------------------------------------------

fn ref_gru_step(
    x_t: &[f32], w: &[f32], r: &[f32], bias_w: &[f32], bias_r: &[f32],
    h: &mut [f32], hs: usize, linear_before_reset: bool,
) {
    let m = 3 * hs;
    let is = x_t.len();
    let mut wc = vec![0.0f32; m];
    let mut rc = vec![0.0f32; m];
    for g in 0..m {
        for k in 0..is { wc[g] += w[g * is + k] * x_t[k]; }
        for k in 0..hs { rc[g] += r[g * hs + k] * h[k]; }
    }

    if linear_before_reset {
        let z_g: Vec<f32> = (0..hs).map(|k| lele::kernels::activations::sigmoid(wc[k] + rc[k] + bias_w[k] + bias_r[k])).collect();
        let r_g: Vec<f32> = (0..hs).map(|k| lele::kernels::activations::sigmoid(wc[hs+k] + rc[hs+k] + bias_w[hs+k] + bias_r[hs+k])).collect();
        for k in 0..hs {
            let h_pre = wc[2*hs+k] + bias_w[2*hs+k] + r_g[k] * (rc[2*hs+k] + bias_r[2*hs+k]);
            let h_gate = lele::kernels::activations::tanh(h_pre);
            h[k] = (1.0 - z_g[k]) * h_gate + z_g[k] * h[k];
        }
    } else {
        let z_g: Vec<f32> = (0..hs).map(|k| lele::kernels::activations::sigmoid(wc[k] + rc[k] + bias_w[k] + bias_r[k])).collect();
        let r_g: Vec<f32> = (0..hs).map(|k| lele::kernels::activations::sigmoid(wc[hs+k] + rc[hs+k] + bias_w[hs+k] + bias_r[hs+k])).collect();
        for k in 0..hs {
            let wh_x = wc[2*hs+k] + bias_w[2*hs+k];
            let r_rh = r_g[k] * (rc[2*hs+k] + bias_r[2*hs+k]);
            let h_gate = lele::kernels::activations::tanh(wh_x + r_rh);
            h[k] = (1.0 - z_g[k]) * h_gate + z_g[k] * h[k];
        }
    }
}

#[test]
fn test_gru_single_step() {
    let (sl, bs, is, hs) = (1, 1, 4, 8);
    let input: Vec<f32> = vec![0.1, 0.2, -0.1, 0.3];
    let w: Vec<f32> = (0..3*hs*is).map(|i| i as f32 * 0.01 - 0.1).collect();
    let r: Vec<f32> = (0..3*hs*hs).map(|i| i as f32 * 0.02 - 0.2).collect();
    let bias: Vec<f32> = (0..6*hs).map(|i| i as f32 * 0.005 - 0.05).collect();

    let inp_t = TensorView::from_slice(&input, vec![sl, bs, is]);
    let w_t = TensorView::from_slice(&w, vec![1, 3*hs, is]);
    let r_t = TensorView::from_slice(&r, vec![1, 3*hs, hs]);
    let b_t = TensorView::from_slice(&bias, vec![1, 6*hs]);

    let mut oy = Vec::new(); let mut oh = Vec::new();
    let (yr, hr) = gru(&inp_t, &w_t, &r_t, Some(&b_t), None, false, &mut oy, &mut oh);

    let mut href = vec![0.0f32; hs];
    let (bw, br) = bias.split_at(3*hs);
    ref_gru_step(&input, &w, &r, bw, br, &mut href, hs, false);

    assert_close(&yr.data, &href, 1e-4, "gru_s_y");
    assert_close(&hr.data, &href, 1e-4, "gru_s_h");
}

#[test]
fn test_gru_multi_step() {
    let (sl, bs, is, hs) = (5, 1, 3, 6);
    let input: Vec<f32> = (0..sl*is).map(|i| ((i*7+3)%20) as f32 * 0.1 - 0.5).collect();
    let w: Vec<f32> = (0..3*hs*is).map(|i| i as f32 * 0.03 - 0.2).collect();
    let r: Vec<f32> = (0..3*hs*hs).map(|i| i as f32 * 0.01 - 0.1).collect();
    let bias: Vec<f32> = vec![0.1; 6*hs];

    let inp_t = TensorView::from_slice(&input, vec![sl, bs, is]);
    let w_t = TensorView::from_slice(&w, vec![1, 3*hs, is]);
    let r_t = TensorView::from_slice(&r, vec![1, 3*hs, hs]);
    let b_t = TensorView::from_slice(&bias, vec![1, 6*hs]);

    let mut oy = Vec::new(); let mut oh = Vec::new();
    let (yr, hr) = gru(&inp_t, &w_t, &r_t, Some(&b_t), None, false, &mut oy, &mut oh);

    let mut href = vec![0.0f32; hs];
    let mut yref = vec![0.0f32; sl*hs];
    let (bw, br) = bias.split_at(3*hs);
    for t in 0..sl {
        ref_gru_step(&input[t*is..(t+1)*is], &w, &r, bw, br, &mut href, hs, false);
        yref[t*hs..(t+1)*hs].copy_from_slice(&href);
    }
    assert_close(&yr.data, &yref, 1e-3, "gru_m_y");
    assert_close(&hr.data, &href, 1e-3, "gru_m_h");
}

#[test]
fn test_gru_linear_before_reset() {
    let (sl, bs, is, hs) = (3, 1, 4, 8);
    let input: Vec<f32> = (0..sl*is).map(|i| i as f32 * 0.15 - 0.3).collect();
    let w: Vec<f32> = (0..3*hs*is).map(|i| i as f32 * 0.01).collect();
    let r: Vec<f32> = (0..3*hs*hs).map(|i| i as f32 * 0.02 - 0.1).collect();
    let bias: Vec<f32> = vec![0.05; 6*hs];

    let inp_t = TensorView::from_slice(&input, vec![sl, bs, is]);
    let w_t = TensorView::from_slice(&w, vec![1, 3*hs, is]);
    let r_t = TensorView::from_slice(&r, vec![1, 3*hs, hs]);
    let b_t = TensorView::from_slice(&bias, vec![1, 6*hs]);

    let mut oy = Vec::new(); let mut oh = Vec::new();
    let (yr, hr) = gru(&inp_t, &w_t, &r_t, Some(&b_t), None, true, &mut oy, &mut oh);

    let mut href = vec![0.0f32; hs];
    let mut yref = vec![0.0f32; sl*hs];
    let (bw, br) = bias.split_at(3*hs);
    for t in 0..sl {
        ref_gru_step(&input[t*is..(t+1)*is], &w, &r, bw, br, &mut href, hs, true);
        yref[t*hs..(t+1)*hs].copy_from_slice(&href);
    }
    assert_close(&yr.data, &yref, 1e-3, "gru_lbr_y");
    assert_close(&hr.data, &href, 1e-3, "gru_lbr_h");
}

#[test]
fn test_gru_no_bias() {
    let (sl, bs, is, hs) = (2, 1, 3, 4);
    let input: Vec<f32> = vec![0.5, -0.3, 0.1, -0.2, 0.4, 0.6];
    let w: Vec<f32> = (0..3*hs*is).map(|i| i as f32 * 0.05).collect();
    let r: Vec<f32> = (0..3*hs*hs).map(|i| i as f32 * 0.03).collect();

    let inp_t = TensorView::from_slice(&input, vec![sl, bs, is]);
    let w_t = TensorView::from_slice(&w, vec![1, 3*hs, is]);
    let r_t = TensorView::from_slice(&r, vec![1, 3*hs, hs]);

    let mut oy = Vec::new(); let mut oh = Vec::new();
    let (yr, hr) = gru(&inp_t, &w_t, &r_t, None, None, false, &mut oy, &mut oh);

    let mut href = vec![0.0f32; hs];
    let mut yref = vec![0.0f32; sl*hs];
    let zw = vec![0.0f32; 3*hs];
    let zr = vec![0.0f32; 3*hs];
    for t in 0..sl {
        ref_gru_step(&input[t*is..(t+1)*is], &w, &r, &zw, &zr, &mut href, hs, false);
        yref[t*hs..(t+1)*hs].copy_from_slice(&href);
    }
    assert_close(&yr.data, &yref, 1e-4, "gru_nb_y");
    assert_close(&hr.data, &href, 1e-4, "gru_nb_h");
}

// ---------------------------------------------------------------------------
// Element-wise math tests
// ---------------------------------------------------------------------------

#[test]
fn test_sub() {
    let a = vec![5.0, 3.0, 1.0, -2.0];
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let mut out = Vec::new();
    let r = sub(&TensorView::from_slice(&a, vec![4]), &TensorView::from_slice(&b, vec![4]), &mut out);
    assert_close(&r.data, &[4.0, 1.0, -2.0, -6.0], 1e-6, "sub");
}

#[test]
fn test_div() {
    let a = vec![10.0, 6.0, 0.0, -8.0];
    let b = vec![2.0, 3.0, 5.0, 4.0];
    let mut out = Vec::new();
    let r = div(&TensorView::from_slice(&a, vec![4]), &TensorView::from_slice(&b, vec![4]), &mut out);
    assert_close(&r.data, &[5.0, 2.0, 0.0, -2.0], 1e-6, "div");
}

#[test]
fn test_clip_both() {
    let x = vec![-5.0, -1.0, 0.0, 0.5, 3.0, 10.0];
    let min_v = TensorView::from_slice(&[-1.0f32], vec![]);
    let max_v = TensorView::from_slice(&[3.0f32], vec![]);
    let mut out = Vec::new();
    let r = clip(&TensorView::from_slice(&x, vec![6]), Some(&min_v), Some(&max_v), &mut out);
    assert_close(&r.data, &[-1.0, -1.0, 0.0, 0.5, 3.0, 3.0], 1e-6, "clip");
}

#[test]
fn test_sqrt() {
    let x: Vec<f32> = (0..=20).map(|i| i as f32 * 0.25).collect();
    let mut out = Vec::new();
    let r = sqrt(&TensorView::from_slice(&x, vec![x.len()]), &mut out);
    let exp: Vec<f32> = x.iter().map(|&v| v.sqrt()).collect();
    assert_close(&r.data, &exp, 1e-5, "sqrt");
}

#[test]
fn test_log() {
    let x: Vec<f32> = (1..=20).map(|i| i as f32 * 0.5).collect();
    let mut out = Vec::new();
    let r = log(&TensorView::from_slice(&x, vec![x.len()]), &mut out);
    let exp: Vec<f32> = x.iter().map(|&v| v.ln()).collect();
    assert_close(&r.data, &exp, 1e-5, "log");
}

#[test]
fn test_exp() {
    let x: Vec<f32> = (-10..=10).map(|i| i as f32 * 0.5).collect();
    let mut out = Vec::new();
    let r = exp(&TensorView::from_slice(&x, vec![x.len()]), &mut out);
    let exp: Vec<f32> = x.iter().map(|&v| v.exp()).collect();
    assert_close(&r.data, &exp, 1e-4, "exp");
}

#[test]
fn test_pow() {
    let a: Vec<f32> = (0..=10).map(|i| i as f32 * 0.5).collect();
    let b = vec![2.0f32; 11];
    let mut out = Vec::new();
    let r = pow(&TensorView::from_slice(&a, vec![11]), &TensorView::from_slice(&b, vec![11]), &mut out);
    let exp: Vec<f32> = a.iter().map(|&v| v.powf(2.0)).collect();
    assert_close(&r.data, &exp, 1e-4, "pow");
}

#[test]
fn test_reduce_sum() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut out = Vec::new();
    let r = reduce_sum(&TensorView::from_slice(&x, vec![2, 3]), &[1], false, &mut out);
    assert_eq!(r.data.as_ref(), &[6.0, 15.0]);
}

#[test]
fn test_reduce_sum_keepdims() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut out = Vec::new();
    let r = reduce_sum(&TensorView::from_slice(&x, vec![2, 3]), &[1], true, &mut out);
    assert_eq!(r.shape.as_ref(), &[2, 1]);
    assert_eq!(r.data.as_ref(), &[6.0, 15.0]);
}

#[test]
fn test_reduce_mean() {
    let x = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
    let mut out = Vec::new();
    let r = reduce_mean(&TensorView::from_slice(&x, vec![2, 3]), &[1], false, &mut out);
    assert_close(&r.data, &[4.0, 10.0], 1e-6, "reduce_mean");
}

#[test]
fn test_reduce_max() {
    let x = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0];
    let mut out = Vec::new();
    let r = reduce_max(&TensorView::from_slice(&x, vec![2, 3]), &[1], false, &mut out);
    assert_close(&r.data, &[4.0, 9.0], 1e-6, "reduce_max");
}

#[test]
fn test_reduce_l2() {
    let x = vec![3.0, 4.0];
    let mut out = Vec::new();
    let r = reduce_l2(&TensorView::from_slice(&x, vec![2]), &[0], false, &mut out);
    assert_close(&r.data, &[5.0], 1e-4, "reduce_l2");
}

#[test]
fn test_tanh() {
    let x: Vec<f32> = (-5..=5).map(|i| i as f32).collect();
    let mut out = Vec::new();
    let r = tanh_kernel(&TensorView::from_slice(&x, vec![x.len()]), &mut out);
    let exp: Vec<f32> = x.iter().map(|&v| v.tanh()).collect();
    assert_close(&r.data, &exp, 1e-5, "tanh");
}

#[test]
fn test_neg() {
    let x = vec![1.0, -2.0, 0.0, 3.5];
    let mut out = Vec::new();
    let r = neg(&TensorView::from_slice(&x, vec![4]), &mut out);
    assert_close(&r.data, &[-1.0, 2.0, 0.0, -3.5], 1e-6, "neg");
}

#[test]
fn test_sigmoid() {
    let x: Vec<f32> = (-10..=10).map(|i| i as f32 * 0.5).collect();
    let mut out = Vec::new();
    let r = sigmoid(&TensorView::from_slice(&x, vec![x.len()]), &mut out);
    let exp: Vec<f32> = x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
    assert_close(&r.data, &exp, 1e-5, "sigmoid");
}

#[test]
fn test_gelu() {
    let x: Vec<f32> = (-5..=5).map(|i| i as f32 * 0.5).collect();
    let mut out = Vec::new();
    let r = gelu(&TensorView::from_slice(&x, vec![x.len()]), &mut out);
    let exp: Vec<f32> = x.iter().map(|&v| v * 0.5 * (1.0 + libm::erff(v / std::f32::consts::SQRT_2))).collect();
    assert_close(&r.data, &exp, 1e-5, "gelu");
}

#[test]
fn test_reciprocal() {
    let x = vec![1.0, 2.0, 4.0, -2.0, 0.5];
    let mut out = Vec::new();
    let r = reciprocal(&TensorView::from_slice(&x, vec![5]), &mut out);
    let exp: Vec<f32> = x.iter().map(|&v| 1.0 / v).collect();
    assert_close(&r.data, &exp, 1e-5, "reciprocal");
}

// ---------------------------------------------------------------------------
// Gemm tests
// ---------------------------------------------------------------------------

#[test]
fn test_gemm_trans_b() {
    let a = TensorView::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = TensorView::from_slice(&[1.0, 3.0, 2.0, 4.0], vec![2, 2]);
    let mut out = Vec::new();
    let r = gemm(&a, &b, None, 1.0, 1.0, false, true, &mut out);
    // A=[[1,2],[3,4]], B^T=[[1,2],[3,4]], A*B^T=[[7,10],[15,22]]
    assert_close(&r.data, &[7.0, 10.0, 15.0, 22.0], 1e-5, "gemm_transB");
}

#[test]
fn test_gemm_with_bias() {
    let a = TensorView::from_slice(&[1.0, 2.0], vec![1, 2]);
    let b = TensorView::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let bias = TensorView::from_slice(&[0.5, -0.5], vec![2]);
    let mut out = Vec::new();
    let r = gemm(&a, &b, Some(&bias), 1.0, 1.0, false, false, &mut out);
    // A[1,2]*B[2,2] = [1*1+2*3, 1*2+2*4] = [7, 10] + [0.5, -0.5] = [7.5, 9.5]
    assert_close(&r.data, &[7.5, 9.5], 1e-5, "gemm_bias");
}

#[test]
fn test_matmul_fused_add() {
    let a = TensorView::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = TensorView::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let bias = TensorView::from_slice(&[0.1, 0.2], vec![2]);
    let mut out = Vec::new();
    let r = matmul_fused_add(&a, &b, &bias, &mut out);
    // [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]] + [[0.1,0.2]]
    assert_close(&r.data, &[22.1, 28.2, 49.1, 64.2], 1e-4, "mm_fused");
}

// ---------------------------------------------------------------------------
// Shape and manipulation tests
// ---------------------------------------------------------------------------

#[test]
fn test_reshape_neg() {
    let t = TensorView::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let r = reshape(&t, &[-1, 2]);
    assert_eq!(r.shape.as_ref(), &[3, 2]);
}

#[test]
fn test_reshape_two_neg() {
    let t = TensorView::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![2, 2, 3]);
    let r = reshape(&t, &[2, -1, 2]);
    assert_eq!(r.shape.as_ref(), &[2, 3, 2]);
}

#[test]
fn test_transpose_3d() {
    let t = TensorView::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3]);
    let mut out = Vec::new();
    let r = transpose(&t, &[0, 2, 1], &mut out);
    assert_eq!(r.shape.as_ref(), &[1, 3, 2]);
    assert_eq!(r.data.as_ref(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_transpose_4d() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = TensorView::from_slice(&data, vec![1, 2, 3, 4]);
    let mut out = Vec::new();
    let r = transpose(&t, &[0, 3, 1, 2], &mut out);
    assert_eq!(r.shape.as_ref(), &[1, 4, 2, 3]);
}

#[test]
fn test_add_scalar_bc() {
    let mut out = Vec::new();
    let r = add(&TensorView::from_slice(&[1.0, 2.0, 3.0], vec![3]), &TensorView::from_slice(&[10.0], vec![1]), &mut out);
    assert_close(&r.data, &[11.0, 12.0, 13.0], 1e-6, "add_bc");
}

// ---------------------------------------------------------------------------
// LSTM test
// ---------------------------------------------------------------------------

#[test]
fn test_lstm_single_step() {
    let (sl, bs, is, hs) = (1, 1, 3, 4);
    let input: Vec<f32> = vec![0.1, -0.2, 0.3];
    let w: Vec<f32> = (0..4*hs*is).map(|i| i as f32 * 0.01).collect();
    let r: Vec<f32> = (0..4*hs*hs).map(|i| i as f32 * 0.02 - 0.1).collect();
    let bias: Vec<f32> = (0..8*hs).map(|i| i as f32 * 0.005).collect();

    let inp_t = TensorView::from_slice(&input, vec![sl, bs, is]);
    let w_t = TensorView::from_slice(&w, vec![1, 4*hs, is]);
    let r_t = TensorView::from_slice(&r, vec![1, 4*hs, hs]);
    let b_t = TensorView::from_slice(&bias, vec![1, 8*hs]);

    let mut oy = Vec::new(); let mut oh = Vec::new(); let mut oc = Vec::new();
    let (yr, hr, cr) = lstm(&inp_t, &w_t, &r_t, Some(&b_t), None, None, None, &mut oy, &mut oh, &mut oc);

    assert_eq!(yr.shape.as_ref(), &[sl, bs, 1, hs]);
    assert_eq!(hr.shape.as_ref(), &[1, bs, hs]);
    assert_eq!(cr.shape.as_ref(), &[1, bs, hs]);
    for v in yr.data.iter() { assert!(v.is_finite()); }
    for v in cr.data.iter() { assert!(v.is_finite()); }
}

// --- Conv1d ---------------------------------------------------------------

/// Naive NCW convolution used as ground truth for the SIMD conv1d paths.
fn ref_conv1d(
    input: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    batch: usize,
    in_ch: usize,
    in_len: usize,
    out_ch: usize,
    k: usize,
    stride: usize,
    pad: usize,
) -> (Vec<f32>, usize) {
    let out_len = (in_len + 2 * pad - k) / stride + 1;
    let mut out = vec![0.0f32; batch * out_ch * out_len];
    for b in 0..batch {
        for oc in 0..out_ch {
            for t in 0..out_len {
                let mut acc = bias.map(|v| v[oc]).unwrap_or(0.0);
                for ic in 0..in_ch {
                    for kk in 0..k {
                        let idx = (t * stride) as isize + kk as isize - pad as isize;
                        if idx >= 0 && (idx as usize) < in_len {
                            acc += input[(b * in_ch + ic) * in_len + idx as usize]
                                * weights[(oc * in_ch + ic) * k + kk];
                        }
                    }
                }
                out[(b * out_ch + oc) * out_len + t] = acc;
            }
        }
    }
    (out, out_len)
}

#[test]
fn test_conv1d_k3_pad1_matches_reference() {
    // Covers the AVX2/NEON k=3 pad=1 direct paths: the channel count is not a
    // multiple of 4 and the length is not a multiple of 8, so both the vector
    // block and the scalar cleanup loops run.
    let (batch, in_ch, in_len, out_ch, k, pad) = (2, 3, 37, 6, 3, 1);
    let input: Vec<f32> = (0..batch * in_ch * in_len)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
        .collect();
    let weights: Vec<f32> = (0..out_ch * in_ch * k)
        .map(|i| ((i * 17 % 29) as f32 - 14.0) / 13.0)
        .collect();
    let bias: Vec<f32> = (0..out_ch).map(|i| i as f32 * 0.1 - 0.25).collect();

    let input_t = TensorView::from_slice(&input, vec![batch, in_ch, in_len]);
    let w_t = TensorView::from_slice(&weights, vec![out_ch, in_ch, k]);
    let b_t = TensorView::from_slice(&bias, vec![out_ch]);

    for stride in [1usize, 2] {
        let (expected, out_len) = ref_conv1d(
            &input, &weights, Some(&bias), batch, in_ch, in_len, out_ch, k, stride, pad,
        );
        let mut buf = Vec::new();
        let got = conv1d(
            &input_t,
            &w_t,
            Some(&b_t),
            &[1],
            1,
            &[pad as i64, pad as i64],
            &[stride as i64],
            &mut buf,
        );
        assert_eq!(got.shape.as_ref(), &[batch, out_ch, out_len]);
        assert_close(&got.data, &expected, 1e-5, &format!("conv1d k3 pad1 s{}", stride));
    }
}

// --- QuantizeLinear / DequantizeLinear ------------------------------------

#[test]
fn test_quantize_linear_per_tensor_rounds_half_even_and_saturates() {
    let x = vec![-1.0, 0.0, 0.25, 0.75, 100.0, -100.0];
    let scale = vec![0.5f32];
    let zp = vec![128.0f32];
    let x_t = TensorView::from_slice(&x, vec![6]);
    let s_t = TensorView::from_slice(&scale, vec![]);
    let z_t = TensorView::from_slice(&zp, vec![]);

    let mut buf = Vec::new();
    let y = quantize_linear(&x_t, &s_t, Some(&z_t), 1, 0, 0.0, 255.0, &mut buf);
    // 0.25/0.5 = 0.5 -> 0 (ties to even), 0.75/0.5 = 1.5 -> 2 (ties to even).
    assert_eq!(y.data.as_ref(), &[126.0, 128.0, 128.0, 130.0, 255.0, 0.0]);
}

#[test]
fn test_dequantize_linear_per_tensor_inverts_quantize() {
    let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.03).collect();
    let scale = vec![0.01f32];
    let zp = vec![-5.0f32];
    let x_t = TensorView::from_slice(&x, vec![8, 8]);
    let s_t = TensorView::from_slice(&scale, vec![1]);
    let z_t = TensorView::from_slice(&zp, vec![1]);

    let mut qbuf = Vec::new();
    let q = quantize_linear(&x_t, &s_t, Some(&z_t), 1, 0, -128.0, 127.0, &mut qbuf);
    let q_owned = q.to_owned();
    let mut dbuf = Vec::new();
    let d = dequantize_linear(&q_owned, &s_t, Some(&z_t), 1, 0, &mut dbuf);

    assert_eq!(d.shape.as_ref(), &[8, 8]);
    // Round-trip error is bounded by half a quantization step.
    assert_close(&d.data, &x, 0.005, "qdq round trip");
}

#[test]
fn test_dequantize_linear_per_axis() {
    // shape [2, 3, 2], per-channel scales along axis 1.
    let x: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let scale = vec![0.5f32, 2.0, 10.0];
    let zp = vec![1.0f32, 0.0, -2.0];
    let x_t = TensorView::from_slice(&x, vec![2, 3, 2]);
    let s_t = TensorView::from_slice(&scale, vec![3]);
    let z_t = TensorView::from_slice(&zp, vec![3]);

    let mut buf = Vec::new();
    let y = dequantize_linear(&x_t, &s_t, Some(&z_t), 1, 0, &mut buf);

    let expected: Vec<f32> = (0..12)
        .map(|i| {
            let c = (i / 2) % 3;
            (i as f32 - zp[c]) * scale[c]
        })
        .collect();
    assert_close(&y.data, &expected, 1e-6, "dequantize per-axis");
}

#[test]
fn test_dequantize_linear_without_zero_point_defaults_to_zero() {
    let x = vec![1.0f32, 2.0, 3.0, 4.0];
    let scale = vec![0.25f32];
    let x_t = TensorView::from_slice(&x, vec![4]);
    let s_t = TensorView::from_slice(&scale, vec![]);

    let mut buf = Vec::new();
    let y = dequantize_linear(&x_t, &s_t, None, 1, 0, &mut buf);
    assert_eq!(y.data.as_ref(), &[0.25, 0.5, 0.75, 1.0]);
}

#[test]
fn test_dequantize_linear_blocked() {
    // shape [4, 2], block_size 2 along axis 0 -> scale shape [2, 2].
    let x: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let scale = vec![1.0f32, 2.0, 10.0, 20.0];
    let x_t = TensorView::from_slice(&x, vec![4, 2]);
    let s_t = TensorView::from_slice(&scale, vec![2, 2]);

    let mut buf = Vec::new();
    let y = dequantize_linear(&x_t, &s_t, None, 0, 2, &mut buf);
    assert_eq!(y.data.as_ref(), &[0.0, 2.0, 2.0, 6.0, 40.0, 100.0, 60.0, 140.0]);
}

#[test]
fn test_quant_range_known_types() {
    assert_eq!(quant_range(2), Some((0.0, 255.0)));
    assert_eq!(quant_range(3), Some((-128.0, 127.0)));
    assert_eq!(quant_range(1), None);
}

/// The compiler collapses `QuantizeLinear` -> `DequantizeLinear` round trips
/// into `fake_quantize_linear`, so the fused kernel must be bit-identical to
/// running the two separately for every parameter layout.
#[test]
fn test_fake_quantize_linear_matches_unfused_round_trip() {
    // Values chosen to straddle the saturation bounds and land on rounding ties.
    let x: Vec<f32> = (0..24).map(|i| (i as f32 - 11.5) * 0.75).collect();
    let x_t = TensorView::from_slice(&x, vec![4, 3, 2]);

    let per_tensor = vec![0.5f32];
    let per_axis = vec![0.25f32, 0.5, 1.0];
    // shape [4, 3, 2] with block_size 2 along axis 0 -> 2 blocks of 6 inner
    // elements, so the scale has shape [2, 3, 2].
    let blocked: Vec<f32> = (0..12).map(|i| 0.25 * (i + 1) as f32).collect();
    let zp_tensor = vec![128.0f32];
    let zp_axis = vec![120.0f32, 128.0, 140.0];

    let cases: [(&str, &[f32], Vec<usize>, Option<&[f32]>, i64, usize); 5] = [
        ("per-tensor", &per_tensor, vec![], Some(&zp_tensor), 1, 0),
        ("per-tensor, no zero point", &per_tensor, vec![], None, 1, 0),
        ("per-axis", &per_axis, vec![3], Some(&zp_axis), 1, 0),
        ("per-axis, no zero point", &per_axis, vec![3], None, 1, 0),
        ("blocked", &blocked, vec![2, 3, 2], None, 0, 2),
    ];

    for (name, scale, scale_shape, zp, axis, block_size) in cases {
        let scale_shape = if scale_shape.is_empty() {
            vec![scale.len()]
        } else {
            scale_shape
        };
        let s_t = TensorView::from_slice(scale, scale_shape);
        let zp_t = zp.map(|z| TensorView::from_slice(z, vec![z.len()]));

        let mut q_buf = Vec::new();
        let mut dq_buf = Vec::new();
        let q = quantize_linear(
            &x_t,
            &s_t,
            zp_t.as_ref(),
            axis,
            block_size,
            0.0,
            255.0,
            &mut q_buf,
        );
        let unfused = dequantize_linear(&q, &s_t, zp_t.as_ref(), axis, block_size, &mut dq_buf);
        let expected = unfused.data.to_vec();

        let mut fused_buf = Vec::new();
        let fused = fake_quantize_linear(
            &x_t,
            &s_t,
            zp_t.as_ref(),
            axis,
            block_size,
            0.0,
            255.0,
            &mut fused_buf,
        );

        assert_eq!(fused.shape, x_t.shape, "{name}: shape");
        assert_eq!(fused.data.as_ref(), expected.as_slice(), "{name}");
    }
}

#[test]
fn test_fake_quantize_linear_saturates() {
    // Inputs far outside the uint8 grid clamp to the ends of the range, which
    // dequantize back to (qmin - zp) * scale and (qmax - zp) * scale.
    let x = vec![-1e6f32, 1e6, 0.0];
    let scale = vec![0.5f32];
    let zp = vec![128.0f32];
    let x_t = TensorView::from_slice(&x, vec![3]);
    let s_t = TensorView::from_slice(&scale, vec![]);
    let z_t = TensorView::from_slice(&zp, vec![]);

    let mut buf = Vec::new();
    let y = fake_quantize_linear(&x_t, &s_t, Some(&z_t), 1, 0, 0.0, 255.0, &mut buf);
    assert_eq!(y.data.as_ref(), &[-64.0, 63.5, 0.0]);
}

/// The compiler fuses `Div -> Erf -> Add -> Mul -> Mul` into `gelu_erf`, so the
/// fused kernel has to reproduce that chain bit for bit. `gelu` folds the
/// division into a reciprocal multiply and reassociates, so it does not.
#[test]
fn test_gelu_erf_matches_unfused_chain() {
    // Spans the saturating tails of erf as well as the linear region, and
    // deliberately has a length that is not a multiple of the 32- or 8-wide
    // blocks so the scalar tail runs too.
    let n = 1000 + 7;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) * 0.01).collect();
    let x_t = TensorView::from_slice(&x, vec![n]);

    let sqrt2 = [std::f32::consts::SQRT_2];
    let one = [1.0f32];
    let half = [0.5f32];
    let sqrt2_t = TensorView::from_slice(&sqrt2, vec![]);
    let one_t = TensorView::from_slice(&one, vec![]);
    let half_t = TensorView::from_slice(&half, vec![]);

    let (mut b0, mut b1, mut b2, mut b3, mut b4) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let scaled = div(&x_t, &sqrt2_t, &mut b0);
    let e = erf(&scaled, &mut b1);
    let shifted = add(&e, &one_t, &mut b2);
    let halved = mul(&half_t, &shifted, &mut b3);
    let expected = mul(&x_t, &halved, &mut b4).data.to_vec();

    let mut fused_buf = Vec::new();
    let fused = gelu_erf(&x_t, &mut fused_buf);

    assert_eq!(fused.shape.as_ref(), &[n]);
    for (i, (&got, &want)) in fused.data.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "gelu_erf differs at index {i} (x = {}): got {got}, want {want}",
            x[i]
        );
    }
}

/// Builds a static-QDQ layer: an activation already snapped onto its
/// quantization grid, and an int8 weight with a per-output-channel scale.
fn qdq_layer(
    m: usize,
    k: usize,
    n: usize,
) -> (Vec<f32>, f32, f32, Vec<u8>, Vec<f32>, Vec<f32>) {
    let a_scale = 0.0037f32;
    let a_zp = 131f32;
    // Activation codes are uint8, so the dequantized value is exactly
    // scale * (code - zero_point).
    let a: Vec<f32> = (0..m * k)
        .map(|i| a_scale * ((i * 37 % 256) as f32 - a_zp))
        .collect();
    let w_i8: Vec<i8> = (0..k * n)
        .map(|i| ((i * 53 % 255) as i32 - 127) as i8)
        .collect();
    let raw: Vec<u8> = w_i8.iter().map(|&v| v as u8).collect();
    let w_scale: Vec<f32> = (0..n).map(|j| 0.001 + (j % 7) as f32 * 1e-4).collect();
    // Reference weight, dequantized the way the unfused graph would.
    let w_f32: Vec<f32> = (0..k * n)
        .map(|i| w_i8[i] as f32 * w_scale[i % n])
        .collect();
    (a, a_scale, a_zp, raw, w_scale, w_f32)
}

#[test]
fn test_qmatmul_i8_matches_dequantized_matmul() {
    // Shapes chosen to exercise the row tail (400 % 6) and the column tail
    // (100 % 64) of the packed kernel, plus one clean shape.
    for &(m, k, n) in &[(7usize, 12usize, 20usize), (13, 36, 100), (64, 128, 64)] {
        let (a, a_scale, a_zp, raw, w_scale, w_f32) = qdq_layer(m, k, n);
        let a_view = TensorView::from_slice(&a, vec![m, k]);
        let w_view = TensorView::from_slice(&w_f32, vec![k, n]);
        let mut expected_buf = Vec::new();
        let expected = lele::kernels::matmul(&a_view, &w_view, &mut expected_buf);

        let qw = lele::kernels::prepare_quantized_weights(&raw, k, n, &w_scale);
        let scale_view = TensorView::from_slice(std::slice::from_ref(&a_scale), vec![]);
        let zp_view = TensorView::from_slice(std::slice::from_ref(&a_zp), vec![]);
        let mut got_buf = Vec::new();
        let got = lele::kernels::qmatmul_i8(&a_view, &scale_view, &zp_view, &qw, &mut got_buf);

        assert_eq!(got.shape.as_ref(), &[m, n]);
        // The integer path accumulates exactly and scales once at the end, so
        // it differs from the f32 reference only by f32 rounding of the inputs.
        let tol = 2e-4 * (k as f32).sqrt();
        for (i, (&g, &e)) in got.data.iter().zip(expected.data.iter()).enumerate() {
            assert!(
                (g - e).abs() <= tol * e.abs().max(1.0),
                "qmatmul_i8 differs at {i} for m={m} k={k} n={n}: got {g}, want {e}"
            );
        }
    }
}

#[test]
fn test_qmatmul_i8_handles_batched_activations() {
    let (m, k, n) = (5usize, 16usize, 32usize);
    let batch = 3usize;
    let (a, a_scale, a_zp, raw, w_scale, w_f32) = qdq_layer(batch * m, k, n);
    let a_view = TensorView::from_slice(&a, vec![batch, m, k]);
    let w_view = TensorView::from_slice(&w_f32, vec![k, n]);
    let mut expected_buf = Vec::new();
    let expected = lele::kernels::matmul(&a_view, &w_view, &mut expected_buf);

    let qw = lele::kernels::prepare_quantized_weights(&raw, k, n, &w_scale);
    let scale_view = TensorView::from_slice(std::slice::from_ref(&a_scale), vec![]);
    let zp_view = TensorView::from_slice(std::slice::from_ref(&a_zp), vec![]);
    let mut got_buf = Vec::new();
    let got = lele::kernels::qmatmul_i8(&a_view, &scale_view, &zp_view, &qw, &mut got_buf);

    assert_eq!(got.shape.as_ref(), &[batch, m, n]);
    for (i, (&g, &e)) in got.data.iter().zip(expected.data.iter()).enumerate() {
        assert!(
            (g - e).abs() <= 1e-3 * e.abs().max(1.0),
            "batched qmatmul_i8 differs at {i}: got {g}, want {e}"
        );
    }
}

#[test]
fn test_qmatmul_i8_accepts_per_tensor_weight_scale() {
    let (m, k, n) = (9usize, 20usize, 24usize);
    let (a, a_scale, a_zp, raw, _, _) = qdq_layer(m, k, n);
    let w_scale = vec![0.0025f32];
    let w_f32: Vec<f32> = raw.iter().map(|&b| (b as i8) as f32 * w_scale[0]).collect();

    let a_view = TensorView::from_slice(&a, vec![m, k]);
    let w_view = TensorView::from_slice(&w_f32, vec![k, n]);
    let mut expected_buf = Vec::new();
    let expected = lele::kernels::matmul(&a_view, &w_view, &mut expected_buf);

    let qw = lele::kernels::prepare_quantized_weights(&raw, k, n, &w_scale);
    let scale_view = TensorView::from_slice(std::slice::from_ref(&a_scale), vec![]);
    let zp_view = TensorView::from_slice(std::slice::from_ref(&a_zp), vec![]);
    let mut got_buf = Vec::new();
    let got = lele::kernels::qmatmul_i8(&a_view, &scale_view, &zp_view, &qw, &mut got_buf);

    for (i, (&g, &e)) in got.data.iter().zip(expected.data.iter()).enumerate() {
        assert!(
            (g - e).abs() <= 1e-3 * e.abs().max(1.0),
            "per-tensor qmatmul_i8 differs at {i}: got {g}, want {e}"
        );
    }
}

/// The tests above pass trivially if every machine takes the f32 fallback, so
/// pin which path was actually chosen to what the CPU reports.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_quantized_weights_take_the_integer_path_when_the_cpu_allows() {
    use lele::kernels::avx::qgemm::has_avx2_int8;
    use lele::kernels::avx512::qgemm::has_vnni;
    let (_, _, _, raw, w_scale, _) = qdq_layer(4, 8, 16);
    let qw = lele::kernels::prepare_quantized_weights(&raw, 8, 16, &w_scale);
    assert_eq!(
        qw.is_integer(),
        has_vnni() || has_avx2_int8(),
        "an integer kernel should run whenever the CPU has VNNI or AVX2"
    );
}

/// Exact check of the packed kernel against integer arithmetic, skipped on
/// machines that cannot run it.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_vnni_qgemm_matches_integer_reference() {
    use lele::kernels::avx512::qgemm::{has_vnni, pack_i8_weights, qgemm_u8s8_f32};
    if !has_vnni() {
        return;
    }
    // K is not a multiple of 4 and N is not a multiple of 64, so both the
    // packing padding and the masked store are exercised.
    let (m, k, n) = (11usize, 37usize, 70usize);
    let a: Vec<u8> = (0..m * k.next_multiple_of(4))
        .map(|i| (i * 37 % 256) as u8)
        .collect();
    let b: Vec<i8> = (0..k * n)
        .map(|i| ((i * 53 % 255) as i32 - 127) as i8)
        .collect();
    let w_scale: Vec<f32> = (0..n).map(|j| 0.001 + (j % 7) as f32 * 1e-4).collect();
    let bias: Vec<f32> = (0..n).map(|j| (j % 11) as f32 * 0.01).collect();
    let (a_scale, a_zp) = (0.0037f32, 131i32);
    let lda = k.next_multiple_of(4);

    let pw = pack_i8_weights(&b, k, n);
    let mut out = vec![0f32; m * n];
    unsafe {
        qgemm_u8s8_f32(
            a.as_ptr(), m, lda, &pw, a_zp, a_scale,
            w_scale.as_ptr(), w_scale.len(), Some(bias.as_ptr()),
            out.as_mut_ptr(), n,
        );
    }

    for i in 0..m {
        for j in 0..n {
            let dot: i32 = (0..k).map(|kk| a[i * lda + kk] as i32 * b[kk * n + j] as i32).sum();
            let col: i32 = (0..k).map(|kk| b[kk * n + j] as i32).sum();
            let want = a_scale * w_scale[j] * (dot - a_zp * col) as f32 + bias[j];
            let got = out[i * n + j];
            assert!(
                (got - want).abs() <= 1e-5 * want.abs().max(1.0),
                "vnni qgemm differs at ({i},{j}): got {got}, want {want}"
            );
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_fake_quantize_per_tensor_matches_exact_division() {
    // The SIMD path multiplies by the reciprocal and refines with an FMA
    // instead of dividing, which can only change the result where the true
    // quotient falls on a rounding tie. Sweep a wide range of magnitudes and
    // scales and require it to agree with an exact division everywhere.
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }
    // Lengths that land on and off the 32- and 8-wide steps, so the vector
    // body and the scalar tail are both exercised.
    for &len in &[1usize, 7, 8, 31, 32, 33, 1000, 4099] {
        for &scale in &[0.0037f32, 1.0, 0.5, 1e-4, 7.25e-3, 123.5] {
            for &zp in &[0.0f32, 128.0, 255.0] {
                let x: Vec<f32> = (0..len)
                    .map(|i| {
                        // Spread over several orders of magnitude and both
                        // signs, and hit exact half-steps of the grid.
                        let t = i as f32;
                        match i % 4 {
                            0 => (t - len as f32 / 2.0) * scale,
                            1 => (t - len as f32 / 2.0) * scale * 0.5,
                            2 => (t * 0.5 + 0.5) * scale,
                            _ => (t - len as f32 / 2.0) * 1e-3,
                        }
                    })
                    .collect();
                let x_view = TensorView::from_slice(&x, vec![len]);
                let s = [scale];
                let z = [zp];
                let s_view = TensorView::from_slice(&s, vec![]);
                let z_view = TensorView::from_slice(&z, vec![]);

                let mut buf = Vec::new();
                let got = lele::kernels::fake_quantize_linear(
                    &x_view,
                    &s_view,
                    Some(&z_view),
                    1,
                    0,
                    0.0,
                    255.0,
                    &mut buf,
                );

                for (i, (&g, &v)) in got.data.iter().zip(x.iter()).enumerate() {
                    let want =
                        (((v / scale).round_ties_even() + zp).clamp(0.0, 255.0) - zp) * scale;
                    assert_eq!(
                        g.to_bits(),
                        want.to_bits(),
                        "len={len} scale={scale} zp={zp} index {i} (x={v}): got {g}, want {want}"
                    );
                }
            }
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_fake_quantize_per_axis_still_divides_exactly() {
    // Only the per-tensor layout takes the SIMD path; a per-channel scale must
    // still go through the general routine.
    let x: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) * 0.031).collect();
    let x_view = TensorView::from_slice(&x, vec![4, 3, 2]);
    let scale = [0.01f32, 0.02, 0.04];
    let zp = [10.0f32, 20.0, 30.0];
    let s_view = TensorView::from_slice(&scale, vec![3]);
    let z_view = TensorView::from_slice(&zp, vec![3]);

    let mut buf = Vec::new();
    let got =
        lele::kernels::fake_quantize_linear(&x_view, &s_view, Some(&z_view), 1, 0, 0.0, 255.0, &mut buf);

    for o in 0..4 {
        for d in 0..3 {
            for i in 0..2 {
                let idx = (o * 3 + d) * 2 + i;
                let (s, z) = (scale[d], zp[d]);
                let want = (((x[idx] / s).round_ties_even() + z).clamp(0.0, 255.0) - z) * s;
                assert_eq!(got.data[idx].to_bits(), want.to_bits(), "index {idx}");
            }
        }
    }
}

/// Exact check of the AVX2 packed kernel against integer arithmetic.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_avx2_qgemm_matches_integer_reference() {
    use lele::kernels::avx::qgemm::{has_avx2_int8, pack_i8_weights_avx2, qgemm_u8s8_f32_avx2};
    if !has_avx2_int8() {
        return;
    }
    // K odd and N not a multiple of 16, so the packing padding, the row tail
    // and the masked store are all exercised.
    let (m, k, n) = (11usize, 37usize, 70usize);
    let lda = k.next_multiple_of(2);
    let codes: Vec<i32> = (0..m * lda).map(|i| (i * 37 % 256) as i32).collect();
    let a: Vec<i16> = codes.iter().map(|&v| v as i16).collect();
    let b: Vec<i8> = (0..k * n)
        .map(|i| ((i * 53 % 255) as i32 - 127) as i8)
        .collect();
    let w_scale: Vec<f32> = (0..n).map(|j| 0.001 + (j % 7) as f32 * 1e-4).collect();
    let bias: Vec<f32> = (0..n).map(|j| (j % 11) as f32 * 0.01).collect();
    let (a_scale, a_zp) = (0.0037f32, 131i32);

    let pw = pack_i8_weights_avx2(&b, k, n);
    let mut out = vec![0f32; m * n];
    unsafe {
        qgemm_u8s8_f32_avx2(
            a.as_ptr(), m, lda, &pw, a_zp, a_scale,
            w_scale.as_ptr(), w_scale.len(), Some(bias.as_ptr()),
            out.as_mut_ptr(), n,
        );
    }

    for i in 0..m {
        for j in 0..n {
            let dot: i32 = (0..k).map(|kk| codes[i * lda + kk] * b[kk * n + j] as i32).sum();
            let col: i32 = (0..k).map(|kk| b[kk * n + j] as i32).sum();
            let want = a_scale * w_scale[j] * (dot - a_zp * col) as f32 + bias[j];
            let got = out[i * n + j];
            assert!(
                (got - want).abs() <= 1e-5 * want.abs().max(1.0),
                "avx2 qgemm differs at ({i},{j}): got {got}, want {want}"
            );
        }
    }
}

#[test]
fn test_reduce_prod_matches_reference() {
    let shape = vec![2usize, 3, 4];
    // Values near one so the products stay in a range f32 represents exactly
    // enough to compare against a scalar reference.
    let data: Vec<f32> = (0..2 * 3 * 4).map(|i| 1.0 + (i % 5) as f32 * 0.25).collect();
    let t = TensorView::from_slice(&data, shape.clone());

    let cases: &[&[i64]] = &[&[-1], &[0], &[1], &[0, 2], &[0, 1, 2]];
    for axes in cases {
        for keepdims in [false, true] {
            let mut buf = Vec::new();
            let got = reduce_prod(&t, axes, keepdims, &mut buf);

            let dims = shape.len();
            let resolved: Vec<usize> = axes
                .iter()
                .map(|&a| if a < 0 { (dims as i64 + a) as usize } else { a as usize })
                .collect();
            let mut want_shape = Vec::new();
            for (i, &d) in shape.iter().enumerate() {
                if !resolved.contains(&i) {
                    want_shape.push(d);
                } else if keepdims {
                    want_shape.push(1);
                }
            }
            assert_eq!(got.shape.as_ref(), want_shape.as_slice(), "axes {axes:?}");

            let kept: Vec<usize> = (0..dims).filter(|i| !resolved.contains(i)).collect();
            let mut want = vec![1.0f32; want_shape.iter().product::<usize>().max(1)];
            let mut coords = vec![0usize; dims];
            for v in &data {
                let mut off = 0usize;
                for &d in &kept {
                    off = off * shape[d] + coords[d];
                }
                want[off] *= v;
                for d in (0..dims).rev() {
                    coords[d] += 1;
                    if coords[d] < shape[d] {
                        break;
                    }
                    coords[d] = 0;
                }
            }
            assert_close(&got.data, &want, 1e-3, &format!("reduce_prod axes {axes:?}"));
        }
    }
}

#[test]
fn test_reduce_prod_of_shape_vector() {
    // How the op actually shows up: an i64 element count pulled out of Shape,
    // which the pooling layer divides by.
    let dims: Vec<i64> = vec![7];
    let t = TensorView::from_slice(&dims, vec![1usize]);
    let mut buf = Vec::new();
    let got = reduce_prod(&t, &[], false, &mut buf);
    assert_eq!(got.data.as_ref(), &[7i64]);
}
