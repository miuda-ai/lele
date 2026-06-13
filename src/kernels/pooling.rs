use crate::tensor::TensorView;
use std::borrow::Cow;

/// GlobalAveragePool for NCHW rank-4 input: [N, C, H, W] -> [N, C, 1, 1]
pub fn global_average_pool<'b, 'a>(
    input: &TensorView<'b>,
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let shape = &input.shape;
    assert!(shape.len() == 4, "GlobalAveragePool: expected rank-4 input");
    let batch = shape[0];
    let channels = shape[1];
    let in_h = shape[2];
    let in_w = shape[3];
    let spatial = in_h * in_w;
    let numel = batch * channels;
    crate::kernels::utils::ensure_capacity(out, numel);
    unsafe {
        out.set_len(numel);
    }
    let data = &input.data;
    let o_slice = out.as_mut_slice();
    for nc in 0..numel {
        let base = nc * spatial;
        let mut sum = 0.0f32;
        for i in 0..spatial {
            sum += data[base + i];
        }
        o_slice[nc] = sum / spatial as f32;
    }
    TensorView {
        data: Cow::Borrowed(out),
        shape: Cow::Owned(vec![batch, channels, 1, 1]),
    }
}

/// AveragePool2d for NCHW rank-4 input with optional count_include_pad.
pub fn average_pool2d<'b, 'a>(
    input: &TensorView<'b>,
    kernel_shape: &[i64],
    strides: &[i64],
    pads: &[i64],
    count_include_pad: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let shape = &input.shape;
    assert!(shape.len() == 4, "AveragePool2d: expected rank-4 input");

    let batch = shape[0];
    let channels = shape[1];
    let in_h = shape[2];
    let in_w = shape[3];

    let kh = kernel_shape[0] as usize;
    let kw = if kernel_shape.len() > 1 {
        kernel_shape[1] as usize
    } else {
        kh
    };

    let sh = if strides.is_empty() { 1 } else { strides[0] as usize };
    let sw = if strides.len() > 1 {
        strides[1] as usize
    } else {
        sh
    };

    let pad_top = if pads.is_empty() { 0 } else { pads[0] as usize };
    let pad_left = if pads.len() > 1 {
        pads[1] as usize
    } else {
        pad_top
    };
    let pad_bottom = if pads.len() > 2 {
        pads[2] as usize
    } else {
        pad_top
    };
    let pad_right = if pads.len() > 3 {
        pads[3] as usize
    } else {
        pad_left
    };

    let out_h =
        ((in_h as i64 + pad_top as i64 + pad_bottom as i64 - kh as i64) / sh as i64 + 1) as usize;
    let out_w =
        ((in_w as i64 + pad_left as i64 + pad_right as i64 - kw as i64) / sw as i64 + 1) as usize;

    let total = batch * channels * out_h * out_w;
    crate::kernels::utils::ensure_capacity(out, total);
    unsafe {
        out.set_len(total);
    }

    let data = &input.data;
    let o_slice = out.as_mut_slice();

    for n in 0..batch {
        for c in 0..channels {
            let in_base = (n * channels + c) * in_h * in_w;
            let out_base = (n * channels + c) * out_h * out_w;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let h_start = oh * sh;
                    let h_end = (h_start + kh).min(in_h + pad_bottom);
                    let w_start = ow * sw;
                    let w_end = (w_start + kw).min(in_w + pad_right);
                    let mut sum = 0.0f32;
                    let mut count = 0;
                    for kh_i in 0..kh {
                        let ih = h_start as i64 + kh_i as i64 - pad_top as i64;
                        if ih < 0 || ih >= in_h as i64 {
                            continue;
                        }
                        for kw_i in 0..kw {
                            let iw = w_start as i64 + kw_i as i64 - pad_left as i64;
                            if iw < 0 || iw >= in_w as i64 {
                                continue;
                            }
                            sum += data[in_base + ih as usize * in_w + iw as usize];
                            count += 1;
                        }
                    }
                    let denom = if count_include_pad {
                        (kh * kw) as f32
                    } else {
                        if count > 0 {
                            count as f32
                        } else {
                            1.0
                        }
                    };
                    o_slice[out_base + oh * out_w + ow] = sum / denom;
                }
            }
        }
    }

    TensorView {
        data: Cow::Borrowed(out),
        shape: Cow::Owned(vec![batch, channels, out_h, out_w]),
    }
}

pub fn adaptive_avg_pool1d(
    input: &[f32],
    output: &mut [f32],
    channels: usize,
    input_len: usize,
    output_len: usize,
) {
    assert_eq!(input.len(), channels * input_len);
    assert_eq!(output.len(), channels * output_len);
    for c in 0..channels {
        let in_offset = c * input_len;
        let out_offset = c * output_len;
        for i in 0..output_len {
            let start_idx_raw = (i * input_len) / output_len;
            let end_idx_raw = ((i + 1) * input_len).div_ceil(output_len);
            let start_idx = start_idx_raw.min(input_len);
            let end_idx = end_idx_raw.min(input_len);
            let kernel_len = end_idx - start_idx;
            if kernel_len == 0 {
                output[out_offset + i] = 0.0;
                continue;
            }
            let mut sum = 0.0;
            for k in start_idx..end_idx {
                sum += input[in_offset + k];
            }
            output[out_offset + i] = sum / kernel_len as f32;
        }
    }
}
