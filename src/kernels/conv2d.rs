use crate::kernels::utils;
use crate::tensor::TensorView;
use faer::linalg::matmul::matmul as faer_matmul;
use faer::mat::{MatMut, MatRef};
use faer::{Accum, Par};

pub fn print_conv_stats() {}

pub fn reset_conv_stats() {}

/// 2D Convolution using im2col + GEMM approach.
/// Input shape: [N, C_in, H, W]
/// Weight shape: [C_out, C_in/groups, kH, kW]
/// Output shape: [N, C_out, H_out, W_out]
pub fn conv2d<'b, 'a>(
    input: &TensorView<'b>,
    weights: &TensorView<'b>,
    bias: Option<&TensorView<'b>>,
    dilations: &[i64],
    group: i64,
    pads: &[i64],
    strides: &[i64],
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    conv2d_fused(input, weights, bias, dilations, group, pads, strides, false, out)
}

/// 2D Convolution with optional fused ReLU activation.
pub fn conv2d_fused<'b, 'a>(
    input: &TensorView<'b>,
    weights: &TensorView<'b>,
    bias: Option<&TensorView<'b>>,
    dilations: &[i64],
    group: i64,
    pads: &[i64],
    strides: &[i64],
    relu: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let in_shape = &input.shape;
    let w_shape = &weights.shape;

    assert!(in_shape.len() == 4, "Conv2d: expected rank-4 input [N,C,H,W], got rank {}", in_shape.len());
    assert!(w_shape.len() == 4, "Conv2d: expected rank-4 weight [C_out,C_in/g,kH,kW], got rank {}", w_shape.len());

    let batch_size = in_shape[0];
    let in_channels = in_shape[1];
    let in_h = in_shape[2];
    let in_w = in_shape[3];

    let out_channels = w_shape[0];
    let kernel_h = w_shape[2];
    let kernel_w = w_shape[3];

    let dilation_h = if dilations.len() >= 2 { dilations[0] as usize } else if dilations.len() == 1 { dilations[0] as usize } else { 1 };
    let dilation_w = if dilations.len() >= 2 { dilations[1] as usize } else if dilations.len() == 1 { dilations[0] as usize } else { 1 };

    let stride_h = if strides.len() >= 2 { strides[0] as usize } else if strides.len() == 1 { strides[0] as usize } else { 1 };
    let stride_w = if strides.len() >= 2 { strides[1] as usize } else if strides.len() == 1 { strides[0] as usize } else { 1 };

    let pad_top = if pads.len() >= 4 { pads[0] as usize } else if pads.len() >= 2 { pads[0] as usize } else { 0 };
    let pad_left = if pads.len() >= 4 { pads[1] as usize } else if pads.len() >= 2 { pads[1] as usize } else { 0 };
    let pad_bottom = if pads.len() >= 4 { pads[2] as usize } else if pads.len() >= 2 { pads[0] as usize } else { 0 };
    let pad_right = if pads.len() >= 4 { pads[3] as usize } else if pads.len() >= 2 { pads[1] as usize } else { 0 };

    let out_h = (in_h + pad_top + pad_bottom - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    let out_w = (in_w + pad_left + pad_right - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    let groups = group as usize;
    let in_channels_per_group = in_channels / groups;
    let out_channels_per_group = out_channels / groups;

    let total_output = batch_size * out_channels * out_h * out_w;
    utils::ensure_capacity(out, total_output);
    unsafe { out.set_len(total_output); }

    let input_data = &input.data;
    let weight_data = &weights.data;

    // Fast path: 1x1 convolution with stride 1, no padding, groups=1
    if kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1
        && pad_top == 0 && pad_left == 0 && pad_bottom == 0 && pad_right == 0
        && groups == 1
    {
        let spatial = in_h * in_w;
        for n in 0..batch_size {
            let in_offset = n * in_channels * spatial;
            let o_offset = n * out_channels * spatial;

            unsafe {
                let w_ptr = weight_data.as_ptr();
                let in_ptr = input_data.as_ptr().add(in_offset);
                let out_ptr = out.as_mut_ptr().add(o_offset);

                let w_mat = MatRef::<f32>::from_raw_parts(
                    w_ptr, out_channels, in_channels, in_channels as isize, 1,
                );
                let in_mat = MatRef::<f32>::from_raw_parts(
                    in_ptr, in_channels, spatial, spatial as isize, 1,
                );
                let out_mat = MatMut::<f32>::from_raw_parts_mut(
                    out_ptr, out_channels, spatial, spatial as isize, 1,
                );

                faer_matmul(out_mat, Accum::Replace, w_mat, in_mat, 1.0, Par::Seq);
            }

            // Apply bias and optional ReLU
            if bias.is_some() || relu {
                for oc in 0..out_channels {
                    let bias_val = if let Some(b) = bias { b.data[oc] } else { 0.0 };
                    let row_start = o_offset + oc * spatial;
                    if bias_val != 0.0 || relu {
                        for j in 0..spatial {
                            let val = out[row_start + j] + bias_val;
                            out[row_start + j] = if relu && val < 0.0 { 0.0 } else { val };
                        }
                    }
                }
            }
        }
        return TensorView::from_slice(out, vec![batch_size, out_channels, out_h, out_w]);
    }

    let col_rows = in_channels_per_group * kernel_h * kernel_w;
    let col_cols = out_h * out_w;

    // Reuse thread-local im2col buffer to avoid repeated allocation
    thread_local! {
        static COL_BUF: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::new());
    }
    COL_BUF.with(|buf_cell| {
    let mut col_buf_ref = buf_cell.borrow_mut();
    let needed = col_rows * col_cols;
    if col_buf_ref.len() < needed {
        col_buf_ref.resize(needed, 0.0);
    }
    let col_buf = &mut col_buf_ref[..needed];

    for n in 0..batch_size {
        for g in 0..groups {
            let in_ch_start = g * in_channels_per_group;
            let out_ch_start = g * out_channels_per_group;

            // im2col: unfold input patch into column matrix
            im2col(
                input_data,
                n, in_ch_start, in_channels_per_group,
                in_h, in_w, in_channels,
                kernel_h, kernel_w,
                stride_h, stride_w,
                pad_top, pad_left,
                dilation_h, dilation_w,
                out_h, out_w,
                col_buf,
            );

            // GEMM: weight_matrix [out_channels_per_group, col_rows] x col_matrix [col_rows, col_cols]
            // = output [out_channels_per_group, out_h * out_w]
            let w_offset = out_ch_start * (in_channels_per_group * kernel_h * kernel_w);
            let o_offset = (n * out_channels + out_ch_start) * out_h * out_w;

            // Use faer for fast GEMM
            unsafe {
                let w_ptr = weight_data.as_ptr().add(w_offset);
                let col_ptr = col_buf.as_ptr();
                let out_ptr = out.as_mut_ptr().add(o_offset);

                // Weight: [out_channels_per_group, col_rows] row-major
                let w_mat = MatRef::<f32>::from_raw_parts(
                    w_ptr, out_channels_per_group, col_rows, col_rows as isize, 1,
                );
                // Col: [col_rows, col_cols] row-major
                let col_mat = MatRef::<f32>::from_raw_parts(
                    col_ptr, col_rows, col_cols, col_cols as isize, 1,
                );
                // Out: [out_channels_per_group, col_cols] row-major
                let out_mat = MatMut::<f32>::from_raw_parts_mut(
                    out_ptr, out_channels_per_group, col_cols, col_cols as isize, 1,
                );

                faer_matmul(out_mat, Accum::Replace, w_mat, col_mat, 1.0, Par::Seq);
            }

            // Apply bias and optional ReLU
            if bias.is_some() || relu {
                for oc in 0..out_channels_per_group {
                    let o_row_start = o_offset + oc * col_cols;
                    let bias_val = if let Some(b) = bias {
                        b.data[out_ch_start + oc]
                    } else {
                        0.0
                    };
                    if bias_val != 0.0 || relu {
                        for j in 0..col_cols {
                            let val = out[o_row_start + j] + bias_val;
                            out[o_row_start + j] = if relu && val < 0.0 { 0.0 } else { val };
                        }
                    }
                }
            }
        }
    }
    }); // COL_BUF.with

    TensorView::from_slice(out, vec![batch_size, out_channels, out_h, out_w])
}

/// im2col: unfold input patches into column matrix for GEMM-based convolution.
#[inline]
fn im2col(
    input: &[f32],
    batch_idx: usize,
    ch_start: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    total_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_left: usize,
    dilation_h: usize,
    dilation_w: usize,
    out_h: usize,
    out_w: usize,
    col: &mut [f32],
) {
    let spatial_cols = out_h * out_w;
    let batch_offset = batch_idx * total_channels * in_h * in_w;

    // Common case: stride=1, dilation=1, 3x3 kernel — use optimized path
    if stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1 {
        for c in 0..channels {
            let ch_offset = batch_offset + (ch_start + c) * in_h * in_w;
            for kh in 0..kernel_h {
                for kw in 0..kernel_w {
                    let col_row_offset = ((c * kernel_h + kh) * kernel_w + kw) * spatial_cols;

                    // Compute the valid range of oh values (where ih is in bounds)
                    // ih = oh + kh - pad_top; valid when 0 <= ih < in_h
                    // oh >= pad_top - kh  and  oh < in_h + pad_top - kh
                    let oh_start = if kh < pad_top { pad_top - kh } else { 0 };
                    let oh_end = (in_h + pad_top).saturating_sub(kh).min(out_h);

                    // Zero the top padding rows
                    unsafe {
                        let col_ptr = col.as_mut_ptr().add(col_row_offset);
                        for oh in 0..oh_start {
                            let row_offset = oh * out_w;
                            for ow in 0..out_w {
                                *col_ptr.add(row_offset + ow) = 0.0;
                            }
                        }
                    }

                    // Process valid rows
                    for oh in oh_start..oh_end {
                        let ih = (oh + kh) as isize - pad_top as isize;
                        let ih = ih as usize;
                        let in_row_offset = ch_offset + ih * in_w;
                        let col_base = col_row_offset + oh * out_w;

                        // Compute valid ow range
                        // iw = ow + kw - pad_left; valid when 0 <= iw < in_w
                        let ow_start = if kw < pad_left { pad_left - kw } else { 0 };
                        let ow_end = (in_w + pad_left).saturating_sub(kw).min(out_w);

                        unsafe {
                            let col_ptr = col.as_mut_ptr().add(col_base);
                            let in_ptr = input.as_ptr().add(in_row_offset);

                            // Zero left padding
                            for ow in 0..ow_start {
                                *col_ptr.add(ow) = 0.0;
                            }
                            // Copy valid region (contiguous with stride=1, dilation=1)
                            let iw_start = (ow_start + kw) as isize - pad_left as isize;
                            let count = ow_end - ow_start;
                            std::ptr::copy_nonoverlapping(
                                in_ptr.add(iw_start as usize),
                                col_ptr.add(ow_start),
                                count,
                            );
                            // Zero right padding
                            for ow in ow_end..out_w {
                                *col_ptr.add(ow) = 0.0;
                            }
                        }
                    }

                    // Zero the bottom padding rows
                    unsafe {
                        let col_ptr = col.as_mut_ptr().add(col_row_offset);
                        for oh in oh_end..out_h {
                            let row_offset = oh * out_w;
                            for ow in 0..out_w {
                                *col_ptr.add(row_offset + ow) = 0.0;
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    // General path for non-unit stride/dilation
    for c in 0..channels {
        let in_ch = ch_start + c;
        let ch_offset = batch_offset + in_ch * in_h * in_w;
        for kh in 0..kernel_h {
            for kw in 0..kernel_w {
                let col_row = (c * kernel_h + kh) * kernel_w + kw;
                let col_row_offset = col_row * spatial_cols;
                for oh in 0..out_h {
                    let ih = (oh * stride_h + kh * dilation_h) as isize - pad_top as isize;
                    let col_oh_offset = col_row_offset + oh * out_w;
                    if ih >= 0 && ih < in_h as isize {
                        let in_row = ch_offset + ih as usize * in_w;
                        for ow in 0..out_w {
                            let iw = (ow * stride_w + kw * dilation_w) as isize - pad_left as isize;
                            unsafe {
                                *col.get_unchecked_mut(col_oh_offset + ow) =
                                    if iw >= 0 && iw < in_w as isize {
                                        *input.get_unchecked(in_row + iw as usize)
                                    } else {
                                        0.0
                                    };
                            }
                        }
                    } else {
                        // Entire row is padding — zero it
                        unsafe {
                            let col_ptr = col.as_mut_ptr().add(col_oh_offset);
                            for ow in 0..out_w {
                                *col_ptr.add(ow) = 0.0;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// 2D Max Pooling.
/// Input shape: [N, C, H, W]
/// Output shape: [N, C, H_out, W_out]
pub fn max_pool2d<'b, 'a>(
    input: &TensorView<'b>,
    kernel_shape: &[i64],
    strides: &[i64],
    pads: &[i64],
    dilations: &[i64],
    ceil_mode: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let shape = &input.shape;
    assert!(shape.len() == 4, "MaxPool2d: expected rank-4 input");

    let batch = shape[0];
    let channels = shape[1];
    let in_h = shape[2];
    let in_w = shape[3];

    let kh = kernel_shape[0] as usize;
    let kw = if kernel_shape.len() > 1 { kernel_shape[1] as usize } else { kh };

    let sh = if strides.is_empty() { 1 } else { strides[0] as usize };
    let sw = if strides.len() > 1 { strides[1] as usize } else { sh };

    let pad_top = if pads.is_empty() { 0 } else { pads[0] as usize };
    let pad_left = if pads.len() > 1 { pads[1] as usize } else { pad_top };
    let pad_bottom = if pads.len() > 2 { pads[2] as usize } else { pad_top };
    let pad_right = if pads.len() > 3 { pads[3] as usize } else { pad_left };

    let dh = if dilations.is_empty() { 1 } else { dilations[0] as usize };
    let dw = if dilations.len() > 1 { dilations[1] as usize } else { dh };

    let effective_kh = dh * (kh - 1) + 1;
    let effective_kw = dw * (kw - 1) + 1;

    let out_h = if ceil_mode {
        (in_h + pad_top + pad_bottom - effective_kh + sh - 1) / sh + 1
    } else {
        (in_h + pad_top + pad_bottom - effective_kh) / sh + 1
    };
    let out_w = if ceil_mode {
        (in_w + pad_left + pad_right - effective_kw + sw - 1) / sw + 1
    } else {
        (in_w + pad_left + pad_right - effective_kw) / sw + 1
    };

    let total = batch * channels * out_h * out_w;
    utils::ensure_capacity(out, total);
    unsafe { out.set_len(total); }

    let data = &input.data;

    for n in 0..batch {
        for c in 0..channels {
            let in_offset = (n * channels + c) * in_h * in_w;
            let out_offset = (n * channels + c) * out_h * out_w;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut max_val = f32::NEG_INFINITY;
                    for ki in 0..kh {
                        let ih = (oh * sh + ki * dh) as isize - pad_top as isize;
                        if ih < 0 || ih >= in_h as isize { continue; }
                        for kj in 0..kw {
                            let iw = (ow * sw + kj * dw) as isize - pad_left as isize;
                            if iw < 0 || iw >= in_w as isize { continue; }
                            let val = data[in_offset + ih as usize * in_w + iw as usize];
                            if val > max_val { max_val = val; }
                        }
                    }
                    out[out_offset + oh * out_w + ow] = max_val;
                }
            }
        }
    }

    TensorView::from_slice(out, vec![batch, channels, out_h, out_w])
}

/// Resize 2D using nearest neighbor interpolation.
/// Input shape: [N, C, H, W]
/// Supports both scales and sizes modes.
/// coordinate_transform_mode: "asymmetric" uses floor(out * in/out) mapping,
/// "half_pixel" uses round((out+0.5)*scale - 0.5) mapping.
pub fn resize_nearest<'b, 'a>(
    input: &TensorView<'b>,
    scales: Option<&[f32]>,
    sizes: Option<&[i64]>,
    coordinate_transform_mode: &str,
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let shape = &input.shape;
    assert!(shape.len() == 4, "Resize: expected rank-4 input");

    let batch = shape[0];
    let channels = shape[1];
    let in_h = shape[2];
    let in_w = shape[3];

    let (out_h, out_w) = if let Some(sizes) = sizes {
        // sizes is [N, C, H, W]
        (sizes[2] as usize, sizes[3] as usize)
    } else if let Some(scales) = scales {
        // scales is [N, C, H, W]
        let sh = if scales.len() >= 3 { scales[2] } else { 1.0 };
        let sw = if scales.len() >= 4 { scales[3] } else { 1.0 };
        ((in_h as f32 * sh) as usize, (in_w as f32 * sw) as usize)
    } else {
        panic!("Resize: either scales or sizes must be provided");
    };

    let total = batch * channels * out_h * out_w;
    utils::ensure_capacity(out, total);
    unsafe { out.set_len(total); }

    let data = &input.data;

    let h_scale = in_h as f32 / out_h as f32;
    let w_scale = in_w as f32 / out_w as f32;

    for n in 0..batch {
        for c in 0..channels {
            let in_offset = (n * channels + c) * in_h * in_w;
            let out_offset = (n * channels + c) * out_h * out_w;
            for oh in 0..out_h {
                let ih = if coordinate_transform_mode == "asymmetric" {
                    // asymmetric + floor: ih = floor(oh * in_h / out_h)
                    (oh as f32 * h_scale).floor().min((in_h - 1) as f32) as usize
                } else {
                    // half_pixel: ih = round((oh+0.5)*scale - 0.5)
                    ((oh as f32 + 0.5) * h_scale - 0.5).round().max(0.0).min((in_h - 1) as f32) as usize
                };
                for ow in 0..out_w {
                    let iw = if coordinate_transform_mode == "asymmetric" {
                        (ow as f32 * w_scale).floor().min((in_w - 1) as f32) as usize
                    } else {
                        ((ow as f32 + 0.5) * w_scale - 0.5).round().max(0.0).min((in_w - 1) as f32) as usize
                    };
                    out[out_offset + oh * out_w + ow] = data[in_offset + ih * in_w + iw];
                }
            }
        }
    }

    TensorView::from_slice(out, vec![batch, channels, out_h, out_w])
}

/// TopK: returns top-k values and indices along the last axis.
pub fn topk<'a>(
    input: &TensorView<'_>,
    k: usize,
    _axis: i64,
    largest: bool,
    _sorted: bool,
    values_buf: &'a mut Vec<f32>,
    indices_buf: &'a mut Vec<f32>,
) -> (TensorView<'a>, TensorView<'a>) {
    let shape = &input.shape;
    let last_dim = *shape.last().unwrap();
    let outer: usize = shape[..shape.len() - 1].iter().product();

    let k = k.min(last_dim);
    let mut out_shape = shape.to_vec();
    *out_shape.last_mut().unwrap() = k;

    let total = outer * k;
    utils::ensure_capacity(values_buf, total);
    utils::ensure_capacity(indices_buf, total);
    unsafe {
        values_buf.set_len(total);
        indices_buf.set_len(total);
    }

    let data = &input.data;

    for o in 0..outer {
        let row_start = o * last_dim;
        let out_start = o * k;

        // Create index-value pairs
        let mut pairs: Vec<(usize, f32)> = (0..last_dim)
            .map(|i| (i, data[row_start + i]))
            .collect();

        if largest {
            pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        for i in 0..k {
            values_buf[out_start + i] = pairs[i].1;
            indices_buf[out_start + i] = pairs[i].0 as f32;
        }
    }

    let values = TensorView::from_slice(values_buf, out_shape.clone());
    let indices = TensorView::from_slice(indices_buf, out_shape);
    (values, indices)
}

/// GatherElements: gather elements along an axis using index tensor.
pub fn gather_elements<'a>(
    input: &TensorView<'_>,
    indices: &TensorView<'_>,
    axis: i64,
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let shape = &input.shape;
    let idx_shape = &indices.shape;
    let rank = shape.len();
    let axis = if axis < 0 { (rank as i64 + axis) as usize } else { axis as usize };

    let total: usize = idx_shape.iter().product();
    utils::ensure_capacity(out, total);
    unsafe { out.set_len(total); }

    let data = &input.data;
    let idx_data = &indices.data;

    // Compute strides for input
    let mut strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Compute strides for index tensor
    let mut idx_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        idx_strides[i] = idx_strides[i + 1] * idx_shape[i + 1];
    }

    for flat_idx in 0..total {
        // Decompose flat_idx into multi-dimensional index in idx_shape
        let mut remaining = flat_idx;
        let mut coords = vec![0usize; rank];
        for d in 0..rank {
            coords[d] = remaining / idx_strides[d];
            remaining %= idx_strides[d];
        }

        // Replace axis coordinate with the index value
        let index_val = idx_data[flat_idx] as i64;
        let index_val = if index_val < 0 {
            (shape[axis] as i64 + index_val) as usize
        } else {
            index_val as usize
        };
        coords[axis] = index_val;

        // Compute flat input index
        let mut in_idx = 0;
        for d in 0..rank {
            in_idx += coords[d] * strides[d];
        }

        out[flat_idx] = data[in_idx];
    }

    TensorView::from_slice(out, idx_shape.to_vec())
}

/// 2D Convolution with zero-point subtraction fused into im2col.
/// Avoids allocating separate dequantized copies of input/weights.
/// Computes: (input - x_zp) conv (weight - w_zp) 
fn conv2d_with_zero_points<'b, 'a>(
    input: &TensorView<'b>,
    weights: &TensorView<'b>,
    bias: Option<&TensorView<'b>>,
    dilations: &[i64],
    group: i64,
    pads: &[i64],
    strides: &[i64],
    x_zp: f32,
    w_zp: f32,
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let in_shape = &input.shape;
    let w_shape = &weights.shape;

    assert!(in_shape.len() == 4);
    assert!(w_shape.len() == 4);

    let batch_size = in_shape[0];
    let in_channels = in_shape[1];
    let in_h = in_shape[2];
    let in_w = in_shape[3];

    let out_channels = w_shape[0];
    let kernel_h = w_shape[2];
    let kernel_w = w_shape[3];

    let dilation_h = if dilations.len() >= 2 { dilations[0] as usize } else if dilations.len() == 1 { dilations[0] as usize } else { 1 };
    let dilation_w = if dilations.len() >= 2 { dilations[1] as usize } else if dilations.len() == 1 { dilations[0] as usize } else { 1 };
    let stride_h = if strides.len() >= 2 { strides[0] as usize } else if strides.len() == 1 { strides[0] as usize } else { 1 };
    let stride_w = if strides.len() >= 2 { strides[1] as usize } else if strides.len() == 1 { strides[0] as usize } else { 1 };
    let pad_top = if pads.len() >= 4 { pads[0] as usize } else if pads.len() >= 2 { pads[0] as usize } else { 0 };
    let pad_left = if pads.len() >= 4 { pads[1] as usize } else if pads.len() >= 2 { pads[1] as usize } else { 0 };
    let pad_bottom = if pads.len() >= 4 { pads[2] as usize } else if pads.len() >= 2 { pads[0] as usize } else { 0 };
    let pad_right = if pads.len() >= 4 { pads[3] as usize } else if pads.len() >= 2 { pads[1] as usize } else { 0 };

    let out_h = (in_h + pad_top + pad_bottom - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    let out_w = (in_w + pad_left + pad_right - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    let groups = group as usize;
    let in_channels_per_group = in_channels / groups;
    let out_channels_per_group = out_channels / groups;

    let total_output = batch_size * out_channels * out_h * out_w;
    utils::ensure_capacity(out, total_output);
    unsafe { out.set_len(total_output); }

    let input_data = &input.data;
    let weight_data = &weights.data;

    // Pre-subtract w_zp from weights using thread-local buffer
    thread_local! {
        static W_ADJ_BUF: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::new());
    }
    
    // Use a raw pointer to extend the thread-local borrow safely.
    // Safety: the thread-local buffer lives for the thread's lifetime,
    // and we only access it within this function call (single-threaded context).
    let mut w_adj_ptr: *const f32 = std::ptr::null();
    let w_data: &[f32] = if w_zp != 0.0 {
        let w_len = weight_data.len();
        W_ADJ_BUF.with(|buf_cell| {
            let mut adj = buf_cell.borrow_mut();
            utils::ensure_capacity(&mut adj, w_len);
            unsafe { adj.set_len(w_len); }
            #[cfg(target_arch = "aarch64")]
            unsafe {
                let zp_vec = core::arch::aarch64::vdupq_n_f32(w_zp);
                let src = weight_data.as_ptr();
                let dst = adj.as_mut_ptr();
                let mut i = 0;
                let simd_end = w_len & !15;
                while i < simd_end {
                    let v0 = core::arch::aarch64::vld1q_f32(src.add(i));
                    let v1 = core::arch::aarch64::vld1q_f32(src.add(i + 4));
                    let v2 = core::arch::aarch64::vld1q_f32(src.add(i + 8));
                    let v3 = core::arch::aarch64::vld1q_f32(src.add(i + 12));
                    core::arch::aarch64::vst1q_f32(dst.add(i), core::arch::aarch64::vsubq_f32(v0, zp_vec));
                    core::arch::aarch64::vst1q_f32(dst.add(i + 4), core::arch::aarch64::vsubq_f32(v1, zp_vec));
                    core::arch::aarch64::vst1q_f32(dst.add(i + 8), core::arch::aarch64::vsubq_f32(v2, zp_vec));
                    core::arch::aarch64::vst1q_f32(dst.add(i + 12), core::arch::aarch64::vsubq_f32(v3, zp_vec));
                    i += 16;
                }
                while i < w_len {
                    *dst.add(i) = *src.add(i) - w_zp;
                    i += 1;
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                for i in 0..w_len {
                    adj[i] = weight_data[i] - w_zp;
                }
            }
            w_adj_ptr = adj.as_ptr();
        });
        unsafe { std::slice::from_raw_parts(w_adj_ptr, w_len) }
    } else {
        weight_data
    };
    let col_rows = in_channels_per_group * kernel_h * kernel_w;
    let col_cols = out_h * out_w;

    // Fast path: 1x1 convolution with stride 1, no padding, groups=1
    if kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1
        && pad_top == 0 && pad_left == 0 && pad_bottom == 0 && pad_right == 0
        && groups == 1
    {
        let spatial = in_h * in_w;

        // Thread-local buffer for input zero-point subtraction in 1x1 conv path
        thread_local! {
            static INPUT_ADJ_BUF: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::new());
        }
        
        for n in 0..batch_size {
            let in_offset = n * in_channels * spatial;
            let o_offset = n * out_channels * spatial;

            if x_zp == 0.0 {
                unsafe {
                    let w_ptr = w_data.as_ptr();
                    let in_ptr = input_data.as_ptr().add(in_offset);
                    let out_ptr = out.as_mut_ptr().add(o_offset);

                    let w_mat = MatRef::<f32>::from_raw_parts(
                        w_ptr, out_channels, in_channels, in_channels as isize, 1,
                    );
                    let in_mat = MatRef::<f32>::from_raw_parts(
                        in_ptr, in_channels, spatial, spatial as isize, 1,
                    );
                    let out_mat = MatMut::<f32>::from_raw_parts_mut(
                        out_ptr, out_channels, spatial, spatial as isize, 1,
                    );

                    faer_matmul(out_mat, Accum::Replace, w_mat, in_mat, 1.0, Par::Seq);
                }
            } else {
                INPUT_ADJ_BUF.with(|buf_cell| {
                    let mut input_adj = buf_cell.borrow_mut();
                    let needed = in_channels * spatial;
                    utils::ensure_capacity(&mut input_adj, needed);
                    unsafe { input_adj.set_len(needed); }
                    #[cfg(target_arch = "aarch64")]
                    unsafe {
                        use core::arch::aarch64::*;
                        let zp_vec = vdupq_n_f32(x_zp);
                        let mut i = 0;
                        let simd_end = needed & !3;
                        let src = input_data.as_ptr().add(in_offset);
                        let dst = input_adj.as_mut_ptr();
                        while i < simd_end {
                            let v = vld1q_f32(src.add(i));
                            vst1q_f32(dst.add(i), vsubq_f32(v, zp_vec));
                            i += 4;
                        }
                        while i < needed {
                            *dst.add(i) = *src.add(i) - x_zp;
                            i += 1;
                        }
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        for i in 0..needed {
                            input_adj[i] = input_data[in_offset + i] - x_zp;
                        }
                    }

                    unsafe {
                        let w_ptr = w_data.as_ptr();
                        let in_ptr = input_adj.as_ptr();
                        let out_ptr = out.as_mut_ptr().add(o_offset);

                        let w_mat = MatRef::<f32>::from_raw_parts(
                            w_ptr, out_channels, in_channels, in_channels as isize, 1,
                        );
                        let in_mat = MatRef::<f32>::from_raw_parts(
                            in_ptr, in_channels, spatial, spatial as isize, 1,
                        );
                        let out_mat = MatMut::<f32>::from_raw_parts_mut(
                            out_ptr, out_channels, spatial, spatial as isize, 1,
                        );

                        faer_matmul(out_mat, Accum::Replace, w_mat, in_mat, 1.0, Par::Seq);
                    }
                });
            }

            if let Some(b) = bias {
                for oc in 0..out_channels {
                    let bias_val = b.data[oc];
                    if bias_val != 0.0 {
                        let row_start = o_offset + oc * spatial;
                        for j in 0..spatial {
                            out[row_start + j] += bias_val;
                        }
                    }
                }
            }
        }

        return TensorView::from_slice(out, vec![batch_size, out_channels, out_h, out_w]);
    }

    // General path with im2col
    // Reuse thread-local im2col buffer to avoid repeated allocation
    thread_local! {
        static COL_BUF_ZP: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::new());
    }

    COL_BUF_ZP.with(|buf_cell| {
    let mut col_buf_ref = buf_cell.borrow_mut();
    let needed = col_rows * col_cols;
    if col_buf_ref.len() < needed {
        col_buf_ref.resize(needed, 0.0);
    }
    let col_buf = &mut col_buf_ref[..needed];

    for n in 0..batch_size {
        for g in 0..groups {
            let in_ch_start = g * in_channels_per_group;
            let out_ch_start = g * out_channels_per_group;

            im2col_with_zp(
                input_data, n, in_ch_start, in_channels_per_group,
                in_h, in_w, in_channels,
                kernel_h, kernel_w, stride_h, stride_w,
                pad_top, pad_left, dilation_h, dilation_w,
                out_h, out_w, x_zp,
                col_buf,
            );
            let w_offset = out_ch_start * col_rows;
            let o_offset = (n * out_channels + out_ch_start) * out_h * out_w;

            unsafe {
                let w_ptr = w_data.as_ptr().add(w_offset);
                let col_ptr = col_buf.as_ptr();
                let out_ptr = out.as_mut_ptr().add(o_offset);

                let w_mat = MatRef::<f32>::from_raw_parts(
                    w_ptr, out_channels_per_group, col_rows, col_rows as isize, 1,
                );
                let col_mat = MatRef::<f32>::from_raw_parts(
                    col_ptr, col_rows, col_cols, col_cols as isize, 1,
                );
                let out_mat = MatMut::<f32>::from_raw_parts_mut(
                    out_ptr, out_channels_per_group, col_cols, col_cols as isize, 1,
                );

                faer_matmul(out_mat, Accum::Replace, w_mat, col_mat, 1.0, Par::Seq);
            }
            if let Some(b) = bias {
                for oc in 0..out_channels_per_group {
                    let o_row_start = o_offset + oc * col_cols;
                    let bias_val = b.data[out_ch_start + oc];
                    if bias_val != 0.0 {
                        for j in 0..col_cols {
                            out[o_row_start + j] += bias_val;
                        }
                    }
                }
            }
        }
    }
    }); // end COL_BUF_ZP.with

    TensorView::from_slice(out, vec![batch_size, out_channels, out_h, out_w])
}

/// im2col with zero-point subtraction fused in.
/// Subtracts x_zp from each input element during the im2col operation.
#[inline]
fn im2col_with_zp(
    input: &[f32],
    batch_idx: usize,
    ch_start: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    total_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_left: usize,
    dilation_h: usize,
    dilation_w: usize,
    out_h: usize,
    out_w: usize,
    x_zp: f32,
    col: &mut [f32],
) {
    let spatial_cols = out_h * out_w;
    let batch_offset = batch_idx * total_channels * in_h * in_w;
    let neg_zp = -x_zp; // pad value is (0 - x_zp)

    // Fast path for stride=1, dilation=1
    if stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1 {
        #[cfg(target_arch = "aarch64")]
        let zp_vec = unsafe { core::arch::aarch64::vdupq_n_f32(x_zp) };

        for c in 0..channels {
            let ch_offset = batch_offset + (ch_start + c) * in_h * in_w;
            for kh in 0..kernel_h {
                for kw in 0..kernel_w {
                    let col_row_offset = ((c * kernel_h + kh) * kernel_w + kw) * spatial_cols;

                    let oh_start = if kh < pad_top { pad_top - kh } else { 0 };
                    let oh_end = (in_h + pad_top).saturating_sub(kh).min(out_h);

                    // Zero-point fill for top padding rows
                    unsafe {
                        let col_ptr = col.as_mut_ptr().add(col_row_offset);
                        for oh in 0..oh_start {
                            let base = oh * out_w;
                            for ow in 0..out_w {
                                *col_ptr.add(base + ow) = neg_zp;
                            }
                        }
                    }

                    for oh in oh_start..oh_end {
                        let ih = (oh + kh) - pad_top;
                        let in_row_offset = ch_offset + ih * in_w;
                        let col_base = col_row_offset + oh * out_w;

                        let ow_start = if kw < pad_left { pad_left - kw } else { 0 };
                        let ow_end = (in_w + pad_left).saturating_sub(kw).min(out_w);

                        unsafe {
                            let col_ptr = col.as_mut_ptr().add(col_base);
                            let in_ptr = input.as_ptr().add(in_row_offset);

                            // Left padding
                            for ow in 0..ow_start {
                                *col_ptr.add(ow) = neg_zp;
                            }

                            // Valid region: copy and subtract zero-point
                            let iw_start = (ow_start + kw) - pad_left;
                            let count = ow_end - ow_start;
                            let src = in_ptr.add(iw_start);
                            let dst = col_ptr.add(ow_start);

                            #[cfg(target_arch = "aarch64")]
                            {
                                let mut j = 0;
                                let simd_end16 = count & !15;
                                while j < simd_end16 {
                                    let v0 = core::arch::aarch64::vld1q_f32(src.add(j));
                                    let v1 = core::arch::aarch64::vld1q_f32(src.add(j + 4));
                                    let v2 = core::arch::aarch64::vld1q_f32(src.add(j + 8));
                                    let v3 = core::arch::aarch64::vld1q_f32(src.add(j + 12));
                                    core::arch::aarch64::vst1q_f32(dst.add(j), core::arch::aarch64::vsubq_f32(v0, zp_vec));
                                    core::arch::aarch64::vst1q_f32(dst.add(j + 4), core::arch::aarch64::vsubq_f32(v1, zp_vec));
                                    core::arch::aarch64::vst1q_f32(dst.add(j + 8), core::arch::aarch64::vsubq_f32(v2, zp_vec));
                                    core::arch::aarch64::vst1q_f32(dst.add(j + 12), core::arch::aarch64::vsubq_f32(v3, zp_vec));
                                    j += 16;
                                }
                                while j + 4 <= count {
                                    let v = core::arch::aarch64::vld1q_f32(src.add(j));
                                    core::arch::aarch64::vst1q_f32(dst.add(j), core::arch::aarch64::vsubq_f32(v, zp_vec));
                                    j += 4;
                                }
                                while j < count {
                                    *dst.add(j) = *src.add(j) - x_zp;
                                    j += 1;
                                }
                            }
                            #[cfg(not(target_arch = "aarch64"))]
                            {
                                for j in 0..count {
                                    *dst.add(j) = *src.add(j) - x_zp;
                                }
                            }

                            // Right padding
                            for ow in ow_end..out_w {
                                *col_ptr.add(ow) = neg_zp;
                            }
                        }
                    }

                    // Bottom padding rows
                    unsafe {
                        let col_ptr = col.as_mut_ptr().add(col_row_offset);
                        for oh in oh_end..out_h {
                            let base = oh * out_w;
                            for ow in 0..out_w {
                                *col_ptr.add(base + ow) = neg_zp;
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    // General path
    for c in 0..channels {
        let in_ch = ch_start + c;
        let ch_offset = batch_offset + in_ch * in_h * in_w;
        for kh in 0..kernel_h {
            for kw in 0..kernel_w {
                let col_row = (c * kernel_h + kh) * kernel_w + kw;
                let col_row_offset = col_row * spatial_cols;
                for oh in 0..out_h {
                    let ih = (oh * stride_h + kh * dilation_h) as isize - pad_top as isize;
                    let col_oh_offset = col_row_offset + oh * out_w;
                    if ih < 0 || ih >= in_h as isize {
                        unsafe {
                            let col_ptr = col.as_mut_ptr().add(col_oh_offset);
                            for ow in 0..out_w {
                                *col_ptr.add(ow) = neg_zp;
                            }
                        }
                    } else {
                        let row_offset = ch_offset + ih as usize * in_w;
                        for ow in 0..out_w {
                            let iw = (ow * stride_w + kw * dilation_w) as isize - pad_left as isize;
                            unsafe {
                                let col_idx = col_oh_offset + ow;
                                *col.get_unchecked_mut(col_idx) =
                                    if iw >= 0 && iw < in_w as isize {
                                        *input.get_unchecked(row_offset + iw as usize) - x_zp
                                    } else {
                                        neg_zp
                                    };
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Integer Convolution (ConvInteger) for quantized models.
/// Performs convolution on quantized int8/uint8 inputs with int8 weights, producing int32 output.
/// 
/// This implementation fuses zero-point subtraction into im2col to avoid
/// allocating separate dequantized copies of the entire input and weight tensors.
///
/// Note: The quantization scale is applied separately after this operation.
pub fn conv_integer<'b, 'a>(
    input: &TensorView<'b>,
    weights: &TensorView<'b>,
    x_zero_point: Option<&TensorView<'b>>,
    w_zero_point: Option<&TensorView<'b>>,
    dilations: &[i64],
    group: i64,
    pads: &[i64],
    strides: &[i64],
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let x_zp: f32 = if let Some(zp) = x_zero_point {
        if zp.data.is_empty() { 0.0 } else { zp.data[0] }
    } else {
        0.0
    };
    let w_zp: f32 = if let Some(zp) = w_zero_point {
        if zp.data.is_empty() { 0.0 } else { zp.data[0] }
    } else {
        0.0
    };
    conv2d_with_zero_points(input, weights, None, dilations, group, pads, strides, x_zp, w_zp, out)
}

/// Fused DQL + conv_integer: takes f32 input, internally quantizes using a
/// thread-local scratch buffer, then runs integer convolution.
/// Returns (conv_output, input_scale) where input_scale is the DQL scale factor.
/// This avoids allocating separate DQL output buffers per call.
pub fn conv_integer_from_f32<'b, 'a>(
    input_f32: &[f32],
    input_shape: &[usize],
    weights: &TensorView<'b>,
    w_zero_point: Option<&TensorView<'b>>,
    dilations: &[i64],
    group: i64,
    pads: &[i64],
    strides: &[i64],
    out: &'a mut Vec<f32>,
) -> (TensorView<'a>, f32) {
    let w_zp: f32 = if let Some(zp) = w_zero_point {
        if zp.data.is_empty() { 0.0 } else { zp.data[0] }
    } else {
        0.0
    };

    // Inline DQL: compute scale, zp, and quantize into thread-local buffer
    thread_local! {
        static DQL_BUF: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::new());
    }

    let len = input_f32.len();

    // Find min/max
    #[cfg(target_arch = "aarch64")]
    let (min_val, max_val) = unsafe {
        use core::arch::aarch64::*;
        let mut min0 = vdupq_n_f32(f32::MAX);
        let mut max0 = vdupq_n_f32(f32::MIN);
        let mut min1 = vdupq_n_f32(f32::MAX);
        let mut max1 = vdupq_n_f32(f32::MIN);
        let mut min2 = vdupq_n_f32(f32::MAX);
        let mut max2 = vdupq_n_f32(f32::MIN);
        let mut min3 = vdupq_n_f32(f32::MAX);
        let mut max3 = vdupq_n_f32(f32::MIN);
        let ptr = input_f32.as_ptr();
        let mut i = 0;
        let simd_end16 = len & !15;
        while i < simd_end16 {
            let v0 = vld1q_f32(ptr.add(i));
            let v1 = vld1q_f32(ptr.add(i + 4));
            let v2 = vld1q_f32(ptr.add(i + 8));
            let v3 = vld1q_f32(ptr.add(i + 12));
            min0 = vminq_f32(min0, v0);
            max0 = vmaxq_f32(max0, v0);
            min1 = vminq_f32(min1, v1);
            max1 = vmaxq_f32(max1, v1);
            min2 = vminq_f32(min2, v2);
            max2 = vmaxq_f32(max2, v2);
            min3 = vminq_f32(min3, v3);
            max3 = vmaxq_f32(max3, v3);
            i += 16;
        }
        let mut min_vec = vminq_f32(vminq_f32(min0, min1), vminq_f32(min2, min3));
        let mut max_vec = vmaxq_f32(vmaxq_f32(max0, max1), vmaxq_f32(max2, max3));
        while i + 4 <= len {
            let v = vld1q_f32(ptr.add(i));
            min_vec = vminq_f32(min_vec, v);
            max_vec = vmaxq_f32(max_vec, v);
            i += 4;
        }
        let mut min_v = vminvq_f32(min_vec);
        let mut max_v = vmaxvq_f32(max_vec);
        while i < len {
            min_v = min_v.min(*ptr.add(i));
            max_v = max_v.max(*ptr.add(i));
            i += 1;
        }
        (min_v, max_v)
    };

    #[cfg(not(target_arch = "aarch64"))]
    let (min_val, max_val) = {
        let mut min_v = f32::MAX;
        let mut max_v = f32::MIN;
        for &v in input_f32.iter() {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        (min_v, max_v)
    };

    let adjusted_max = max_val.max(0.0);
    let adjusted_min = min_val.min(0.0);
    let range = (adjusted_max - adjusted_min).max(1e-5);
    let scale = range / 255.0;
    let zp = (-adjusted_min / scale).round().clamp(0.0, 255.0);

    // Quantize into thread-local buffer and immediately run conv
    DQL_BUF.with(|buf_cell| {
        let mut dql_buf = buf_cell.borrow_mut();
        utils::ensure_capacity(&mut dql_buf, len);
        unsafe { dql_buf.set_len(len); }

        // Quantize
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use core::arch::aarch64::*;
            let inv_scale_vec = vdupq_n_f32(1.0 / scale);
            let zp_vec = vdupq_n_f32(zp);
            let zero_vec = vdupq_n_f32(0.0);
            let max_vec = vdupq_n_f32(255.0);
            let src = input_f32.as_ptr();
            let dst = dql_buf.as_mut_ptr();
            let mut i = 0;
            let simd_end16 = len & !15;
            while i < simd_end16 {
                let v0 = vld1q_f32(src.add(i));
                let v1 = vld1q_f32(src.add(i + 4));
                let v2 = vld1q_f32(src.add(i + 8));
                let v3 = vld1q_f32(src.add(i + 12));
                let s0 = vrndnq_f32(vaddq_f32(vmulq_f32(v0, inv_scale_vec), zp_vec));
                let s1 = vrndnq_f32(vaddq_f32(vmulq_f32(v1, inv_scale_vec), zp_vec));
                let s2 = vrndnq_f32(vaddq_f32(vmulq_f32(v2, inv_scale_vec), zp_vec));
                let s3 = vrndnq_f32(vaddq_f32(vmulq_f32(v3, inv_scale_vec), zp_vec));
                vst1q_f32(dst.add(i), vminq_f32(vmaxq_f32(s0, zero_vec), max_vec));
                vst1q_f32(dst.add(i + 4), vminq_f32(vmaxq_f32(s1, zero_vec), max_vec));
                vst1q_f32(dst.add(i + 8), vminq_f32(vmaxq_f32(s2, zero_vec), max_vec));
                vst1q_f32(dst.add(i + 12), vminq_f32(vmaxq_f32(s3, zero_vec), max_vec));
                i += 16;
            }
            while i + 4 <= len {
                let v = vld1q_f32(src.add(i));
                let scaled = vmulq_f32(v, inv_scale_vec);
                let rounded = vrndnq_f32(vaddq_f32(scaled, zp_vec));
                vst1q_f32(dst.add(i), vminq_f32(vmaxq_f32(rounded, zero_vec), max_vec));
                i += 4;
            }
            while i < len {
                *dst.add(i) = (*src.add(i) / scale + zp).round().clamp(0.0, 255.0);
                i += 1;
            }
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            let inv_scale = 1.0 / scale;
            for i in 0..len {
                dql_buf[i] = (input_f32[i] * inv_scale + zp).round().clamp(0.0, 255.0);
            }
        }

        // Create TensorView borrowing the thread-local buffer
        let quantized_view = TensorView::from_slice(&dql_buf, input_shape.to_vec());
        let result = conv2d_with_zero_points(
            &quantized_view, weights, None, dilations, group, pads, strides, zp, w_zp, out,
        );
        // Return the shape from result, dropping the borrow
        let out_shape = result.shape.to_vec();
        (TensorView::from_slice(out, out_shape), scale)
    })
}

/// Fused concat + DQL + conv_integer for 1×1 convolutions: takes multiple f32 source
/// tensors (which would be concatenated along channels), internally quantizes them
/// into a single contiguous buffer, then runs 1×1 integer convolution via GEMM.
/// This avoids materializing the concatenated tensor entirely.
///
/// All source tensors must have the same [N, C_i, H, W] layout where H and W are
/// identical. The concat axis is channels (dim 1).
///
/// Returns (conv_output, input_scale).
pub fn conv_integer_from_f32_multi<'b, 'a>(
    sources: &[&TensorView<'b>],
    weights: &TensorView<'b>,
    w_zero_point: Option<&TensorView<'b>>,
    out: &'a mut Vec<f32>,
) -> (TensorView<'a>, f32) {
    let w_zp: f32 = if let Some(zp) = w_zero_point {
        if zp.data.is_empty() { 0.0 } else { zp.data[0] }
    } else {
        0.0
    };

    // Extract dimensions from first source (all must share N, H, W)
    let batch_size = sources[0].shape[0];
    let height = sources[0].shape[2];
    let width = sources[0].shape[3];
    let spatial = height * width;

    // Collect channel counts from each source
    let source_channels: Vec<usize> = sources.iter().map(|s| s.shape[1]).collect();
    let total_channels: usize = source_channels.iter().sum();
    let total_len = batch_size * total_channels * spatial;

    thread_local! {
        static DQL_BUF_MULTI: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::new());
    }

    // Find global min/max across ALL sources
    #[cfg(target_arch = "aarch64")]
    let (min_val, max_val) = unsafe {
        use core::arch::aarch64::*;
        let mut gmin0 = vdupq_n_f32(f32::MAX);
        let mut gmax0 = vdupq_n_f32(f32::MIN);
        let mut gmin1 = vdupq_n_f32(f32::MAX);
        let mut gmax1 = vdupq_n_f32(f32::MIN);
        let mut gmin2 = vdupq_n_f32(f32::MAX);
        let mut gmax2 = vdupq_n_f32(f32::MIN);
        let mut gmin3 = vdupq_n_f32(f32::MAX);
        let mut gmax3 = vdupq_n_f32(f32::MIN);
        for src in sources.iter() {
            let ptr = src.data.as_ptr();
            let slen = src.data.len();
            let mut i = 0;
            let simd_end16 = slen & !15;
            while i < simd_end16 {
                let v0 = vld1q_f32(ptr.add(i));
                let v1 = vld1q_f32(ptr.add(i + 4));
                let v2 = vld1q_f32(ptr.add(i + 8));
                let v3 = vld1q_f32(ptr.add(i + 12));
                gmin0 = vminq_f32(gmin0, v0);
                gmax0 = vmaxq_f32(gmax0, v0);
                gmin1 = vminq_f32(gmin1, v1);
                gmax1 = vmaxq_f32(gmax1, v1);
                gmin2 = vminq_f32(gmin2, v2);
                gmax2 = vmaxq_f32(gmax2, v2);
                gmin3 = vminq_f32(gmin3, v3);
                gmax3 = vmaxq_f32(gmax3, v3);
                i += 16;
            }
            while i + 4 <= slen {
                let v = vld1q_f32(ptr.add(i));
                gmin0 = vminq_f32(gmin0, v);
                gmax0 = vmaxq_f32(gmax0, v);
                i += 4;
            }
            while i < slen {
                let val = *ptr.add(i);
                if val < vgetq_lane_f32(gmin0, 0) {
                    gmin0 = vsetq_lane_f32(val, gmin0, 0);
                }
                if val > vgetq_lane_f32(gmax0, 0) {
                    gmax0 = vsetq_lane_f32(val, gmax0, 0);
                }
                i += 1;
            }
        }
        let min_vec = vminq_f32(vminq_f32(gmin0, gmin1), vminq_f32(gmin2, gmin3));
        let max_vec = vmaxq_f32(vmaxq_f32(gmax0, gmax1), vmaxq_f32(gmax2, gmax3));
        (vminvq_f32(min_vec), vmaxvq_f32(max_vec))
    };

    #[cfg(not(target_arch = "aarch64"))]
    let (min_val, max_val) = {
        let mut min_v = f32::MAX;
        let mut max_v = f32::MIN;
        for src in sources.iter() {
            for &v in src.data.iter() {
                min_v = min_v.min(v);
                max_v = max_v.max(v);
            }
        }
        (min_v, max_v)
    };

    let adjusted_max = max_val.max(0.0);
    let adjusted_min = min_val.min(0.0);
    let range = (adjusted_max - adjusted_min).max(1e-5);
    let scale = range / 255.0;
    let zp = (-adjusted_min / scale).round().clamp(0.0, 255.0);

    // Quantize from multiple sources into DQL buffer in channel-concat order
    DQL_BUF_MULTI.with(|buf_cell| {
        let mut dql_buf = buf_cell.borrow_mut();
        utils::ensure_capacity(&mut dql_buf, total_len);
        unsafe { dql_buf.set_len(total_len); }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            use core::arch::aarch64::*;
            let inv_scale_vec = vdupq_n_f32(1.0 / scale);
            let zp_vec = vdupq_n_f32(zp);
            let zero_vec = vdupq_n_f32(0.0);
            let max_clamp = vdupq_n_f32(255.0);

            let dst_base = dql_buf.as_mut_ptr();
            for n in 0..batch_size {
                let mut ch_offset = 0usize;
                for (src_idx, src) in sources.iter().enumerate() {
                    let src_ch = source_channels[src_idx];
                    let src_spatial = src_ch * spatial;
                    // Source data for this batch: src.data[n * src_ch * spatial .. (n+1) * src_ch * spatial]
                    let src_ptr = src.data.as_ptr().add(n * src_spatial);
                    // Destination: dql_buf[(n * total_channels + ch_offset) * spatial ..]
                    let dst_ptr = dst_base.add((n * total_channels + ch_offset) * spatial);

                    let copy_len = src_spatial;
                    let mut i = 0;
                    let simd_end16 = copy_len & !15;
                    while i < simd_end16 {
                        let v0 = vld1q_f32(src_ptr.add(i));
                        let v1 = vld1q_f32(src_ptr.add(i + 4));
                        let v2 = vld1q_f32(src_ptr.add(i + 8));
                        let v3 = vld1q_f32(src_ptr.add(i + 12));
                        let s0 = vrndnq_f32(vaddq_f32(vmulq_f32(v0, inv_scale_vec), zp_vec));
                        let s1 = vrndnq_f32(vaddq_f32(vmulq_f32(v1, inv_scale_vec), zp_vec));
                        let s2 = vrndnq_f32(vaddq_f32(vmulq_f32(v2, inv_scale_vec), zp_vec));
                        let s3 = vrndnq_f32(vaddq_f32(vmulq_f32(v3, inv_scale_vec), zp_vec));
                        vst1q_f32(dst_ptr.add(i), vminq_f32(vmaxq_f32(s0, zero_vec), max_clamp));
                        vst1q_f32(dst_ptr.add(i + 4), vminq_f32(vmaxq_f32(s1, zero_vec), max_clamp));
                        vst1q_f32(dst_ptr.add(i + 8), vminq_f32(vmaxq_f32(s2, zero_vec), max_clamp));
                        vst1q_f32(dst_ptr.add(i + 12), vminq_f32(vmaxq_f32(s3, zero_vec), max_clamp));
                        i += 16;
                    }
                    while i + 4 <= copy_len {
                        let v = vld1q_f32(src_ptr.add(i));
                        let s = vrndnq_f32(vaddq_f32(vmulq_f32(v, inv_scale_vec), zp_vec));
                        vst1q_f32(dst_ptr.add(i), vminq_f32(vmaxq_f32(s, zero_vec), max_clamp));
                        i += 4;
                    }
                    while i < copy_len {
                        *dst_ptr.add(i) = (*src_ptr.add(i) / scale + zp).round().clamp(0.0, 255.0);
                        i += 1;
                    }
                    ch_offset += src_ch;
                }
            }
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            let inv_scale = 1.0 / scale;
            for n in 0..batch_size {
                let mut ch_offset = 0usize;
                for (src_idx, src) in sources.iter().enumerate() {
                    let src_ch = source_channels[src_idx];
                    let src_spatial = src_ch * spatial;
                    let src_start = n * src_spatial;
                    let dst_start = (n * total_channels + ch_offset) * spatial;
                    for i in 0..src_spatial {
                        dql_buf[dst_start + i] = (src.data[src_start + i] * inv_scale + zp).round().clamp(0.0, 255.0);
                    }
                    ch_offset += src_ch;
                }
            }
        }

        // Now run 1×1 conv using conv2d_with_zero_points (1×1 fast path)
        let input_shape = vec![batch_size, total_channels, height, width];
        let quantized_view = TensorView::from_slice(&dql_buf, input_shape);
        let result = conv2d_with_zero_points(
            &quantized_view, weights, None,
            &[1, 1],  // dilations (1×1 conv)
            1,         // groups
            &[0, 0, 0, 0], // pads
            &[1, 1],  // strides
            zp, w_zp, out,
        );
        let out_shape = result.shape.to_vec();
        (TensorView::from_slice(out, out_shape), scale)
    })
}

/// Fused scale + bias + SiLU activation applied in-place on conv output.
/// Input shape: [N, C, H, W], scale: scalar, bias: [C]
/// Computes: x = data * scale + bias[c], output = x * sigmoid(x)
#[inline]
pub fn fused_scale_bias_silu(data: &mut [f32], shape: &[usize], scale: f32, bias: &[f32]) {
    let channels = shape[1];
    let spatial = shape[2] * shape[3];
    let batch_size = shape[0];

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let scale_v = vdupq_n_f32(scale);
        let one_v = vdupq_n_f32(1.0);
        let zero_v = vdupq_n_f32(0.0);

        for n in 0..batch_size {
            for c in 0..channels {
                let bias_v = vdupq_n_f32(bias[c]);
                let offset = (n * channels + c) * spatial;
                let ptr = data.as_mut_ptr().add(offset);
                
                let mut i = 0;
                let simd_end = spatial & !15;
                // Process 16 elements at a time (4x unrolled)
                while i < simd_end {
                    let q0 = vld1q_f32(ptr.add(i));
                    let q1 = vld1q_f32(ptr.add(i + 4));
                    let q2 = vld1q_f32(ptr.add(i + 8));
                    let q3 = vld1q_f32(ptr.add(i + 12));
                    let x0 = vaddq_f32(vmulq_f32(q0, scale_v), bias_v);
                    let x1 = vaddq_f32(vmulq_f32(q1, scale_v), bias_v);
                    let x2 = vaddq_f32(vmulq_f32(q2, scale_v), bias_v);
                    let x3 = vaddq_f32(vmulq_f32(q3, scale_v), bias_v);
                    // SiLU using fast NEON exp approximation
                    let neg_x0 = vsubq_f32(zero_v, x0);
                    let neg_x1 = vsubq_f32(zero_v, x1);
                    let neg_x2 = vsubq_f32(zero_v, x2);
                    let neg_x3 = vsubq_f32(zero_v, x3);
                    let e0 = crate::kernels::neon::math::neon_exp_f32x4(neg_x0);
                    let e1 = crate::kernels::neon::math::neon_exp_f32x4(neg_x1);
                    let e2 = crate::kernels::neon::math::neon_exp_f32x4(neg_x2);
                    let e3 = crate::kernels::neon::math::neon_exp_f32x4(neg_x3);
                    let sig0 = vdivq_f32(one_v, vaddq_f32(one_v, e0));
                    let sig1 = vdivq_f32(one_v, vaddq_f32(one_v, e1));
                    let sig2 = vdivq_f32(one_v, vaddq_f32(one_v, e2));
                    let sig3 = vdivq_f32(one_v, vaddq_f32(one_v, e3));
                    vst1q_f32(ptr.add(i), vmulq_f32(x0, sig0));
                    vst1q_f32(ptr.add(i + 4), vmulq_f32(x1, sig1));
                    vst1q_f32(ptr.add(i + 8), vmulq_f32(x2, sig2));
                    vst1q_f32(ptr.add(i + 12), vmulq_f32(x3, sig3));
                    i += 16;
                }
                while i < spatial {
                    let x = *ptr.add(i) * scale + bias[c];
                    *ptr.add(i) = x / (1.0 + (-x).exp());
                    i += 1;
                }
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for n in 0..batch_size {
            for c in 0..channels {
                let b = bias[c];
                let offset = (n * channels + c) * spatial;
                for i in 0..spatial {
                    let x = data[offset + i] * scale + b;
                    data[offset + i] = x / (1.0 + (-x).exp());
                }
            }
        }
    }
}

/// Fused scale + bias applied in-place (no activation).
#[inline]
pub fn fused_scale_bias(data: &mut [f32], shape: &[usize], scale: f32, bias: &[f32]) {
    let channels = shape[1];
    let spatial = shape[2] * shape[3];
    let batch_size = shape[0];

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let scale_v = vdupq_n_f32(scale);
        for n in 0..batch_size {
            for c in 0..channels {
                let bias_v = vdupq_n_f32(bias[c]);
                let offset = (n * channels + c) * spatial;
                let ptr = data.as_mut_ptr().add(offset);
                let mut i = 0;
                let simd_end16 = spatial & !15;
                while i < simd_end16 {
                    let q0 = vld1q_f32(ptr.add(i));
                    let q1 = vld1q_f32(ptr.add(i + 4));
                    let q2 = vld1q_f32(ptr.add(i + 8));
                    let q3 = vld1q_f32(ptr.add(i + 12));
                    vst1q_f32(ptr.add(i), vaddq_f32(vmulq_f32(q0, scale_v), bias_v));
                    vst1q_f32(ptr.add(i + 4), vaddq_f32(vmulq_f32(q1, scale_v), bias_v));
                    vst1q_f32(ptr.add(i + 8), vaddq_f32(vmulq_f32(q2, scale_v), bias_v));
                    vst1q_f32(ptr.add(i + 12), vaddq_f32(vmulq_f32(q3, scale_v), bias_v));
                    i += 16;
                }
                while i + 4 <= spatial {
                    let q = vld1q_f32(ptr.add(i));
                    vst1q_f32(ptr.add(i), vaddq_f32(vmulq_f32(q, scale_v), bias_v));
                    i += 4;
                }
                while i < spatial {
                    *ptr.add(i) = *ptr.add(i) * scale + bias[c];
                    i += 1;
                }
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for n in 0..batch_size {
            for c in 0..channels {
                let b = bias[c];
                let offset = (n * channels + c) * spatial;
                for i in 0..spatial {
                    data[offset + i] = data[offset + i] * scale + b;
                }
            }
        }
    }
}

/// Integer GEMM: C[M×N] = A[M×K] × B[K×N] where A=weights, B=input
/// A is row-major [M, K], B is row-major [K, N]  
/// Uses i16 inputs with i32 accumulation for exact results.
#[inline]
#[allow(dead_code)]
fn conv_integer_gemm_i16(
    input: &[i16],   // [K, N] row-major (input_channels × spatial)
    weights: &[i16], // [M, K] row-major (output_channels × input_channels)
    m: usize,        // output_channels
    k: usize,        // input_channels
    n: usize,        // spatial
    out: &mut [f32],  // [M, N] row-major
) {
    #[cfg(target_arch = "aarch64")]
    {
        conv_integer_gemm_i16_neon(input, weights, m, k, n, out);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        conv_integer_gemm_i16_scalar(input, weights, m, k, n, out);
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn conv_integer_gemm_i16_scalar(
    input: &[i16],
    weights: &[i16],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) {
    for oc in 0..m {
        let w_row = &weights[oc * k..(oc + 1) * k];
        let o_row = &mut out[oc * n..(oc + 1) * n];
        for j in 0..n {
            let mut acc: i32 = 0;
            for ic in 0..k {
                acc += w_row[ic] as i32 * input[ic * n + j] as i32;
            }
            o_row[j] = acc as f32;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
fn conv_integer_gemm_i16_neon(
    input: &[i16],   // [K, N] row-major
    weights: &[i16], // [M, K] row-major
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) {
    // For the 1x1 conv case, weights are [M, K] and input is [K, N].
    // We compute C[i,j] = sum_over_k(W[i,k] * I[k,j])
    // 
    // The input layout is [K, N] row-major, so I[k,j] = input[k*n + j]
    // This means consecutive j values are contiguous in memory for the same k.
    //
    // Strategy: for each output channel, iterate over K, loading 1 weight value
    // and broadcasting it across 8 spatial positions of input.
    
    unsafe {
        use core::arch::aarch64::*;
        
        for oc in 0..m {
            let w_row = weights.as_ptr().add(oc * k);
            let o_row = out.as_mut_ptr().add(oc * n);
            
            let mut j = 0;
            // Process 8 spatial positions at a time
            while j + 8 <= n {
                let mut acc0 = vdupq_n_s32(0);
                let mut acc1 = vdupq_n_s32(0);
                
                for ic in 0..k {
                    let w_val = *w_row.add(ic);
                    let w_vec = vdupq_n_s16(w_val);
                    let in_ptr = input.as_ptr().add(ic * n + j);
                    
                    let in_lo = vld1_s16(in_ptr);          // 4 i16 values
                    let in_hi = vld1_s16(in_ptr.add(4));   // 4 i16 values
                    
                    acc0 = vmlal_s16(acc0, vget_low_s16(w_vec), in_lo);
                    acc1 = vmlal_s16(acc1, vget_low_s16(w_vec), in_hi);
                }
                
                // Convert i32 accumulators to f32 and store
                let f0 = vcvtq_f32_s32(acc0);
                let f1 = vcvtq_f32_s32(acc1);
                vst1q_f32(o_row.add(j), f0);
                vst1q_f32(o_row.add(j + 4), f1);
                
                j += 8;
            }
            
            // Process 4 at a time
            while j + 4 <= n {
                let mut acc0 = vdupq_n_s32(0);
                
                for ic in 0..k {
                    let w_val = *w_row.add(ic);
                    let w_vec = vdup_n_s16(w_val);
                    let in_ptr = input.as_ptr().add(ic * n + j);
                    let in_lo = vld1_s16(in_ptr);
                    acc0 = vmlal_s16(acc0, w_vec, in_lo);
                }
                
                let f0 = vcvtq_f32_s32(acc0);
                vst1q_f32(o_row.add(j), f0);
                j += 4;
            }
            
            // Handle remaining spatial positions
            while j < n {
                let mut acc: i32 = 0;
                for ic in 0..k {
                    acc += *w_row.add(ic) as i32 * *input.as_ptr().add(ic * n + j) as i32;
                }
                *o_row.add(j) = acc as f32;
                j += 1;
            }
        }
    }
}

/// im2col for i16 data (zero-point already subtracted from valid positions)
#[allow(dead_code)]
fn im2col_i16(
    input: &[i16],
    batch_idx: usize,
    ch_start: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    total_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_left: usize,
    dilation_h: usize,
    dilation_w: usize,
    out_h: usize,
    out_w: usize,
    pad_val: i16,
    col: &mut [i16],
) {
    let spatial_cols = out_h * out_w;
    let batch_offset = batch_idx * total_channels * in_h * in_w;

    for c in 0..channels {
        let in_ch = ch_start + c;
        let ch_offset = batch_offset + in_ch * in_h * in_w;
        for kh in 0..kernel_h {
            for kw in 0..kernel_w {
                let col_row = (c * kernel_h + kh) * kernel_w + kw;
                let col_row_offset = col_row * spatial_cols;
                for oh in 0..out_h {
                    let ih = (oh * stride_h + kh * dilation_h) as isize - pad_top as isize;
                    let col_oh_offset = col_row_offset + oh * out_w;
                    if ih < 0 || ih >= in_h as isize {
                        for ow in 0..out_w {
                            col[col_oh_offset + ow] = pad_val;
                        }
                    } else {
                        let row_offset = ch_offset + ih as usize * in_w;
                        for ow in 0..out_w {
                            let iw = (ow * stride_w + kw * dilation_w) as isize - pad_left as isize;
                            if iw >= 0 && iw < in_w as isize {
                                col[col_oh_offset + ow] = input[row_offset + iw as usize];
                            } else {
                                col[col_oh_offset + ow] = pad_val;
                            }
                        }
                    }
                }
            }
        }
    }
}
