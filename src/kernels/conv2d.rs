#![allow(unsafe_op_in_unsafe_fn)]
use crate::kernels::bias_act::bias_act_inplace;
use crate::kernels::timing;
use crate::kernels::utils;
#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
use crate::kernels::matmul::{Accum, MatMut, MatRef, Par, matmul as strided_matmul};
use crate::kernels::window2d::{self, Window};
use crate::tensor::TensorView;
use fearless_simd::Level;

// Apple Accelerate framework bindings for AMX-accelerated GEMM on macOS aarch64
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
mod accelerate {
    pub const CBLAS_ROW_MAJOR: i32 = 101;
    pub const CBLAS_NO_TRANS: i32 = 111;
    pub const CBLAS_TRANS: i32 = 112;

    unsafe extern "C" {
        pub fn cblas_sgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }
}

/// Thin wrapper: C = A * B (row-major, no-transpose, no bias accumulation)
/// A: [M, K], B: [K, N], C: [M, N]
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
#[inline(always)]
unsafe fn accel_sgemm(m: usize, n: usize, k: usize, a: *const f32, b: *const f32, c: *mut f32) {
    unsafe {
        accelerate::cblas_sgemm(
            accelerate::CBLAS_ROW_MAJOR,
            accelerate::CBLAS_NO_TRANS,
            accelerate::CBLAS_NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0_f32,
            a,
            k as i32,
            b,
            n as i32,
            0.0_f32,
            c,
            n as i32,
        );
    }
}

pub fn print_conv_stats() {}
/// C = A^T * B (row-major, A transposed).
/// A stored as [k, m] row-major, B as [k, n], C as [m, n].
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
#[inline(always)]
unsafe fn accel_sgemm_ta(m: usize, n: usize, k: usize, a: *const f32, b: *const f32, c: *mut f32) {
    unsafe {
        accelerate::cblas_sgemm(
            accelerate::CBLAS_ROW_MAJOR,
            accelerate::CBLAS_TRANS,
            accelerate::CBLAS_NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0_f32,
            a,
            m as i32, // lda = number of cols in A as stored in memory (A stored as [k, m])
            b,
            n as i32,
            0.0_f32,
            c,
            n as i32,
        );
    }
}

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
    return conv2d_fused(
        input, weights, bias, dilations, group, pads, strides, false, out,
    );
}

/// 2D Convolution with fused SiLU activation (x * sigmoid(x)).
/// Avoids separate sigmoid + mul passes over the output.
pub fn conv2d_silu<'b, 'a>(
    input: &TensorView<'b>,
    weights: &TensorView<'b>,
    bias: Option<&TensorView<'b>>,
    dilations: &[i64],
    group: i64,
    pads: &[i64],
    strides: &[i64],
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    return conv2d_activation(
        Level::new(),
        input,
        weights,
        bias,
        dilations,
        group,
        pads,
        strides,
        Activation::SiLU,
        out,
    );
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Activation {
    None,
    Relu,
    SiLU,
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
    let act = if relu {
        Activation::Relu
    } else {
        Activation::None
    };
    return conv2d_activation(
        Level::new(),
        input,
        weights,
        bias,
        dilations,
        group,
        pads,
        strides,
        act,
        out,
    );
}

fn conv2d_activation<'b, 'a>(
    level: Level,
    input: &TensorView<'b>,
    weights: &TensorView<'b>,
    bias: Option<&TensorView<'b>>,
    dilations: &[i64],
    group: i64,
    pads: &[i64],
    strides: &[i64],
    act: Activation,
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let _relu = act == Activation::Relu;
    let _t0 = if timing::TIMING_ENABLED {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let in_shape = &input.shape;
    let w_shape = &weights.shape;

    assert!(
        in_shape.len() == 4,
        "Conv2d: expected rank-4 input [N,C,H,W], got rank {}",
        in_shape.len()
    );
    assert!(
        w_shape.len() == 4,
        "Conv2d: expected rank-4 weight [C_out,C_in/g,kH,kW], got rank {}",
        w_shape.len()
    );

    let batch_size = in_shape[0];
    let in_channels = in_shape[1];
    let in_h = in_shape[2];
    let in_w = in_shape[3];

    let out_channels = w_shape[0];
    let kernel_h = w_shape[2];
    let kernel_w = w_shape[3];

    let dilation_h = if dilations.len() >= 2 {
        dilations[0] as usize
    } else if dilations.len() == 1 {
        dilations[0] as usize
    } else {
        1
    };
    let dilation_w = if dilations.len() >= 2 {
        dilations[1] as usize
    } else if dilations.len() == 1 {
        dilations[0] as usize
    } else {
        1
    };

    let stride_h = if strides.len() >= 2 {
        strides[0] as usize
    } else if strides.len() == 1 {
        strides[0] as usize
    } else {
        1
    };
    let stride_w = if strides.len() >= 2 {
        strides[1] as usize
    } else if strides.len() == 1 {
        strides[0] as usize
    } else {
        1
    };

    let pad_top = if pads.len() >= 4 {
        pads[0] as usize
    } else if pads.len() >= 2 {
        pads[0] as usize
    } else {
        0
    };
    let pad_left = if pads.len() >= 4 {
        pads[1] as usize
    } else if pads.len() >= 2 {
        pads[1] as usize
    } else {
        0
    };
    let pad_bottom = if pads.len() >= 4 {
        pads[2] as usize
    } else if pads.len() >= 2 {
        pads[0] as usize
    } else {
        0
    };
    let pad_right = if pads.len() >= 4 {
        pads[3] as usize
    } else if pads.len() >= 2 {
        pads[1] as usize
    } else {
        0
    };

    let out_h = (in_h as u64 + pad_top as u64 + pad_bottom as u64
        - dilation_h as u64 * (kernel_h as u64 - 1)
        - 1) as usize
        / stride_h
        + 1;
    let out_w = (in_w as u64 + pad_left as u64 + pad_right as u64
        - dilation_w as u64 * (kernel_w as u64 - 1)
        - 1) as usize
        / stride_w
        + 1;

    assert!(
        out_h > 0 && out_w > 0,
        "conv2d: output dimensions must be positive, got out_h={} out_w={}",
        out_h,
        out_w
    );
    let total_output = batch_size as u64 * out_channels as u64 * out_h as u64 * out_w as u64;
    assert!(
        total_output <= isize::MAX as u64,
        "conv2d: output size overflow"
    );
    let total_output = total_output as usize;

    let groups = group as usize;
    let in_channels_per_group = in_channels / groups;
    let out_channels_per_group = out_channels / groups;

    utils::ensure_capacity(out, total_output);
    unsafe {
        out.set_len(total_output);
    }

    let input_data = &input.data;
    let weight_data = &weights.data;

    // Fast path: 1x1 convolution with stride 1, no padding, groups=1
    if kernel_h == 1
        && kernel_w == 1
        && stride_h == 1
        && stride_w == 1
        && pad_top == 0
        && pad_left == 0
        && pad_bottom == 0
        && pad_right == 0
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

                #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
                accel_sgemm(out_channels, spatial, in_channels, w_ptr, in_ptr, out_ptr);

                #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
                {
                    let w_mat = MatRef::<f32>::from_raw_parts(
                        w_ptr,
                        out_channels,
                        in_channels,
                        in_channels as isize,
                        1,
                    );
                    let in_mat = MatRef::<f32>::from_raw_parts(
                        in_ptr,
                        in_channels,
                        spatial,
                        spatial as isize,
                        1,
                    );
                    let out_mat = MatMut::<f32>::from_raw_parts_mut(
                        out_ptr,
                        out_channels,
                        spatial,
                        spatial as isize,
                        1,
                    );
                    strided_matmul(out_mat, Accum::Replace, w_mat, in_mat, 1.0, Par::Seq);
                }
            }

            // Apply bias and optional activation
            for oc in 0..out_channels {
                let bias_val = if let Some(b) = bias { b.data[oc] } else { 0.0 };
                if bias_val != 0.0 || act != Activation::None {
                    let row_start = o_offset + oc * spatial;
                    bias_act_inplace(&mut out[row_start..row_start + spatial], bias_val, act);
                }
            }
        }
        return {
            if timing::TIMING_ENABLED {
                let ns = _t0.unwrap().elapsed().as_nanos() as u64;
                timing::CONV1X1_NS.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
            }
            TensorView::from_slice(out, vec![batch_size, out_channels, out_h, out_w])
        };
    }

    // Depthwise: every channel convolved with its own kernel, straight from the input.
    if groups == in_channels && in_channels_per_group == 1 && out_channels_per_group == 1 {
        let window = Window {
            channels: in_channels,
            in_h,
            in_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            pad_top,
            pad_left,
            out_h,
            out_w,
        };
        let bias = bias.map(|b| &b.data[..]);
        window2d::depthwise(level, &window, input_data, weight_data, bias, act, out);
        if timing::TIMING_ENABLED {
            let ns = _t0.unwrap().elapsed().as_nanos() as u64;
            timing::CONV_DW_NS.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
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
                    n,
                    in_ch_start,
                    in_channels_per_group,
                    in_h,
                    in_w,
                    in_channels,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    pad_top,
                    pad_left,
                    dilation_h,
                    dilation_w,
                    out_h,
                    out_w,
                    col_buf,
                );

                // GEMM: weight_matrix [out_channels_per_group, col_rows] x col_matrix [col_rows, col_cols]
                // = output [out_channels_per_group, out_h * out_w]
                let w_offset = out_ch_start * (in_channels_per_group * kernel_h * kernel_w);
                let o_offset = (n * out_channels + out_ch_start) * out_h * out_w;

                // GEMM over the im2col columns
                unsafe {
                    let w_ptr = weight_data.as_ptr().add(w_offset);
                    let col_ptr = col_buf.as_ptr();
                    let out_ptr = out.as_mut_ptr().add(o_offset);

                    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
                    accel_sgemm(
                        out_channels_per_group,
                        col_cols,
                        col_rows,
                        w_ptr,
                        col_ptr,
                        out_ptr,
                    );

                    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
                    {
                        // Weight: [out_channels_per_group, col_rows] row-major
                        let w_mat = MatRef::<f32>::from_raw_parts(
                            w_ptr,
                            out_channels_per_group,
                            col_rows,
                            col_rows as isize,
                            1,
                        );
                        // Col: [col_rows, col_cols] row-major
                        let col_mat = MatRef::<f32>::from_raw_parts(
                            col_ptr,
                            col_rows,
                            col_cols,
                            col_cols as isize,
                            1,
                        );
                        // Out: [out_channels_per_group, col_cols] row-major
                        let out_mat = MatMut::<f32>::from_raw_parts_mut(
                            out_ptr,
                            out_channels_per_group,
                            col_cols,
                            col_cols as isize,
                            1,
                        );
                        strided_matmul(out_mat, Accum::Replace, w_mat, col_mat, 1.0, Par::Seq);
                    }
                }

                // Apply bias and optional activation
                for oc in 0..out_channels_per_group {
                    let bias_val = if let Some(b) = bias {
                        b.data[out_ch_start + oc]
                    } else {
                        0.0
                    };
                    if bias_val != 0.0 || act != Activation::None {
                        let row_start = o_offset + oc * col_cols;
                        bias_act_inplace(&mut out[row_start..row_start + col_cols], bias_val, act);
                    }
                }
            }
        }
    }); // COL_BUF.with

    if timing::TIMING_ENABLED {
        let ns = _t0.unwrap().elapsed().as_nanos() as u64;
        let kh = weights.shape[2];
        let kw = weights.shape[3];
        if kh == 3 && kw == 3 {
            timing::CONV3X3_NS.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
        } else {
            timing::CONV_OTHER_NS.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
        }
    }
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

    // Row-contiguous path: any row stride, unit column stride and dilation. One
    // output row is then a single memcpy out of one input row, so the strided
    // downsampling convolutions in a ResNet stem never touch the per-element
    // general path below.
    if stride_w == 1 && dilation_h == 1 && dilation_w == 1 {
        for c in 0..channels {
            let ch_offset = batch_offset + (ch_start + c) * in_h * in_w;
            for kh in 0..kernel_h {
                for kw in 0..kernel_w {
                    let col_row_offset = ((c * kernel_h + kh) * kernel_w + kw) * spatial_cols;

                    // Columns whose source `iw = ow + kw - pad_left` is in range.
                    let ow_start = pad_left.saturating_sub(kw).min(out_w);
                    let ow_end = (in_w + pad_left).saturating_sub(kw).min(out_w);
                    let iw_start = (ow_start + kw).saturating_sub(pad_left);
                    let count = ow_end.saturating_sub(ow_start);

                    for oh in 0..out_h {
                        let ih = (oh * stride_h + kh * dilation_h) as isize - pad_top as isize;
                        let col_base = col_row_offset + oh * out_w;
                        unsafe {
                            let col_ptr = col.as_mut_ptr().add(col_base);
                            if ih < 0 || ih >= in_h as isize {
                                std::ptr::write_bytes(col_ptr, 0, out_w);
                                continue;
                            }
                            let in_ptr = input.as_ptr().add(ch_offset + ih as usize * in_w);
                            if ow_start > 0 {
                                std::ptr::write_bytes(col_ptr, 0, ow_start);
                            }
                            if count > 0 {
                                std::ptr::copy_nonoverlapping(
                                    in_ptr.add(iw_start),
                                    col_ptr.add(ow_start),
                                    count,
                                );
                            }
                            if ow_end < out_w {
                                std::ptr::write_bytes(col_ptr.add(ow_end), 0, out_w - ow_end);
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    // General path for non-unit column stride/dilation
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
                            std::ptr::write_bytes(col_ptr, 0, out_w);
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
    max_pool2d_at(Level::new(), input, kernel_shape, strides, pads, dilations, ceil_mode, out)
}

fn max_pool2d_at<'b, 'a>(
    level: Level,
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
    let kw = if kernel_shape.len() > 1 {
        kernel_shape[1] as usize
    } else {
        kh
    };

    let sh = if strides.is_empty() {
        1
    } else {
        strides[0] as usize
    };
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

    let dh = if dilations.is_empty() {
        1
    } else {
        dilations[0] as usize
    };
    let dw = if dilations.len() > 1 {
        dilations[1] as usize
    } else {
        dh
    };

    let effective_kh = dh * (kh - 1) + 1;
    let effective_kw = dw * (kw - 1) + 1;

    let out_h = if ceil_mode {
        ((in_h as i64 + pad_top as i64 + pad_bottom as i64 - effective_kh as i64 + sh as i64 - 1)
            / sh as i64
            + 1) as usize
    } else {
        ((in_h as i64 + pad_top as i64 + pad_bottom as i64 - effective_kh as i64) / sh as i64 + 1)
            as usize
    };
    let out_w = if ceil_mode {
        ((in_w as i64 + pad_left as i64 + pad_right as i64 - effective_kw as i64 + sw as i64 - 1)
            / sw as i64
            + 1) as usize
    } else {
        ((in_w as i64 + pad_left as i64 + pad_right as i64 - effective_kw as i64) / sw as i64 + 1)
            as usize
    };

    let total = batch as u64 * channels as u64 * out_h as u64 * out_w as u64;
    assert!(
        total <= isize::MAX as u64,
        "pool/compute output size overflow"
    );
    let total = total as usize;
    utils::ensure_capacity(out, total);
    unsafe {
        out.set_len(total);
    }

    let window = Window {
        channels,
        in_h,
        in_w,
        kernel_h: kh,
        kernel_w: kw,
        stride_h: sh,
        stride_w: sw,
        dilation_h: dh,
        dilation_w: dw,
        pad_top,
        pad_left,
        out_h,
        out_w,
    };
    window2d::max_pool(level, &window, &input.data, out);
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
    let _t0 = if crate::kernels::timing::TIMING_ENABLED {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let r = resize_nearest_inner(input, scales, sizes, coordinate_transform_mode, out);
    if crate::kernels::timing::TIMING_ENABLED {
        crate::kernels::timing::RESIZE_NS.fetch_add(
            _t0.unwrap().elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    r
}
fn resize_nearest_inner<'b, 'a>(
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
        assert!(
            sizes.len() >= 4,
            "Resize: sizes must have at least 4 elements"
        );
        assert!(
            sizes[2] > 0 && sizes[3] > 0,
            "Resize: sizes H and W must be positive"
        );
        (sizes[2] as u64, sizes[3] as u64)
    } else if let Some(scales) = scales {
        // scales is [N, C, H, W]
        let sh = if scales.len() >= 3 { scales[2] } else { 1.0 };
        let sw = if scales.len() >= 4 { scales[3] } else { 1.0 };
        assert!(sh > 0.0 && sw > 0.0, "Resize: scales must be positive");
        (
            (in_h as f64 * sh as f64) as u64,
            (in_w as f64 * sw as f64) as u64,
        )
    } else {
        panic!("Resize: either scales or sizes must be provided");
    };

    assert!(
        out_h > 0 && out_w > 0,
        "Resize: output dimensions must be positive, got out_h={} out_w={}",
        out_h,
        out_w
    );
    let total = batch as u64 * channels as u64 * out_h * out_w;
    assert!(
        total <= isize::MAX as u64,
        "Resize: output size overflow ({}x{}x{}x{})",
        batch,
        channels,
        out_h,
        out_w
    );
    let total = total as usize;
    let out_h = out_h as usize;
    let out_w = out_w as usize;
    utils::ensure_capacity(out, total);
    unsafe {
        out.set_len(total);
    }

    let data = &input.data;

    let h_scale = in_h as f32 / out_h as f32;
    let w_scale = in_w as f32 / out_w as f32;

    let asym = coordinate_transform_mode == "asymmetric";

    let row_map: Vec<usize> = (0..out_h)
        .map(|oh| {
            if asym {
                (oh as f32 * h_scale).floor().min((in_h - 1) as f32) as usize
            } else {
                ((oh as f32 + 0.5) * h_scale - 0.5)
                    .round()
                    .max(0.0)
                    .min((in_h - 1) as f32) as usize
            }
        })
        .collect();

    let col_map: Vec<usize> = (0..out_w)
        .map(|ow| {
            if asym {
                (ow as f32 * w_scale).floor().min((in_w - 1) as f32) as usize
            } else {
                ((ow as f32 + 0.5) * w_scale - 0.5)
                    .round()
                    .max(0.0)
                    .min((in_w - 1) as f32) as usize
            }
        })
        .collect();

    for n in 0..batch {
        for c in 0..channels {
            let in_offset = (n * channels + c) * in_h * in_w;
            let out_offset = (n * channels + c) * out_h * out_w;
            for oh in 0..out_h {
                let ih = row_map[oh];
                let in_row = in_offset + ih * in_w;
                let out_row = out_offset + oh * out_w;
                for ow in 0..out_w {
                    out[out_row + ow] = data[in_row + col_map[ow]];
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
        let mut pairs: Vec<(usize, f32)> =
            (0..last_dim).map(|i| (i, data[row_start + i])).collect();

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
pub fn gather_elements<'a, 'b, T: Clone + Copy + std::fmt::Debug, U: crate::kernels::ElementOps>(
    input: &TensorView<'b, T>,
    indices: &TensorView<'b, U>,
    axis: i64,
    out: &'a mut Vec<T>,
) -> TensorView<'a, T> {
    let shape = &input.shape;
    let idx_shape = &indices.shape;
    let rank = shape.len();
    let axis = if axis < 0 {
        (rank as i64 + axis) as usize
    } else {
        axis as usize
    };

    let total: usize = idx_shape.iter().product();
    utils::ensure_capacity(out, total);
    unsafe {
        out.set_len(total);
    }

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
        let index_val = idx_data[flat_idx].as_f32() as i64;
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

/// Output shape: [N, C_out, H_out, W_out]
///
/// Output size formula:
/// output_size = (input_size - 1) * strides - 2 * pads + dilations * (kernel_size - 1) + output_padding + 1
pub fn conv_transpose<'b, 'a>(
    input: &TensorView<'b>,
    weights: &TensorView<'b>,
    bias: Option<&TensorView<'b>>,
    dilations: &[i64],
    group: i64,
    pads: &[i64],
    strides: &[i64],
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let _t0 = if crate::kernels::timing::TIMING_ENABLED {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let r = if input.shape.len() == 3 {
        conv_transpose_1d(input, weights, bias, dilations, group, pads, strides, out)
    } else {
        conv_transpose_inner(input, weights, bias, dilations, group, pads, strides, out)
    };
    if crate::kernels::timing::TIMING_ENABLED {
        crate::kernels::timing::CONV_TRANS_NS.fetch_add(
            _t0.unwrap().elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    r
}
fn conv_transpose_1d<'b, 'a>(
    input: &TensorView<'b>,
    weights: &TensorView<'b>,
    bias: Option<&TensorView<'b>>,
    dilations: &[i64],
    group: i64,
    pads: &[i64],
    strides: &[i64],
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let in_shape = &input.shape;
    let w_shape = &weights.shape;

    // Input: [N, C_in, L_in]
    // Weights: [C_in, C_out/group, K]
    let batch_size = in_shape[0];
    let in_channels = in_shape[1];
    let l_in = in_shape[2];

    let out_channels_per_group = w_shape[1] as usize;
    let kernel = w_shape[2] as usize;
    let group = group as usize;
    let out_channels = out_channels_per_group * group;
    let in_channels_per_group = in_channels / group;

    let stride = strides.get(0).copied().unwrap_or(1) as usize;
    let dilation = dilations.get(0).copied().unwrap_or(1) as usize;
    let pad_begin = pads.get(0).copied().unwrap_or(0) as usize;
    let pad_end = pads.get(1).copied().unwrap_or(0) as usize;

    let l_out = (l_in as i64 - 1) * stride as i64
        - (pad_begin as i64 + pad_end as i64)
        + dilation as i64 * (kernel as i64 - 1)
        + 1;
    assert!(
        l_out > 0,
        "conv_transpose_1d: output length must be positive, got {}",
        l_out
    );
    let l_out = l_out as usize;

    let out_size = batch_size * out_channels * l_out;
    if out.len() != out_size {
        out.resize(out_size, 0.0);
    }
    out[..out_size].fill(0.0);

    // Weight layout: [C_in, C_out_per_group, K] (C-order)
    let w_stride_oc = kernel;

    for n in 0..batch_size {
        for g in 0..group {
            for ic_local in 0..in_channels_per_group {
                let ic_global = g * in_channels_per_group + ic_local;
                let in_base = n * in_channels * l_in + ic_global * l_in;

                for oc_local in 0..out_channels_per_group {
                    let oc_global = g * out_channels_per_group + oc_local;
                    let out_base = n * out_channels * l_out + oc_global * l_out;
                    let w_base = ic_global * out_channels_per_group * kernel + oc_local * w_stride_oc;

                    for l in 0..l_in {
                        let in_val = input.data[in_base + l];
                        for k in 0..kernel {
                            let v = l * stride + k * dilation;
                            if v >= pad_begin && v < l_out + pad_begin {
                                let v_valid = v - pad_begin;
                                out[out_base + v_valid] += weights.data[w_base + k] * in_val;
                            }
                        }
                    }
                }
            }
        }

        if let Some(b) = bias {
            for oc in 0..out_channels {
                let bv = b.data[oc];
                let ch_off = n * out_channels * l_out + oc * l_out;
                for i in 0..l_out {
                    out[ch_off + i] += bv;
                }
            }
        }
    }

    TensorView::from_slice(out, vec![batch_size, out_channels, l_out])
}

fn conv_transpose_inner<'b, 'a>(
    input: &TensorView<'b>,
    weights: &TensorView<'b>,
    bias: Option<&TensorView<'b>>,
    dilations: &[i64],
    group: i64,
    pads: &[i64],
    strides: &[i64],
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let in_shape = &input.shape;
    let w_shape = &weights.shape;

    assert!(
        in_shape.len() == 4,
        "ConvTranspose: expected rank-4 input [N,C,H,W], got rank {}",
        in_shape.len()
    );
    assert!(
        w_shape.len() == 4,
        "ConvTranspose: expected rank-4 weight [C_in,C_out/g,kH,kW], got rank {}",
        w_shape.len()
    );

    let batch_size = in_shape[0];
    let in_channels = in_shape[1];
    let in_h = in_shape[2];
    let in_w = in_shape[3];

    let kernel_h = w_shape[2] as usize;
    let kernel_w = w_shape[3] as usize;
    let out_channels = (w_shape[1] as i64 * group) as usize;

    let stride_h = strides.get(0).copied().unwrap_or(1) as usize;
    let stride_w = strides.get(1).copied().unwrap_or(1) as usize;
    let dilation_h = dilations.get(0).copied().unwrap_or(1) as usize;
    let dilation_w = dilations.get(1).copied().unwrap_or(1) as usize;
    let pad_top = pads.get(0).copied().unwrap_or(0) as usize;
    let pad_left = pads.get(1).copied().unwrap_or(0) as usize;
    let pad_bottom = pads.get(2).copied().unwrap_or(pad_top as i64) as usize;
    let pad_right = pads.get(3).copied().unwrap_or(pad_left as i64) as usize;

    // Calculate output size: (in - 1) * stride - 2*pad + dilation * (kernel - 1) + output_padding + 1
    let out_h = (in_h as i64 - 1) * stride_h as i64 - (pad_top as i64 + pad_bottom as i64)
        + dilation_h as i64 * (kernel_h as i64 - 1)
        + 1;
    let out_w = (in_w as i64 - 1) * stride_w as i64 - (pad_left as i64 + pad_right as i64)
        + dilation_w as i64 * (kernel_w as i64 - 1)
        + 1;
    assert!(
        out_h > 0 && out_w > 0,
        "conv_transpose: output dimensions must be positive, got out_h={} out_w={}",
        out_h,
        out_w
    );
    let out_h = out_h as usize;
    let out_w = out_w as usize;
    let out_size = batch_size * out_channels * out_h * out_w;
    if out.len() != out_size {
        out.resize(out_size, 0.0);
    }
    // Always zero the entire output buffer before accumulation,
    // since resize only fills NEW elements (old data may remain).
    out[..out_size].fill(0.0);

    // Groups are not supported for simplicity in this implementation
    assert!(group == 1, "ConvTranspose: group > 1 not supported yet");

    // GEMM + col2im:
    // col[oc*kh*kw, ih*iw] = W^T[oc*kh*kw, ic] * input[ic, ih*iw]
    // then scatter col into output according to stride/dilation/padding.
    let hw = in_h * in_w;
    let col_rows = out_channels * kernel_h * kernel_w;

    thread_local! {
        static CT_COL_BUF: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::new());
    }
    let col_len = col_rows * hw;
    CT_COL_BUF.with(|buf_cell| {
        let mut col_ref = buf_cell.borrow_mut();
        if col_ref.len() < col_len {
            col_ref.resize(col_len, 0.0);
        }
        let col = &mut col_ref[..col_len];

    for n in 0..batch_size {
        let batch_offset = n * out_channels * out_h * out_w;
        let input_ptr = unsafe { input.data.as_ptr().add(n * in_channels * hw) };

        // GEMM: col[col_rows x hw] = W^T[col_rows x in_channels] * input[in_channels x hw]
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        unsafe {
            accel_sgemm_ta(
                col_rows,
                hw,
                in_channels,
                weights.data.as_ptr(),
                input_ptr,
                col.as_mut_ptr(),
            );
        }
        #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
        {
            for r in 0..col_rows {
                let oc = r / (kernel_h * kernel_w);
                let k_idx = r % (kernel_h * kernel_w);
                let kh = k_idx / kernel_w;
                let kw = k_idx % kernel_w;
                for ih in 0..in_h {
                    for iw in 0..in_w {
                        let mut sum = 0.0f32;
                        for ic in 0..in_channels {
                            let w_val = weights.data
                                [ic * col_rows + oc * kernel_h * kernel_w + kh * kernel_w + kw];
                            let in_val =
                                input.data[n * in_channels * hw + ic * hw + ih * in_w + iw];
                            sum += w_val * in_val;
                        }
                        col[r * hw + ih * in_w + iw] = sum;
                    }
                }
            }
        }

        for oc in 0..out_channels {
            let oc_out_base = batch_offset + oc * out_h * out_w;
            for kh in 0..kernel_h {
                let oh_shift = kh * dilation_h;
                for kw in 0..kernel_w {
                    let ow_shift = kw * dilation_w;
                    let r = oc * kernel_h * kernel_w + kh * kernel_w + kw;
                    let col_row_base = r * hw;
                    for ih in 0..in_h {
                        let oh = ih * stride_h + oh_shift;
                        if oh >= pad_top && oh < out_h + pad_top {
                            let oh_valid = oh - pad_top;
                            for iw in 0..in_w {
                                let ow = iw * stride_w + ow_shift;
                                if ow >= pad_left && ow < out_w + pad_left {
                                    let ow_valid = ow - pad_left;
                                    out[oc_out_base + oh_valid * out_w + ow_valid] +=
                                        col[col_row_base + ih * in_w + iw];
                                }
                            }
                        }
                    }
                }
            }
        }

        if let Some(b) = bias {
            for oc in 0..out_channels {
                let ch_offset = batch_offset + oc * out_h * out_w;
                let bv = b.data[oc];
                for i in 0..out_h * out_w {
                    out[ch_offset + i] += bv;
                }
            }
        }
    }

    }); // end CT_COL_BUF.with

    TensorView::from_slice(out, vec![batch_size, out_channels, out_h, out_w])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::test_util::{assert_close, levels, Rng};

    /// `[channels, h, w]`, `[kh, kw]`, strides, dilations and pads `[top, left, bottom, right]`.
    type WindowCase = ([usize; 3], [usize; 2], [usize; 2], [usize; 2], [usize; 4]);

    /// Every depthwise and pooling path: 3x3 and 5x5 "same" windows, stride 2,
    /// stride 3, dilation, asymmetric padding, rows shorter than a vector and
    /// planes shorter than the kernel.
    const WINDOW_CASES: &[WindowCase] = &[
        ([3, 7, 9], [3, 3], [1, 1], [1, 1], [1, 1, 1, 1]),
        ([2, 20, 20], [3, 3], [1, 1], [1, 1], [1, 1, 1, 1]),
        ([2, 12, 21], [5, 5], [1, 1], [1, 1], [2, 2, 2, 2]),
        ([2, 10, 18], [5, 5], [1, 1], [1, 1], [2, 2, 2, 2]),
        ([2, 6, 40], [5, 5], [1, 1], [1, 1], [2, 2, 2, 2]),
        ([4, 9, 33], [3, 3], [2, 2], [1, 1], [1, 1, 1, 1]),
        ([2, 8, 17], [3, 3], [2, 2], [1, 1], [1, 1, 1, 1]),
        ([2, 6, 40], [2, 2], [2, 2], [1, 1], [0, 0, 0, 0]),
        ([2, 5, 40], [3, 3], [1, 1], [2, 2], [2, 2, 2, 2]),
        ([2, 6, 40], [3, 3], [2, 2], [2, 2], [2, 2, 2, 2]),
        ([2, 9, 20], [3, 3], [2, 1], [1, 1], [1, 1, 1, 1]),
        ([2, 7, 37], [3, 3], [1, 2], [1, 1], [1, 1, 1, 1]),
        ([1, 6, 50], [3, 5], [3, 3], [1, 1], [0, 1, 2, 2]),
        ([3, 5, 6], [1, 1], [1, 1], [1, 1], [0, 0, 0, 0]),
        ([2, 3, 3], [3, 3], [1, 1], [1, 1], [1, 1, 1, 1]),
        ([1, 3, 70], [7, 7], [1, 1], [1, 1], [3, 3, 3, 3]),
        ([2, 4, 34], [2, 2], [1, 1], [1, 1], [0, 0, 1, 1]),
    ];

    fn out_dims(case: &WindowCase) -> (usize, usize) {
        let ([_, h, w], [kh, kw], [sh, sw], [dh, dw], [pt, pl, pb, pr]) = *case;
        ((h + pt + pb - dh * (kh - 1) - 1) / sh + 1, (w + pl + pr - dw * (kw - 1) - 1) / sw + 1)
    }

    /// Each output's valid taps `(input index, tap index)` for one channel plane.
    fn window_taps(case: &WindowCase, oh: usize, ow: usize) -> Vec<(usize, usize)> {
        let ([_, h, w], [kh, kw], [sh, sw], [dh, dw], [pt, pl, _, _]) = *case;
        let mut taps = Vec::new();
        for ki in 0..kh {
            for kj in 0..kw {
                let ih = (oh * sh + ki * dh) as isize - pt as isize;
                let iw = (ow * sw + kj * dw) as isize - pl as isize;
                if (0..h as isize).contains(&ih) && (0..w as isize).contains(&iw) {
                    taps.push((ih as usize * w + iw as usize, ki * kw + kj));
                }
            }
        }
        taps
    }

    #[test]
    fn test_depthwise_conv2d_matches_reference() {
        let mut rng = Rng::new(11);
        for case in WINDOW_CASES {
            let ([c, h, w], [kh, kw], strides, dilations, pads) = *case;
            let (oh_n, ow_n) = out_dims(case);
            let batch = 2;
            let x = rng.vec(batch * c * h * w, -1.0, 1.0);
            let wt = rng.vec(c * kh * kw, -1.0, 1.0);
            let b = rng.vec(c, -1.0, 1.0);
            for act in [Activation::None, Activation::Relu, Activation::SiLU] {
                for bias in [None, Some(&b)] {
                    let mut want = Vec::new();
                    for n in 0..batch {
                        for ch in 0..c {
                            let plane = &x[(n * c + ch) * h * w..][..h * w];
                            for oh in 0..oh_n {
                                for ow in 0..ow_n {
                                    let mut s = bias.map_or(0.0, |b| b[ch] as f64);
                                    for (i, k) in window_taps(case, oh, ow) {
                                        s += plane[i] as f64 * wt[ch * kh * kw + k] as f64;
                                    }
                                    want.push(match act {
                                        Activation::None => s,
                                        Activation::Relu => s.max(0.0),
                                        Activation::SiLU => s / (1.0 + (-s).exp()),
                                    });
                                }
                            }
                        }
                    }
                    let to_i64 = |v: &[usize]| v.iter().map(|&v| v as i64).collect::<Vec<_>>();
                    let input = TensorView::from_slice(&x, vec![batch, c, h, w]);
                    let weights = TensorView::from_slice(&wt, vec![c, 1, kh, kw]);
                    let bias_view = bias.map(|b| TensorView::from_slice(b, vec![c]));
                    for level in levels() {
                        let mut out = Vec::new();
                        let got = conv2d_activation(
                            level,
                            &input,
                            &weights,
                            bias_view.as_ref(),
                            &to_i64(&dilations),
                            c as i64,
                            &to_i64(&pads),
                            &to_i64(&strides),
                            act,
                            &mut out,
                        );
                        assert_eq!(got.shape.as_ref(), &[batch, c, oh_n, ow_n]);
                        let what = format!("{level:?} {case:?} {act:?} bias {}", bias.is_some());
                        assert_close(&got.data, &want, 1e-5, &what);
                    }
                }
            }
        }
    }

    #[test]
    fn test_max_pool2d_matches_reference() {
        let mut rng = Rng::new(12);
        for case in WINDOW_CASES {
            let ([c, h, w], [kh, kw], strides, dilations, pads) = *case;
            // The padding must stay smaller than the window.
            if pads.iter().any(|&p| p >= kh.min(kw)) {
                continue;
            }
            let (oh_n, ow_n) = out_dims(case);
            let x = rng.vec(c * h * w, -1.0, 1.0);
            let mut want = Vec::new();
            for ch in 0..c {
                let plane = &x[ch * h * w..][..h * w];
                for oh in 0..oh_n {
                    for ow in 0..ow_n {
                        let m = window_taps(case, oh, ow)
                            .iter()
                            .fold(f32::NEG_INFINITY, |m, &(i, _)| m.max(plane[i]));
                        want.push(m as f64);
                    }
                }
            }
            let to_i64 = |v: &[usize]| v.iter().map(|&v| v as i64).collect::<Vec<_>>();
            let input = TensorView::from_slice(&x, vec![1, c, h, w]);
            for level in levels() {
                let mut out = Vec::new();
                let got = max_pool2d_at(
                    level,
                    &input,
                    &[kh as i64, kw as i64],
                    &to_i64(&strides),
                    &to_i64(&pads),
                    &to_i64(&dilations),
                    false,
                    &mut out,
                );
                assert_eq!(got.shape.as_ref(), &[1, c, oh_n, ow_n], "{case:?}");
                assert_close(&got.data, &want, 0.0, &format!("{level:?} {case:?}"));
            }
        }
    }

    #[test]
    fn test_conv_transpose_output_size_stride2_no_padding() {
        // Input: [1, 64, 80, 80], kernel 2x2, stride 2, no padding
        // Expected output: (80-1)*2 - 0 + 1*(2-1) + 1 = 160
        let input_data = vec![0.0f32; 1 * 64 * 80 * 80];
        let weight_data = vec![1.0f32; 64 * 64 * 2 * 2];
        let bias_data = vec![0.0f32; 64];

        let input = TensorView::from_slice(&input_data, vec![1, 64, 80, 80]);
        let weights = TensorView::from_slice(&weight_data, vec![64, 64, 2, 2]);
        let bias = TensorView::from_slice(&bias_data, vec![64]);

        let mut out = Vec::new();
        let result = conv_transpose(
            &input,
            &weights,
            Some(&bias),
            &[1, 1],
            1,
            &[0, 0, 0, 0],
            &[2, 2],
            &mut out,
        );

        assert_eq!(
            result.shape.as_ref(),
            &[1, 64, 160, 160],
            "conv_transpose with stride=2 should upsample 80x80 -> 160x160"
        );
    }

    #[test]
    fn test_conv_transpose_output_size_stride2_with_padding() {
        // Input: [1, 32, 40, 40], kernel 3x3, stride 2, padding 1x1
        // Expected: (40-1)*2 - 2 + 1*(3-1) + 1 = 78 - 2 + 2 + 1 = 79
        let input_data = vec![0.0f32; 1 * 32 * 40 * 40];
        let weight_data = vec![1.0f32; 32 * 32 * 3 * 3];
        let bias_data = vec![0.0f32; 32];

        let input = TensorView::from_slice(&input_data, vec![1, 32, 40, 40]);
        let weights = TensorView::from_slice(&weight_data, vec![32, 32, 3, 3]);
        let bias = TensorView::from_slice(&bias_data, vec![32]);

        let mut out = Vec::new();
        let result = conv_transpose(
            &input,
            &weights,
            Some(&bias),
            &[1, 1],
            1,
            &[1, 1, 1, 1],
            &[2, 2],
            &mut out,
        );

        assert_eq!(result.shape.as_ref(), &[1, 32, 79, 79]);
    }

    #[test]
    fn test_conv_transpose_stride1_no_padding() {
        // Input: [1, 16, 10, 10], kernel 3x3, stride 1, no padding
        // Expected: (10-1)*1 - 0 + 1*(3-1) + 1 = 9 + 2 + 1 = 12
        let input_data = vec![0.0f32; 1 * 16 * 10 * 10];
        let weight_data = vec![1.0f32; 16 * 16 * 3 * 3];
        let bias_data = vec![0.0f32; 16];

        let input = TensorView::from_slice(&input_data, vec![1, 16, 10, 10]);
        let weights = TensorView::from_slice(&weight_data, vec![16, 16, 3, 3]);
        let bias = TensorView::from_slice(&bias_data, vec![16]);

        let mut out = Vec::new();
        let result = conv_transpose(
            &input,
            &weights,
            Some(&bias),
            &[1, 1],
            1,
            &[0, 0, 0, 0],
            &[1, 1],
            &mut out,
        );

        assert_eq!(result.shape.as_ref(), &[1, 16, 12, 12]);
    }

    #[test]
    fn test_conv_transpose_values_simple() {
        // Minimal: 1x1 input, 1x1 kernel, stride 1, no padding, no bias
        let input = TensorView::from_slice(&[1.0f32], vec![1, 1, 1, 1]);
        let weights = TensorView::from_slice(&[2.0f32], vec![1, 1, 1, 1]);

        let mut out = Vec::new();
        let result = conv_transpose(
            &input,
            &weights,
            None,
            &[1],
            1,
            &[0, 0, 0, 0],
            &[1, 1],
            &mut out,
        );

        assert_eq!(result.shape.as_ref(), &[1, 1, 1, 1]);
        assert!((result.data.as_ref()[0] - 2.0).abs() < 1e-6);
    }

    // ==================== resize_nearest tests ====================

    #[test]
    fn test_resize_nearest_identity() {
        // 1x1x2x2 input, scale 1.0 -> same output
        let input = TensorView::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let mut out = Vec::new();
        let result = resize_nearest(
            &input,
            Some(&[1.0, 1.0, 1.0, 1.0]),
            None,
            "asymmetric",
            &mut out,
        );
        assert_eq!(result.shape.as_ref(), &[1, 1, 2, 2]);
        let data = result.data.as_ref();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
        assert!((data[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_resize_nearest_upscale_2x() {
        // 1x1x2x2 input, scale 2.0 -> 1x1x4x4 output
        let input = TensorView::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let mut out = Vec::new();
        let result = resize_nearest(
            &input,
            Some(&[1.0, 1.0, 2.0, 2.0]),
            None,
            "asymmetric",
            &mut out,
        );
        assert_eq!(result.shape.as_ref(), &[1, 1, 4, 4]);
        let data = result.data.as_ref();
        // With asymmetric mode: ih = floor(oh * 2/4) = floor(oh * 0.5)
        // oh=0 -> ih=0, oh=1 -> ih=0, oh=2 -> ih=1, oh=3 -> ih=1
        // Expected: [1,1,2,2, 1,1,2,2, 3,3,4,4, 3,3,4,4]
        let expected = [
            1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
        ];
        for (i, &e) in expected.iter().enumerate() {
            assert!(
                (data[i] - e).abs() < 1e-6,
                "mismatch at index {}: got {} expected {}",
                i,
                data[i],
                e
            );
        }
    }

    #[test]
    fn test_resize_nearest_with_sizes() {
        // 1x1x2x2 input, target sizes [1,1,3,3]
        let input = TensorView::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let mut out = Vec::new();
        let result = resize_nearest(&input, None, Some(&[1, 1, 3, 3]), "asymmetric", &mut out);
        assert_eq!(result.shape.as_ref(), &[1, 1, 3, 3]);
        let data = result.data.as_ref();
        assert_eq!(data.len(), 9);
    }

    #[test]
    fn test_resize_nearest_multichannel() {
        // 1x2x2x2 input (2 channels), upscale to 1x2x4x4
        let input_data: Vec<f32> = (0..8).map(|v| v as f32).collect();
        let input = TensorView::from_slice(&input_data, vec![1, 2, 2, 2]);
        let mut out = Vec::new();
        let result = resize_nearest(
            &input,
            Some(&[1.0, 1.0, 2.0, 2.0]),
            None,
            "asymmetric",
            &mut out,
        );
        assert_eq!(result.shape.as_ref(), &[1, 2, 4, 4]);
        assert_eq!(result.data.len(), 32);
    }

    #[test]
    fn test_resize_nearest_half_pixel() {
        // 1x1x2x2 input, upscale 2x with half_pixel mode
        let input = TensorView::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let mut out = Vec::new();
        let result = resize_nearest(
            &input,
            Some(&[1.0, 1.0, 2.0, 2.0]),
            None,
            "half_pixel",
            &mut out,
        );
        assert_eq!(result.shape.as_ref(), &[1, 1, 4, 4]);
        let data = result.data.as_ref();
        assert_eq!(data.len(), 16);
        // All values should be valid (from input)
        for &v in data.iter() {
            assert!(v >= 1.0 && v <= 4.0, "value {} out of range", v);
        }
    }

    #[test]
    fn test_resize_nearest_large_size_no_overflow() {
        // Test that large sizes don't overflow - use sizes mode with moderately large dimensions
        // 1x1x1x1 input -> 1x1x100x100
        let input = TensorView::from_slice(&[42.0f32], vec![1, 1, 1, 1]);
        let mut out = Vec::new();
        let result = resize_nearest(
            &input,
            None,
            Some(&[1, 1, 100, 100]),
            "asymmetric",
            &mut out,
        );
        assert_eq!(result.shape.as_ref(), &[1, 1, 100, 100]);
        // All values should be 42.0 (single pixel upscaled)
        for &v in result.data.as_ref() {
            assert!((v - 42.0).abs() < 1e-6);
        }
    }

    #[test]
    #[should_panic(expected = "sizes H and W must be positive")]
    fn test_resize_nearest_negative_sizes_panics() {
        let input = TensorView::from_slice(&[1.0f32], vec![1, 1, 1, 1]);
        let mut out = Vec::new();
        let _ = resize_nearest(&input, None, Some(&[1, 1, -1, 10]), "asymmetric", &mut out);
    }

    // ==================== max_pool2d tests ====================

    #[test]
    fn test_max_pool2d_simple() {
        // 1x1x4x4 input, 2x2 kernel, stride 2, no padding
        let input_data: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let input = TensorView::from_slice(&input_data, vec![1, 1, 4, 4]);
        let mut out = Vec::new();
        let result = max_pool2d(
            &input,
            &[2, 2],
            &[2, 2],
            &[0, 0, 0, 0],
            &[1, 1],
            false,
            &mut out,
        );
        assert_eq!(result.shape.as_ref(), &[1, 1, 2, 2]);
        let data = result.data.as_ref();
        // [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]
        // pool 2x2 stride 2: max(0,1,4,5)=5, max(2,3,6,7)=7, max(8,9,12,13)=13, max(10,11,14,15)=15
        assert!((data[0] - 5.0).abs() < 1e-6, "got {}", data[0]);
        assert!((data[1] - 7.0).abs() < 1e-6, "got {}", data[1]);
        assert!((data[2] - 13.0).abs() < 1e-6, "got {}", data[2]);
        assert!((data[3] - 15.0).abs() < 1e-6, "got {}", data[3]);
    }

    #[test]
    fn test_max_pool2d_stride1() {
        // 1x1x3x3 input, 2x2 kernel, stride 1
        let input_data: Vec<f32> = (0..9).map(|v| v as f32).collect();
        let input = TensorView::from_slice(&input_data, vec![1, 1, 3, 3]);
        let mut out = Vec::new();
        let result = max_pool2d(
            &input,
            &[2, 2],
            &[1, 1],
            &[0, 0, 0, 0],
            &[1, 1],
            false,
            &mut out,
        );
        assert_eq!(result.shape.as_ref(), &[1, 1, 2, 2]);
        let data = result.data.as_ref();
        // [[0,1,2],[3,4,5],[6,7,8]]
        // pool 2x2 stride 1: max(0,1,3,4)=4, max(1,2,4,5)=5, max(3,4,6,7)=7, max(4,5,7,8)=8
        assert!((data[0] - 4.0).abs() < 1e-6);
        assert!((data[1] - 5.0).abs() < 1e-6);
        assert!((data[2] - 7.0).abs() < 1e-6);
        assert!((data[3] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_pool2d_with_padding() {
        // 1x1x2x2 input, 2x2 kernel, stride 1, padding 1
        let input = TensorView::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let mut out = Vec::new();
        let result = max_pool2d(
            &input,
            &[2, 2],
            &[1, 1],
            &[1, 1, 1, 1],
            &[1, 1],
            false,
            &mut out,
        );
        assert_eq!(result.shape.as_ref(), &[1, 1, 3, 3]);
        let data = result.data.as_ref();
        // With padding, the 2x2 input becomes 4x4 padded, then 2x2 pool stride 1 -> 3x3 output
        // Position (0,0): max(pad, pad, pad, 1) = 1
        assert!((data[0] - 1.0).abs() < 1e-6);
        // Center position (1,1): max(1,2,3,4) = 4
        assert!((data[4] - 4.0).abs() < 1e-6);
        // Position (2,2): max(4, pad, pad, pad) = 4
        assert!((data[8] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_pool2d_multichannel() {
        // 1x2x4x4 input (2 channels)
        let input_data: Vec<f32> = (0..32).map(|v| v as f32).collect();
        let input = TensorView::from_slice(&input_data, vec![1, 2, 4, 4]);
        let mut out = Vec::new();
        let result = max_pool2d(
            &input,
            &[2, 2],
            &[2, 2],
            &[0, 0, 0, 0],
            &[1, 1],
            false,
            &mut out,
        );
        assert_eq!(result.shape.as_ref(), &[1, 2, 2, 2]);
        assert_eq!(result.data.len(), 8);
    }

    #[test]
    fn test_max_pool2d_no_overflow_large_dims() {
        // Test with moderately large dimensions to verify no overflow in intermediate calculations
        let batch = 1usize;
        let channels = 32usize;
        let h = 80usize;
        let w = 80usize;
        let total = batch * channels * h * w;
        let input_data = vec![1.0f32; total];
        let input = TensorView::from_slice(&input_data, vec![batch, channels, h, w]);
        let mut out = Vec::new();
        let result = max_pool2d(
            &input,
            &[2, 2],
            &[2, 2],
            &[0, 0, 0, 0],
            &[1, 1],
            false,
            &mut out,
        );
        assert_eq!(result.shape.as_ref(), &[1, 32, 40, 40]);
        // All values should be 1.0
        for &v in result.data.as_ref().iter().take(10) {
            assert!((v - 1.0).abs() < 1e-6, "got {}", v);
        }
    }
}
