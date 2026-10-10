#![allow(unsafe_op_in_unsafe_fn)]
use crate::kernels::bias_act::bias_act_inplace;
use crate::kernels::timing;
use crate::kernels::utils;
#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
use crate::kernels::matmul::{Accum, MatMut, MatRef, Par, matmul as strided_matmul};
use crate::tensor::TensorView;

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

#[derive(Clone, Copy, PartialEq)]
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
        input, weights, bias, dilations, group, pads, strides, act, out,
    );
}

fn conv2d_activation<'b, 'a>(
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

    // Fast path: depthwise convolution (groups == channels, in_channels_per_group == 1)
    // Skips im2col overhead and does direct computation with SIMD.
    if groups == in_channels && in_channels_per_group == 1 && out_channels_per_group == 1
        && dilation_h == 1 && dilation_w == 1
    {
        let input_data = &input.data;
        let weight_data = &weights.data;
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr(), total_output) };

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            if kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1
                && pad_top == 1 && pad_left == 1
            {
                let bias_slice: Option<&[f32]> = bias.map(|b| b.data.as_ref());
                unsafe {
                    depthwise_conv2d_3x3_s1_avx2(
                        input_data, weight_data, bias_slice, out_slice,
                        batch_size, in_channels, in_h, in_w, out_h, out_w, act,
                    );
                }
                return TensorView::from_slice(out, vec![batch_size, out_channels, out_h, out_w]);
            }
            unsafe { depthwise_conv2d_avx2(
                input_data, weight_data, out_slice,
                batch_size, in_channels, in_h, in_w,
                kernel_h, kernel_w, stride_h, stride_w,
                pad_top, pad_left, out_h, out_w, act,
            ); }
            return TensorView::from_slice(out, vec![batch_size, out_channels, out_h, out_w]);
        }

        #[cfg(target_arch = "aarch64")]
        {
            if kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1
                && pad_top == 1 && pad_left == 1
            {
                let bias_slice: Option<&[f32]> = bias.map(|b| b.data.as_ref());
                unsafe {
                    depthwise_conv2d_3x3_s1_neon(
                        input_data, weight_data, bias_slice, out_slice,
                        batch_size, in_channels, in_h, in_w, out_h, out_w, act,
                    );
                }
                if timing::TIMING_ENABLED {
                    let ns = _t0.unwrap().elapsed().as_nanos() as u64;
                    timing::CONV_DW_NS.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
                }
                return TensorView::from_slice(out, vec![batch_size, out_channels, out_h, out_w]);
            }
            let bias_slice: Option<&[f32]> = bias.map(|b| b.data.as_ref());
            unsafe {
                depthwise_conv2d_neon(
                    input_data, weight_data, bias_slice, out_slice,
                    batch_size, in_channels, in_h, in_w,
                    kernel_h, kernel_w, stride_h, stride_w,
                    pad_top, pad_left, out_h, out_w, act,
                );
            }
            if timing::TIMING_ENABLED {
                let ns = _t0.unwrap().elapsed().as_nanos() as u64;
                timing::CONV_DW_NS.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
            }
            return TensorView::from_slice(out, vec![batch_size, out_channels, out_h, out_w]);
        }

        // Scalar depthwise fallback
        for n in 0..batch_size {
            for c in 0..in_channels {
                let in_base = n * in_channels * in_h * in_w + c * in_h * in_w;
                let w_base = c * kernel_h * kernel_w;
                let out_base = n * out_channels * out_h * out_w + c * out_h * out_w;
                let bias_val = if let Some(b) = bias { b.data[c] } else { 0.0 };
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0f32;
                        for kh in 0..kernel_h {
                            let ih = (oh * stride_h + kh) as isize - pad_top as isize;
                            if ih < 0 || ih >= in_h as isize { continue; }
                            for kw in 0..kernel_w {
                                let iw = (ow * stride_w + kw) as isize - pad_left as isize;
                                if iw < 0 || iw >= in_w as isize { continue; }
                                sum += input_data[in_base + ih as usize * in_w + iw as usize]
                                    * weight_data[w_base + kh * kernel_w + kw];
                            }
                        }
                        sum += bias_val;
                        out_slice[out_base + oh * out_w + ow] = match act {
                            Activation::Relu => if sum < 0.0 { 0.0 } else { sum },
                            _ => sum,
                        };
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

    // Common case: stride=1, dilation=1 — use optimized path
    if stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1 {
        // For small kernels with padding, use AVX2 for zeroing when on x86_64
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && out_w >= 8 {
            // Use AVX2 optimized path
            unsafe {
                crate::kernels::avx::conv2d::im2col_avx2(
                    input,
                    batch_offset,
                    ch_start,
                    channels,
                    in_h,
                    in_w,
                    kernel_h,
                    kernel_w,
                    pad_top,
                    pad_left,
                    out_h,
                    out_w,
                    spatial_cols,
                    col,
                );
            }
            return;
        }
    }

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

    let data = &input.data;

    if kh == 2
        && kw == 2
        && sh == 2
        && sw == 2
        && pad_top == 0
        && pad_left == 0
        && pad_bottom == 0
        && pad_right == 0
        && dh == 1
        && dw == 1
        && out_w >= 8
    {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    use std::arch::x86_64::*;
                    let idx = _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0);
                    for n in 0..batch {
                        for c in 0..channels {
                            let in_base =
                                (n * channels + c) * in_h * in_w;
                            let out_base =
                                (n * channels + c) * out_h * out_w;
                            for oh in 0..out_h {
                                let r0 = data
                                    .as_ptr()
                                    .add(in_base + oh * 2 * in_w);
                                let r1 = data
                                    .as_ptr()
                                    .add(in_base + (oh * 2 + 1) * in_w);
                                let dst = out.as_mut_ptr().add(
                                    out_base + oh * out_w,
                                );
                                let mut ow: usize = 0;
                                while ow + 8 <= out_w {
                                    let col = ow * 2;
                                    let r0_lo = _mm256_loadu_ps(r0.add(col));
                                    let r0_hi = _mm256_loadu_ps(r0.add(col + 8));
                                    let r1_lo = _mm256_loadu_ps(r1.add(col));
                                    let r1_hi = _mm256_loadu_ps(r1.add(col + 8));
                                    let m_lo = _mm256_max_ps(r0_lo, r1_lo);
                                    let m_hi = _mm256_max_ps(r0_hi, r1_hi);
                                    let s_lo = _mm256_shuffle_ps(m_lo, m_lo, 0xB1);
                                    let s_hi = _mm256_shuffle_ps(m_hi, m_hi, 0xB1);
                                    let p_lo = _mm256_max_ps(m_lo, s_lo);
                                    let p_hi = _mm256_max_ps(m_hi, s_hi);
                                    let r_lo = _mm256_permutevar8x32_ps(p_lo, idx);
                                    let r_hi = _mm256_permutevar8x32_ps(p_hi, idx);
                                    let res = _mm256_permute2f128_ps(r_lo, r_hi, 0x20);
                                    _mm256_storeu_ps(dst.add(ow), res);
                                    ow += 8;
                                }
                                while ow < out_w {
                                    let col = ow * 2;
                                    let v00 = *r0.add(col);
                                    let v01 = *r0.add(col + 1);
                                    let v10 = *r1.add(col);
                                    let v11 = *r1.add(col + 1);
                                    *dst.add(ow) =
                                        v00.max(v01).max(v10).max(v11);
                                    ow += 1;
                                }
                            }
                        }
                    }
                }
                return TensorView::from_slice(
                    out,
                    vec![batch, channels, out_h, out_w],
                );
            }
        }
    }

    for n in 0..batch {
        for c in 0..channels {
            let in_offset = (n as u64 * channels as u64 + c as u64) * in_h as u64 * in_w as u64;
            let out_offset = (n as u64 * channels as u64 + c as u64) * out_h as u64 * out_w as u64;
            let in_offset = in_offset as usize;
            let out_offset = out_offset as usize;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut max_val = f32::NEG_INFINITY;
                    for ki in 0..kh {
                        let ih = (oh * sh + ki * dh) as isize - pad_top as isize;
                        if ih < 0 || ih >= in_h as isize {
                            continue;
                        }
                        for kj in 0..kw {
                            let iw = (ow * sw + kj * dw) as isize - pad_left as isize;
                            if iw < 0 || iw >= in_w as isize {
                                continue;
                            }
                            let val = data[in_offset + ih as usize * in_w + iw as usize];
                            if val > max_val {
                                max_val = val;
                            }
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

#[cfg(target_arch = "aarch64")]
unsafe fn depthwise_conv2d_3x3_s1_neon(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
    batch: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    act: Activation,
) {
    use core::arch::aarch64::*;
    let zero_v = vdupq_n_f32(0.0);
    let pad = 1usize;
    // Same trick as the AVX2 kernel: the top and bottom output rows read one
    // input row that does not exist. Point those taps at a row of zeros so
    // every output row stays on the same vector body; a scalar edge path
    // costs 2/in_h of the work, and in_h here is as small as 5.
    let zero_row = vec![0.0f32; in_w];

    for n in 0..batch {
        for c in 0..channels {
            let in_base = n * channels * in_h * in_w + c * in_h * in_w;
            let w_base = c * 9;
            let out_base = n * channels * out_h * out_w + c * out_h * out_w;

            let w0 = vld1q_dup_f32(weight.as_ptr().add(w_base));
            let w1 = vld1q_dup_f32(weight.as_ptr().add(w_base + 1));
            let w2 = vld1q_dup_f32(weight.as_ptr().add(w_base + 2));
            let w3 = vld1q_dup_f32(weight.as_ptr().add(w_base + 3));
            let w4 = vld1q_dup_f32(weight.as_ptr().add(w_base + 4));
            let w5 = vld1q_dup_f32(weight.as_ptr().add(w_base + 5));
            let w6 = vld1q_dup_f32(weight.as_ptr().add(w_base + 6));
            let w7 = vld1q_dup_f32(weight.as_ptr().add(w_base + 7));
            let w8 = vld1q_dup_f32(weight.as_ptr().add(w_base + 8));

            let bias_val = bias.map(|b| *b.get_unchecked(c)).unwrap_or(0.0f32);
            let bias_v = vdupq_n_f32(bias_val);

            for oh in 0..out_h {
                let out_row = out_base + oh * out_w;
                let mut ow = 0usize;

                {
                    let row = |ki: usize| {
                        let ih = oh + ki;
                        if ih >= pad && ih < in_h + pad {
                            input.as_ptr().add(in_base + (ih - pad) * in_w)
                        } else {
                            zero_row.as_ptr()
                        }
                    };
                    let r0 = row(0);
                    let r1 = row(1);
                    let r2 = row(2);

                    // Scalar: ow=0 (left pad)
                    {
                        let mut s = bias_val;
                        for ki in 0..3usize {
                            let rp = [r0, r1, r2][ki];
                            for kj in 0..3usize {
                                let iw = (kj as isize) - 1;
                                if iw >= 0 && (iw as usize) < in_w {
                                    s += *rp.add(iw as usize) * *weight.get_unchecked(w_base + ki * 3 + kj);
                                }
                            }
                        }
                        let v = if act == Activation::Relu && s < 0.0 { 0.0 } else { s };
                        *out.get_unchecked_mut(out_row + ow) = v;
                        ow += 1;
                    }

                    // NEON middle: 16 pixels (4 accumulators for FMA latency hiding)
                    while ow + 16 <= out_w && ow + 16 < in_w {
                        let off = ow - 1;
                        let mut a0 = bias_v;
                        let mut a1 = bias_v;
                        let mut a2 = bias_v;
                        let mut a3 = bias_v;
                        // Row 0
                        a0 = vfmaq_f32(a0, w0, vld1q_f32(r0.add(off)));
                        a1 = vfmaq_f32(a1, w0, vld1q_f32(r0.add(off + 4)));
                        a2 = vfmaq_f32(a2, w0, vld1q_f32(r0.add(off + 8)));
                        a3 = vfmaq_f32(a3, w0, vld1q_f32(r0.add(off + 12)));
                        a0 = vfmaq_f32(a0, w1, vld1q_f32(r0.add(off + 1)));
                        a1 = vfmaq_f32(a1, w1, vld1q_f32(r0.add(off + 5)));
                        a2 = vfmaq_f32(a2, w1, vld1q_f32(r0.add(off + 9)));
                        a3 = vfmaq_f32(a3, w1, vld1q_f32(r0.add(off + 13)));
                        a0 = vfmaq_f32(a0, w2, vld1q_f32(r0.add(off + 2)));
                        a1 = vfmaq_f32(a1, w2, vld1q_f32(r0.add(off + 6)));
                        a2 = vfmaq_f32(a2, w2, vld1q_f32(r0.add(off + 10)));
                        a3 = vfmaq_f32(a3, w2, vld1q_f32(r0.add(off + 14)));
                        // Row 1
                        a0 = vfmaq_f32(a0, w3, vld1q_f32(r1.add(off)));
                        a1 = vfmaq_f32(a1, w3, vld1q_f32(r1.add(off + 4)));
                        a2 = vfmaq_f32(a2, w3, vld1q_f32(r1.add(off + 8)));
                        a3 = vfmaq_f32(a3, w3, vld1q_f32(r1.add(off + 12)));
                        a0 = vfmaq_f32(a0, w4, vld1q_f32(r1.add(off + 1)));
                        a1 = vfmaq_f32(a1, w4, vld1q_f32(r1.add(off + 5)));
                        a2 = vfmaq_f32(a2, w4, vld1q_f32(r1.add(off + 9)));
                        a3 = vfmaq_f32(a3, w4, vld1q_f32(r1.add(off + 13)));
                        a0 = vfmaq_f32(a0, w5, vld1q_f32(r1.add(off + 2)));
                        a1 = vfmaq_f32(a1, w5, vld1q_f32(r1.add(off + 6)));
                        a2 = vfmaq_f32(a2, w5, vld1q_f32(r1.add(off + 10)));
                        a3 = vfmaq_f32(a3, w5, vld1q_f32(r1.add(off + 14)));
                        // Row 2
                        a0 = vfmaq_f32(a0, w6, vld1q_f32(r2.add(off)));
                        a1 = vfmaq_f32(a1, w6, vld1q_f32(r2.add(off + 4)));
                        a2 = vfmaq_f32(a2, w6, vld1q_f32(r2.add(off + 8)));
                        a3 = vfmaq_f32(a3, w6, vld1q_f32(r2.add(off + 12)));
                        a0 = vfmaq_f32(a0, w7, vld1q_f32(r2.add(off + 1)));
                        a1 = vfmaq_f32(a1, w7, vld1q_f32(r2.add(off + 5)));
                        a2 = vfmaq_f32(a2, w7, vld1q_f32(r2.add(off + 9)));
                        a3 = vfmaq_f32(a3, w7, vld1q_f32(r2.add(off + 13)));
                        a0 = vfmaq_f32(a0, w8, vld1q_f32(r2.add(off + 2)));
                        a1 = vfmaq_f32(a1, w8, vld1q_f32(r2.add(off + 6)));
                        a2 = vfmaq_f32(a2, w8, vld1q_f32(r2.add(off + 10)));
                        a3 = vfmaq_f32(a3, w8, vld1q_f32(r2.add(off + 14)));
                        if act == Activation::Relu {
                            a0 = vmaxq_f32(a0, zero_v);
                            a1 = vmaxq_f32(a1, zero_v);
                            a2 = vmaxq_f32(a2, zero_v);
                            a3 = vmaxq_f32(a3, zero_v);
                        }
                        let out_ptr = out.as_mut_ptr().add(out_row + ow);
                        vst1q_f32(out_ptr, a0);
                        vst1q_f32(out_ptr.add(4), a1);
                        vst1q_f32(out_ptr.add(8), a2);
                        vst1q_f32(out_ptr.add(12), a3);
                        ow += 16;
                    }

                    // 8-pixel fallback
                    while ow + 8 <= out_w && ow + 8 < in_w {
                        let off = ow - 1;
                        let mut acc0 = bias_v;
                        let mut acc1 = bias_v;
                        acc0 = vfmaq_f32(acc0, w0, vld1q_f32(r0.add(off)));
                        acc1 = vfmaq_f32(acc1, w0, vld1q_f32(r0.add(off + 4)));
                        acc0 = vfmaq_f32(acc0, w1, vld1q_f32(r0.add(off + 1)));
                        acc1 = vfmaq_f32(acc1, w1, vld1q_f32(r0.add(off + 5)));
                        acc0 = vfmaq_f32(acc0, w2, vld1q_f32(r0.add(off + 2)));
                        acc1 = vfmaq_f32(acc1, w2, vld1q_f32(r0.add(off + 6)));
                        acc0 = vfmaq_f32(acc0, w3, vld1q_f32(r1.add(off)));
                        acc1 = vfmaq_f32(acc1, w3, vld1q_f32(r1.add(off + 4)));
                        acc0 = vfmaq_f32(acc0, w4, vld1q_f32(r1.add(off + 1)));
                        acc1 = vfmaq_f32(acc1, w4, vld1q_f32(r1.add(off + 5)));
                        acc0 = vfmaq_f32(acc0, w5, vld1q_f32(r1.add(off + 2)));
                        acc1 = vfmaq_f32(acc1, w5, vld1q_f32(r1.add(off + 6)));
                        acc0 = vfmaq_f32(acc0, w6, vld1q_f32(r2.add(off)));
                        acc1 = vfmaq_f32(acc1, w6, vld1q_f32(r2.add(off + 4)));
                        acc0 = vfmaq_f32(acc0, w7, vld1q_f32(r2.add(off + 1)));
                        acc1 = vfmaq_f32(acc1, w7, vld1q_f32(r2.add(off + 5)));
                        acc0 = vfmaq_f32(acc0, w8, vld1q_f32(r2.add(off + 2)));
                        acc1 = vfmaq_f32(acc1, w8, vld1q_f32(r2.add(off + 6)));
                        if act == Activation::Relu {
                            acc0 = vmaxq_f32(acc0, zero_v);
                            acc1 = vmaxq_f32(acc1, zero_v);
                        }
                        vst1q_f32(out.as_mut_ptr().add(out_row + ow), acc0);
                        vst1q_f32(out.as_mut_ptr().add(out_row + ow + 4), acc1);
                        ow += 8;
                    }

                    // 4-pixel tail of the vectorized region
                    while ow + 4 <= out_w && ow + 4 < in_w {
                        let off = ow - 1;
                        let mut acc = bias_v;
                        acc = vfmaq_f32(acc, w0, vld1q_f32(r0.add(off)));
                        acc = vfmaq_f32(acc, w1, vld1q_f32(r0.add(off + 1)));
                        acc = vfmaq_f32(acc, w2, vld1q_f32(r0.add(off + 2)));
                        acc = vfmaq_f32(acc, w3, vld1q_f32(r1.add(off)));
                        acc = vfmaq_f32(acc, w4, vld1q_f32(r1.add(off + 1)));
                        acc = vfmaq_f32(acc, w5, vld1q_f32(r1.add(off + 2)));
                        acc = vfmaq_f32(acc, w6, vld1q_f32(r2.add(off)));
                        acc = vfmaq_f32(acc, w7, vld1q_f32(r2.add(off + 1)));
                        acc = vfmaq_f32(acc, w8, vld1q_f32(r2.add(off + 2)));
                        if act == Activation::Relu {
                            acc = vmaxq_f32(acc, zero_v);
                        }
                        vst1q_f32(out.as_mut_ptr().add(out_row + ow), acc);
                        ow += 4;
                    }

                    // Scalar tail (handles right edge with bounds checking)
                    while ow < out_w {
                        let mut s = bias_val;
                        for ki in 0..3usize {
                            let rp = [r0, r1, r2][ki];
                            for kj in 0..3usize {
                                let iw = (ow as isize) + (kj as isize) - 1;
                                if iw >= 0 && (iw as usize) < in_w {
                                    s += *rp.add(iw as usize) * *weight.get_unchecked(w_base + ki * 3 + kj);
                                }
                            }
                        }
                        let v = if act == Activation::Relu && s < 0.0 { 0.0 } else { s };
                        *out.get_unchecked_mut(out_row + ow) = v;
                        ow += 1;
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn depthwise_conv2d_neon(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
    batch: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    pad_top: usize,
    pad_left: usize,
    out_h: usize,
    out_w: usize,
    act: Activation,
) {
    use core::arch::aarch64::*;
    let zero_v = vdupq_n_f32(0.0);

    let max_vec_ow = (in_w + pad_left).saturating_sub(kw + 3);
    let vec_ow_end = if sw == 1 { max_vec_ow } else { 0 };
    let vec_ow_end = (vec_ow_end / 4) * 4;
    let vec_ow_start = if sw == 1 { pad_left } else { out_w };

    for n in 0..batch {
        for c in 0..channels {
            let in_base = n * channels * in_h * in_w + c * in_h * in_w;
            let w_base = c * kh * kw;
            let out_base = n * channels * out_h * out_w + c * out_h * out_w;
            let bias_val = bias.map(|b| *b.get_unchecked(c)).unwrap_or(0.0f32);
            let bias_v = vdupq_n_f32(bias_val);

            for oh in 0..out_h {
                let mut ow = 0usize;

                // Scalar prefix
                while ow < vec_ow_start.min(out_w) {
                    let mut s = bias_val;
                    for ki in 0..kh {
                        let ih = (oh * sh + ki) as isize - pad_top as isize;
                        if ih < 0 || ih >= in_h as isize { continue; }
                        for kj in 0..kw {
                            let iw = (ow * sw + kj) as isize - pad_left as isize;
                            if iw < 0 || iw >= in_w as isize { continue; }
                            s += input.get_unchecked(in_base + ih as usize * in_w + iw as usize)
                                * weight.get_unchecked(w_base + ki * kw + kj);
                        }
                    }
                    let v = if act == Activation::Relu && s < 0.0 { 0.0 } else { s };
                    *out.get_unchecked_mut(out_base + oh * out_w + ow) = v;
                    ow += 1;
                }

                // NEON stride-2 4-pixel block using vld2q for decimation
                while sw == 2 && sh == 1 && kh <= 5 && kw <= 5 && oh > 0 && oh < in_h && ow + 4 <= out_w {
                    let base_iw = ow * 2 - pad_left;
                    let max_read = base_iw + (kw - 1) + 7;
                    if max_read >= in_w { break; }
                    if base_iw < 0 { break; }
                    let mut acc = bias_v;
                    let all_rows_in = oh >= 1 && oh + kh - 1 < in_h + pad_top && oh >= pad_top;
                    if all_rows_in {
                        for ki in 0..kh {
                            let ih_valid = oh + ki - pad_top;
                            let row_ptr = input.as_ptr().add(in_base + ih_valid * in_w);
                            for kj in 0..kw {
                                let pair = vld2q_f32(row_ptr.add(base_iw + kj));
                                let w_val = vdupq_n_f32(*weight.get_unchecked(w_base + ki * kw + kj));
                                acc = vfmaq_f32(acc, w_val, pair.0);
                            }
                        }
                    } else {
                        for ki in 0..kh {
                            let ih = (oh * sh + ki) as isize - pad_top as isize;
                            if ih < 0 || ih >= in_h as isize { continue; }
                            let row_ptr = input.as_ptr().add(in_base + ih as usize * in_w);
                            for kj in 0..kw {
                                let pair = vld2q_f32(row_ptr.add(base_iw + kj));
                                let w_val = vdupq_n_f32(*weight.get_unchecked(w_base + ki * kw + kj));
                                acc = vfmaq_f32(acc, w_val, pair.0);
                            }
                        }
                    }
                    if act == Activation::Relu {
                        acc = vmaxq_f32(acc, zero_v);
                    }
                    vst1q_f32(out.as_mut_ptr().add(out_base + oh * out_w + ow), acc);
                    ow += 4;
                }

                // NEON 16-pixel block (4 accumulators for FMA latency hiding)
                while sw == 1 && ow + 16 <= vec_ow_end && ow + 16 <= out_w {
                    let mut a0 = bias_v;
                    let mut a1 = bias_v;
                    let mut a2 = bias_v;
                    let mut a3 = bias_v;
                    for ki in 0..kh {
                        let ih = oh * sh + ki;
                        if ih < pad_top || ih >= in_h + pad_top { continue; }
                        let ih_valid = ih - pad_top;
                        let row_ptr = input.as_ptr().add(in_base + ih_valid * in_w);
                        for kj in 0..kw {
                            let off = ow + kj - pad_left;
                            let w_val = vld1q_dup_f32(weight.as_ptr().add(w_base + ki * kw + kj));
                            a0 = vfmaq_f32(a0, w_val, vld1q_f32(row_ptr.add(off)));
                            a1 = vfmaq_f32(a1, w_val, vld1q_f32(row_ptr.add(off + 4)));
                            a2 = vfmaq_f32(a2, w_val, vld1q_f32(row_ptr.add(off + 8)));
                            a3 = vfmaq_f32(a3, w_val, vld1q_f32(row_ptr.add(off + 12)));
                        }
                    }
                    if act == Activation::Relu {
                        a0 = vmaxq_f32(a0, zero_v);
                        a1 = vmaxq_f32(a1, zero_v);
                        a2 = vmaxq_f32(a2, zero_v);
                        a3 = vmaxq_f32(a3, zero_v);
                    }
                    let out_ptr = out.as_mut_ptr().add(out_base + oh * out_w + ow);
                    vst1q_f32(out_ptr, a0);
                    vst1q_f32(out_ptr.add(4), a1);
                    vst1q_f32(out_ptr.add(8), a2);
                    vst1q_f32(out_ptr.add(12), a3);
                    ow += 16;
                }

                // NEON 8-pixel block (2 accumulators)
                while sw == 1 && ow + 8 <= vec_ow_end && ow + 8 <= out_w {
                    let mut a0 = bias_v;
                    let mut a1 = bias_v;
                    for ki in 0..kh {
                        let ih = oh * sh + ki;
                        if ih < pad_top || ih >= in_h + pad_top { continue; }
                        let ih_valid = ih - pad_top;
                        let row_ptr = input.as_ptr().add(in_base + ih_valid * in_w);
                        for kj in 0..kw {
                            let off = ow + kj - pad_left;
                            let w_val = vld1q_dup_f32(weight.as_ptr().add(w_base + ki * kw + kj));
                            a0 = vfmaq_f32(a0, w_val, vld1q_f32(row_ptr.add(off)));
                            a1 = vfmaq_f32(a1, w_val, vld1q_f32(row_ptr.add(off + 4)));
                        }
                    }
                    if act == Activation::Relu {
                        a0 = vmaxq_f32(a0, zero_v);
                        a1 = vmaxq_f32(a1, zero_v);
                    }
                    vst1q_f32(out.as_mut_ptr().add(out_base + oh * out_w + ow), a0);
                    vst1q_f32(out.as_mut_ptr().add(out_base + oh * out_w + ow + 4), a1);
                    ow += 8;
                }

                // NEON 4-pixel block (1 accumulator)
                while sw == 1 && ow + 4 <= vec_ow_end && ow + 4 <= out_w {
                    let mut acc = bias_v;
                    for ki in 0..kh {
                        let ih = oh * sh + ki;
                        if ih < pad_top || ih >= in_h + pad_top { continue; }
                        let ih_valid = ih - pad_top;
                        let row_ptr = input.as_ptr().add(in_base + ih_valid * in_w);
                        for kj in 0..kw {
                            let iw_valid = ow + kj - pad_left;
                            let w_val = vld1q_dup_f32(weight.as_ptr().add(w_base + ki * kw + kj));
                            acc = vfmaq_f32(acc, w_val, vld1q_f32(row_ptr.add(iw_valid)));
                        }
                    }
                    if act == Activation::Relu {
                        acc = vmaxq_f32(acc, zero_v);
                    }
                    vst1q_f32(out.as_mut_ptr().add(out_base + oh * out_w + ow), acc);
                    ow += 4;
                }

                // Scalar tail
                while ow < out_w {
                    let mut s = bias_val;
                    for ki in 0..kh {
                        let ih = (oh * sh + ki) as isize - pad_top as isize;
                        if ih < 0 || ih >= in_h as isize { continue; }
                        for kj in 0..kw {
                            let iw = (ow * sw + kj) as isize - pad_left as isize;
                            if iw < 0 || iw >= in_w as isize { continue; }
                            s += input.get_unchecked(in_base + ih as usize * in_w + iw as usize)
                                * weight.get_unchecked(w_base + ki * kw + kj);
                        }
                    }
                    let v = if act == Activation::Relu && s < 0.0 { 0.0 } else { s };
                    *out.get_unchecked_mut(out_base + oh * out_w + ow) = v;
                    ow += 1;
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn depthwise_conv2d_avx2(
    input: &[f32],
    weight: &[f32],
    out: &mut [f32],
    batch: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    pad_top: usize,
    pad_left: usize,
    out_h: usize,
    out_w: usize,
    act: Activation,
) {
    use std::arch::x86_64::*;

    let zero_vec = _mm256_setzero_ps();

    // For the vectorized sw==1 path, we need to ensure all 8 loads are in bounds.
    // Input pixel for output ow with kernel offset kj: ow + kj - pad_left
    // Max read index: ow + (kw-1) - pad_left + 7
    // Must be < in_w, so: ow + kw - 1 - pad_left + 7 < in_w
    // => ow < in_w + pad_left - kw - 6
    // We also need: ow + 0 - pad_left >= 0 => ow >= pad_left
    let vec_ow_end = if sw == 1 {
        in_w + pad_left.saturating_sub(kw - 1 + 7)
    } else {
        0
    };
    // Round down to multiple of 8
    let vec_ow_end = (vec_ow_end / 8) * 8;
    let vec_ow_start = if sw == 1 { pad_left } else { out_w };

    for n in 0..batch {
        for c in 0..channels {
            let in_base = n * channels * in_h * in_w + c * in_h * in_w;
            let w_base = c * kh * kw;
            let out_base = n * channels * out_h * out_w + c * out_h * out_w;

            for oh in 0..out_h {
                let mut ow = 0usize;

                // Scalar prefix (before vectorized region)
                while ow < vec_ow_start.min(out_w) {
                    depthwise_scalar_px(
                        input, weight, out, in_base, w_base, out_base,
                        oh, ow, in_h, in_w, kh, kw, sh, sw, pad_top, pad_left, out_w, act,
                    );
                    ow += 1;
                }

                // Vectorized middle (sw==1 only)
                while sw == 1 && ow + 8 <= vec_ow_end && ow + 8 <= out_w {
                    let mut acc = _mm256_setzero_ps();
                    for ki in 0..kh {
                        let ih = oh * sh + ki;
                        if ih < pad_top || ih >= in_h + pad_top { continue; }
                        let ih_valid = ih - pad_top;
                        let row_ptr = input.as_ptr().add(in_base + ih_valid * in_w);
                        for kj in 0..kw {
                            let iw_valid = ow + kj - pad_left;
                            let w_val = _mm256_set1_ps(*weight.get_unchecked(w_base + ki * kw + kj));
                            let v = _mm256_loadu_ps(row_ptr.add(iw_valid));
                            acc = _mm256_fmadd_ps(w_val, v, acc);
                        }
                    }
                    if act == Activation::Relu {
                        acc = _mm256_max_ps(acc, zero_vec);
                    }
                    _mm256_storeu_ps(out.as_mut_ptr().add(out_base + oh * out_w + ow), acc);
                    ow += 8;
                }

                // Scalar tail
                while ow < out_w {
                    depthwise_scalar_px(
                        input, weight, out, in_base, w_base, out_base,
                        oh, ow, in_h, in_w, kh, kw, sh, sw, pad_top, pad_left, out_w, act,
                    );
                    ow += 1;
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn depthwise_conv2d_3x3_s1_avx2(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
    batch: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    act: Activation,
) {
    use std::arch::x86_64::*;
    let zero_v = _mm256_setzero_ps();
    let pad = 1usize;
    // The top and bottom output rows read one input row that does not exist.
    // Pointing those taps at a row of zeros keeps every output row on the same
    // vector path; branching to a scalar path for them instead costs 2/in_h of
    // the work, and in_h here is as small as 5.
    let zero_row = vec![0.0f32; in_w];

    for n in 0..batch {
        for c in 0..channels {
            let in_base = n * channels * in_h * in_w + c * in_h * in_w;
            let w_base = c * 9;
            let out_base = n * channels * out_h * out_w + c * out_h * out_w;

            let w0 = _mm256_set1_ps(*weight.get_unchecked(w_base));
            let w1 = _mm256_set1_ps(*weight.get_unchecked(w_base + 1));
            let w2 = _mm256_set1_ps(*weight.get_unchecked(w_base + 2));
            let w3 = _mm256_set1_ps(*weight.get_unchecked(w_base + 3));
            let w4 = _mm256_set1_ps(*weight.get_unchecked(w_base + 4));
            let w5 = _mm256_set1_ps(*weight.get_unchecked(w_base + 5));
            let w6 = _mm256_set1_ps(*weight.get_unchecked(w_base + 6));
            let w7 = _mm256_set1_ps(*weight.get_unchecked(w_base + 7));
            let w8 = _mm256_set1_ps(*weight.get_unchecked(w_base + 8));

            let bias_v = bias.map(|b| _mm256_set1_ps(*b.get_unchecked(c)));

            let vec_end = (in_w / 8) * 8;

            for oh in 0..out_h {
                let out_row = out_base + oh * out_w;
                let mut ow = 0usize;

                {
                    let row = |ki: usize| {
                        let ih = oh + ki;
                        if ih >= pad && ih < in_h + pad {
                            input.as_ptr().add(in_base + (ih - pad) * in_w)
                        } else {
                            zero_row.as_ptr()
                        }
                    };
                    let r0 = row(0);
                    let r1 = row(1);
                    let r2 = row(2);

                    // Scalar: ow=0 (left pad)
                    {
                        let mut s = bias.map(|b| *b.get_unchecked(c)).unwrap_or(0.0f32);
                        for ki in 0..3usize {
                            let rp = [r0, r1, r2][ki];
                            for kj in 0..3usize {
                                let iw = ow + kj;
                                if iw > 0 && iw - 1 < in_w {
                                    s += *rp.add(iw - 1) * *weight.get_unchecked(w_base + ki * 3 + kj);
                                }
                            }
                        }
                        let v = if act == Activation::Relu && s < 0.0 { 0.0 } else { s };
                        *out.get_unchecked_mut(out_row + ow) = v;
                        ow += 1;
                    }

                    // AVX2 middle: ow=1..vec_end
                    while ow + 8 <= vec_end && ow + 8 <= out_w {
                        let mut acc = bias_v.unwrap_or(zero_v);
                        let off = ow - 1;
                        acc = _mm256_fmadd_ps(w0, _mm256_loadu_ps(r0.add(off)),     acc);
                        acc = _mm256_fmadd_ps(w1, _mm256_loadu_ps(r0.add(off + 1)), acc);
                        acc = _mm256_fmadd_ps(w2, _mm256_loadu_ps(r0.add(off + 2)), acc);
                        acc = _mm256_fmadd_ps(w3, _mm256_loadu_ps(r1.add(off)),     acc);
                        acc = _mm256_fmadd_ps(w4, _mm256_loadu_ps(r1.add(off + 1)), acc);
                        acc = _mm256_fmadd_ps(w5, _mm256_loadu_ps(r1.add(off + 2)), acc);
                        acc = _mm256_fmadd_ps(w6, _mm256_loadu_ps(r2.add(off)),     acc);
                        acc = _mm256_fmadd_ps(w7, _mm256_loadu_ps(r2.add(off + 1)), acc);
                        acc = _mm256_fmadd_ps(w8, _mm256_loadu_ps(r2.add(off + 2)), acc);
                        if act == Activation::Relu {
                            acc = _mm256_max_ps(acc, zero_v);
                        }
                        _mm256_storeu_ps(out.as_mut_ptr().add(out_row + ow), acc);
                        ow += 8;
                    }

                    // Scalar tail
                    while ow < out_w {
                        let off = ow - 1;
                        let mut s = bias.map(|b| *b.get_unchecked(c)).unwrap_or(0.0f32);
                        if off + 3 <= in_w {
                            // Whole 3-wide window is inside the row.
                            s += *r0.add(off)     * *weight.get_unchecked(w_base);
                            s += *r0.add(off + 1) * *weight.get_unchecked(w_base + 1);
                            s += *r0.add(off + 2) * *weight.get_unchecked(w_base + 2);
                            s += *r1.add(off)     * *weight.get_unchecked(w_base + 3);
                            s += *r1.add(off + 1) * *weight.get_unchecked(w_base + 4);
                            s += *r1.add(off + 2) * *weight.get_unchecked(w_base + 5);
                            s += *r2.add(off)     * *weight.get_unchecked(w_base + 6);
                            s += *r2.add(off + 1) * *weight.get_unchecked(w_base + 7);
                            s += *r2.add(off + 2) * *weight.get_unchecked(w_base + 8);
                        } else {
                            // Right edge: taps past the row are zero padding.
                            for ki in 0..3usize {
                                let rp = [r0, r1, r2][ki];
                                for kj in 0..3usize {
                                    if off + kj < in_w {
                                        s += *rp.add(off + kj)
                                            * *weight.get_unchecked(w_base + ki * 3 + kj);
                                    }
                                }
                            }
                        }
                        let v = if act == Activation::Relu && s < 0.0 { 0.0 } else { s };
                        *out.get_unchecked_mut(out_row + ow) = v;
                        ow += 1;
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn depthwise_scalar_px(
    input: &[f32],
    weight: &[f32],
    out: &mut [f32],
    in_base: usize,
    w_base: usize,
    out_base: usize,
    oh: usize,
    ow: usize,
    in_h: usize,
    in_w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    pad_top: usize,
    pad_left: usize,
    out_w: usize,
    act: Activation,
) {
    let mut sum = 0.0f32;
    for ki in 0..kh {
        let ih = (oh * sh + ki) as isize - pad_top as isize;
        if ih < 0 || ih >= in_h as isize { continue; }
        for kj in 0..kw {
            let iw = (ow * sw + kj) as isize - pad_left as isize;
            if iw < 0 || iw >= in_w as isize { continue; }
            sum += *input.get_unchecked(in_base + ih as usize * in_w + iw as usize)
                * *weight.get_unchecked(w_base + ki * kw + kj);
        }
    }
    out[out_base + oh * out_w + ow] = if act == Activation::Relu && sum < 0.0 { 0.0 } else { sum };
}

#[cfg(test)]
mod tests {
    use super::*;

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
