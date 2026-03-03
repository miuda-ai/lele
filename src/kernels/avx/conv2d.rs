/// AVX2-accelerated direct 2D convolution for common cases.
///
/// This module bypasses the im2col→GEMM pipeline for convolutions that benefit
/// from direct computation, particularly 3×3 stride-1 convolutions where im2col
/// would expand memory 9× and cause L2/L3 cache thrashing.
///
/// Strategy: output-stationary tiles.
///   For each output channel (oc) and spatial output row (oh):
///     For each input channel (ic) and filter tap (kh, kw):
///       Accumulate: out[oc, oh, ow..ow+8] += w[oc,ic,kh,kw] * in[ic, oh+kh-pad, ow+kw..ow+kw+8]
///   This keeps the partial sums in AVX registers across all ic/tap iterations,
///   then stores once — much better than im2col which needs 9× more bandwidth.
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Direct 3×3 convolution, stride=1, dilation=1, groups=1.
/// Input: [N, C_in, H, W] (NCHW)
/// Weight: [C_out, C_in, 3, 3]
/// Output: [N, C_out, H_out, W_out]
/// Padding: (pad_top, pad_left, pad_bottom, pad_right) with H_out = H + pad_top + pad_bottom - 2
///
/// # Safety
/// Caller must ensure all pointers are valid and sizes match.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn conv2d_3x3_direct_avx2(
    input: *const f32,
    weight: *const f32,
    bias: *const f32, // nullptr → no bias
    output: *mut f32,
    batch_size: usize,
    in_channels: usize,
    out_channels: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    pad_top: usize,
    pad_left: usize,
) {
    unsafe {
        let in_spatial = in_h * in_w;
        let out_spatial = out_h * out_w;

        for n in 0..batch_size {
            let in_base = n * in_channels * in_spatial;
            let out_base = n * out_channels * out_spatial;

            for oc in 0..out_channels {
                let out_ch_base = out_base + oc * out_spatial;
                let w_oc_base = oc * in_channels * 9; // 9 = 3*3

                // Initialize output row with bias or zero
                let bias_val = if !bias.is_null() { *bias.add(oc) } else { 0.0 };
                let bias_v = _mm256_set1_ps(bias_val);

                for oh in 0..out_h {
                    let out_row_base = out_ch_base + oh * out_w;

                    // Initialize output row with bias
                    let mut ow = 0usize;
                    while ow + 8 <= out_w {
                        _mm256_storeu_ps(output.add(out_row_base + ow), bias_v);
                        ow += 8;
                    }
                    while ow < out_w {
                        *output.add(out_row_base + ow) = bias_val;
                        ow += 1;
                    }

                    // Accumulate contributions from all input channels and filter taps
                    for ic in 0..in_channels {
                        let in_ch_base = in_base + ic * in_spatial;
                        let w_ic_base = w_oc_base + ic * 9;

                        for kh in 0..3usize {
                            let ih = oh as isize + kh as isize - pad_top as isize;
                            if ih < 0 || ih >= in_h as isize {
                                continue;
                            }
                            let ih = ih as usize;
                            let in_row_base = in_ch_base + ih * in_w;

                            for kw in 0..3usize {
                                let tap_idx = kh * 3 + kw;
                                let w_val = *weight.add(w_ic_base + tap_idx);
                                let w_v = _mm256_set1_ps(w_val);

                                // iw = ow + kw - pad_left
                                // Valid range: ow in [pad_left - kw, in_w + pad_left - kw)
                                // i.e. iw = ow + kw - pad_left must be in [0, in_w)
                                let ow_start_isize = pad_left as isize - kw as isize;

                                let mut ow = 0usize;

                                // Process 8 outputs at a time
                                while ow + 8 <= out_w {
                                    let iw_start = ow as isize + kw as isize - pad_left as isize;
                                    let iw_end = iw_start + 8;

                                    if iw_start >= 0 && iw_end <= in_w as isize {
                                        // All 8 in bounds — fast path
                                        let in_ptr = input.add(in_row_base + iw_start as usize);
                                        let inp = _mm256_loadu_ps(in_ptr);
                                        let acc = _mm256_loadu_ps(output.add(out_row_base + ow));
                                        _mm256_storeu_ps(
                                            output.add(out_row_base + ow),
                                            _mm256_fmadd_ps(inp, w_v, acc),
                                        );
                                    } else {
                                        // Partial — scalar fallback for this chunk
                                        for k in 0..8usize {
                                            let iw = ow as isize + k as isize + kw as isize
                                                - pad_left as isize;
                                            if iw >= 0 && iw < in_w as isize {
                                                *output.add(out_row_base + ow + k) +=
                                                    w_val * *input.add(in_row_base + iw as usize);
                                            }
                                        }
                                    }
                                    ow += 8;
                                }

                                // Scalar tail
                                while ow < out_w {
                                    let iw =
                                        ow as isize + kw as isize - pad_left as isize;
                                    if iw >= 0 && iw < in_w as isize {
                                        *output.add(out_row_base + ow) +=
                                            w_val * *input.add(in_row_base + iw as usize);
                                    }
                                    ow += 1;
                                }

                                let _ = ow_start_isize; // suppress unused warning
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Check if this conv can use the direct AVX2 3×3 path on x86_64.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn can_use_direct_3x3(
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    groups: usize,
) -> bool {
    kernel_h == 3
        && kernel_w == 3
        && stride_h == 1
        && stride_w == 1
        && dilation_h == 1
        && dilation_w == 1
        && groups == 1
        && is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
}

/// AVX2-optimized im2col for stride=1, dilation=1
/// Uses SIMD for zeroing and copying
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn im2col_avx2(
    input: &[f32],
    batch_offset: usize,
    ch_start: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    pad_top: usize,
    pad_left: usize,
    out_h: usize,
    out_w: usize,
    spatial_cols: usize,
    col: &mut [f32],
) {
    let zero = _mm256_setzero_ps();

    for c in 0..channels {
        let ch_offset = batch_offset + (ch_start + c) * in_h * in_w;
        for kh in 0..kernel_h {
            for kw in 0..kernel_w {
                let col_row_offset = ((c * kernel_h + kh) * kernel_w + kw) * spatial_cols;

                let oh_start = if kh < pad_top { pad_top - kh } else { 0 };
                let oh_end = (in_h + pad_top).saturating_sub(kh).min(out_h);

                // Zero top padding rows using AVX2
                let col_ptr = col.as_mut_ptr().add(col_row_offset);
                for oh in 0..oh_start {
                    let row_offset = oh * out_w;
                    let ptr = col_ptr.add(row_offset);
                    let mut ow = 0usize;
                    while ow + 8 <= out_w {
                        _mm256_storeu_ps(ptr.add(ow), zero);
                        ow += 8;
                    }
                    while ow < out_w {
                        *ptr.add(ow) = 0.0;
                        ow += 1;
                    }
                }

                // Process valid rows
                for oh in oh_start..oh_end {
                    let ih = (oh + kh) as isize - pad_top as isize;
                    let ih = ih as usize;
                    let in_row_offset = ch_offset + ih * in_w;
                    let col_base = col_row_offset + oh * out_w;

                    let ow_start = if kw < pad_left { pad_left - kw } else { 0 };
                    let ow_end = (in_w + pad_left).saturating_sub(kw).min(out_w);

                    let col_ptr_row = col.as_mut_ptr().add(col_base);
                    let in_ptr = input.as_ptr().add(in_row_offset);

                    // Zero left padding
                    let mut ow = 0usize;
                    while ow + 8 <= ow_start {
                        _mm256_storeu_ps(col_ptr_row.add(ow), zero);
                        ow += 8;
                    }
                    while ow < ow_start {
                        *col_ptr_row.add(ow) = 0.0;
                        ow += 1;
                    }

                    // Copy valid region using AVX2
                    let iw_start = (ow_start + kw) as isize - pad_left as isize;
                    let count = ow_end - ow_start;
                    let src_ptr = in_ptr.add(iw_start as usize);
                    let mut copy_ow = 0usize;
                    while copy_ow + 8 <= count {
                        let v = _mm256_loadu_ps(src_ptr.add(copy_ow));
                        _mm256_storeu_ps(col_ptr_row.add(ow_start + copy_ow), v);
                        copy_ow += 8;
                    }
                    while copy_ow < count {
                        *col_ptr_row.add(ow_start + copy_ow) = *src_ptr.add(copy_ow);
                        copy_ow += 1;
                    }

                    // Zero right padding
                    ow = ow_end;
                    while ow + 8 <= out_w {
                        _mm256_storeu_ps(col_ptr_row.add(ow), zero);
                        ow += 8;
                    }
                    while ow < out_w {
                        *col_ptr_row.add(ow) = 0.0;
                        ow += 1;
                    }
                }

                // Zero bottom padding rows
                let col_ptr = col.as_mut_ptr().add(col_row_offset);
                for oh in oh_end..out_h {
                    let row_offset = oh * out_w;
                    let ptr = col_ptr.add(row_offset);
                    let mut ow = 0usize;
                    while ow + 8 <= out_w {
                        _mm256_storeu_ps(ptr.add(ow), zero);
                        ow += 8;
                    }
                    while ow < out_w {
                        *ptr.add(ow) = 0.0;
                        ow += 1;
                    }
                }
            }
        }
    }
}
