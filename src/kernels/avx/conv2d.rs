#![allow(unsafe_op_in_unsafe_fn)]
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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
