#![allow(unsafe_op_in_unsafe_fn)]
//! WASM SIMD128 normalization kernel implementations.
//! Only compiled when `target_arch = "wasm32"`.

use std::arch::wasm32::*;

/// Horizontal sum of f32x4 → f32
#[inline(always)]
pub unsafe fn hsum_f32x4(v: v128) -> f32 {
    // v = [a, b, c, d]
    // shuffle to get [c, d, ?, ?] and add → [a+c, b+d, ?, ?]
    let hi = i32x4_shuffle::<2, 3, 0, 1>(v, v);
    let sum2 = f32x4_add(v, hi);
    // shuffle to get [b+d, ?, ?, ?] and add → [a+b+c+d, ?, ?, ?]
    let hi2 = i32x4_shuffle::<1, 0, 2, 3>(sum2, sum2);
    let sum4 = f32x4_add(sum2, hi2);
    f32x4_extract_lane::<0>(sum4)
}

/// WASM SIMD128 softmax over a contiguous slice (inner_size == 1 case).
pub unsafe fn softmax(src: &[f32], dst: &mut [f32]) {
    let len = src.len();
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let simd_end = (len / 4) * 4;

    // 1. Find max
    let mut max_vec = f32x4_splat(f32::MIN);
    let mut j = 0;
    while j < simd_end {
        let v = v128_load(src_ptr.add(j) as *const v128);
        max_vec = f32x4_max(max_vec, v);
        j += 4;
    }
    let mut max_val = f32x4_extract_lane::<0>(max_vec)
        .max(f32x4_extract_lane::<1>(max_vec))
        .max(f32x4_extract_lane::<2>(max_vec))
        .max(f32x4_extract_lane::<3>(max_vec));
    for k in simd_end..len {
        max_val = max_val.max(*src_ptr.add(k));
    }

    // 2. exp(x - max) and sum
    let max_broadcast = f32x4_splat(max_val);
    let mut sum_vec = f32x4_splat(0.0);
    j = 0;
    while j < simd_end {
        let v = v128_load(src_ptr.add(j) as *const v128);
        let shifted = f32x4_sub(v, max_broadcast);
        let exp_val = crate::kernels::wasm::math::exp_f32x4(shifted);
        v128_store(dst_ptr.add(j) as *mut v128, exp_val);
        sum_vec = f32x4_add(sum_vec, exp_val);
        j += 4;
    }
    let mut sum = hsum_f32x4(sum_vec);
    for k in simd_end..len {
        let e = (*src_ptr.add(k) - max_val).exp();
        *dst_ptr.add(k) = e;
        sum += e;
    }

    // 3. Normalize
    let inv_sum = 1.0 / sum;
    let inv_sum_vec = f32x4_splat(inv_sum);
    j = 0;
    while j < simd_end {
        let v = v128_load(dst_ptr.add(j) as *const v128);
        let normalized = f32x4_mul(v, inv_sum_vec);
        v128_store(dst_ptr.add(j) as *mut v128, normalized);
        j += 4;
    }
    for k in simd_end..len {
        *dst_ptr.add(k) *= inv_sum;
    }
}

/// WASM SIMD128 LayerNorm: (x - mean) * inv_std * gamma + beta
pub unsafe fn layer_norm(
    src: *const f32,
    gamma: *const f32,
    beta: *const f32,
    out: *mut f32,
    norm_size: usize,
    outer_size: usize,
    epsilon: f32,
) {
    let simd_end = (norm_size / 4) * 4;

    for i in 0..outer_size {
        let offset = i * norm_size;
        let chunk = src.add(offset);

        // Compute mean using SIMD
        let mut sum_vec = f32x4_splat(0.0);
        let mut j = 0;
        while j < simd_end {
            let v = v128_load(chunk.add(j) as *const v128);
            sum_vec = f32x4_add(sum_vec, v);
            j += 4;
        }
        let mut sum = hsum_f32x4(sum_vec);
        for k in simd_end..norm_size {
            sum += *chunk.add(k);
        }
        let mean = sum / norm_size as f32;

        // Compute variance using SIMD
        let mean_vec = f32x4_splat(mean);
        let mut var_vec = f32x4_splat(0.0);
        j = 0;
        while j < simd_end {
            let v = v128_load(chunk.add(j) as *const v128);
            let d = f32x4_sub(v, mean_vec);
            var_vec = f32x4_add(var_vec, f32x4_mul(d, d));
            j += 4;
        }
        let mut var_sum = hsum_f32x4(var_vec);
        for k in simd_end..norm_size {
            let d = *chunk.add(k) - mean;
            var_sum += d * d;
        }
        let inv_std = 1.0 / (var_sum / norm_size as f32 + epsilon).sqrt();

        // Normalize: (x - mean) * inv_std * gamma + beta
        let inv_std_vec = f32x4_splat(inv_std);
        let out_chunk = out.add(offset);
        j = 0;
        while j < simd_end {
            let x = v128_load(chunk.add(j) as *const v128);
            let g = v128_load(gamma.add(j) as *const v128);
            let b = v128_load(beta.add(j) as *const v128);
            let d = f32x4_sub(x, mean_vec);
            let normed = f32x4_mul(d, inv_std_vec);
            let result = f32x4_add(f32x4_mul(normed, g), b);
            v128_store(out_chunk.add(j) as *mut v128, result);
            j += 4;
        }
        for k in simd_end..norm_size {
            *out_chunk.add(k) = (*chunk.add(k) - mean) * inv_std * *gamma.add(k) + *beta.add(k);
        }
    }
}
