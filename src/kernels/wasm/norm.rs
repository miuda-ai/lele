#![allow(unsafe_op_in_unsafe_fn)]
//! WASM SIMD128 normalization kernel implementations.
//! Only compiled when `target_arch = "wasm32"`.

use std::arch::wasm32::*;

/// Fused multiply-add: a*b + acc.
/// Uses relaxed_madd when the `relaxed-simd` target feature is enabled.
#[inline(always)]
unsafe fn fmadd(a: v128, b: v128, acc: v128) -> v128 {
    #[cfg(target_feature = "relaxed-simd")]
    {
        f32x4_relaxed_madd(a, b, acc)
    }
    #[cfg(not(target_feature = "relaxed-simd"))]
    {
        f32x4_add(acc, f32x4_mul(a, b))
    }
}

/// Fast reciprocal sqrt using a Newton-Raphson step seeded by the WASM f32x4.sqrt
/// approximation.  One NR step achieves ~23-bit accuracy (same as NEON rsqrte+NR).
/// For norm ops a single pass is sufficient.
#[inline(always)]
unsafe fn fast_rsqrt_f32x4(x: f32) -> f32 {
    // Use scalar — we only call this once per row, so SIMD overhead isn't worth it.
    // f32 sqrt is fast on modern CPUs; reciprocal avoids a second fdiv later.
    1.0 / x.sqrt()
}

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
/// Single-pass: computes sum and sum² simultaneously (mirrors neon/normalization.rs).
/// Uses relaxed FMA when available, falls back to mul+add.
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
    let inv_n = f32x4_splat(1.0 / norm_size as f32);

    for i in 0..outer_size {
        let offset = i * norm_size;
        let chunk = src.add(offset);

        // Single-pass: accumulate sum and sum-of-squares together (fewer memory accesses)
        let mut sum_vec = f32x4_splat(0.0);
        let mut sumsq_vec = f32x4_splat(0.0);
        let mut j = 0;
        // 2× unrolled inner loop for better ILP
        let simd_end2 = (simd_end / 8) * 8;
        while j < simd_end2 {
            let v0 = v128_load(chunk.add(j) as *const v128);
            let v1 = v128_load(chunk.add(j + 4) as *const v128);
            sum_vec = f32x4_add(sum_vec, v0);
            sumsq_vec = fmadd(v0, v0, sumsq_vec);
            sum_vec = f32x4_add(sum_vec, v1);
            sumsq_vec = fmadd(v1, v1, sumsq_vec);
            j += 8;
        }
        while j < simd_end {
            let v = v128_load(chunk.add(j) as *const v128);
            sum_vec = f32x4_add(sum_vec, v);
            sumsq_vec = fmadd(v, v, sumsq_vec);
            j += 4;
        }

        let mut sum = hsum_f32x4(sum_vec);
        let mut sumsq = hsum_f32x4(sumsq_vec);
        for k in simd_end..norm_size {
            let x = *chunk.add(k);
            sum += x;
            sumsq += x * x;
        }

        let inv_n_s = 1.0 / norm_size as f32;
        let mean = sum * inv_n_s;
        let var = sumsq * inv_n_s - mean * mean;
        let inv_std = fast_rsqrt_f32x4(var + epsilon);

        // Normalize, scale, bias — fused SIMD pass
        let mean_vec = f32x4_splat(mean);
        let inv_std_vec = f32x4_splat(inv_std);
        let out_chunk = out.add(offset);
        j = 0;
        while j < simd_end2 {
            let x0 = v128_load(chunk.add(j) as *const v128);
            let x1 = v128_load(chunk.add(j + 4) as *const v128);
            let g0 = v128_load(gamma.add(j) as *const v128);
            let g1 = v128_load(gamma.add(j + 4) as *const v128);
            let b0 = v128_load(beta.add(j) as *const v128);
            let b1 = v128_load(beta.add(j + 4) as *const v128);
            // (x - mean) * inv_std * gamma + beta
            let d0 = f32x4_mul(f32x4_sub(x0, mean_vec), inv_std_vec);
            let d1 = f32x4_mul(f32x4_sub(x1, mean_vec), inv_std_vec);
            v128_store(out_chunk.add(j) as *mut v128, fmadd(d0, g0, b0));
            v128_store(out_chunk.add(j + 4) as *mut v128, fmadd(d1, g1, b1));
            j += 8;
        }
        while j < simd_end {
            let x = v128_load(chunk.add(j) as *const v128);
            let g = v128_load(gamma.add(j) as *const v128);
            let b = v128_load(beta.add(j) as *const v128);
            let d = f32x4_mul(f32x4_sub(x, mean_vec), inv_std_vec);
            v128_store(out_chunk.add(j) as *mut v128, fmadd(d, g, b));
            j += 4;
        }
        for k in simd_end..norm_size {
            let x = *chunk.add(k);
            let g = *gamma.add(k);
            let b = *beta.add(k);
            *out_chunk.add(k) = (x - mean) * inv_std * g + b;
        }
    }
    let _ = inv_n; // suppress unused warning
}

/// WASM SIMD128 RMSNorm: x / rms(x) * weight
/// Mirrors neon/normalization.rs::rms_norm_neon.
/// Uses single-pass sum-of-squares accumulation with FMA + fast rsqrt.
pub unsafe fn rms_norm(
    src: *const f32,
    weight: *const f32,
    out: *mut f32,
    norm_size: usize,
    outer_size: usize,
    epsilon: f32,
) {
    let simd_end = (norm_size / 4) * 4;
    let inv_n = 1.0 / norm_size as f32;

    for i in 0..outer_size {
        let offset = i * norm_size;
        let row = src.add(offset);
        let out_row = out.add(offset);

        // Phase 1: sum of squares with 2× unrolling (mirrors NEON version)
        let mut sumsq_vec = f32x4_splat(0.0);
        let mut sumsq_vec2 = f32x4_splat(0.0);
        let simd_end2 = (simd_end / 8) * 8;
        let mut j = 0;
        while j < simd_end2 {
            let v0 = v128_load(row.add(j) as *const v128);
            let v1 = v128_load(row.add(j + 4) as *const v128);
            sumsq_vec = fmadd(v0, v0, sumsq_vec);
            sumsq_vec2 = fmadd(v1, v1, sumsq_vec2);
            j += 8;
        }
        let mut sumsq_acc = f32x4_add(sumsq_vec, sumsq_vec2);
        while j < simd_end {
            let v = v128_load(row.add(j) as *const v128);
            sumsq_acc = fmadd(v, v, sumsq_acc);
            j += 4;
        }
        let mut sumsq = hsum_f32x4(sumsq_acc);
        for k in simd_end..norm_size {
            let x = *row.add(k);
            sumsq += x * x;
        }

        // Phase 2: rms_inv = 1 / sqrt(mean(x²) + eps)
        let mean_sq = sumsq * inv_n;
        let rms_inv = fast_rsqrt_f32x4(mean_sq + epsilon);
        let rms_vec = f32x4_splat(rms_inv);

        // Phase 3: normalize and scale — fused SIMD
        j = 0;
        while j < simd_end2 {
            let x0 = v128_load(row.add(j) as *const v128);
            let x1 = v128_load(row.add(j + 4) as *const v128);
            let w0 = v128_load(weight.add(j) as *const v128);
            let w1 = v128_load(weight.add(j + 4) as *const v128);
            v128_store(
                out_row.add(j) as *mut v128,
                f32x4_mul(f32x4_mul(x0, rms_vec), w0),
            );
            v128_store(
                out_row.add(j + 4) as *mut v128,
                f32x4_mul(f32x4_mul(x1, rms_vec), w1),
            );
            j += 8;
        }
        while j < simd_end {
            let x = v128_load(row.add(j) as *const v128);
            let w = v128_load(weight.add(j) as *const v128);
            v128_store(
                out_row.add(j) as *mut v128,
                f32x4_mul(f32x4_mul(x, rms_vec), w),
            );
            j += 4;
        }
        for k in simd_end..norm_size {
            *out_row.add(k) = *row.add(k) * rms_inv * *weight.add(k);
        }
    }
}

/// WASM SIMD128 fused scale+bias for a contiguous spatial slice:
/// out[i] = data[i] * scale + bias_val
pub unsafe fn scale_bias_spatial(
    src: *const f32,
    out: *mut f32,
    scale: f32,
    bias_val: f32,
    len: usize,
) {
    let vs = f32x4_splat(scale);
    let vb = f32x4_splat(bias_val);
    let mut i = 0;
    let end16 = (len / 16) * 16;
    while i < end16 {
        let v0 = v128_load(src.add(i) as *const v128);
        let v1 = v128_load(src.add(i + 4) as *const v128);
        let v2 = v128_load(src.add(i + 8) as *const v128);
        let v3 = v128_load(src.add(i + 12) as *const v128);
        v128_store(out.add(i) as *mut v128, fmadd(v0, vs, vb));
        v128_store(out.add(i + 4) as *mut v128, fmadd(v1, vs, vb));
        v128_store(out.add(i + 8) as *mut v128, fmadd(v2, vs, vb));
        v128_store(out.add(i + 12) as *mut v128, fmadd(v3, vs, vb));
        i += 16;
    }
    while i + 4 <= len {
        let v = v128_load(src.add(i) as *const v128);
        v128_store(out.add(i) as *mut v128, fmadd(v, vs, vb));
        i += 4;
    }
    while i < len {
        *out.add(i) = *src.add(i) * scale + bias_val;
        i += 1;
    }
}
