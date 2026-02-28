// Allow unsafe operations in unsafe functions without explicit unsafe blocks
// This is safe because the entire function body is already in an unsafe context
#![allow(unsafe_op_in_unsafe_fn)]

/// Optimized LayerNorm using NEON SIMD intrinsics on aarch64.
///
/// Key optimizations ported from candle's CPU CustomOp3 impl:
/// 1. Single-pass mean/variance: accumulate sum + sum² in one pass
/// 2. Full NEON SIMD for both accumulation and normalization phases
/// 3. Fused normalize-scale-bias in the output pass
/// 4. Fast rsqrt using NEON rsqrte + Newton-Raphson refinement

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// Fast 1/sqrt(x) using NEON rsqrte + Newton-Raphson refinement
/// Accuracy: ~1e-6 relative error
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fast_rsqrt_f32(x: f32) -> f32 {
    // Use scalar rsqrte for single value
    let x_vec = vdupq_n_f32(x);
    let y = vrsqrteq_f32(x_vec);
    // Newton-Raphson: y_new = y * (3 - x * y^2) / 2
    // One iteration gives ~23 bits of mantissa accuracy
    let y2 = vmulq_f32(y, y);
    let x_y2 = vmulq_f32(x_vec, y2);
    let three = vdupq_n_f32(3.0);
    let half = vdupq_n_f32(0.5);
    let refined = vmulq_f32(y, vmulq_f32(half, vsubq_f32(three, x_y2)));
    vgetq_lane_f32(refined, 0)
}

/// Optimized RMSNorm using NEON SIMD intrinsics on aarch64.
///
/// RMSNorm is faster than LayerNorm because it doesn't compute mean.
/// Formula: output = (x / sqrt(mean(x^2) + eps)) * weight
///
/// # Safety
/// - All pointers must be valid for their respective lengths
#[cfg(target_arch = "aarch64")]
pub unsafe fn rms_norm_neon(
    src: *const f32,
    weight: *const f32,
    out: *mut f32,
    norm_size: usize,
    outer_size: usize,
    epsilon: f32,
) {
    let simd_width = 4usize;
    let simd_end = norm_size & !(simd_width - 1);
    let inv_n = 1.0f32 / norm_size as f32;

    for i in 0..outer_size {
        let offset = i * norm_size;
        let row = src.add(offset);
        let out_row = out.add(offset);

        // Phase 1: Sum of squares using NEON
        let mut sumsq_vec = vdupq_n_f32(0.0);

        // Vectorized accumulation with 2x unrolling for better ILP
        let simd_end2 = simd_end / 8 * 8;
        let mut j = 0;

        while j < simd_end2 {
            let v0 = vld1q_f32(row.add(j));
            let v1 = vld1q_f32(row.add(j + 4));
            sumsq_vec = vfmaq_f32(sumsq_vec, v0, v0);
            sumsq_vec = vfmaq_f32(sumsq_vec, v1, v1);
            j += 8;
        }

        while j < simd_end {
            let v = vld1q_f32(row.add(j));
            sumsq_vec = vfmaq_f32(sumsq_vec, v, v);
            j += 4;
        }

        // Horizontal reduction
        let mut sumsq = vaddvq_f32(sumsq_vec);

        // Scalar tail
        for k in simd_end..norm_size {
            let x = *row.add(k);
            sumsq += x * x;
        }

        // Phase 2: Compute RMS
        let mean_sq = sumsq * inv_n;
        let rms_inv = fast_rsqrt_f32(mean_sq + epsilon);

        // Phase 3: Normalize and scale — fully NEON-vectorized
        let rms_inv_vec = vdupq_n_f32(rms_inv);

        j = 0;
        while j < simd_end2 {
            let x0 = vld1q_f32(row.add(j));
            let x1 = vld1q_f32(row.add(j + 4));
            let w0 = vld1q_f32(weight.add(j));
            let w1 = vld1q_f32(weight.add(j + 4));

            let normed0 = vmulq_f32(x0, rms_inv_vec);
            let normed1 = vmulq_f32(x1, rms_inv_vec);
            let result0 = vmulq_f32(normed0, w0);
            let result1 = vmulq_f32(normed1, w1);

            vst1q_f32(out_row.add(j), result0);
            vst1q_f32(out_row.add(j + 4), result1);
            j += 8;
        }

        while j < simd_end {
            let x = vld1q_f32(row.add(j));
            let w = vld1q_f32(weight.add(j));
            let normed = vmulq_f32(x, rms_inv_vec);
            let result = vmulq_f32(normed, w);
            vst1q_f32(out_row.add(j), result);
            j += 4;
        }

        // Scalar tail
        for k in simd_end..norm_size {
            let x = *row.add(k);
            let w = *weight.add(k);
            *out_row.add(k) = x * rms_inv * w;
        }
    }
}

/// Single-pass layer normalization using NEON intrinsics.
/// Processes each row with a single data scan for mean+variance (sum, sum²),
/// then a second scan to normalize, scale, and add bias — all NEON-vectorized.
///
/// # Safety
/// - All pointers must be valid for their respective lengths
/// - `src`, `gamma`, `beta`, and `out` must be aligned for NEON loads/stores
#[cfg(target_arch = "aarch64")]
pub unsafe fn layer_norm_neon(
    src: *const f32,
    gamma: *const f32,
    beta: *const f32,
    out: *mut f32,
    norm_size: usize,
    outer_size: usize,
    epsilon: f32,
) {
    let simd_width = 4usize;
    let simd_end = norm_size & !(simd_width - 1); // round down to multiple of 4
    let inv_n = 1.0f32 / norm_size as f32;

    for i in 0..outer_size {
        let offset = i * norm_size;
        let row = src.add(offset);
        let out_row = out.add(offset);

        // Phase 1: Single-pass sum and sum of squares using NEON
        let mut sum_vec = vdupq_n_f32(0.0);
        let mut sumsq_vec = vdupq_n_f32(0.0);

        // Vectorized accumulation
        for j in (0..simd_end).step_by(4) {
            let v = vld1q_f32(row.add(j));
            sum_vec = vaddq_f32(sum_vec, v);
            sumsq_vec = vfmaq_f32(sumsq_vec, v, v); // sumsq += v * v
        }

        // Horizontal reduction of SIMD accumulators
        let mut sum = vaddvq_f32(sum_vec);
        let mut sumsq = vaddvq_f32(sumsq_vec);

        // Scalar tail for remaining elements
        for j in simd_end..norm_size {
            let x = *row.add(j);
            sum += x;
            sumsq += x * x;
        }

        // Phase 2: Compute mean, variance, inv_std
        let mean = sum * inv_n;
        let var = sumsq * inv_n - mean * mean;
        let var_eps = var + epsilon;

        // Use fast rsqrt for 1/sqrt(var_eps)
        let inv_std = fast_rsqrt_f32(var_eps);

        // Phase 3: Normalize, scale, and bias — fully NEON-vectorized
        let mean_vec = vdupq_n_f32(mean);
        let inv_std_vec = vdupq_n_f32(inv_std);

        for j in (0..simd_end).step_by(4) {
            let x = vld1q_f32(row.add(j));
            let g = vld1q_f32(gamma.add(j));
            let b = vld1q_f32(beta.add(j));

            let d = vsubq_f32(x, mean_vec);
            let normed = vmulq_f32(d, inv_std_vec);
            let result = vfmaq_f32(b, normed, g); // result = normed * g + b

            vst1q_f32(out_row.add(j), result);
        }

        // Scalar tail
        for j in simd_end..norm_size {
            let x = *row.add(j);
            let g = *gamma.add(j);
            let b = *beta.add(j);
            *out_row.add(j) = (x - mean) * inv_std * g + b;
        }
    }
}
