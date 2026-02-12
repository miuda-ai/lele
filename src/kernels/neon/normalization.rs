/// Optimized LayerNorm using NEON SIMD intrinsics on aarch64.
///
/// Key optimizations ported from candle's CPU CustomOp3 impl:
/// 1. Single-pass mean/variance: accumulate sum + sum² in one pass
/// 2. Full NEON SIMD for both accumulation and normalization phases
/// 3. Fused normalize-scale-bias in the output pass

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// Single-pass layer normalization using NEON intrinsics.
/// Processes each row with a single data scan for mean+variance (sum, sum²),
/// then a second scan to normalize, scale, and add bias — all NEON-vectorized.
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn layer_norm_neon(
    src: *const f32,
    gamma: *const f32,
    beta: *const f32,
    dst: *mut f32,
    norm_size: usize,
    outer_size: usize,
    epsilon: f32,
) {
    let simd_width = 4usize;
    let simd_end = norm_size & !(simd_width - 1); // round down to multiple of 4
    let inv_n = 1.0f32 / norm_size as f32;
    let eps_vec = vdupq_n_f32(epsilon);

    for i in 0..outer_size {
        let offset = i * norm_size;
        let row = src.add(offset);
        let out_row = dst.add(offset);

        // Phase 1: Single-pass accumulate sum and sum² with NEON
        let mut sum_v0 = vdupq_n_f32(0.0);
        let mut sum_v1 = vdupq_n_f32(0.0);
        let mut sumsq_v0 = vdupq_n_f32(0.0);
        let mut sumsq_v1 = vdupq_n_f32(0.0);

        let mut j = 0usize;
        // Process 8 floats per iteration (2x4 NEON)
        let simd_end_8 = norm_size & !(7usize);
        while j < simd_end_8 {
            let v0 = vld1q_f32(row.add(j));
            let v1 = vld1q_f32(row.add(j + 4));
            sum_v0 = vaddq_f32(sum_v0, v0);
            sum_v1 = vaddq_f32(sum_v1, v1);
            sumsq_v0 = vfmaq_f32(sumsq_v0, v0, v0);
            sumsq_v1 = vfmaq_f32(sumsq_v1, v1, v1);
            j += 8;
        }
        // Process remaining groups of 4
        while j < simd_end {
            let v = vld1q_f32(row.add(j));
            sum_v0 = vaddq_f32(sum_v0, v);
            sumsq_v0 = vfmaq_f32(sumsq_v0, v, v);
            j += 4;
        }
        // Reduce NEON vectors to scalars
        sum_v0 = vaddq_f32(sum_v0, sum_v1);
        sumsq_v0 = vaddq_f32(sumsq_v0, sumsq_v1);
        let mut sum = vaddvq_f32(sum_v0);
        let mut sumsq = vaddvq_f32(sumsq_v0);
        // Handle scalar tail
        while j < norm_size {
            let v = *row.add(j);
            sum += v;
            sumsq += v * v;
            j += 1;
        }

        // Phase 2: Compute mean, variance, inv_std
        let mean = sum * inv_n;
        let var = sumsq * inv_n - mean * mean;
        // Use NEON rsqrt estimate + Newton-Raphson refinement for fast 1/sqrt
        let var_eps = var + epsilon;
        let inv_std = 1.0 / var_eps.sqrt();

        // Phase 3: Normalize, scale, and bias — fully NEON-vectorized
        let mean_vec = vdupq_n_f32(mean);
        let inv_std_vec = vdupq_n_f32(inv_std);

        j = 0;
        while j < simd_end_8 {
            // Load input
            let x0 = vld1q_f32(row.add(j));
            let x1 = vld1q_f32(row.add(j + 4));
            // Load gamma, beta
            let g0 = vld1q_f32(gamma.add(j));
            let g1 = vld1q_f32(gamma.add(j + 4));
            let b0 = vld1q_f32(beta.add(j));
            let b1 = vld1q_f32(beta.add(j + 4));
            // (x - mean) * inv_std
            let d0 = vmulq_f32(vsubq_f32(x0, mean_vec), inv_std_vec);
            let d1 = vmulq_f32(vsubq_f32(x1, mean_vec), inv_std_vec);
            // d * gamma + beta
            let r0 = vfmaq_f32(b0, d0, g0);
            let r1 = vfmaq_f32(b1, d1, g1);
            vst1q_f32(out_row.add(j), r0);
            vst1q_f32(out_row.add(j + 4), r1);
            j += 8;
        }
        while j < simd_end {
            let x = vld1q_f32(row.add(j));
            let g = vld1q_f32(gamma.add(j));
            let b = vld1q_f32(beta.add(j));
            let d = vmulq_f32(vsubq_f32(x, mean_vec), inv_std_vec);
            let r = vfmaq_f32(b, d, g);
            vst1q_f32(out_row.add(j), r);
            j += 4;
        }
        // Scalar tail
        while j < norm_size {
            let x = *row.add(j);
            let g = *gamma.add(j);
            let b = *beta.add(j);
            *out_row.add(j) = (x - mean) * inv_std * g + b;
            j += 1;
        }
    }
}
