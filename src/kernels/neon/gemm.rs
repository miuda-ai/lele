//! High-performance FP32 GEMM kernel using ARM NEON intrinsics.
//!
//! Implements C = alpha * A * B + beta * C with row-major storage.
//!
//! Key optimizations:
//! - 4x12 micro-kernel using NEON FMLA (fused multiply-add)
//! - Double accumulators to reduce dependency chains
//! - 8x loop unrolling for better ILP
//! - Interleaved load/compute to hide memory latency

// Allow unsafe operations in unsafe functions without explicit unsafe blocks
#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// 4x12 NEON micro-kernel with double accumulators
/// This reduces FMLA dependency chains and improves instruction-level parallelism
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn gemm_4x12_kernel(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    k: usize,
    n: usize,
    alpha: f32,
) {
    // Double accumulators for each output to reduce dependency chains
    // This allows the CPU to execute FMLA instructions in parallel
    let mut acc00_0 = vdupq_n_f32(0.0);
    let mut acc00_1 = vdupq_n_f32(0.0);
    let mut acc01_0 = vdupq_n_f32(0.0);
    let mut acc01_1 = vdupq_n_f32(0.0);
    let mut acc02_0 = vdupq_n_f32(0.0);
    let mut acc02_1 = vdupq_n_f32(0.0);

    let mut acc10_0 = vdupq_n_f32(0.0);
    let mut acc10_1 = vdupq_n_f32(0.0);
    let mut acc11_0 = vdupq_n_f32(0.0);
    let mut acc11_1 = vdupq_n_f32(0.0);
    let mut acc12_0 = vdupq_n_f32(0.0);
    let mut acc12_1 = vdupq_n_f32(0.0);

    let mut acc20_0 = vdupq_n_f32(0.0);
    let mut acc20_1 = vdupq_n_f32(0.0);
    let mut acc21_0 = vdupq_n_f32(0.0);
    let mut acc21_1 = vdupq_n_f32(0.0);
    let mut acc22_0 = vdupq_n_f32(0.0);
    let mut acc22_1 = vdupq_n_f32(0.0);

    let mut acc30_0 = vdupq_n_f32(0.0);
    let mut acc30_1 = vdupq_n_f32(0.0);
    let mut acc31_0 = vdupq_n_f32(0.0);
    let mut acc31_1 = vdupq_n_f32(0.0);
    let mut acc32_0 = vdupq_n_f32(0.0);
    let mut acc32_1 = vdupq_n_f32(0.0);

    let a0 = a;
    let a1 = a.add(k);
    let a2 = a.add(k * 2);
    let a3 = a.add(k * 3);

    // 8x unrolling with double accumulators
    let k8 = k / 8 * 8;
    let mut kk = 0;

    while kk < k8 {
        // Iteration 1 - use _0 accumulators
        let b_row = b.add(kk * n);
        let vb0 = vld1q_f32(b_row);
        let vb1 = vld1q_f32(b_row.add(4));
        let vb2 = vld1q_f32(b_row.add(8));
        let va0 = *a0.add(kk);
        let va1 = *a1.add(kk);
        let va2 = *a2.add(kk);
        let va3 = *a3.add(kk);
        acc00_0 = vfmaq_n_f32(acc00_0, vb0, va0);
        acc01_0 = vfmaq_n_f32(acc01_0, vb1, va0);
        acc02_0 = vfmaq_n_f32(acc02_0, vb2, va0);
        acc10_0 = vfmaq_n_f32(acc10_0, vb0, va1);
        acc11_0 = vfmaq_n_f32(acc11_0, vb1, va1);
        acc12_0 = vfmaq_n_f32(acc12_0, vb2, va1);
        acc20_0 = vfmaq_n_f32(acc20_0, vb0, va2);
        acc21_0 = vfmaq_n_f32(acc21_0, vb1, va2);
        acc22_0 = vfmaq_n_f32(acc22_0, vb2, va2);
        acc30_0 = vfmaq_n_f32(acc30_0, vb0, va3);
        acc31_0 = vfmaq_n_f32(acc31_0, vb1, va3);
        acc32_0 = vfmaq_n_f32(acc32_0, vb2, va3);

        // Iteration 2 - use _1 accumulators
        let b_row = b.add((kk + 1) * n);
        let vb0 = vld1q_f32(b_row);
        let vb1 = vld1q_f32(b_row.add(4));
        let vb2 = vld1q_f32(b_row.add(8));
        let va0 = *a0.add(kk + 1);
        let va1 = *a1.add(kk + 1);
        let va2 = *a2.add(kk + 1);
        let va3 = *a3.add(kk + 1);
        acc00_1 = vfmaq_n_f32(acc00_1, vb0, va0);
        acc01_1 = vfmaq_n_f32(acc01_1, vb1, va0);
        acc02_1 = vfmaq_n_f32(acc02_1, vb2, va0);
        acc10_1 = vfmaq_n_f32(acc10_1, vb0, va1);
        acc11_1 = vfmaq_n_f32(acc11_1, vb1, va1);
        acc12_1 = vfmaq_n_f32(acc12_1, vb2, va1);
        acc20_1 = vfmaq_n_f32(acc20_1, vb0, va2);
        acc21_1 = vfmaq_n_f32(acc21_1, vb1, va2);
        acc22_1 = vfmaq_n_f32(acc22_1, vb2, va2);
        acc30_1 = vfmaq_n_f32(acc30_1, vb0, va3);
        acc31_1 = vfmaq_n_f32(acc31_1, vb1, va3);
        acc32_1 = vfmaq_n_f32(acc32_1, vb2, va3);

        // Iteration 3 - use _0 accumulators
        let b_row = b.add((kk + 2) * n);
        let vb0 = vld1q_f32(b_row);
        let vb1 = vld1q_f32(b_row.add(4));
        let vb2 = vld1q_f32(b_row.add(8));
        let va0 = *a0.add(kk + 2);
        let va1 = *a1.add(kk + 2);
        let va2 = *a2.add(kk + 2);
        let va3 = *a3.add(kk + 2);
        acc00_0 = vfmaq_n_f32(acc00_0, vb0, va0);
        acc01_0 = vfmaq_n_f32(acc01_0, vb1, va0);
        acc02_0 = vfmaq_n_f32(acc02_0, vb2, va0);
        acc10_0 = vfmaq_n_f32(acc10_0, vb0, va1);
        acc11_0 = vfmaq_n_f32(acc11_0, vb1, va1);
        acc12_0 = vfmaq_n_f32(acc12_0, vb2, va1);
        acc20_0 = vfmaq_n_f32(acc20_0, vb0, va2);
        acc21_0 = vfmaq_n_f32(acc21_0, vb1, va2);
        acc22_0 = vfmaq_n_f32(acc22_0, vb2, va2);
        acc30_0 = vfmaq_n_f32(acc30_0, vb0, va3);
        acc31_0 = vfmaq_n_f32(acc31_0, vb1, va3);
        acc32_0 = vfmaq_n_f32(acc32_0, vb2, va3);

        // Iteration 4 - use _1 accumulators
        let b_row = b.add((kk + 3) * n);
        let vb0 = vld1q_f32(b_row);
        let vb1 = vld1q_f32(b_row.add(4));
        let vb2 = vld1q_f32(b_row.add(8));
        let va0 = *a0.add(kk + 3);
        let va1 = *a1.add(kk + 3);
        let va2 = *a2.add(kk + 3);
        let va3 = *a3.add(kk + 3);
        acc00_1 = vfmaq_n_f32(acc00_1, vb0, va0);
        acc01_1 = vfmaq_n_f32(acc01_1, vb1, va0);
        acc02_1 = vfmaq_n_f32(acc02_1, vb2, va0);
        acc10_1 = vfmaq_n_f32(acc10_1, vb0, va1);
        acc11_1 = vfmaq_n_f32(acc11_1, vb1, va1);
        acc12_1 = vfmaq_n_f32(acc12_1, vb2, va1);
        acc20_1 = vfmaq_n_f32(acc20_1, vb0, va2);
        acc21_1 = vfmaq_n_f32(acc21_1, vb1, va2);
        acc22_1 = vfmaq_n_f32(acc22_1, vb2, va2);
        acc30_1 = vfmaq_n_f32(acc30_1, vb0, va3);
        acc31_1 = vfmaq_n_f32(acc31_1, vb1, va3);
        acc32_1 = vfmaq_n_f32(acc32_1, vb2, va3);

        // Iteration 5 - use _0 accumulators
        let b_row = b.add((kk + 4) * n);
        let vb0 = vld1q_f32(b_row);
        let vb1 = vld1q_f32(b_row.add(4));
        let vb2 = vld1q_f32(b_row.add(8));
        let va0 = *a0.add(kk + 4);
        let va1 = *a1.add(kk + 4);
        let va2 = *a2.add(kk + 4);
        let va3 = *a3.add(kk + 4);
        acc00_0 = vfmaq_n_f32(acc00_0, vb0, va0);
        acc01_0 = vfmaq_n_f32(acc01_0, vb1, va0);
        acc02_0 = vfmaq_n_f32(acc02_0, vb2, va0);
        acc10_0 = vfmaq_n_f32(acc10_0, vb0, va1);
        acc11_0 = vfmaq_n_f32(acc11_0, vb1, va1);
        acc12_0 = vfmaq_n_f32(acc12_0, vb2, va1);
        acc20_0 = vfmaq_n_f32(acc20_0, vb0, va2);
        acc21_0 = vfmaq_n_f32(acc21_0, vb1, va2);
        acc22_0 = vfmaq_n_f32(acc22_0, vb2, va2);
        acc30_0 = vfmaq_n_f32(acc30_0, vb0, va3);
        acc31_0 = vfmaq_n_f32(acc31_0, vb1, va3);
        acc32_0 = vfmaq_n_f32(acc32_0, vb2, va3);

        // Iteration 6 - use _1 accumulators
        let b_row = b.add((kk + 5) * n);
        let vb0 = vld1q_f32(b_row);
        let vb1 = vld1q_f32(b_row.add(4));
        let vb2 = vld1q_f32(b_row.add(8));
        let va0 = *a0.add(kk + 5);
        let va1 = *a1.add(kk + 5);
        let va2 = *a2.add(kk + 5);
        let va3 = *a3.add(kk + 5);
        acc00_1 = vfmaq_n_f32(acc00_1, vb0, va0);
        acc01_1 = vfmaq_n_f32(acc01_1, vb1, va0);
        acc02_1 = vfmaq_n_f32(acc02_1, vb2, va0);
        acc10_1 = vfmaq_n_f32(acc10_1, vb0, va1);
        acc11_1 = vfmaq_n_f32(acc11_1, vb1, va1);
        acc12_1 = vfmaq_n_f32(acc12_1, vb2, va1);
        acc20_1 = vfmaq_n_f32(acc20_1, vb0, va2);
        acc21_1 = vfmaq_n_f32(acc21_1, vb1, va2);
        acc22_1 = vfmaq_n_f32(acc22_1, vb2, va2);
        acc30_1 = vfmaq_n_f32(acc30_1, vb0, va3);
        acc31_1 = vfmaq_n_f32(acc31_1, vb1, va3);
        acc32_1 = vfmaq_n_f32(acc32_1, vb2, va3);

        // Iteration 7 - use _0 accumulators
        let b_row = b.add((kk + 6) * n);
        let vb0 = vld1q_f32(b_row);
        let vb1 = vld1q_f32(b_row.add(4));
        let vb2 = vld1q_f32(b_row.add(8));
        let va0 = *a0.add(kk + 6);
        let va1 = *a1.add(kk + 6);
        let va2 = *a2.add(kk + 6);
        let va3 = *a3.add(kk + 6);
        acc00_0 = vfmaq_n_f32(acc00_0, vb0, va0);
        acc01_0 = vfmaq_n_f32(acc01_0, vb1, va0);
        acc02_0 = vfmaq_n_f32(acc02_0, vb2, va0);
        acc10_0 = vfmaq_n_f32(acc10_0, vb0, va1);
        acc11_0 = vfmaq_n_f32(acc11_0, vb1, va1);
        acc12_0 = vfmaq_n_f32(acc12_0, vb2, va1);
        acc20_0 = vfmaq_n_f32(acc20_0, vb0, va2);
        acc21_0 = vfmaq_n_f32(acc21_0, vb1, va2);
        acc22_0 = vfmaq_n_f32(acc22_0, vb2, va2);
        acc30_0 = vfmaq_n_f32(acc30_0, vb0, va3);
        acc31_0 = vfmaq_n_f32(acc31_0, vb1, va3);
        acc32_0 = vfmaq_n_f32(acc32_0, vb2, va3);

        // Iteration 8 - use _1 accumulators
        let b_row = b.add((kk + 7) * n);
        let vb0 = vld1q_f32(b_row);
        let vb1 = vld1q_f32(b_row.add(4));
        let vb2 = vld1q_f32(b_row.add(8));
        let va0 = *a0.add(kk + 7);
        let va1 = *a1.add(kk + 7);
        let va2 = *a2.add(kk + 7);
        let va3 = *a3.add(kk + 7);
        acc00_1 = vfmaq_n_f32(acc00_1, vb0, va0);
        acc01_1 = vfmaq_n_f32(acc01_1, vb1, va0);
        acc02_1 = vfmaq_n_f32(acc02_1, vb2, va0);
        acc10_1 = vfmaq_n_f32(acc10_1, vb0, va1);
        acc11_1 = vfmaq_n_f32(acc11_1, vb1, va1);
        acc12_1 = vfmaq_n_f32(acc12_1, vb2, va1);
        acc20_1 = vfmaq_n_f32(acc20_1, vb0, va2);
        acc21_1 = vfmaq_n_f32(acc21_1, vb1, va2);
        acc22_1 = vfmaq_n_f32(acc22_1, vb2, va2);
        acc30_1 = vfmaq_n_f32(acc30_1, vb0, va3);
        acc31_1 = vfmaq_n_f32(acc31_1, vb1, va3);
        acc32_1 = vfmaq_n_f32(acc32_1, vb2, va3);

        kk += 8;
    }

    // Handle remaining 4 iterations
    let k4 = k / 4 * 4;
    while kk < k4 {
        let b_row = b.add(kk * n);
        let vb0 = vld1q_f32(b_row);
        let vb1 = vld1q_f32(b_row.add(4));
        let vb2 = vld1q_f32(b_row.add(8));
        let va0 = *a0.add(kk);
        let va1 = *a1.add(kk);
        let va2 = *a2.add(kk);
        let va3 = *a3.add(kk);
        acc00_0 = vfmaq_n_f32(acc00_0, vb0, va0);
        acc01_0 = vfmaq_n_f32(acc01_0, vb1, va0);
        acc02_0 = vfmaq_n_f32(acc02_0, vb2, va0);
        acc10_0 = vfmaq_n_f32(acc10_0, vb0, va1);
        acc11_0 = vfmaq_n_f32(acc11_0, vb1, va1);
        acc12_0 = vfmaq_n_f32(acc12_0, vb2, va1);
        acc20_0 = vfmaq_n_f32(acc20_0, vb0, va2);
        acc21_0 = vfmaq_n_f32(acc21_0, vb1, va2);
        acc22_0 = vfmaq_n_f32(acc22_0, vb2, va2);
        acc30_0 = vfmaq_n_f32(acc30_0, vb0, va3);
        acc31_0 = vfmaq_n_f32(acc31_0, vb1, va3);
        acc32_0 = vfmaq_n_f32(acc32_0, vb2, va3);
        kk += 1;
    }

    // Handle final remaining iterations
    while kk < k {
        let b_row = b.add(kk * n);
        let vb0 = vld1q_f32(b_row);
        let vb1 = vld1q_f32(b_row.add(4));
        let vb2 = vld1q_f32(b_row.add(8));
        let va0 = *a0.add(kk);
        let va1 = *a1.add(kk);
        let va2 = *a2.add(kk);
        let va3 = *a3.add(kk);
        acc00_0 = vfmaq_n_f32(acc00_0, vb0, va0);
        acc01_0 = vfmaq_n_f32(acc01_0, vb1, va0);
        acc02_0 = vfmaq_n_f32(acc02_0, vb2, va0);
        acc10_0 = vfmaq_n_f32(acc10_0, vb0, va1);
        acc11_0 = vfmaq_n_f32(acc11_0, vb1, va1);
        acc12_0 = vfmaq_n_f32(acc12_0, vb2, va1);
        acc20_0 = vfmaq_n_f32(acc20_0, vb0, va2);
        acc21_0 = vfmaq_n_f32(acc21_0, vb1, va2);
        acc22_0 = vfmaq_n_f32(acc22_0, vb2, va2);
        acc30_0 = vfmaq_n_f32(acc30_0, vb0, va3);
        acc31_0 = vfmaq_n_f32(acc31_0, vb1, va3);
        acc32_0 = vfmaq_n_f32(acc32_0, vb2, va3);
        kk += 1;
    }

    // Merge double accumulators
    let acc00 = vaddq_f32(acc00_0, acc00_1);
    let acc01 = vaddq_f32(acc01_0, acc01_1);
    let acc02 = vaddq_f32(acc02_0, acc02_1);
    let acc10 = vaddq_f32(acc10_0, acc10_1);
    let acc11 = vaddq_f32(acc11_0, acc11_1);
    let acc12 = vaddq_f32(acc12_0, acc12_1);
    let acc20 = vaddq_f32(acc20_0, acc20_1);
    let acc21 = vaddq_f32(acc21_0, acc21_1);
    let acc22 = vaddq_f32(acc22_0, acc22_1);
    let acc30 = vaddq_f32(acc30_0, acc30_1);
    let acc31 = vaddq_f32(acc31_0, acc31_1);
    let acc32 = vaddq_f32(acc32_0, acc32_1);

    // Apply alpha scaling and store
    let alpha_vec = vdupq_n_f32(alpha);
    vst1q_f32(c, vmulq_f32(acc00, alpha_vec));
    vst1q_f32(c.add(4), vmulq_f32(acc01, alpha_vec));
    vst1q_f32(c.add(8), vmulq_f32(acc02, alpha_vec));

    let c1 = c.add(n);
    vst1q_f32(c1, vmulq_f32(acc10, alpha_vec));
    vst1q_f32(c1.add(4), vmulq_f32(acc11, alpha_vec));
    vst1q_f32(c1.add(8), vmulq_f32(acc12, alpha_vec));

    let c2 = c.add(n * 2);
    vst1q_f32(c2, vmulq_f32(acc20, alpha_vec));
    vst1q_f32(c2.add(4), vmulq_f32(acc21, alpha_vec));
    vst1q_f32(c2.add(8), vmulq_f32(acc22, alpha_vec));

    let c3 = c.add(n * 3);
    vst1q_f32(c3, vmulq_f32(acc30, alpha_vec));
    vst1q_f32(c3.add(4), vmulq_f32(acc31, alpha_vec));
    vst1q_f32(c3.add(8), vmulq_f32(acc32, alpha_vec));
}

/// 4x4 NEON micro-kernel with double accumulators
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn gemm_4x4_kernel(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    k: usize,
    n: usize,
    alpha: f32,
) {
    // Double accumulators
    let mut acc0_0 = vdupq_n_f32(0.0);
    let mut acc0_1 = vdupq_n_f32(0.0);
    let mut acc1_0 = vdupq_n_f32(0.0);
    let mut acc1_1 = vdupq_n_f32(0.0);
    let mut acc2_0 = vdupq_n_f32(0.0);
    let mut acc2_1 = vdupq_n_f32(0.0);
    let mut acc3_0 = vdupq_n_f32(0.0);
    let mut acc3_1 = vdupq_n_f32(0.0);

    let a0 = a;
    let a1 = a.add(k);
    let a2 = a.add(k * 2);
    let a3 = a.add(k * 3);

    let k2 = k / 2 * 2;
    let mut kk = 0;

    // 2x unrolled loop with double accumulators
    while kk < k2 {
        let b_row0 = b.add(kk * n);
        let vb0 = vld1q_f32(b_row0);
        let va0_0 = *a0.add(kk);
        let va1_0 = *a1.add(kk);
        let va2_0 = *a2.add(kk);
        let va3_0 = *a3.add(kk);
        acc0_0 = vfmaq_n_f32(acc0_0, vb0, va0_0);
        acc1_0 = vfmaq_n_f32(acc1_0, vb0, va1_0);
        acc2_0 = vfmaq_n_f32(acc2_0, vb0, va2_0);
        acc3_0 = vfmaq_n_f32(acc3_0, vb0, va3_0);

        let b_row1 = b.add((kk + 1) * n);
        let vb1 = vld1q_f32(b_row1);
        let va0_1 = *a0.add(kk + 1);
        let va1_1 = *a1.add(kk + 1);
        let va2_1 = *a2.add(kk + 1);
        let va3_1 = *a3.add(kk + 1);
        acc0_1 = vfmaq_n_f32(acc0_1, vb1, va0_1);
        acc1_1 = vfmaq_n_f32(acc1_1, vb1, va1_1);
        acc2_1 = vfmaq_n_f32(acc2_1, vb1, va2_1);
        acc3_1 = vfmaq_n_f32(acc3_1, vb1, va3_1);

        kk += 2;
    }

    // Handle remaining
    while kk < k {
        let b_row = b.add(kk * n);
        let vb = vld1q_f32(b_row);
        let va0 = *a0.add(kk);
        let va1 = *a1.add(kk);
        let va2 = *a2.add(kk);
        let va3 = *a3.add(kk);
        acc0_0 = vfmaq_n_f32(acc0_0, vb, va0);
        acc1_0 = vfmaq_n_f32(acc1_0, vb, va1);
        acc2_0 = vfmaq_n_f32(acc2_0, vb, va2);
        acc3_0 = vfmaq_n_f32(acc3_0, vb, va3);
        kk += 1;
    }

    // Merge and store
    let alpha_vec = vdupq_n_f32(alpha);
    vst1q_f32(c, vmulq_f32(vaddq_f32(acc0_0, acc0_1), alpha_vec));
    vst1q_f32(c.add(n), vmulq_f32(vaddq_f32(acc1_0, acc1_1), alpha_vec));
    vst1q_f32(
        c.add(n * 2),
        vmulq_f32(vaddq_f32(acc2_0, acc2_1), alpha_vec),
    );
    vst1q_f32(
        c.add(n * 3),
        vmulq_f32(vaddq_f32(acc3_0, acc3_1), alpha_vec),
    );
}

/// 1x12 NEON micro-kernel for single row (FFN layer optimization)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn gemm_1x12_kernel(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    k: usize,
    n: usize,
    alpha: f32,
) {
    // Double accumulators for better ILP
    let mut acc0_0 = vdupq_n_f32(0.0);
    let mut acc0_1 = vdupq_n_f32(0.0);
    let mut acc1_0 = vdupq_n_f32(0.0);
    let mut acc1_1 = vdupq_n_f32(0.0);
    let mut acc2_0 = vdupq_n_f32(0.0);
    let mut acc2_1 = vdupq_n_f32(0.0);

    let k2 = k / 2 * 2;
    let mut kk = 0;

    while kk < k2 {
        let va0 = *a.add(kk);
        let b_row0 = b.add(kk * n);
        let vb0_0 = vld1q_f32(b_row0);
        let vb1_0 = vld1q_f32(b_row0.add(4));
        let vb2_0 = vld1q_f32(b_row0.add(8));
        acc0_0 = vfmaq_n_f32(acc0_0, vb0_0, va0);
        acc1_0 = vfmaq_n_f32(acc1_0, vb1_0, va0);
        acc2_0 = vfmaq_n_f32(acc2_0, vb2_0, va0);

        let va1 = *a.add(kk + 1);
        let b_row1 = b.add((kk + 1) * n);
        let vb0_1 = vld1q_f32(b_row1);
        let vb1_1 = vld1q_f32(b_row1.add(4));
        let vb2_1 = vld1q_f32(b_row1.add(8));
        acc0_1 = vfmaq_n_f32(acc0_1, vb0_1, va1);
        acc1_1 = vfmaq_n_f32(acc1_1, vb1_1, va1);
        acc2_1 = vfmaq_n_f32(acc2_1, vb2_1, va1);

        kk += 2;
    }

    while kk < k {
        let va = *a.add(kk);
        let b_row = b.add(kk * n);
        let vb0 = vld1q_f32(b_row);
        let vb1 = vld1q_f32(b_row.add(4));
        let vb2 = vld1q_f32(b_row.add(8));
        acc0_0 = vfmaq_n_f32(acc0_0, vb0, va);
        acc1_0 = vfmaq_n_f32(acc1_0, vb1, va);
        acc2_0 = vfmaq_n_f32(acc2_0, vb2, va);
        kk += 1;
    }

    // Merge and store
    let alpha_vec = vdupq_n_f32(alpha);
    vst1q_f32(c, vmulq_f32(vaddq_f32(acc0_0, acc0_1), alpha_vec));
    vst1q_f32(c.add(4), vmulq_f32(vaddq_f32(acc1_0, acc1_1), alpha_vec));
    vst1q_f32(c.add(8), vmulq_f32(vaddq_f32(acc2_0, acc2_1), alpha_vec));
}

/// 1x4 NEON micro-kernel for single row with 4 columns
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn gemm_1x4_kernel(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    k: usize,
    n: usize,
    alpha: f32,
) {
    let mut acc_0 = vdupq_n_f32(0.0);
    let mut acc_1 = vdupq_n_f32(0.0);

    let k2 = k / 2 * 2;
    let mut kk = 0;

    while kk < k2 {
        let va0 = *a.add(kk);
        let vb0 = vld1q_f32(b.add(kk * n));
        acc_0 = vfmaq_n_f32(acc_0, vb0, va0);

        let va1 = *a.add(kk + 1);
        let vb1 = vld1q_f32(b.add((kk + 1) * n));
        acc_1 = vfmaq_n_f32(acc_1, vb1, va1);

        kk += 2;
    }

    while kk < k {
        let va = *a.add(kk);
        let vb = vld1q_f32(b.add(kk * n));
        acc_0 = vfmaq_n_f32(acc_0, vb, va);
        kk += 1;
    }

    vst1q_f32(c, vmulq_f32(vaddq_f32(acc_0, acc_1), vdupq_n_f32(alpha)));
}

/// Main GEMM function - dispatches to optimized kernels
/// C = alpha * A @ B
/// All matrices are row-major
/// Args: (a_ptr, b_ptr, c_ptr, m, k, n, alpha)
#[cfg(target_arch = "aarch64")]
pub unsafe fn gemm_neon(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
) {
    let n12 = n / 12 * 12;
    let n4 = n / 4 * 4;
    let m4 = m / 4 * 4;

    // Process 4 rows at a time with 4x12 kernel
    for i in (0..m4).step_by(4) {
        // Process 12 columns at a time
        for j in (0..n12).step_by(12) {
            gemm_4x12_kernel(a.add(i * k), b.add(j), c.add(i * n + j), k, n, alpha);
        }

        // Handle remaining 4-column blocks
        for j in (n12..n4).step_by(4) {
            gemm_4x4_kernel(a.add(i * k), b.add(j), c.add(i * n + j), k, n, alpha);
        }

        // Handle remaining columns (scalar)
        for j in n4..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += *a.add(i * k + kk) * *b.add(kk * n + j);
            }
            *c.add(i * n + j) = alpha * sum;
            let mut sum1 = 0.0f32;
            for kk in 0..k {
                sum1 += *a.add((i + 1) * k + kk) * *b.add(kk * n + j);
            }
            *c.add((i + 1) * n + j) = alpha * sum1;
            let mut sum2 = 0.0f32;
            for kk in 0..k {
                sum2 += *a.add((i + 2) * k + kk) * *b.add(kk * n + j);
            }
            *c.add((i + 2) * n + j) = alpha * sum2;
            let mut sum3 = 0.0f32;
            for kk in 0..k {
                sum3 += *a.add((i + 3) * k + kk) * *b.add(kk * n + j);
            }
            *c.add((i + 3) * n + j) = alpha * sum3;
        }
    }

    // Handle remaining rows
    for i in m4..m {
        // Process 12 columns at a time with 1x12 kernel
        for j in (0..n12).step_by(12) {
            gemm_1x12_kernel(a.add(i * k), b.add(j), c.add(i * n + j), k, n, alpha);
        }

        // Process remaining 4-column blocks
        for j in (n12..n4).step_by(4) {
            gemm_1x4_kernel(a.add(i * k), b.add(j), c.add(i * n + j), k, n, alpha);
        }

        // Handle remaining columns (scalar)
        for j in n4..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += *a.add(i * k + kk) * *b.add(kk * n + j);
            }
            *c.add(i * n + j) = alpha * sum;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn gemm_neon(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
) {
    // Scalar fallback
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += *a.add(i * k + kk) * *b.add(kk * n + j);
            }
            *c.add(i * n + j) = alpha * sum;
        }
    }
}

/// GEMM with accumulation: C = alpha * A @ B + beta * C
#[cfg(target_arch = "aarch64")]
pub unsafe fn gemm_neon_accum(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    _beta: f32,
) {
    let n12 = n / 12 * 12;
    let n4 = n / 4 * 4;
    let m4 = m / 4 * 4;

    // Process 4 rows at a time with 4x12 kernel (accumulate version)
    for i in (0..m4).step_by(4) {
        for j in (0..n12).step_by(12) {
            gemm_4x12_kernel_accum(a.add(i * k), b.add(j), c.add(i * n + j), k, n, alpha);
        }

        for j in (n12..n4).step_by(4) {
            gemm_4x4_kernel_accum(a.add(i * k), b.add(j), c.add(i * n + j), k, n, alpha);
        }

        // Scalar tail for columns
        for j in n4..n {
            for ri in 0..4 {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += *a.add((i + ri) * k + kk) * *b.add(kk * n + j);
                }
                *c.add((i + ri) * n + j) += alpha * sum;
            }
        }
    }

    // Handle remaining rows
    for i in m4..m {
        for j in (0..n12).step_by(12) {
            gemm_1x12_kernel_accum(a.add(i * k), b.add(j), c.add(i * n + j), k, n, alpha);
        }

        for j in (n12..n4).step_by(4) {
            gemm_1x4_kernel_accum(a.add(i * k), b.add(j), c.add(i * n + j), k, n, alpha);
        }

        for j in n4..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += *a.add(i * k + kk) * *b.add(kk * n + j);
            }
            *c.add(i * n + j) += alpha * sum;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn gemm_neon_accum(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    _beta: f32,
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += *a.add(i * k + kk) * *b.add(kk * n + j);
            }
            *c.add(i * n + j) += alpha * sum;
        }
    }
}

/// 4x12 kernel with accumulation
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn gemm_4x12_kernel_accum(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    k: usize,
    n: usize,
    alpha: f32,
) {
    // Load existing C values first
    let mut acc00_0 = vld1q_f32(c);
    let mut acc00_1 = vdupq_n_f32(0.0);
    let mut acc01_0 = vld1q_f32(c.add(4));
    let mut acc01_1 = vdupq_n_f32(0.0);
    let mut acc02_0 = vld1q_f32(c.add(8));
    let mut acc02_1 = vdupq_n_f32(0.0);

    let c1 = c.add(n);
    let mut acc10_0 = vld1q_f32(c1);
    let mut acc10_1 = vdupq_n_f32(0.0);
    let mut acc11_0 = vld1q_f32(c1.add(4));
    let mut acc11_1 = vdupq_n_f32(0.0);
    let mut acc12_0 = vld1q_f32(c1.add(8));
    let mut acc12_1 = vdupq_n_f32(0.0);

    let c2 = c.add(n * 2);
    let mut acc20_0 = vld1q_f32(c2);
    let mut acc20_1 = vdupq_n_f32(0.0);
    let mut acc21_0 = vld1q_f32(c2.add(4));
    let mut acc21_1 = vdupq_n_f32(0.0);
    let mut acc22_0 = vld1q_f32(c2.add(8));
    let mut acc22_1 = vdupq_n_f32(0.0);

    let c3 = c.add(n * 3);
    let mut acc30_0 = vld1q_f32(c3);
    let mut acc30_1 = vdupq_n_f32(0.0);
    let mut acc31_0 = vld1q_f32(c3.add(4));
    let mut acc31_1 = vdupq_n_f32(0.0);
    let mut acc32_0 = vld1q_f32(c3.add(8));
    let mut acc32_1 = vdupq_n_f32(0.0);

    let a0 = a;
    let a1 = a.add(k);
    let a2 = a.add(k * 2);
    let a3 = a.add(k * 3);

    let k2 = k / 2 * 2;
    let mut kk = 0;

    while kk < k2 {
        let b_row0 = b.add(kk * n);
        let vb0_0 = vld1q_f32(b_row0);
        let vb1_0 = vld1q_f32(b_row0.add(4));
        let vb2_0 = vld1q_f32(b_row0.add(8));
        let va0_0 = *a0.add(kk);
        let va1_0 = *a1.add(kk);
        let va2_0 = *a2.add(kk);
        let va3_0 = *a3.add(kk);
        acc00_0 = vfmaq_n_f32(acc00_0, vb0_0, va0_0);
        acc01_0 = vfmaq_n_f32(acc01_0, vb1_0, va0_0);
        acc02_0 = vfmaq_n_f32(acc02_0, vb2_0, va0_0);
        acc10_0 = vfmaq_n_f32(acc10_0, vb0_0, va1_0);
        acc11_0 = vfmaq_n_f32(acc11_0, vb1_0, va1_0);
        acc12_0 = vfmaq_n_f32(acc12_0, vb2_0, va1_0);
        acc20_0 = vfmaq_n_f32(acc20_0, vb0_0, va2_0);
        acc21_0 = vfmaq_n_f32(acc21_0, vb1_0, va2_0);
        acc22_0 = vfmaq_n_f32(acc22_0, vb2_0, va2_0);
        acc30_0 = vfmaq_n_f32(acc30_0, vb0_0, va3_0);
        acc31_0 = vfmaq_n_f32(acc31_0, vb1_0, va3_0);
        acc32_0 = vfmaq_n_f32(acc32_0, vb2_0, va3_0);

        let b_row1 = b.add((kk + 1) * n);
        let vb0_1 = vld1q_f32(b_row1);
        let vb1_1 = vld1q_f32(b_row1.add(4));
        let vb2_1 = vld1q_f32(b_row1.add(8));
        let va0_1 = *a0.add(kk + 1);
        let va1_1 = *a1.add(kk + 1);
        let va2_1 = *a2.add(kk + 1);
        let va3_1 = *a3.add(kk + 1);
        acc00_1 = vfmaq_n_f32(acc00_1, vb0_1, va0_1);
        acc01_1 = vfmaq_n_f32(acc01_1, vb1_1, va0_1);
        acc02_1 = vfmaq_n_f32(acc02_1, vb2_1, va0_1);
        acc10_1 = vfmaq_n_f32(acc10_1, vb0_1, va1_1);
        acc11_1 = vfmaq_n_f32(acc11_1, vb1_1, va1_1);
        acc12_1 = vfmaq_n_f32(acc12_1, vb2_1, va1_1);
        acc20_1 = vfmaq_n_f32(acc20_1, vb0_1, va2_1);
        acc21_1 = vfmaq_n_f32(acc21_1, vb1_1, va2_1);
        acc22_1 = vfmaq_n_f32(acc22_1, vb2_1, va2_1);
        acc30_1 = vfmaq_n_f32(acc30_1, vb0_1, va3_1);
        acc31_1 = vfmaq_n_f32(acc31_1, vb1_1, va3_1);
        acc32_1 = vfmaq_n_f32(acc32_1, vb2_1, va3_1);

        kk += 2;
    }

    while kk < k {
        let b_row = b.add(kk * n);
        let vb0 = vld1q_f32(b_row);
        let vb1 = vld1q_f32(b_row.add(4));
        let vb2 = vld1q_f32(b_row.add(8));
        let va0 = *a0.add(kk);
        let va1 = *a1.add(kk);
        let va2 = *a2.add(kk);
        let va3 = *a3.add(kk);
        acc00_0 = vfmaq_n_f32(acc00_0, vb0, va0);
        acc01_0 = vfmaq_n_f32(acc01_0, vb1, va0);
        acc02_0 = vfmaq_n_f32(acc02_0, vb2, va0);
        acc10_0 = vfmaq_n_f32(acc10_0, vb0, va1);
        acc11_0 = vfmaq_n_f32(acc11_0, vb1, va1);
        acc12_0 = vfmaq_n_f32(acc12_0, vb2, va1);
        acc20_0 = vfmaq_n_f32(acc20_0, vb0, va2);
        acc21_0 = vfmaq_n_f32(acc21_0, vb1, va2);
        acc22_0 = vfmaq_n_f32(acc22_0, vb2, va2);
        acc30_0 = vfmaq_n_f32(acc30_0, vb0, va3);
        acc31_0 = vfmaq_n_f32(acc31_0, vb1, va3);
        acc32_0 = vfmaq_n_f32(acc32_0, vb2, va3);
        kk += 1;
    }

    let alpha_vec = vdupq_n_f32(alpha);
    vst1q_f32(
        c,
        vaddq_f32(acc00_0, vfmaq_f32(vdupq_n_f32(0.0), acc00_1, alpha_vec)),
    );
    vst1q_f32(
        c.add(4),
        vaddq_f32(acc01_0, vfmaq_f32(vdupq_n_f32(0.0), acc01_1, alpha_vec)),
    );
    vst1q_f32(
        c.add(8),
        vaddq_f32(acc02_0, vfmaq_f32(vdupq_n_f32(0.0), acc02_1, alpha_vec)),
    );

    vst1q_f32(
        c1,
        vaddq_f32(acc10_0, vfmaq_f32(vdupq_n_f32(0.0), acc10_1, alpha_vec)),
    );
    vst1q_f32(
        c1.add(4),
        vaddq_f32(acc11_0, vfmaq_f32(vdupq_n_f32(0.0), acc11_1, alpha_vec)),
    );
    vst1q_f32(
        c1.add(8),
        vaddq_f32(acc12_0, vfmaq_f32(vdupq_n_f32(0.0), acc12_1, alpha_vec)),
    );

    vst1q_f32(
        c2,
        vaddq_f32(acc20_0, vfmaq_f32(vdupq_n_f32(0.0), acc20_1, alpha_vec)),
    );
    vst1q_f32(
        c2.add(4),
        vaddq_f32(acc21_0, vfmaq_f32(vdupq_n_f32(0.0), acc21_1, alpha_vec)),
    );
    vst1q_f32(
        c2.add(8),
        vaddq_f32(acc22_0, vfmaq_f32(vdupq_n_f32(0.0), acc22_1, alpha_vec)),
    );

    vst1q_f32(
        c3,
        vaddq_f32(acc30_0, vfmaq_f32(vdupq_n_f32(0.0), acc30_1, alpha_vec)),
    );
    vst1q_f32(
        c3.add(4),
        vaddq_f32(acc31_0, vfmaq_f32(vdupq_n_f32(0.0), acc31_1, alpha_vec)),
    );
    vst1q_f32(
        c3.add(8),
        vaddq_f32(acc32_0, vfmaq_f32(vdupq_n_f32(0.0), acc32_1, alpha_vec)),
    );
}

/// 4x4 kernel with accumulation
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn gemm_4x4_kernel_accum(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    k: usize,
    n: usize,
    alpha: f32,
) {
    let mut acc0_0 = vld1q_f32(c);
    let mut acc0_1 = vdupq_n_f32(0.0);
    let mut acc1_0 = vld1q_f32(c.add(n));
    let mut acc1_1 = vdupq_n_f32(0.0);
    let mut acc2_0 = vld1q_f32(c.add(n * 2));
    let mut acc2_1 = vdupq_n_f32(0.0);
    let mut acc3_0 = vld1q_f32(c.add(n * 3));
    let mut acc3_1 = vdupq_n_f32(0.0);

    let a0 = a;
    let a1 = a.add(k);
    let a2 = a.add(k * 2);
    let a3 = a.add(k * 3);

    let k2 = k / 2 * 2;
    let mut kk = 0;

    while kk < k2 {
        let vb0 = vld1q_f32(b.add(kk * n));
        let va0_0 = *a0.add(kk);
        let va1_0 = *a1.add(kk);
        let va2_0 = *a2.add(kk);
        let va3_0 = *a3.add(kk);
        acc0_0 = vfmaq_n_f32(acc0_0, vb0, va0_0);
        acc1_0 = vfmaq_n_f32(acc1_0, vb0, va1_0);
        acc2_0 = vfmaq_n_f32(acc2_0, vb0, va2_0);
        acc3_0 = vfmaq_n_f32(acc3_0, vb0, va3_0);

        let vb1 = vld1q_f32(b.add((kk + 1) * n));
        let va0_1 = *a0.add(kk + 1);
        let va1_1 = *a1.add(kk + 1);
        let va2_1 = *a2.add(kk + 1);
        let va3_1 = *a3.add(kk + 1);
        acc0_1 = vfmaq_n_f32(acc0_1, vb1, va0_1);
        acc1_1 = vfmaq_n_f32(acc1_1, vb1, va1_1);
        acc2_1 = vfmaq_n_f32(acc2_1, vb1, va2_1);
        acc3_1 = vfmaq_n_f32(acc3_1, vb1, va3_1);

        kk += 2;
    }

    while kk < k {
        let vb = vld1q_f32(b.add(kk * n));
        let va0 = *a0.add(kk);
        let va1 = *a1.add(kk);
        let va2 = *a2.add(kk);
        let va3 = *a3.add(kk);
        acc0_0 = vfmaq_n_f32(acc0_0, vb, va0);
        acc1_0 = vfmaq_n_f32(acc1_0, vb, va1);
        acc2_0 = vfmaq_n_f32(acc2_0, vb, va2);
        acc3_0 = vfmaq_n_f32(acc3_0, vb, va3);
        kk += 1;
    }

    let alpha_vec = vdupq_n_f32(alpha);
    vst1q_f32(
        c,
        vaddq_f32(acc0_0, vfmaq_f32(vdupq_n_f32(0.0), acc0_1, alpha_vec)),
    );
    vst1q_f32(
        c.add(n),
        vaddq_f32(acc1_0, vfmaq_f32(vdupq_n_f32(0.0), acc1_1, alpha_vec)),
    );
    vst1q_f32(
        c.add(n * 2),
        vaddq_f32(acc2_0, vfmaq_f32(vdupq_n_f32(0.0), acc2_1, alpha_vec)),
    );
    vst1q_f32(
        c.add(n * 3),
        vaddq_f32(acc3_0, vfmaq_f32(vdupq_n_f32(0.0), acc3_1, alpha_vec)),
    );
}

/// 1x12 kernel with accumulation
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn gemm_1x12_kernel_accum(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    k: usize,
    n: usize,
    alpha: f32,
) {
    let mut acc0_0 = vld1q_f32(c);
    let mut acc0_1 = vdupq_n_f32(0.0);
    let mut acc1_0 = vld1q_f32(c.add(4));
    let mut acc1_1 = vdupq_n_f32(0.0);
    let mut acc2_0 = vld1q_f32(c.add(8));
    let mut acc2_1 = vdupq_n_f32(0.0);

    let k2 = k / 2 * 2;
    let mut kk = 0;

    while kk < k2 {
        let va0 = *a.add(kk);
        let b_row0 = b.add(kk * n);
        let vb0_0 = vld1q_f32(b_row0);
        let vb1_0 = vld1q_f32(b_row0.add(4));
        let vb2_0 = vld1q_f32(b_row0.add(8));
        acc0_0 = vfmaq_n_f32(acc0_0, vb0_0, va0);
        acc1_0 = vfmaq_n_f32(acc1_0, vb1_0, va0);
        acc2_0 = vfmaq_n_f32(acc2_0, vb2_0, va0);

        let va1 = *a.add(kk + 1);
        let b_row1 = b.add((kk + 1) * n);
        let vb0_1 = vld1q_f32(b_row1);
        let vb1_1 = vld1q_f32(b_row1.add(4));
        let vb2_1 = vld1q_f32(b_row1.add(8));
        acc0_1 = vfmaq_n_f32(acc0_1, vb0_1, va1);
        acc1_1 = vfmaq_n_f32(acc1_1, vb1_1, va1);
        acc2_1 = vfmaq_n_f32(acc2_1, vb2_1, va1);

        kk += 2;
    }

    while kk < k {
        let va = *a.add(kk);
        let b_row = b.add(kk * n);
        let vb0 = vld1q_f32(b_row);
        let vb1 = vld1q_f32(b_row.add(4));
        let vb2 = vld1q_f32(b_row.add(8));
        acc0_0 = vfmaq_n_f32(acc0_0, vb0, va);
        acc1_0 = vfmaq_n_f32(acc1_0, vb1, va);
        acc2_0 = vfmaq_n_f32(acc2_0, vb2, va);
        kk += 1;
    }

    let alpha_vec = vdupq_n_f32(alpha);
    vst1q_f32(
        c,
        vaddq_f32(acc0_0, vfmaq_f32(vdupq_n_f32(0.0), acc0_1, alpha_vec)),
    );
    vst1q_f32(
        c.add(4),
        vaddq_f32(acc1_0, vfmaq_f32(vdupq_n_f32(0.0), acc1_1, alpha_vec)),
    );
    vst1q_f32(
        c.add(8),
        vaddq_f32(acc2_0, vfmaq_f32(vdupq_n_f32(0.0), acc2_1, alpha_vec)),
    );
}

/// 1x4 kernel with accumulation
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn gemm_1x4_kernel_accum(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    k: usize,
    n: usize,
    alpha: f32,
) {
    let mut acc_0 = vld1q_f32(c);
    let mut acc_1 = vdupq_n_f32(0.0);

    let k2 = k / 2 * 2;
    let mut kk = 0;

    while kk < k2 {
        let va0 = *a.add(kk);
        let vb0 = vld1q_f32(b.add(kk * n));
        acc_0 = vfmaq_n_f32(acc_0, vb0, va0);

        let va1 = *a.add(kk + 1);
        let vb1 = vld1q_f32(b.add((kk + 1) * n));
        acc_1 = vfmaq_n_f32(acc_1, vb1, va1);

        kk += 2;
    }

    while kk < k {
        let va = *a.add(kk);
        let vb = vld1q_f32(b.add(kk * n));
        acc_0 = vfmaq_n_f32(acc_0, vb, va);
        kk += 1;
    }

    let alpha_vec = vdupq_n_f32(alpha);
    vst1q_f32(
        c,
        vaddq_f32(acc_0, vfmaq_f32(vdupq_n_f32(0.0), acc_1, alpha_vec)),
    );
}

/// Optimized GEMM with A transposed: C = alpha * A^T @ B
#[cfg(target_arch = "aarch64")]
pub unsafe fn gemm_at_neon(
    m: usize,
    n: usize,
    k: usize,
    a: *const f32, // [K, M] - will be transposed
    b: *const f32, // [K, N]
    c: *mut f32,   // [M, N]
    alpha: f32,
    _beta: f32,
) {
    // For A^T @ B where A is [K, M] and B is [K, N]
    // Result C is [M, N] where C[i,j] = sum_k A[k,i] * B[k,j]
    // This is equivalent to computing dot products of columns of A with columns of B

    let n4 = n / 4 * 4;

    for i in 0..m {
        // Process 4 columns at a time
        for j in (0..n4).step_by(4) {
            let mut acc_0 = vdupq_n_f32(0.0);
            let mut acc_1 = vdupq_n_f32(0.0);

            let k2 = k / 2 * 2;
            let mut kk = 0;

            while kk < k2 {
                // Load A[kk, i] and A[kk+1, i] (column i of A)
                let a_k0 = *a.add(kk * m + i);
                let a_k1 = *a.add((kk + 1) * m + i);

                // Load B[kk, j:j+4] and B[kk+1, j:j+4]
                let b_k0 = vld1q_f32(b.add(kk * n + j));
                let b_k1 = vld1q_f32(b.add((kk + 1) * n + j));

                acc_0 = vfmaq_n_f32(acc_0, b_k0, a_k0);
                acc_1 = vfmaq_n_f32(acc_1, b_k1, a_k1);

                kk += 2;
            }

            while kk < k {
                let a_k = *a.add(kk * m + i);
                let b_k = vld1q_f32(b.add(kk * n + j));
                acc_0 = vfmaq_n_f32(acc_0, b_k, a_k);
                kk += 1;
            }

            vst1q_f32(
                c.add(i * n + j),
                vmulq_f32(vaddq_f32(acc_0, acc_1), vdupq_n_f32(alpha)),
            );
        }

        // Handle remaining columns
        for j in n4..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += *a.add(kk * m + i) * *b.add(kk * n + j);
            }
            *c.add(i * n + j) = alpha * sum;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn gemm_at_neon(
    m: usize,
    n: usize,
    k: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    alpha: f32,
    _beta: f32,
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += *a.add(kk * m + i) * *b.add(kk * n + j);
            }
            *c.add(i * n + j) = alpha * sum;
        }
    }
}

/// Optimized GEMM with B transposed: C = alpha * A @ B^T
#[cfg(target_arch = "aarch64")]
pub unsafe fn gemm_bt_neon(
    m: usize,
    n: usize,
    k: usize,
    a: *const f32, // [M, K]
    b: *const f32, // [N, K] - will be transposed
    c: *mut f32,   // [M, N]
    alpha: f32,
    _beta: f32,
) {
    // For A @ B^T where A is [M, K] and B is [N, K]
    // Result C is [M, N] where C[i,j] = sum_k A[i,k] * B[j,k]
    // This is a dot product of row i of A and row j of B

    for i in 0..m {
        let a_row = a.add(i * k);

        for j in 0..n {
            let b_row = b.add(j * k);

            let mut acc_0 = vdupq_n_f32(0.0);
            let mut acc_1 = vdupq_n_f32(0.0);

            let k8 = k / 8 * 8;
            let mut kk = 0;

            // 8x unrolled with double accumulators
            while kk < k8 {
                let a0 = vld1q_f32(a_row.add(kk));
                let b0 = vld1q_f32(b_row.add(kk));
                acc_0 = vfmaq_f32(acc_0, a0, b0);

                let a1 = vld1q_f32(a_row.add(kk + 4));
                let b1 = vld1q_f32(b_row.add(kk + 4));
                acc_1 = vfmaq_f32(acc_1, a1, b1);

                kk += 8;
            }

            let k4 = k / 4 * 4;
            while kk < k4 {
                let av = vld1q_f32(a_row.add(kk));
                let bv = vld1q_f32(b_row.add(kk));
                acc_0 = vfmaq_f32(acc_0, av, bv);
                kk += 4;
            }

            // Horizontal sum
            let mut sum = vaddvq_f32(vaddq_f32(acc_0, acc_1));

            // Handle remaining elements
            while kk < k {
                sum += *a_row.add(kk) * *b_row.add(kk);
                kk += 1;
            }

            *c.add(i * n + j) = alpha * sum;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn gemm_bt_neon(
    m: usize,
    n: usize,
    k: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    alpha: f32,
    _beta: f32,
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += *a.add(i * k + kk) * *b.add(j * k + kk);
            }
            *c.add(i * n + j) = alpha * sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemm_neon_simple() {
        let m = 4;
        let n = 4;
        let k = 4;

        let a: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let b: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let mut c = vec![0.0f32; m * n];

        unsafe {
            gemm_neon(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, k, n, 1.0);
        }

        // A @ I = A
        for i in 0..m {
            for j in 0..n {
                assert!((c[i * n + j] - a[i * n + j]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_gemm_neon_larger() {
        let m = 16;
        let n = 16;
        let k = 16;

        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
        let mut c = vec![0.0f32; m * n];

        unsafe {
            gemm_neon(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, k, n, 1.0);
        }

        // Verify first element
        let expected_00: f32 = (0..k).map(|kk| a[kk] * b[kk * n]).sum();
        assert!((c[0] - expected_00).abs() < 1e-3);
    }

    #[test]
    fn test_gemm_m1() {
        // Test single-row GEMM (FFN layer scenario)
        let m = 1;
        let n = 512;
        let k = 256;

        let a: Vec<f32> = vec![0.5; m * k];
        let b: Vec<f32> = vec![1.0; k * n];
        let mut c = vec![0.0f32; m * n];

        unsafe {
            gemm_neon(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, k, n, 1.0);
        }

        // Each output should be k * 0.5 * 1.0 = k/2 = 128.0
        for i in 0..n {
            assert!((c[i] - 128.0).abs() < 1e-3);
        }
    }
}

/// Fused GEMM + Bias: C = alpha * A @ B + bias
/// This is more efficient than separate GEMM + add because it only traverses output memory once
#[cfg(target_arch = "aarch64")]
pub unsafe fn gemm_neon_fused_bias(
    a: *const f32,
    b: *const f32,
    bias: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
) {
    let n12 = n / 12 * 12;
    let n4 = n / 4 * 4;
    let m4 = m / 4 * 4;

    // Process 4 rows at a time with 4x12 fused kernel
    for i in (0..m4).step_by(4) {
        for j in (0..n12).step_by(12) {
            gemm_4x12_kernel_fused_bias(
                a.add(i * k),
                b.add(j),
                bias.add(j),
                c.add(i * n + j),
                k,
                n,
                alpha,
            );
        }

        for j in (n12..n4).step_by(4) {
            gemm_4x4_kernel_fused_bias(
                a.add(i * k),
                b.add(j),
                bias.add(j),
                c.add(i * n + j),
                k,
                n,
                alpha,
            );
        }

        // Scalar tail for columns
        for j in n4..n {
            for ri in 0..4 {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += *a.add((i + ri) * k + kk) * *b.add(kk * n + j);
                }
                *c.add((i + ri) * n + j) = alpha * sum + *bias.add(j);
            }
        }
    }

    // Handle remaining rows
    for i in m4..m {
        for j in (0..n12).step_by(12) {
            gemm_1x12_kernel_fused_bias(
                a.add(i * k),
                b.add(j),
                bias.add(j),
                c.add(i * n + j),
                k,
                n,
                alpha,
            );
        }

        for j in (n12..n4).step_by(4) {
            gemm_1x4_kernel_fused_bias(
                a.add(i * k),
                b.add(j),
                bias.add(j),
                c.add(i * n + j),
                k,
                n,
                alpha,
            );
        }

        for j in n4..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += *a.add(i * k + kk) * *b.add(kk * n + j);
            }
            *c.add(i * n + j) = alpha * sum + *bias.add(j);
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn gemm_neon_fused_bias(
    a: *const f32,
    b: *const f32,
    bias: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += *a.add(i * k + kk) * *b.add(kk * n + j);
            }
            *c.add(i * n + j) = alpha * sum + *bias.add(j);
        }
    }
}

/// 4x12 kernel with fused bias addition
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn gemm_4x12_kernel_fused_bias(
    a: *const f32,
    b: *const f32,
    bias: *const f32,
    c: *mut f32,
    k: usize,
    n: usize,
    alpha: f32,
) {
    // Load bias vectors once at the beginning
    let bias0 = vld1q_f32(bias);
    let bias1 = vld1q_f32(bias.add(4));
    let bias2 = vld1q_f32(bias.add(8));

    // Initialize accumulators with bias
    let mut acc00_0 = bias0;
    let mut acc00_1 = vdupq_n_f32(0.0);
    let mut acc01_0 = bias1;
    let mut acc01_1 = vdupq_n_f32(0.0);
    let mut acc02_0 = bias2;
    let mut acc02_1 = vdupq_n_f32(0.0);

    let mut acc10_0 = bias0;
    let mut acc10_1 = vdupq_n_f32(0.0);
    let mut acc11_0 = bias1;
    let mut acc11_1 = vdupq_n_f32(0.0);
    let mut acc12_0 = bias2;
    let mut acc12_1 = vdupq_n_f32(0.0);

    let mut acc20_0 = bias0;
    let mut acc20_1 = vdupq_n_f32(0.0);
    let mut acc21_0 = bias1;
    let mut acc21_1 = vdupq_n_f32(0.0);
    let mut acc22_0 = bias2;
    let mut acc22_1 = vdupq_n_f32(0.0);

    let mut acc30_0 = bias0;
    let mut acc30_1 = vdupq_n_f32(0.0);
    let mut acc31_0 = bias1;
    let mut acc31_1 = vdupq_n_f32(0.0);
    let mut acc32_0 = bias2;
    let mut acc32_1 = vdupq_n_f32(0.0);

    let a0 = a;
    let a1 = a.add(k);
    let a2 = a.add(k * 2);
    let a3 = a.add(k * 3);

    let k8 = k / 8 * 8;
    let mut kk = 0;

    while kk < k8 {
        // 8x unrolled with double accumulators
        let b_row0 = b.add(kk * n);
        let vb0_0 = vld1q_f32(b_row0);
        let vb1_0 = vld1q_f32(b_row0.add(4));
        let vb2_0 = vld1q_f32(b_row0.add(8));
        let va0_0 = *a0.add(kk);
        let va1_0 = *a1.add(kk);
        let va2_0 = *a2.add(kk);
        let va3_0 = *a3.add(kk);
        acc00_0 = vfmaq_n_f32(acc00_0, vb0_0, va0_0);
        acc01_0 = vfmaq_n_f32(acc01_0, vb1_0, va0_0);
        acc02_0 = vfmaq_n_f32(acc02_0, vb2_0, va0_0);
        acc10_0 = vfmaq_n_f32(acc10_0, vb0_0, va1_0);
        acc11_0 = vfmaq_n_f32(acc11_0, vb1_0, va1_0);
        acc12_0 = vfmaq_n_f32(acc12_0, vb2_0, va1_0);
        acc20_0 = vfmaq_n_f32(acc20_0, vb0_0, va2_0);
        acc21_0 = vfmaq_n_f32(acc21_0, vb1_0, va2_0);
        acc22_0 = vfmaq_n_f32(acc22_0, vb2_0, va2_0);
        acc30_0 = vfmaq_n_f32(acc30_0, vb0_0, va3_0);
        acc31_0 = vfmaq_n_f32(acc31_0, vb1_0, va3_0);
        acc32_0 = vfmaq_n_f32(acc32_0, vb2_0, va3_0);

        let b_row1 = b.add((kk + 1) * n);
        let vb0_1 = vld1q_f32(b_row1);
        let vb1_1 = vld1q_f32(b_row1.add(4));
        let vb2_1 = vld1q_f32(b_row1.add(8));
        let va0_1 = *a0.add(kk + 1);
        let va1_1 = *a1.add(kk + 1);
        let va2_1 = *a2.add(kk + 1);
        let va3_1 = *a3.add(kk + 1);
        acc00_1 = vfmaq_n_f32(acc00_1, vb0_1, va0_1);
        acc01_1 = vfmaq_n_f32(acc01_1, vb1_1, va0_1);
        acc02_1 = vfmaq_n_f32(acc02_1, vb2_1, va0_1);
        acc10_1 = vfmaq_n_f32(acc10_1, vb0_1, va1_1);
        acc11_1 = vfmaq_n_f32(acc11_1, vb1_1, va1_1);
        acc12_1 = vfmaq_n_f32(acc12_1, vb2_1, va1_1);
        acc20_1 = vfmaq_n_f32(acc20_1, vb0_1, va2_1);
        acc21_1 = vfmaq_n_f32(acc21_1, vb1_1, va2_1);
        acc22_1 = vfmaq_n_f32(acc22_1, vb2_1, va2_1);
        acc30_1 = vfmaq_n_f32(acc30_1, vb0_1, va3_1);
        acc31_1 = vfmaq_n_f32(acc31_1, vb1_1, va3_1);
        acc32_1 = vfmaq_n_f32(acc32_1, vb2_1, va3_1);

        // Continue for remaining 6 iterations...
        for offset in 2..8 {
            let b_row = b.add((kk + offset) * n);
            let vb0 = vld1q_f32(b_row);
            let vb1 = vld1q_f32(b_row.add(4));
            let vb2 = vld1q_f32(b_row.add(8));
            let va0 = *a0.add(kk + offset);
            let va1 = *a1.add(kk + offset);
            let va2 = *a2.add(kk + offset);
            let va3 = *a3.add(kk + offset);
            if offset % 2 == 0 {
                acc00_0 = vfmaq_n_f32(acc00_0, vb0, va0);
                acc01_0 = vfmaq_n_f32(acc01_0, vb1, va0);
                acc02_0 = vfmaq_n_f32(acc02_0, vb2, va0);
                acc10_0 = vfmaq_n_f32(acc10_0, vb0, va1);
                acc11_0 = vfmaq_n_f32(acc11_0, vb1, va1);
                acc12_0 = vfmaq_n_f32(acc12_0, vb2, va1);
                acc20_0 = vfmaq_n_f32(acc20_0, vb0, va2);
                acc21_0 = vfmaq_n_f32(acc21_0, vb1, va2);
                acc22_0 = vfmaq_n_f32(acc22_0, vb2, va2);
                acc30_0 = vfmaq_n_f32(acc30_0, vb0, va3);
                acc31_0 = vfmaq_n_f32(acc31_0, vb1, va3);
                acc32_0 = vfmaq_n_f32(acc32_0, vb2, va3);
            } else {
                acc00_1 = vfmaq_n_f32(acc00_1, vb0, va0);
                acc01_1 = vfmaq_n_f32(acc01_1, vb1, va0);
                acc02_1 = vfmaq_n_f32(acc02_1, vb2, va0);
                acc10_1 = vfmaq_n_f32(acc10_1, vb0, va1);
                acc11_1 = vfmaq_n_f32(acc11_1, vb1, va1);
                acc12_1 = vfmaq_n_f32(acc12_1, vb2, va1);
                acc20_1 = vfmaq_n_f32(acc20_1, vb0, va2);
                acc21_1 = vfmaq_n_f32(acc21_1, vb1, va2);
                acc22_1 = vfmaq_n_f32(acc22_1, vb2, va2);
                acc30_1 = vfmaq_n_f32(acc30_1, vb0, va3);
                acc31_1 = vfmaq_n_f32(acc31_1, vb1, va3);
                acc32_1 = vfmaq_n_f32(acc32_1, vb2, va3);
            }
        }

        kk += 8;
    }

    // Handle remaining iterations
    while kk < k {
        let b_row = b.add(kk * n);
        let vb0 = vld1q_f32(b_row);
        let vb1 = vld1q_f32(b_row.add(4));
        let vb2 = vld1q_f32(b_row.add(8));
        let va0 = *a0.add(kk);
        let va1 = *a1.add(kk);
        let va2 = *a2.add(kk);
        let va3 = *a3.add(kk);
        acc00_0 = vfmaq_n_f32(acc00_0, vb0, va0);
        acc01_0 = vfmaq_n_f32(acc01_0, vb1, va0);
        acc02_0 = vfmaq_n_f32(acc02_0, vb2, va0);
        acc10_0 = vfmaq_n_f32(acc10_0, vb0, va1);
        acc11_0 = vfmaq_n_f32(acc11_0, vb1, va1);
        acc12_0 = vfmaq_n_f32(acc12_0, vb2, va1);
        acc20_0 = vfmaq_n_f32(acc20_0, vb0, va2);
        acc21_0 = vfmaq_n_f32(acc21_0, vb1, va2);
        acc22_0 = vfmaq_n_f32(acc22_0, vb2, va2);
        acc30_0 = vfmaq_n_f32(acc30_0, vb0, va3);
        acc31_0 = vfmaq_n_f32(acc31_0, vb1, va3);
        acc32_0 = vfmaq_n_f32(acc32_0, vb2, va3);
        kk += 1;
    }

    // Merge and store with alpha scaling
    let alpha_vec = vdupq_n_f32(alpha);
    vst1q_f32(
        c,
        vaddq_f32(vmulq_f32(acc00_0, alpha_vec), vmulq_f32(acc00_1, alpha_vec)),
    );
    vst1q_f32(
        c.add(4),
        vaddq_f32(vmulq_f32(acc01_0, alpha_vec), vmulq_f32(acc01_1, alpha_vec)),
    );
    vst1q_f32(
        c.add(8),
        vaddq_f32(vmulq_f32(acc02_0, alpha_vec), vmulq_f32(acc02_1, alpha_vec)),
    );

    let c1 = c.add(n);
    vst1q_f32(
        c1,
        vaddq_f32(vmulq_f32(acc10_0, alpha_vec), vmulq_f32(acc10_1, alpha_vec)),
    );
    vst1q_f32(
        c1.add(4),
        vaddq_f32(vmulq_f32(acc11_0, alpha_vec), vmulq_f32(acc11_1, alpha_vec)),
    );
    vst1q_f32(
        c1.add(8),
        vaddq_f32(vmulq_f32(acc12_0, alpha_vec), vmulq_f32(acc12_1, alpha_vec)),
    );

    let c2 = c.add(n * 2);
    vst1q_f32(
        c2,
        vaddq_f32(vmulq_f32(acc20_0, alpha_vec), vmulq_f32(acc20_1, alpha_vec)),
    );
    vst1q_f32(
        c2.add(4),
        vaddq_f32(vmulq_f32(acc21_0, alpha_vec), vmulq_f32(acc21_1, alpha_vec)),
    );
    vst1q_f32(
        c2.add(8),
        vaddq_f32(vmulq_f32(acc22_0, alpha_vec), vmulq_f32(acc22_1, alpha_vec)),
    );

    let c3 = c.add(n * 3);
    vst1q_f32(
        c3,
        vaddq_f32(vmulq_f32(acc30_0, alpha_vec), vmulq_f32(acc30_1, alpha_vec)),
    );
    vst1q_f32(
        c3.add(4),
        vaddq_f32(vmulq_f32(acc31_0, alpha_vec), vmulq_f32(acc31_1, alpha_vec)),
    );
    vst1q_f32(
        c3.add(8),
        vaddq_f32(vmulq_f32(acc32_0, alpha_vec), vmulq_f32(acc32_1, alpha_vec)),
    );
}

/// 4x4 kernel with fused bias
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn gemm_4x4_kernel_fused_bias(
    a: *const f32,
    b: *const f32,
    bias: *const f32,
    c: *mut f32,
    k: usize,
    n: usize,
    alpha: f32,
) {
    let bias_vec = vld1q_f32(bias);
    let mut acc0_0 = bias_vec;
    let mut acc0_1 = vdupq_n_f32(0.0);
    let mut acc1_0 = bias_vec;
    let mut acc1_1 = vdupq_n_f32(0.0);
    let mut acc2_0 = bias_vec;
    let mut acc2_1 = vdupq_n_f32(0.0);
    let mut acc3_0 = bias_vec;
    let mut acc3_1 = vdupq_n_f32(0.0);

    let a0 = a;
    let a1 = a.add(k);
    let a2 = a.add(k * 2);
    let a3 = a.add(k * 3);

    let k2 = k / 2 * 2;
    let mut kk = 0;

    while kk < k2 {
        let vb0 = vld1q_f32(b.add(kk * n));
        let va0_0 = *a0.add(kk);
        let va1_0 = *a1.add(kk);
        let va2_0 = *a2.add(kk);
        let va3_0 = *a3.add(kk);
        acc0_0 = vfmaq_n_f32(acc0_0, vb0, va0_0);
        acc1_0 = vfmaq_n_f32(acc1_0, vb0, va1_0);
        acc2_0 = vfmaq_n_f32(acc2_0, vb0, va2_0);
        acc3_0 = vfmaq_n_f32(acc3_0, vb0, va3_0);

        let vb1 = vld1q_f32(b.add((kk + 1) * n));
        let va0_1 = *a0.add(kk + 1);
        let va1_1 = *a1.add(kk + 1);
        let va2_1 = *a2.add(kk + 1);
        let va3_1 = *a3.add(kk + 1);
        acc0_1 = vfmaq_n_f32(acc0_1, vb1, va0_1);
        acc1_1 = vfmaq_n_f32(acc1_1, vb1, va1_1);
        acc2_1 = vfmaq_n_f32(acc2_1, vb1, va2_1);
        acc3_1 = vfmaq_n_f32(acc3_1, vb1, va3_1);

        kk += 2;
    }

    while kk < k {
        let vb = vld1q_f32(b.add(kk * n));
        let va0 = *a0.add(kk);
        let va1 = *a1.add(kk);
        let va2 = *a2.add(kk);
        let va3 = *a3.add(kk);
        acc0_0 = vfmaq_n_f32(acc0_0, vb, va0);
        acc1_0 = vfmaq_n_f32(acc1_0, vb, va1);
        acc2_0 = vfmaq_n_f32(acc2_0, vb, va2);
        acc3_0 = vfmaq_n_f32(acc3_0, vb, va3);
        kk += 1;
    }

    let alpha_vec = vdupq_n_f32(alpha);
    vst1q_f32(
        c,
        vaddq_f32(vmulq_f32(acc0_0, alpha_vec), vmulq_f32(acc0_1, alpha_vec)),
    );
    vst1q_f32(
        c.add(n),
        vaddq_f32(vmulq_f32(acc1_0, alpha_vec), vmulq_f32(acc1_1, alpha_vec)),
    );
    vst1q_f32(
        c.add(n * 2),
        vaddq_f32(vmulq_f32(acc2_0, alpha_vec), vmulq_f32(acc2_1, alpha_vec)),
    );
    vst1q_f32(
        c.add(n * 3),
        vaddq_f32(vmulq_f32(acc3_0, alpha_vec), vmulq_f32(acc3_1, alpha_vec)),
    );
}

/// 1x12 kernel with fused bias
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn gemm_1x12_kernel_fused_bias(
    a: *const f32,
    b: *const f32,
    bias: *const f32,
    c: *mut f32,
    k: usize,
    n: usize,
    alpha: f32,
) {
    let mut acc0_0 = vld1q_f32(bias);
    let mut acc0_1 = vdupq_n_f32(0.0);
    let mut acc1_0 = vld1q_f32(bias.add(4));
    let mut acc1_1 = vdupq_n_f32(0.0);
    let mut acc2_0 = vld1q_f32(bias.add(8));
    let mut acc2_1 = vdupq_n_f32(0.0);

    let k2 = k / 2 * 2;
    let mut kk = 0;

    while kk < k2 {
        let va0 = *a.add(kk);
        let b_row0 = b.add(kk * n);
        let vb0_0 = vld1q_f32(b_row0);
        let vb1_0 = vld1q_f32(b_row0.add(4));
        let vb2_0 = vld1q_f32(b_row0.add(8));
        acc0_0 = vfmaq_n_f32(acc0_0, vb0_0, va0);
        acc1_0 = vfmaq_n_f32(acc1_0, vb1_0, va0);
        acc2_0 = vfmaq_n_f32(acc2_0, vb2_0, va0);

        let va1 = *a.add(kk + 1);
        let b_row1 = b.add((kk + 1) * n);
        let vb0_1 = vld1q_f32(b_row1);
        let vb1_1 = vld1q_f32(b_row1.add(4));
        let vb2_1 = vld1q_f32(b_row1.add(8));
        acc0_1 = vfmaq_n_f32(acc0_1, vb0_1, va1);
        acc1_1 = vfmaq_n_f32(acc1_1, vb1_1, va1);
        acc2_1 = vfmaq_n_f32(acc2_1, vb2_1, va1);

        kk += 2;
    }

    while kk < k {
        let va = *a.add(kk);
        let b_row = b.add(kk * n);
        let vb0 = vld1q_f32(b_row);
        let vb1 = vld1q_f32(b_row.add(4));
        let vb2 = vld1q_f32(b_row.add(8));
        acc0_0 = vfmaq_n_f32(acc0_0, vb0, va);
        acc1_0 = vfmaq_n_f32(acc1_0, vb1, va);
        acc2_0 = vfmaq_n_f32(acc2_0, vb2, va);
        kk += 1;
    }

    let alpha_vec = vdupq_n_f32(alpha);
    vst1q_f32(
        c,
        vaddq_f32(vmulq_f32(acc0_0, alpha_vec), vmulq_f32(acc0_1, alpha_vec)),
    );
    vst1q_f32(
        c.add(4),
        vaddq_f32(vmulq_f32(acc1_0, alpha_vec), vmulq_f32(acc1_1, alpha_vec)),
    );
    vst1q_f32(
        c.add(8),
        vaddq_f32(vmulq_f32(acc2_0, alpha_vec), vmulq_f32(acc2_1, alpha_vec)),
    );
}

/// 1x4 kernel with fused bias
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn gemm_1x4_kernel_fused_bias(
    a: *const f32,
    b: *const f32,
    bias: *const f32,
    c: *mut f32,
    k: usize,
    n: usize,
    alpha: f32,
) {
    let mut acc_0 = vld1q_f32(bias);
    let mut acc_1 = vdupq_n_f32(0.0);

    let k2 = k / 2 * 2;
    let mut kk = 0;

    while kk < k2 {
        let va0 = *a.add(kk);
        let vb0 = vld1q_f32(b.add(kk * n));
        acc_0 = vfmaq_n_f32(acc_0, vb0, va0);

        let va1 = *a.add(kk + 1);
        let vb1 = vld1q_f32(b.add((kk + 1) * n));
        acc_1 = vfmaq_n_f32(acc_1, vb1, va1);

        kk += 2;
    }

    while kk < k {
        let va = *a.add(kk);
        let vb = vld1q_f32(b.add(kk * n));
        acc_0 = vfmaq_n_f32(acc_0, vb, va);
        kk += 1;
    }

    let alpha_vec = vdupq_n_f32(alpha);
    vst1q_f32(
        c,
        vaddq_f32(vmulq_f32(acc_0, alpha_vec), vmulq_f32(acc_1, alpha_vec)),
    );
}
