// Allow unsafe operations in unsafe functions without explicit unsafe blocks
#![allow(unsafe_op_in_unsafe_fn)]

use crate::kernels::utils;
use crate::tensor::TensorView;
use core::arch::aarch64::*;
use core::arch::asm;
use std::borrow::Cow;

#[inline(always)]
unsafe fn vdotq_u32_custom(mut acc: uint32x4_t, a: uint8x16_t, b: uint8x16_t) -> uint32x4_t {
    unsafe {
        asm!(
            "udot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(nostack)
        );
        acc
    }
}

pub struct PreparedWeightsArm {
    /// Pre-packed B data in UDOT format
    pub packed_b: Vec<u8>,
    /// Column sums of the u8 weight values (for zero-point correction)
    pub col_sums: Vec<i32>,
    /// Pre-computed f32 column sums (padded to multiple of 16) for vectorized loading
    pub col_sums_f32: Vec<f32>,
    /// Original K dimension
    pub k: usize,
    /// K padded to multiple of 4
    pub k_aligned: usize,
    /// Original N dimension
    pub n: usize,
    /// Stride per strip in bytes = k_aligned * 16
    pub strip_stride: usize,
    /// Raw u8 weights in K×N row-major layout (for fp32 dequantization on macOS)
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    pub raw_b: Vec<u8>,
    /// Lazily dequantized fp32 weights [K, N] for Accelerate cblas_sgemm
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    pub dequantized_f32: std::sync::OnceLock<Vec<f32>>,
}

impl PreparedWeightsArm {
    /// Get or compute the dequantized fp32 weight matrix [K, N] row-major
    /// w_f32[k*n+j] = (raw_b[k*n+j] as f32 - zp_b as f32) * weight_scale[j]
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    pub fn get_dequantized_weights(
        &self,
        b_zero_point: Option<u8>,
        weight_scale: &crate::tensor::TensorView<'_, f32>,
    ) -> &[f32] {
        self.dequantized_f32.get_or_init(|| {
            let k = self.k;
            let n = self.n;
            let zp_b = b_zero_point.unwrap_or(0) as f32;
            let ws = &weight_scale.data;
            let per_channel = ws.len() > 1;
            let mut out = vec![0.0f32; k * n];

            if per_channel {
                // Per-channel: w_f32[k*n+j] = (raw_b[k*n+j] - zp_b) * ws[j]
                for kk in 0..k {
                    let row_off = kk * n;
                    let mut j = 0;
                    unsafe {
                        let zp_v = vdupq_n_f32(zp_b);
                        while j + 4 <= n {
                            let b_u8_0 = self.raw_b[row_off + j] as f32;
                            let b_u8_1 = self.raw_b[row_off + j + 1] as f32;
                            let b_u8_2 = self.raw_b[row_off + j + 2] as f32;
                            let b_u8_3 = self.raw_b[row_off + j + 3] as f32;
                            let b_f32 = vld1q_f32([b_u8_0, b_u8_1, b_u8_2, b_u8_3].as_ptr());
                            let scale_v = vld1q_f32(ws.as_ptr().add(j));
                            let result = vmulq_f32(vsubq_f32(b_f32, zp_v), scale_v);
                            vst1q_f32(out.as_mut_ptr().add(row_off + j), result);
                            j += 4;
                        }
                    }
                    while j < n {
                        out[row_off + j] = (self.raw_b[row_off + j] as f32 - zp_b) * ws[j];
                        j += 1;
                    }
                }
            } else {
                // Per-tensor: w_f32[i] = (raw_b[i] - zp_b) * ws[0]
                let scale = ws[0];
                let total = k * n;
                let mut i = 0;
                unsafe {
                    let zp_v = vdupq_n_f32(zp_b);
                    let s_v = vdupq_n_f32(scale);
                    while i + 16 <= total {
                        let b0 = self.raw_b[i] as f32;
                        let b1 = self.raw_b[i + 1] as f32;
                        let b2 = self.raw_b[i + 2] as f32;
                        let b3 = self.raw_b[i + 3] as f32;
                        let b4 = self.raw_b[i + 4] as f32;
                        let b5 = self.raw_b[i + 5] as f32;
                        let b6 = self.raw_b[i + 6] as f32;
                        let b7 = self.raw_b[i + 7] as f32;
                        let b8 = self.raw_b[i + 8] as f32;
                        let b9 = self.raw_b[i + 9] as f32;
                        let b10 = self.raw_b[i + 10] as f32;
                        let b11 = self.raw_b[i + 11] as f32;
                        let b12 = self.raw_b[i + 12] as f32;
                        let b13 = self.raw_b[i + 13] as f32;
                        let b14 = self.raw_b[i + 14] as f32;
                        let b15 = self.raw_b[i + 15] as f32;
                        let v0 = vld1q_f32([b0, b1, b2, b3].as_ptr());
                        let v1 = vld1q_f32([b4, b5, b6, b7].as_ptr());
                        let v2 = vld1q_f32([b8, b9, b10, b11].as_ptr());
                        let v3 = vld1q_f32([b12, b13, b14, b15].as_ptr());
                        vst1q_f32(out.as_mut_ptr().add(i), vmulq_f32(vsubq_f32(v0, zp_v), s_v));
                        vst1q_f32(
                            out.as_mut_ptr().add(i + 4),
                            vmulq_f32(vsubq_f32(v1, zp_v), s_v),
                        );
                        vst1q_f32(
                            out.as_mut_ptr().add(i + 8),
                            vmulq_f32(vsubq_f32(v2, zp_v), s_v),
                        );
                        vst1q_f32(
                            out.as_mut_ptr().add(i + 12),
                            vmulq_f32(vsubq_f32(v3, zp_v), s_v),
                        );
                        i += 16;
                    }
                }
                while i < total {
                    out[i] = (self.raw_b[i] as f32 - zp_b) * scale;
                    i += 1;
                }
            }
            out
        })
    }
}

pub fn prepare_weights_arm(b_u8: &[u8], k: usize, n: usize) -> PreparedWeightsArm {
    let k_aligned = (k + 3) & !3;
    let strip_stride = k_aligned * 16;
    let n_full_strips = n / 16;
    let n_rem = n % 16;
    let total_strips = n_full_strips + if n_rem > 0 { 1 } else { 0 };
    let mut packed_b = vec![0u8; total_strips * strip_stride];
    let mut col_sums = vec![0i32; n];

    // Compute column sums - simple loop (runs once at model load)
    for kk in 0..k {
        let row_off = kk * n;
        for jj in 0..n {
            col_sums[jj] += b_u8[row_off + jj] as i32;
        }
    }

    // Pack full 16-col strips
    for strip in 0..n_full_strips {
        let j = strip * 16;
        let dst_base = strip * strip_stride;

        for k_block in 0..(k_aligned / 4) {
            let kk = k_block * 4;
            let dst_off = dst_base + k_block * 64;

            // Load 4 rows × 16 cols, transpose to UDOT format
            unsafe {
                let rows_left = k.saturating_sub(kk);
                let v0 = if rows_left > 0 {
                    vld1q_u8(b_u8.as_ptr().add(kk * n + j))
                } else {
                    vdupq_n_u8(0)
                };
                let v1 = if rows_left > 1 {
                    vld1q_u8(b_u8.as_ptr().add((kk + 1) * n + j))
                } else {
                    vdupq_n_u8(0)
                };
                let v2 = if rows_left > 2 {
                    vld1q_u8(b_u8.as_ptr().add((kk + 2) * n + j))
                } else {
                    vdupq_n_u8(0)
                };
                let v3 = if rows_left > 3 {
                    vld1q_u8(b_u8.as_ptr().add((kk + 3) * n + j))
                } else {
                    vdupq_n_u8(0)
                };

                // 4×16 → 16×4 transpose for UDOT
                let t0 = vzip1q_u8(v0, v1);
                let t1 = vzip2q_u8(v0, v1);
                let t2 = vzip1q_u8(v2, v3);
                let t3 = vzip2q_u8(v2, v3);

                let res0 = vzip1q_u16(vreinterpretq_u16_u8(t0), vreinterpretq_u16_u8(t2));
                let res1 = vzip2q_u16(vreinterpretq_u16_u8(t0), vreinterpretq_u16_u8(t2));
                let res2 = vzip1q_u16(vreinterpretq_u16_u8(t1), vreinterpretq_u16_u8(t3));
                let res3 = vzip2q_u16(vreinterpretq_u16_u8(t1), vreinterpretq_u16_u8(t3));

                let dst = packed_b.as_mut_ptr().add(dst_off);
                vst1q_u8(dst, vreinterpretq_u8_u16(res0));
                vst1q_u8(dst.add(16), vreinterpretq_u8_u16(res1));
                vst1q_u8(dst.add(32), vreinterpretq_u8_u16(res2));
                vst1q_u8(dst.add(48), vreinterpretq_u8_u16(res3));
            }
        }
    }

    // Pack remainder strip (< 16 cols) with zero padding
    if n_rem > 0 {
        let j = n_full_strips * 16;
        let dst_base = n_full_strips * strip_stride;
        for k_block in 0..(k_aligned / 4) {
            let kk = k_block * 4;
            let dst_off = dst_base + k_block * 64;
            // Scalar pack with zero padding for missing columns
            for dk in 0..4 {
                let src_k = kk + dk;
                for col_group in 0..4 {
                    for col_in_group in 0..4 {
                        let src_col = j + col_group * 4 + col_in_group;
                        let byte_idx = col_group * 16 + col_in_group * 4 + dk;
                        packed_b[dst_off + byte_idx] = if src_k < k && src_col < n {
                            b_u8[src_k * n + src_col]
                        } else {
                            0
                        };
                    }
                }
            }
        }
    }

    // Pre-compute f32 col_sums padded to multiple of 16 for NEON loads
    let n_padded = (n + 15) & !15;
    let mut col_sums_f32 = vec![0.0f32; n_padded];
    for j in 0..n {
        col_sums_f32[j] = col_sums[j] as f32;
    }

    PreparedWeightsArm {
        packed_b,
        col_sums,
        col_sums_f32,
        k,
        k_aligned,
        n,
        strip_stride,
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        raw_b: b_u8.to_vec(),
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        dequantized_f32: std::sync::OnceLock::new(),
    }
}

#[inline(always)]
pub fn mat_mul_integer_prepared_neon<'a>(
    a_u8: &[u8],
    m: usize,
    k: usize,
    pw: &PreparedWeightsArm,
    zp_a: i32,
    zp_b: i32,
    scale: Option<&TensorView<f32>>,
    bias: Option<&TensorView<f32>>,
    apply_relu: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    let n = pw.n;
    let output_len = m * n;
    utils::ensure_capacity(out, output_len);
    out.resize(output_len, 0.0);

    let constant_term = (k as f32) * (zp_a as f32) * (zp_b as f32);

    // Pre-compute row sums for A (needed for zero-point correction)
    let mut row_sums_a = vec![0i32; m];
    if zp_b != 0 {
        for r in 0..m {
            let row = &a_u8[r * k..(r + 1) * k];
            let mut s: i32;
            unsafe {
                let mut i = 0;
                // Use UADALP for fast horizontal u8→u16→u32 accumulation
                let mut acc0 = vdupq_n_u32(0);
                while i + 16 <= k {
                    let v = vld1q_u8(row.as_ptr().add(i));
                    // vpaddlq: pairwise add u8→u16, then vpadalq: pairwise add u16→u32
                    let v16 = vpaddlq_u8(v);
                    acc0 = vpadalq_u16(acc0, v16);
                    i += 16;
                }
                s = vaddvq_u32(acc0) as i32;
                while i < k {
                    s += row[i] as i32;
                    i += 1;
                }
            }
            row_sums_a[r] = s;
        }
    }

    let n_full_strips = n / 16;
    let n_rem = n % 16;

    // Process 16-column strips
    for strip in 0..n_full_strips {
        let j = strip * 16;
        let pb_base = pw.packed_b.as_ptr();
        let pb_strip = unsafe { pb_base.add(strip * pw.strip_stride) };

        // Pre-load correction vectors for this strip (hoisted out of row loop)
        let (strip_corr0, strip_corr1, strip_corr2, strip_corr3) = if zp_a != 0 {
            unsafe {
                let zp_a_f32 = zp_a as f32;
                let cs0 = vld1q_f32(pw.col_sums_f32.as_ptr().add(j));
                let cs1 = vld1q_f32(pw.col_sums_f32.as_ptr().add(j + 4));
                let cs2 = vld1q_f32(pw.col_sums_f32.as_ptr().add(j + 8));
                let cs3 = vld1q_f32(pw.col_sums_f32.as_ptr().add(j + 12));
                (
                    vmulq_n_f32(cs0, zp_a_f32),
                    vmulq_n_f32(cs1, zp_a_f32),
                    vmulq_n_f32(cs2, zp_a_f32),
                    vmulq_n_f32(cs3, zp_a_f32),
                )
            }
        } else {
            unsafe {
                let z = vdupq_n_f32(0.0);
                (z, z, z, z)
            }
        };

        // === 4-row micro-kernel ===
        let mut i = 0;
        while i + 4 <= m {
            unsafe {
                // 16 accumulators: acc[row][col_group]
                let mut acc00 = vdupq_n_u32(0);
                let mut acc01 = vdupq_n_u32(0);
                let mut acc02 = vdupq_n_u32(0);
                let mut acc03 = vdupq_n_u32(0);
                let mut acc10 = vdupq_n_u32(0);
                let mut acc11 = vdupq_n_u32(0);
                let mut acc12 = vdupq_n_u32(0);
                let mut acc13 = vdupq_n_u32(0);
                let mut acc20 = vdupq_n_u32(0);
                let mut acc21 = vdupq_n_u32(0);
                let mut acc22 = vdupq_n_u32(0);
                let mut acc23 = vdupq_n_u32(0);
                let mut acc30 = vdupq_n_u32(0);
                let mut acc31 = vdupq_n_u32(0);
                let mut acc32 = vdupq_n_u32(0);
                let mut acc33 = vdupq_n_u32(0);

                let a0_ptr = a_u8.as_ptr().add(i * k);
                let a1_ptr = a_u8.as_ptr().add((i + 1) * k);
                let a2_ptr = a_u8.as_ptr().add((i + 2) * k);
                let a3_ptr = a_u8.as_ptr().add((i + 3) * k);

                let mut pb_ptr = pb_strip;

                // K-loop unrolled 2x: process 8 K-elements per iteration
                let k_full_blocks = k / 4;
                let k_pairs = k_full_blocks / 2;
                let mut kk = 0usize;
                for _ in 0..k_pairs {
                    // K-step 0
                    let va0 = vreinterpretq_u8_u32(vdupq_n_u32(core::ptr::read_unaligned(
                        a0_ptr.add(kk) as *const u32,
                    )));
                    let va1 = vreinterpretq_u8_u32(vdupq_n_u32(core::ptr::read_unaligned(
                        a1_ptr.add(kk) as *const u32,
                    )));
                    let va2 = vreinterpretq_u8_u32(vdupq_n_u32(core::ptr::read_unaligned(
                        a2_ptr.add(kk) as *const u32,
                    )));
                    let va3 = vreinterpretq_u8_u32(vdupq_n_u32(core::ptr::read_unaligned(
                        a3_ptr.add(kk) as *const u32,
                    )));

                    let vb0 = vld1q_u8(pb_ptr);
                    let vb1 = vld1q_u8(pb_ptr.add(16));
                    let vb2 = vld1q_u8(pb_ptr.add(32));
                    let vb3 = vld1q_u8(pb_ptr.add(48));

                    acc00 = vdotq_u32_custom(acc00, va0, vb0);
                    acc01 = vdotq_u32_custom(acc01, va0, vb1);
                    acc02 = vdotq_u32_custom(acc02, va0, vb2);
                    acc03 = vdotq_u32_custom(acc03, va0, vb3);
                    acc10 = vdotq_u32_custom(acc10, va1, vb0);
                    acc11 = vdotq_u32_custom(acc11, va1, vb1);
                    acc12 = vdotq_u32_custom(acc12, va1, vb2);
                    acc13 = vdotq_u32_custom(acc13, va1, vb3);
                    acc20 = vdotq_u32_custom(acc20, va2, vb0);
                    acc21 = vdotq_u32_custom(acc21, va2, vb1);
                    acc22 = vdotq_u32_custom(acc22, va2, vb2);
                    acc23 = vdotq_u32_custom(acc23, va2, vb3);
                    acc30 = vdotq_u32_custom(acc30, va3, vb0);
                    acc31 = vdotq_u32_custom(acc31, va3, vb1);
                    acc32 = vdotq_u32_custom(acc32, va3, vb2);
                    acc33 = vdotq_u32_custom(acc33, va3, vb3);

                    // K-step 1
                    let va0 = vreinterpretq_u8_u32(vdupq_n_u32(core::ptr::read_unaligned(
                        a0_ptr.add(kk + 4) as *const u32,
                    )));
                    let va1 = vreinterpretq_u8_u32(vdupq_n_u32(core::ptr::read_unaligned(
                        a1_ptr.add(kk + 4) as *const u32,
                    )));
                    let va2 = vreinterpretq_u8_u32(vdupq_n_u32(core::ptr::read_unaligned(
                        a2_ptr.add(kk + 4) as *const u32,
                    )));
                    let va3 = vreinterpretq_u8_u32(vdupq_n_u32(core::ptr::read_unaligned(
                        a3_ptr.add(kk + 4) as *const u32,
                    )));

                    let vb0 = vld1q_u8(pb_ptr.add(64));
                    let vb1 = vld1q_u8(pb_ptr.add(80));
                    let vb2 = vld1q_u8(pb_ptr.add(96));
                    let vb3 = vld1q_u8(pb_ptr.add(112));

                    acc00 = vdotq_u32_custom(acc00, va0, vb0);
                    acc01 = vdotq_u32_custom(acc01, va0, vb1);
                    acc02 = vdotq_u32_custom(acc02, va0, vb2);
                    acc03 = vdotq_u32_custom(acc03, va0, vb3);
                    acc10 = vdotq_u32_custom(acc10, va1, vb0);
                    acc11 = vdotq_u32_custom(acc11, va1, vb1);
                    acc12 = vdotq_u32_custom(acc12, va1, vb2);
                    acc13 = vdotq_u32_custom(acc13, va1, vb3);
                    acc20 = vdotq_u32_custom(acc20, va2, vb0);
                    acc21 = vdotq_u32_custom(acc21, va2, vb1);
                    acc22 = vdotq_u32_custom(acc22, va2, vb2);
                    acc23 = vdotq_u32_custom(acc23, va2, vb3);
                    acc30 = vdotq_u32_custom(acc30, va3, vb0);
                    acc31 = vdotq_u32_custom(acc31, va3, vb1);
                    acc32 = vdotq_u32_custom(acc32, va3, vb2);
                    acc33 = vdotq_u32_custom(acc33, va3, vb3);

                    pb_ptr = pb_ptr.add(128);
                    kk += 8;
                }

                // Handle odd remaining K-block if k_full_blocks is odd
                if k_full_blocks & 1 != 0 {
                    let va0 = vreinterpretq_u8_u32(vdupq_n_u32(core::ptr::read_unaligned(
                        a0_ptr.add(kk) as *const u32,
                    )));
                    let va1 = vreinterpretq_u8_u32(vdupq_n_u32(core::ptr::read_unaligned(
                        a1_ptr.add(kk) as *const u32,
                    )));
                    let va2 = vreinterpretq_u8_u32(vdupq_n_u32(core::ptr::read_unaligned(
                        a2_ptr.add(kk) as *const u32,
                    )));
                    let va3 = vreinterpretq_u8_u32(vdupq_n_u32(core::ptr::read_unaligned(
                        a3_ptr.add(kk) as *const u32,
                    )));

                    let vb0 = vld1q_u8(pb_ptr);
                    let vb1 = vld1q_u8(pb_ptr.add(16));
                    let vb2 = vld1q_u8(pb_ptr.add(32));
                    let vb3 = vld1q_u8(pb_ptr.add(48));

                    acc00 = vdotq_u32_custom(acc00, va0, vb0);
                    acc01 = vdotq_u32_custom(acc01, va0, vb1);
                    acc02 = vdotq_u32_custom(acc02, va0, vb2);
                    acc03 = vdotq_u32_custom(acc03, va0, vb3);

                    acc10 = vdotq_u32_custom(acc10, va1, vb0);
                    acc11 = vdotq_u32_custom(acc11, va1, vb1);
                    acc12 = vdotq_u32_custom(acc12, va1, vb2);
                    acc13 = vdotq_u32_custom(acc13, va1, vb3);

                    acc20 = vdotq_u32_custom(acc20, va2, vb0);
                    acc21 = vdotq_u32_custom(acc21, va2, vb1);
                    acc22 = vdotq_u32_custom(acc22, va2, vb2);
                    acc23 = vdotq_u32_custom(acc23, va2, vb3);

                    acc30 = vdotq_u32_custom(acc30, va3, vb0);
                    acc31 = vdotq_u32_custom(acc31, va3, vb1);
                    acc32 = vdotq_u32_custom(acc32, va3, vb2);
                    acc33 = vdotq_u32_custom(acc33, va3, vb3);

                    pb_ptr = pb_ptr.add(64);
                    kk += 4;
                }

                // K remainder: at most one partial block (K%4 != 0)
                if kk < k {
                    let load_partial = |ptr: *const u8, off: usize, remaining: usize| -> u32 {
                        let mut tmp = [0u8; 4];
                        for x in 0..remaining {
                            tmp[x] = *ptr.add(off + x);
                        }
                        u32::from_ne_bytes(tmp)
                    };
                    let rem = k - kk;
                    let va0 = vreinterpretq_u8_u32(vdupq_n_u32(load_partial(a0_ptr, kk, rem)));
                    let va1 = vreinterpretq_u8_u32(vdupq_n_u32(load_partial(a1_ptr, kk, rem)));
                    let va2 = vreinterpretq_u8_u32(vdupq_n_u32(load_partial(a2_ptr, kk, rem)));
                    let va3 = vreinterpretq_u8_u32(vdupq_n_u32(load_partial(a3_ptr, kk, rem)));

                    let vb0 = vld1q_u8(pb_ptr);
                    let vb1 = vld1q_u8(pb_ptr.add(16));
                    let vb2 = vld1q_u8(pb_ptr.add(32));
                    let vb3 = vld1q_u8(pb_ptr.add(48));

                    acc00 = vdotq_u32_custom(acc00, va0, vb0);
                    acc01 = vdotq_u32_custom(acc01, va0, vb1);
                    acc02 = vdotq_u32_custom(acc02, va0, vb2);
                    acc03 = vdotq_u32_custom(acc03, va0, vb3);
                    acc10 = vdotq_u32_custom(acc10, va1, vb0);
                    acc11 = vdotq_u32_custom(acc11, va1, vb1);
                    acc12 = vdotq_u32_custom(acc12, va1, vb2);
                    acc13 = vdotq_u32_custom(acc13, va1, vb3);
                    acc20 = vdotq_u32_custom(acc20, va2, vb0);
                    acc21 = vdotq_u32_custom(acc21, va2, vb1);
                    acc22 = vdotq_u32_custom(acc22, va2, vb2);
                    acc23 = vdotq_u32_custom(acc23, va2, vb3);
                    acc30 = vdotq_u32_custom(acc30, va3, vb0);
                    acc31 = vdotq_u32_custom(acc31, va3, vb1);
                    acc32 = vdotq_u32_custom(acc32, va3, vb2);
                    acc33 = vdotq_u32_custom(acc33, va3, vb3);
                }

                // Apply zero-point corrections and write results for 4 rows
                // Unrolled: process each of the 4 rows inline to avoid array/iter overhead
                macro_rules! finalize_row {
                    ($ri:expr, $a0:expr, $a1:expr, $a2:expr, $a3:expr, $row_sum:expr) => {{
                        let out_ptr = out.as_mut_ptr().add((i + $ri) * n + j);
                        let base_corr =
                            vdupq_n_f32(constant_term - ($row_sum as f32) * (zp_b as f32));
                        let corr0 = vsubq_f32(base_corr, strip_corr0);
                        let corr1 = vsubq_f32(base_corr, strip_corr1);
                        let corr2 = vsubq_f32(base_corr, strip_corr2);
                        let corr3 = vsubq_f32(base_corr, strip_corr3);

                        let mut result0 = vaddq_f32(corr0, vcvtq_f32_u32($a0));
                        let mut result1 = vaddq_f32(corr1, vcvtq_f32_u32($a1));
                        let mut result2 = vaddq_f32(corr2, vcvtq_f32_u32($a2));
                        let mut result3 = vaddq_f32(corr3, vcvtq_f32_u32($a3));

                        if let Some(scale_data) = scale {
                            if scale_data.data.len() == 1 {
                                let sv = vdupq_n_f32(scale_data.data[0]);
                                result0 = vmulq_f32(result0, sv);
                                result1 = vmulq_f32(result1, sv);
                                result2 = vmulq_f32(result2, sv);
                                result3 = vmulq_f32(result3, sv);
                            } else {
                                result0 =
                                    vmulq_f32(result0, vld1q_f32(scale_data.data.as_ptr().add(j)));
                                result1 = vmulq_f32(
                                    result1,
                                    vld1q_f32(scale_data.data.as_ptr().add(j + 4)),
                                );
                                result2 = vmulq_f32(
                                    result2,
                                    vld1q_f32(scale_data.data.as_ptr().add(j + 8)),
                                );
                                result3 = vmulq_f32(
                                    result3,
                                    vld1q_f32(scale_data.data.as_ptr().add(j + 12)),
                                );
                            }
                        }

                        if let Some(bias_data) = bias {
                            result0 = vaddq_f32(result0, vld1q_f32(bias_data.data.as_ptr().add(j)));
                            result1 =
                                vaddq_f32(result1, vld1q_f32(bias_data.data.as_ptr().add(j + 4)));
                            result2 =
                                vaddq_f32(result2, vld1q_f32(bias_data.data.as_ptr().add(j + 8)));
                            result3 =
                                vaddq_f32(result3, vld1q_f32(bias_data.data.as_ptr().add(j + 12)));
                        }

                        if apply_relu {
                            let zero = vdupq_n_f32(0.0);
                            result0 = vmaxq_f32(result0, zero);
                            result1 = vmaxq_f32(result1, zero);
                            result2 = vmaxq_f32(result2, zero);
                            result3 = vmaxq_f32(result3, zero);
                        }

                        vst1q_f32(out_ptr, result0);
                        vst1q_f32(out_ptr.add(4), result1);
                        vst1q_f32(out_ptr.add(8), result2);
                        vst1q_f32(out_ptr.add(12), result3);
                    }};
                }

                finalize_row!(0, acc00, acc01, acc02, acc03, row_sums_a[i]);
                finalize_row!(1, acc10, acc11, acc12, acc13, row_sums_a[i + 1]);
                finalize_row!(2, acc20, acc21, acc22, acc23, row_sums_a[i + 2]);
                finalize_row!(3, acc30, acc31, acc32, acc33, row_sums_a[i + 3]);
            }
            i += 4;
        }

        // Handle remaining 1-3 rows with 1×16 kernel
        while i < m {
            unsafe {
                let mut acc0 = vdupq_n_u32(0);
                let mut acc1 = vdupq_n_u32(0);
                let mut acc2 = vdupq_n_u32(0);
                let mut acc3 = vdupq_n_u32(0);

                let a_ptr = a_u8.as_ptr().add(i * k);
                let mut pb_ptr = pb_strip;

                // Fast path: full 4-byte K-blocks (branchless)
                let k_full_blocks = k / 4;
                let mut kk = 0usize;
                for _ in 0..k_full_blocks {
                    let va = vreinterpretq_u8_u32(vdupq_n_u32(core::ptr::read_unaligned(
                        a_ptr.add(kk) as *const u32,
                    )));

                    acc0 = vdotq_u32_custom(acc0, va, vld1q_u8(pb_ptr));
                    acc1 = vdotq_u32_custom(acc1, va, vld1q_u8(pb_ptr.add(16)));
                    acc2 = vdotq_u32_custom(acc2, va, vld1q_u8(pb_ptr.add(32)));
                    acc3 = vdotq_u32_custom(acc3, va, vld1q_u8(pb_ptr.add(48)));

                    pb_ptr = pb_ptr.add(64);
                    kk += 4;
                }
                // K remainder
                if kk < k {
                    let mut tmp = [0u8; 4];
                    for x in 0..(k - kk) {
                        tmp[x] = *a_ptr.add(kk + x);
                    }
                    let va = vreinterpretq_u8_u32(vdupq_n_u32(u32::from_ne_bytes(tmp)));
                    acc0 = vdotq_u32_custom(acc0, va, vld1q_u8(pb_ptr));
                    acc1 = vdotq_u32_custom(acc1, va, vld1q_u8(pb_ptr.add(16)));
                    acc2 = vdotq_u32_custom(acc2, va, vld1q_u8(pb_ptr.add(32)));
                    acc3 = vdotq_u32_custom(acc3, va, vld1q_u8(pb_ptr.add(48)));
                }

                let out_ptr = out.as_mut_ptr().add(i * n + j);
                let base_corr = vdupq_n_f32(constant_term - (row_sums_a[i] as f32) * (zp_b as f32));

                let mut result0 = vaddq_f32(vsubq_f32(base_corr, strip_corr0), vcvtq_f32_u32(acc0));
                let mut result1 = vaddq_f32(vsubq_f32(base_corr, strip_corr1), vcvtq_f32_u32(acc1));
                let mut result2 = vaddq_f32(vsubq_f32(base_corr, strip_corr2), vcvtq_f32_u32(acc2));
                let mut result3 = vaddq_f32(vsubq_f32(base_corr, strip_corr3), vcvtq_f32_u32(acc3));

                if let Some(scale_data) = scale {
                    if scale_data.data.len() == 1 {
                        let sv = vdupq_n_f32(scale_data.data[0]);
                        result0 = vmulq_f32(result0, sv);
                        result1 = vmulq_f32(result1, sv);
                        result2 = vmulq_f32(result2, sv);
                        result3 = vmulq_f32(result3, sv);
                    } else {
                        result0 = vmulq_f32(result0, vld1q_f32(scale_data.data.as_ptr().add(j)));
                        result1 =
                            vmulq_f32(result1, vld1q_f32(scale_data.data.as_ptr().add(j + 4)));
                        result2 =
                            vmulq_f32(result2, vld1q_f32(scale_data.data.as_ptr().add(j + 8)));
                        result3 =
                            vmulq_f32(result3, vld1q_f32(scale_data.data.as_ptr().add(j + 12)));
                    }
                }

                if let Some(bias_data) = bias {
                    result0 = vaddq_f32(result0, vld1q_f32(bias_data.data.as_ptr().add(j)));
                    result1 = vaddq_f32(result1, vld1q_f32(bias_data.data.as_ptr().add(j + 4)));
                    result2 = vaddq_f32(result2, vld1q_f32(bias_data.data.as_ptr().add(j + 8)));
                    result3 = vaddq_f32(result3, vld1q_f32(bias_data.data.as_ptr().add(j + 12)));
                }

                if apply_relu {
                    let zero = vdupq_n_f32(0.0);
                    result0 = vmaxq_f32(result0, zero);
                    result1 = vmaxq_f32(result1, zero);
                    result2 = vmaxq_f32(result2, zero);
                    result3 = vmaxq_f32(result3, zero);
                }

                vst1q_f32(out_ptr, result0);
                vst1q_f32(out_ptr.add(4), result1);
                vst1q_f32(out_ptr.add(8), result2);
                vst1q_f32(out_ptr.add(12), result3);
            }
            i += 1;
        }
    }

    // Remainder columns (< 16)
    if n_rem > 0 {
        let j_start = n_full_strips * 16;
        for i in 0..m {
            let row_a = &a_u8[i * k..(i + 1) * k];
            for j in j_start..n {
                // Scalar dot product from packed remainder strip
                let mut dot: i32 = 0;
                // Access the remainder strip
                let strip_idx = n_full_strips;
                let col_in_strip = j - j_start;
                let pb_base = strip_idx * pw.strip_stride;
                // Scalar dot product from packed format
                for kk in 0..k {
                    let k_block = kk / 4;
                    let k_in_block = kk % 4;
                    let col_group = col_in_strip / 4;
                    let col_in_group = col_in_strip % 4;
                    let byte_idx =
                        pb_base + k_block * 64 + col_group * 16 + col_in_group * 4 + k_in_block;
                    dot += (row_a[kk] as i32) * (pw.packed_b[byte_idx] as i32);
                }

                let mut val = dot as f32 + constant_term;
                if zp_b != 0 {
                    val -= (row_sums_a[i] as f32) * (zp_b as f32);
                }
                if zp_a != 0 {
                    val -= (pw.col_sums[j] as f32) * (zp_a as f32);
                }

                if let Some(scale_data) = scale {
                    if scale_data.data.len() == 1 {
                        val *= scale_data.data[0];
                    } else {
                        val *= scale_data.data[j];
                    }
                }
                if let Some(bias_data) = bias {
                    val += bias_data.data[j];
                }
                if apply_relu && val < 0.0 {
                    val = 0.0;
                }
                out[i * n + j] = val;
            }
        }
    }

    TensorView {
        data: Cow::Borrowed(out),
        shape: Cow::Owned(vec![m, n]),
    }
}

#[inline(always)]
pub fn fused_dq_gemm_neon<'a>(
    input: &[f32], // fp32 input data, row-major [M*K]
    m: usize,
    k: usize,
    pw: &PreparedWeightsArm,
    zp_b: i32,
    weight_scale: &[f32], // per-channel weight scale [N]
    bias: Option<&[f32]>, // per-channel bias [N]
    apply_relu: bool,
    a_u8_scratch: &mut Vec<u8>,   // reusable buffer for quantized A
    scale_scratch: &mut Vec<f32>, // reusable buffer for combined_scale
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    let n = pw.n;
    let len = m * k;

    let (min_val, max_val) = unsafe {
        let ptr = input.as_ptr();
        let mut v_min0 = vdupq_n_f32(f32::MAX);
        let mut v_max0 = vdupq_n_f32(f32::MIN);
        let mut v_min1 = vdupq_n_f32(f32::MAX);
        let mut v_max1 = vdupq_n_f32(f32::MIN);
        let mut i = 0;
        while i + 8 <= len {
            let va = vld1q_f32(ptr.add(i));
            let vb = vld1q_f32(ptr.add(i + 4));
            v_min0 = vminq_f32(v_min0, va);
            v_max0 = vmaxq_f32(v_max0, va);
            v_min1 = vminq_f32(v_min1, vb);
            v_max1 = vmaxq_f32(v_max1, vb);
            i += 8;
        }
        v_min0 = vminq_f32(v_min0, v_min1);
        v_max0 = vmaxq_f32(v_max0, v_max1);
        while i + 4 <= len {
            let v = vld1q_f32(ptr.add(i));
            v_min0 = vminq_f32(v_min0, v);
            v_max0 = vmaxq_f32(v_max0, v);
            i += 4;
        }
        // Horizontal reduction
        let min_arr = [
            vgetq_lane_f32(v_min0, 0),
            vgetq_lane_f32(v_min0, 1),
            vgetq_lane_f32(v_min0, 2),
            vgetq_lane_f32(v_min0, 3),
        ];
        let max_arr = [
            vgetq_lane_f32(v_max0, 0),
            vgetq_lane_f32(v_max0, 1),
            vgetq_lane_f32(v_max0, 2),
            vgetq_lane_f32(v_max0, 3),
        ];
        let mut min_v = min_arr.iter().fold(f32::MAX, |a, &b| a.min(b));
        let mut max_v = max_arr.iter().fold(f32::MIN, |a, &b| a.max(b));
        for j in i..len {
            let v = *ptr.add(j);
            if v < min_v {
                min_v = v;
            }
            if v > max_v {
                max_v = v;
            }
        }
        (min_v, max_v)
    };

    let adjusted_max = max_val.max(0.0);
    let adjusted_min = min_val.min(0.0);
    let range = (adjusted_max - adjusted_min).max(1e-5);
    let a_scale = range / 255.0;
    let a_zp = (-adjusted_min / a_scale).round().clamp(0.0, 255.0);
    let zp_a = a_zp as i32;
    let inv_scale = 1.0 / a_scale;

    utils::ensure_capacity(a_u8_scratch, len);

    unsafe {
        let src = input.as_ptr();
        let dst = a_u8_scratch.as_mut_ptr();
        let v_inv_scale = vdupq_n_f32(inv_scale);
        let v_zp = vdupq_n_f32(a_zp);
        let v_zero = vdupq_n_f32(0.0);
        let v_255 = vdupq_n_f32(255.0);
        let mut i = 0;

        while i + 16 <= len {
            // Quantize: round(val * inv_scale + zp), clamp [0,255]
            // Same sequence as dynamic_quantize_linear for bit-exact match
            let s0 = vmulq_f32(vld1q_f32(src.add(i)), v_inv_scale);
            let s1 = vmulq_f32(vld1q_f32(src.add(i + 4)), v_inv_scale);
            let s2 = vmulq_f32(vld1q_f32(src.add(i + 8)), v_inv_scale);
            let s3 = vmulq_f32(vld1q_f32(src.add(i + 12)), v_inv_scale);

            let r0 = vrndnq_f32(vaddq_f32(s0, v_zp));
            let r1 = vrndnq_f32(vaddq_f32(s1, v_zp));
            let r2 = vrndnq_f32(vaddq_f32(s2, v_zp));
            let r3 = vrndnq_f32(vaddq_f32(s3, v_zp));

            let c0 = vminq_f32(vmaxq_f32(r0, v_zero), v_255);
            let c1 = vminq_f32(vmaxq_f32(r1, v_zero), v_255);
            let c2 = vminq_f32(vmaxq_f32(r2, v_zero), v_255);
            let c3 = vminq_f32(vmaxq_f32(r3, v_zero), v_255);

            // Convert to u32 (truncate — value is already integer)
            let u0 = vcvtq_u32_f32(c0);
            let u1 = vcvtq_u32_f32(c1);
            let u2 = vcvtq_u32_f32(c2);
            let u3 = vcvtq_u32_f32(c3);

            // Saturating narrow: u32 → u16 → u8
            let n0 = vqmovn_u32(u0);
            let n1 = vqmovn_u32(u1);
            let n2 = vqmovn_u32(u2);
            let n3 = vqmovn_u32(u3);
            let nn0 = vcombine_u16(n0, n1);
            let nn1 = vcombine_u16(n2, n3);
            let b0 = vqmovn_u16(nn0);
            let b1 = vqmovn_u16(nn1);
            vst1q_u8(dst.add(i), vcombine_u8(b0, b1));

            i += 16;
        }

        // Scalar remainder (same formula as dynamic_quantize_linear)
        while i < len {
            let val = (*src.add(i) / a_scale + a_zp).round().clamp(0.0, 255.0);
            *dst.add(i) = val as u8;
            i += 1;
        }
    }

    let scalar_combined: f32;
    let use_scalar_scale = weight_scale.len() == 1;

    if use_scalar_scale {
        scalar_combined = a_scale * weight_scale[0];
    } else {
        scalar_combined = 0.0; // unused
        let scale_padded = (n + 15) & !15;
        utils::ensure_capacity(scale_scratch, scale_padded);

        unsafe {
            let mut j = 0;
            let v_a_scale = vdupq_n_f32(a_scale);
            while j + 4 <= n {
                let ws = vld1q_f32(weight_scale.as_ptr().add(j));
                vst1q_f32(scale_scratch.as_mut_ptr().add(j), vmulq_f32(v_a_scale, ws));
                j += 4;
            }
            while j < n {
                scale_scratch[j] = a_scale * weight_scale[j];
                j += 1;
            }
            for j in n..scale_padded {
                scale_scratch[j] = 0.0;
            }
        }
    }

    let scalar_buf;
    let shape_1 = [1usize];
    let shape_n = [n];
    let scale_tv = if use_scalar_scale {
        scalar_buf = [scalar_combined];
        TensorView {
            data: Cow::Borrowed(&scalar_buf[..]),
            shape: Cow::Borrowed(&shape_1[..]),
        }
    } else {
        TensorView {
            data: Cow::Borrowed(&scale_scratch[..n]),
            shape: Cow::Borrowed(&shape_n[..]),
        }
    };

    let bias_tv = bias.map(|b| TensorView {
        data: Cow::Borrowed(b),
        shape: Cow::Owned(vec![b.len()]),
    });

    mat_mul_integer_prepared_neon(
        &a_u8_scratch[..len],
        m,
        k,
        pw,
        zp_a,
        zp_b,
        Some(&scale_tv),
        bias_tv.as_ref(),
        apply_relu,
        out,
    )
}

pub fn dynamic_quantize_linear<'a, 'b>(
    x: &TensorView<'b, f32>,
    out_y: &'a mut Vec<f32>,
    out_scale: &'a mut Vec<f32>,
    out_zp: &'a mut Vec<f32>,
) -> (
    TensorView<'a, f32>,
    TensorView<'a, f32>,
    TensorView<'a, f32>,
) {
    // Legacy fake quantization fallback
    let len = x.data.len();
    if len == 0 {
        return (
            TensorView::from_owned(vec![], x.shape.to_vec()),
            TensorView::from_owned(vec![1.0], vec![1]),
            TensorView::from_owned(vec![0.0], vec![1]),
        );
    }

    // SIMD-optimized min/max finding
    #[cfg(target_arch = "aarch64")]
    let (min_val, max_val) = unsafe {
        let mut min_vec = vdupq_n_f32(f32::MAX);
        let mut max_vec = vdupq_n_f32(f32::MIN);

        let mut i = 0;
        let simd_end = (len / 4) * 4;

        while i < simd_end {
            let v = vld1q_f32(x.data.as_ptr().add(i));
            min_vec = vminq_f32(min_vec, v);
            max_vec = vmaxq_f32(max_vec, v);
            i += 4;
        }

        // Horizontal min/max
        let min_arr = [
            vgetq_lane_f32(min_vec, 0),
            vgetq_lane_f32(min_vec, 1),
            vgetq_lane_f32(min_vec, 2),
            vgetq_lane_f32(min_vec, 3),
        ];
        let max_arr = [
            vgetq_lane_f32(max_vec, 0),
            vgetq_lane_f32(max_vec, 1),
            vgetq_lane_f32(max_vec, 2),
            vgetq_lane_f32(max_vec, 3),
        ];

        let mut min_val = min_arr.iter().fold(f32::MAX, |a, &b| a.min(b));
        let mut max_val = max_arr.iter().fold(f32::MIN, |a, &b| a.max(b));

        // Process remaining elements
        for &v in &x.data[simd_end..] {
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }

        (min_val, max_val)
    };

    #[cfg(not(target_arch = "aarch64"))]
    let (min_val, max_val) = {
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        for &v in x.data.iter() {
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }
        (min_val, max_val)
    };

    let adjusted_max = max_val.max(0.0);
    let adjusted_min = min_val.min(0.0);
    let range = (adjusted_max - adjusted_min).max(1e-5);
    let scale = range / 255.0;
    let zp = (-adjusted_min / scale).round().clamp(0.0, 255.0);

    utils::ensure_capacity(out_scale, 1);
    out_scale[0] = scale;
    utils::ensure_capacity(out_zp, 1);
    out_zp[0] = zp;
    utils::ensure_capacity(out_y, len);

    // SIMD-optimized quantization
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let inv_scale_vec = vdupq_n_f32(1.0 / scale);
        let zp_vec = vdupq_n_f32(zp);
        let zero_vec = vdupq_n_f32(0.0);
        let max_vec = vdupq_n_f32(255.0);

        let mut i = 0;
        let simd_end = (len / 4) * 4;

        while i < simd_end {
            let v = vld1q_f32(x.data.as_ptr().add(i));
            let scaled = vmulq_f32(v, inv_scale_vec);
            let rounded = vrndnq_f32(vaddq_f32(scaled, zp_vec));
            let clamped = vminq_f32(vmaxq_f32(rounded, zero_vec), max_vec);
            vst1q_f32(out_y.as_mut_ptr().add(i), clamped);
            i += 4;
        }

        // Process remaining elements
        for j in simd_end..len {
            out_y[j] = (x.data[j] / scale + zp).round().clamp(0.0, 255.0);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let inv_scale = 1.0 / scale;
        for i in 0..len {
            out_y[i] = (x.data[i] * inv_scale + zp).round().clamp(0.0, 255.0);
        }
    }

    (
        TensorView {
            data: Cow::Borrowed(out_y),
            shape: Cow::Owned(x.shape.to_vec()),
        },
        TensorView {
            data: Cow::Borrowed(out_scale),
            shape: Cow::Owned(vec![1]),
        },
        TensorView {
            data: Cow::Borrowed(out_zp),
            shape: Cow::Owned(vec![1]),
        },
    )
}

pub fn dynamic_quantize_linear_u8<'a, 'b>(
    x: &TensorView<'b, f32>,
    out_y_storage: &'a mut Vec<f32>,
    out_scale: &'a mut Vec<f32>,
    out_zp: &'a mut Vec<f32>,
) -> (TensorView<'a, u8>, TensorView<'a, f32>, TensorView<'a, u8>) {
    let len = x.data.len();
    // We store u8 data in a f32 vector by casting the pointer.
    // Ensure enough capacity: 1 f32 = 4 u8.
    let cap_bytes = out_y_storage.capacity() * 4;
    if cap_bytes < len {
        out_y_storage.reserve(len.div_ceil(4));
    }

    // IMPORTANT: This reuses the memory of a Vec<f32> to store u8.
    // The caller must be aware that `out_y_storage` now contains raw u8 bytes
    // effectively, even though it's typed as Vec<f32>.
    let ptr = out_y_storage.as_mut_ptr() as *mut u8;
    let out_u8 = unsafe { std::slice::from_raw_parts_mut(ptr, len) };

    #[allow(unused)]
    let mut min_val = f32::MAX;
    #[allow(unused)]
    let mut max_val = f32::MIN;

    // Vectorized Min/Max
    let mut cur = 0;
    unsafe {
        let mut v_min = vdupq_n_f32(f32::MAX);
        let mut v_max = vdupq_n_f32(f32::MIN);

        while cur + 4 <= len {
            let v_data = vld1q_f32(x.data.as_ptr().add(cur));
            v_min = vminnmq_f32(v_min, v_data);
            v_max = vmaxnmq_f32(v_max, v_data);
            cur += 4;
        }

        // Reduce min
        min_val = vminvq_f32(v_min);
        max_val = vmaxvq_f32(v_max);
    }

    // Remainder Min/Max
    for j in cur..len {
        let v = x.data[j];
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    }

    let adjusted_max = max_val.max(0.0);
    let adjusted_min = min_val.min(0.0);
    let range = (adjusted_max - adjusted_min).max(1e-5);
    let scale = range / 255.0;
    let zp = (-adjusted_min / scale).round().clamp(0.0, 255.0);
    let inv_scale = 1.0 / scale;

    utils::ensure_capacity(out_scale, 1);
    out_scale[0] = scale;

    utils::ensure_capacity(out_zp, 1);
    let ptr_zp = out_zp.as_mut_ptr() as *mut u8;
    let out_zp_u8 = unsafe { std::slice::from_raw_parts_mut(ptr_zp, 1) };
    out_zp_u8[0] = zp as u8;

    // Vectorized Quantization
    let mut cur = 0;
    unsafe {
        let v_inv_scale = vdupq_n_f32(inv_scale);
        let v_zp = vdupq_n_f32(zp);

        while cur + 16 <= len {
            // Process 16 float elements -> 16 u8 elements (1 vector)
            let v0 = vld1q_f32(x.data.as_ptr().add(cur));
            let v1 = vld1q_f32(x.data.as_ptr().add(cur + 4));
            let v2 = vld1q_f32(x.data.as_ptr().add(cur + 8));
            let v3 = vld1q_f32(x.data.as_ptr().add(cur + 12));

            // Mul + Add
            let r0 = vfmaq_f32(v_zp, v0, v_inv_scale);
            let r1 = vfmaq_f32(v_zp, v1, v_inv_scale);
            let r2 = vfmaq_f32(v_zp, v2, v_inv_scale);
            let r3 = vfmaq_f32(v_zp, v3, v_inv_scale);

            // Convert to u32 (Round to nearest)
            let i0 = vcvtaq_u32_f32(r0);
            let i1 = vcvtaq_u32_f32(r1);
            let i2 = vcvtaq_u32_f32(r2);
            let i3 = vcvtaq_u32_f32(r3);

            // Narrow u32 -> u16 (Saturate)
            let n0 = vqmovn_u32(i0); // 4x u16
            let n1 = vqmovn_u32(i1);
            let n2 = vqmovn_u32(i2);
            let n3 = vqmovn_u32(i3);

            // Combine n0/n1 -> 8x u16, n2/n3 -> 8x u16
            let nn0 = vcombine_u16(n0, n1);
            let nn1 = vcombine_u16(n2, n3);

            // Narrow u16 -> u8 (Saturate)
            let b0 = vqmovn_u16(nn0); // 8x u8
            let b1 = vqmovn_u16(nn1);

            // Combine -> 16x u8
            let res = vcombine_u8(b0, b1);

            vst1q_u8(out_u8.as_mut_ptr().add(cur), res);
            cur += 16;
        }
    }

    for j in cur..len {
        out_u8[j] = (x.data[j] * inv_scale + zp).round().clamp(0.0, 255.0) as u8;
    }

    (
        TensorView {
            data: Cow::Borrowed(out_u8),
            shape: Cow::Owned(x.shape.to_vec()),
        },
        TensorView::new(out_scale, &[1]),
        TensorView::new(out_zp_u8, &[1]),
    )
}

use std::sync::Arc;

pub fn mat_mul_integer_u8<'a, 'b, 'c>(
    a: &TensorView<'b, u8>,
    b: &TensorView<'c, u8>,
    a_zero_point: Option<&TensorView<'b, u8>>,
    b_zero_point: Option<&TensorView<'c, u8>>,
    scale: Option<&TensorView<'b, f32>>,
    bias: Option<&TensorView<'b, f32>>,
    apply_relu: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    let zp_a = a_zero_point.map(|z| z.data[0] as i32).unwrap_or(0);
    let zp_b = b_zero_point.map(|z| z.data[0] as i32).unwrap_or(0);

    let a_dims = a.shape.len();
    let b_dims = b.shape.len();

    let m = a.shape[a_dims - 2];
    let k = a.shape[a_dims - 1];
    let n = b.shape[b_dims - 1];

    let batch_a: usize = a.shape[..a_dims - 2].iter().product();
    let batch_b: usize = b.shape[..b_dims - 2].iter().product();
    let final_batch = batch_a.max(batch_b).max(1);
    let output_len = final_batch * m * n;

    utils::ensure_capacity(out, output_len);
    out.resize(output_len, 0.0);

    let stride_a = m * k;
    let stride_b = k * n;
    let stride_out = m * n;

    // Thread-local cache for PreparedWeightsArm, keyed by (b_data_ptr, k, n).
    // For static model weights (Cow::Borrowed), the pointer is stable across calls.
    use std::cell::RefCell;
    use std::collections::HashMap;
    thread_local! {
        static WEIGHT_CACHE: RefCell<HashMap<(usize, usize, usize), Arc<PreparedWeightsArm>>> =
            RefCell::new(HashMap::new());
    }

    for b_i in 0..final_batch {
        let a_offset = if batch_a <= 1 { 0 } else { b_i * stride_a };
        let b_offset = if batch_b <= 1 { 0 } else { b_i * stride_b };

        let a_slice = &a.data[a_offset..a_offset + stride_a];
        let b_slice = &b.data[b_offset..b_offset + stride_b];

        // Try to use cached PreparedWeightsArm
        let b_ptr = b_slice.as_ptr() as usize;
        let cache_key = (b_ptr, k, n);

        let pw = WEIGHT_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            cache
                .entry(cache_key)
                .or_insert_with(|| Arc::new(prepare_weights_arm(b_slice, k, n)))
                .clone()
        });

        if final_batch == 1 {
            mat_mul_integer_prepared_neon(
                a_slice, m, k, &pw, zp_a, zp_b, scale, bias, apply_relu, out,
            );
        } else {
            let mut batch_out = vec![0f32; stride_out];
            mat_mul_integer_prepared_neon(
                a_slice,
                m,
                k,
                &pw,
                zp_a,
                zp_b,
                scale,
                bias,
                apply_relu,
                &mut batch_out,
            );
            let out_offset = b_i * stride_out;
            out[out_offset..out_offset + stride_out].copy_from_slice(&batch_out);
        }
    }

    let mut output_shape = if batch_a >= batch_b {
        a.shape[..a_dims - 2].to_vec()
    } else {
        b.shape[..b_dims - 2].to_vec()
    };
    output_shape.push(m);
    output_shape.push(n);

    TensorView {
        data: Cow::Borrowed(out),
        shape: Cow::Owned(output_shape),
    }
}
