//! AVX2 quantized GEMM: u8 activations against i8 weights.
//!
//! The VNNI counterpart in [`crate::kernels::avx512::qgemm`] is the one to use
//! where `vpdpbusd` exists. This is the same algorithm for machines without it,
//! built on `vpmaddwd`, which multiplies eight pairs of i16 and adds adjacent
//! products into four i32 lanes. The multiply is exact — the intermediate is
//! 32-bit — so unlike `vpmaddubsw` it cannot saturate, which matters: across a
//! typical model's weights the largest adjacent pair of magnitudes is well
//! past the 128 that `vpmaddubsw` needs to stay in range.
//!
//! `vpmaddwd` reduces along K inside the register, which would normally force
//! one accumulator per output element and a horizontal sum at the end. Packing
//! B with two consecutive K values interleaved per column sidesteps that: a
//! broadcast of two A values against such a panel accumulates straight into
//! per-column i32 lanes, exactly as `vpdpbusd` does with four. Six rows by
//! sixteen columns then fills twelve of the sixteen registers with
//! accumulators, leaving room for the two B operands and the broadcast, and
//! issues twelve `vpmaddwd` per two K values against two multiply ports.

use std::arch::x86_64::*;

/// `ymm` lanes of output columns handled by one pass of the microkernel.
pub const NR: usize = 2;
/// Output columns handled by one pass of the microkernel.
pub const N_PANEL: usize = NR * 8;
/// Rows handled by one pass of the microkernel.
///
/// `MR * NR` accumulators plus one B operand and one broadcast have to live in
/// sixteen registers, and the loop wants as many multiplies per load as it can
/// get: loads are `NR + MR` against `MR * NR` multiplies.
pub const MR: usize = 5;

/// Weights packed for `vpmaddwd`, widened to i16 at pack time so the inner loop
/// never pays for the conversion, plus the column sums the zero-point
/// correction needs.
pub struct PackedI8WeightsAvx2 {
    /// `[n_pad / N_PANEL][k2 / 2][N_PANEL][2]`, zero padded.
    pub data: Vec<i16>,
    /// `sum_k qb[k, j]`, length `n_pad`.
    pub col_sums: Vec<i32>,
    pub k: usize,
    /// `k` rounded up to a multiple of 2.
    pub k2: usize,
    pub n: usize,
    /// `n` rounded up to a multiple of [`N_PANEL`].
    pub n_pad: usize,
}

/// Packs a row-major `[k, n]` i8 weight matrix. One-shot and cached, like the
/// VNNI variant.
pub fn pack_i8_weights_avx2(b: &[i8], k: usize, n: usize) -> PackedI8WeightsAvx2 {
    let k2 = k.next_multiple_of(2);
    let n_pad = n.next_multiple_of(N_PANEL);
    let kb_count = k2 / 2;
    let mut data = vec![0i16; n_pad * k2];

    for jb in 0..n_pad / N_PANEL {
        for kb in 0..kb_count {
            let block = (jb * kb_count + kb) * (N_PANEL * 2);
            for j in 0..N_PANEL {
                let col = jb * N_PANEL + j;
                if col >= n {
                    continue;
                }
                for t in 0..2 {
                    let kk = kb * 2 + t;
                    if kk < k {
                        data[block + j * 2 + t] = b[kk * n + col] as i16;
                    }
                }
            }
        }
    }

    let mut col_sums = vec![0i32; n_pad];
    for kk in 0..k {
        for (col, &v) in b[kk * n..kk * n + n].iter().enumerate() {
            col_sums[col] += v as i32;
        }
    }

    PackedI8WeightsAvx2 {
        data,
        col_sums,
        k,
        k2,
        n,
        n_pad,
    }
}

/// Accumulates `M` rows against one 16-column panel over the whole K range.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn accumulate_panel<const M: usize>(
    a: *const i16,
    lda: usize,
    bp: *const i16,
    kb_count: usize,
) -> [[__m256i; NR]; M] {
    unsafe {
        let mut acc = [[_mm256_setzero_si256(); NR]; M];
        // The B panel is held across the row loop and each broadcast is used
        // and discarded, so only NR + 1 registers are live beyond the
        // accumulators. Holding the broadcasts instead costs M, which is worse
        // whenever there are more rows than column groups, and spilling an
        // accumulator undoes far more than the loads save.
        let mut bv = [_mm256_setzero_si256(); NR];
        for kb in 0..kb_count {
            let block = bp.add(kb * (N_PANEL * 2));
            for (g, b) in bv.iter_mut().enumerate() {
                *b = _mm256_loadu_si256(block.add(g * 16) as *const __m256i);
            }
            for (r, accr) in acc.iter_mut().enumerate() {
                // Two consecutive i16 of this row, splatted across the vector,
                // to line up with the two K values interleaved in each column.
                let pair = (a.add(r * lda + kb * 2) as *const i32).read_unaligned();
                let av = _mm256_set1_epi32(pair);
                for g in 0..NR {
                    accr[g] = _mm256_add_epi32(accr[g], _mm256_madd_epi16(av, bv[g]));
                }
            }
        }
        acc
    }
}

/// Applies the zero-point correction, the combined scale and the optional bias,
/// then stores, masking off any columns past `n`.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn store_panel<const M: usize>(
    acc: &[[__m256i; NR]; M],
    col_base: usize,
    n: usize,
    col_sums: *const i32,
    a_zero_point: i32,
    a_scale: f32,
    w_scale: *const f32,
    w_scale_len: usize,
    bias: Option<*const f32>,
    out: *mut f32,
    ldc: usize,
) {
    unsafe {
        let zp = _mm256_set1_epi32(a_zero_point);
        let sa = _mm256_set1_ps(a_scale);
        for sub in 0..NR {
            let col = col_base + sub * 8;
            if col >= n {
                break;
            }
            let rem = n - col;
            // AVX2 has no mask registers; a lane is written when its mask word
            // has the high bit set.
            let mut lanes = [0i32; 8];
            for (i, l) in lanes.iter_mut().enumerate() {
                *l = if i < rem { -1 } else { 0 };
            }
            let mask = _mm256_loadu_si256(lanes.as_ptr() as *const __m256i);

            let cs = _mm256_loadu_si256(col_sums.add(col) as *const __m256i);
            let corr = _mm256_mullo_epi32(cs, zp);
            let scale = if w_scale_len == 1 {
                _mm256_mul_ps(sa, _mm256_set1_ps(*w_scale))
            } else {
                _mm256_mul_ps(sa, _mm256_maskload_ps(w_scale.add(col), mask))
            };
            let bias_v = match bias {
                Some(p) => _mm256_maskload_ps(p.add(col), mask),
                None => _mm256_setzero_ps(),
            };
            for (r, accr) in acc.iter().enumerate() {
                let v = _mm256_sub_epi32(accr[sub], corr);
                let f = _mm256_cvtepi32_ps(v);
                let y = _mm256_fmadd_ps(f, scale, bias_v);
                _mm256_maskstore_ps(out.add(r * ldc + col), mask, y);
            }
        }
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn tile<const M: usize>(
    a: *const i16,
    lda: usize,
    bp: *const i16,
    kb_count: usize,
    col_base: usize,
    n: usize,
    col_sums: *const i32,
    a_zero_point: i32,
    a_scale: f32,
    w_scale: *const f32,
    w_scale_len: usize,
    bias: Option<*const f32>,
    out: *mut f32,
    ldc: usize,
) {
    unsafe {
        let acc = accumulate_panel::<M>(a, lda, bp, kb_count);
        store_panel::<M>(
            &acc,
            col_base,
            n,
            col_sums,
            a_zero_point,
            a_scale,
            w_scale,
            w_scale_len,
            bias,
            out,
            ldc,
        );
    }
}

/// `out[m, n] = a_scale * w_scale[j] * (qa @ qb - a_zp * colsum(qb)) + bias`.
///
/// `a` holds the activation codes widened to i16, `pw.k2` of them per row.
///
/// # Safety
/// Caller guarantees the pointers are valid for the shapes described, `out` is
/// writable for `m * ldc` floats, and the CPU supports AVX2 and FMA.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn qgemm_u8s8_f32_avx2(
    a: *const i16,
    m: usize,
    lda: usize,
    pw: &PackedI8WeightsAvx2,
    a_zero_point: i32,
    a_scale: f32,
    w_scale: *const f32,
    w_scale_len: usize,
    bias: Option<*const f32>,
    out: *mut f32,
    ldc: usize,
) {
    unsafe {
        let kb_count = pw.k2 / 2;
        let panel_stride = kb_count * (N_PANEL * 2);
        let col_sums = pw.col_sums.as_ptr();

        for jb in 0..pw.n_pad / N_PANEL {
            let bp = pw.data.as_ptr().add(jb * panel_stride);
            let col_base = jb * N_PANEL;

            let mut i = 0;
            while i + MR <= m {
                tile::<MR>(
                    a.add(i * lda),
                    lda,
                    bp,
                    kb_count,
                    col_base,
                    pw.n,
                    col_sums,
                    a_zero_point,
                    a_scale,
                    w_scale,
                    w_scale_len,
                    bias,
                    out.add(i * ldc),
                    ldc,
                );
                i += MR;
            }

            macro_rules! tail {
                ($m:literal) => {
                    tile::<$m>(
                        a.add(i * lda),
                        lda,
                        bp,
                        kb_count,
                        col_base,
                        pw.n,
                        col_sums,
                        a_zero_point,
                        a_scale,
                        w_scale,
                        w_scale_len,
                        bias,
                        out.add(i * ldc),
                        ldc,
                    )
                };
            }
            match m - i {
                0 => {}
                1 => tail!(1),
                2 => tail!(2),
                3 => tail!(3),
                4 => tail!(4),
                _ => tail!(5),
            }
        }
    }
}

/// True when this CPU can run [`qgemm_u8s8_f32_avx2`].
#[inline]
pub fn has_avx2_int8() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}
