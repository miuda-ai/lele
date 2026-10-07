//! AVX-512 VNNI quantized GEMM: u8 activations against i8 weights.
//!
//! `vpdpbusd` multiplies 64 u8/i8 pairs and accumulates straight into 16 i32
//! lanes, so unlike the AVX2 `vpmaddwd` path it neither saturates nor needs an
//! i16 intermediate. It issues on two ports, giving 128 MAC/cycle against 32
//! for f32 FMA — the reason an int8 path is worth having at all on this class
//! of machine.
//!
//! Weights are quantized symmetrically (zero point 0) with a per-output-channel
//! scale, which is what ONNX QDQ graphs produced by the standard quantizers
//! emit, so the only correction term is the activation zero point:
//!
//! ```text
//! y[i,j] = a_scale * w_scale[j] * (sum_k qa[i,k]*qb[k,j] - a_zp * sum_k qb[k,j])
//! ```

use std::arch::x86_64::*;

/// Output columns handled by one pass of the microkernel: four `zmm` lanes.
pub const N_PANEL: usize = 64;
/// Rows handled by one pass of the microkernel. `MR * N_PANEL / 16` = 24
/// accumulators, leaving registers for the four B operands and the broadcast.
pub const MR: usize = 6;

/// Weights packed into the layout `vpdpbusd` wants, plus the column sums the
/// zero-point correction needs. Built once per weight tensor and cached.
pub struct PackedI8Weights {
    /// `[n_pad / N_PANEL][k4 / 4][N_PANEL][4]`, zero padded.
    pub data: Vec<i8>,
    /// `sum_k qb[k, j]`, length `n_pad`.
    pub col_sums: Vec<i32>,
    pub k: usize,
    /// `k` rounded up to a multiple of 4.
    pub k4: usize,
    pub n: usize,
    /// `n` rounded up to a multiple of [`N_PANEL`].
    pub n_pad: usize,
}

/// Packs a row-major `[k, n]` i8 weight matrix. Scalar and one-shot: the result
/// is cached for the life of the model, so only the kernel below is hot.
pub fn pack_i8_weights(b: &[i8], k: usize, n: usize) -> PackedI8Weights {
    let k4 = k.next_multiple_of(4);
    let n_pad = n.next_multiple_of(N_PANEL);
    let kb_count = k4 / 4;
    let mut data = vec![0i8; n_pad * k4];

    for jb in 0..n_pad / N_PANEL {
        for kb in 0..kb_count {
            let block = (jb * kb_count + kb) * (N_PANEL * 4);
            for sub in 0..4 {
                for j in 0..16 {
                    let col = jb * N_PANEL + sub * 16 + j;
                    if col >= n {
                        continue;
                    }
                    for t in 0..4 {
                        let kk = kb * 4 + t;
                        if kk < k {
                            data[block + sub * 64 + j * 4 + t] = b[kk * n + col];
                        }
                    }
                }
            }
        }
    }

    let mut col_sums = vec![0i32; n_pad];
    for kk in 0..k {
        let row = &b[kk * n..kk * n + n];
        for (col, &v) in row.iter().enumerate() {
            col_sums[col] += v as i32;
        }
    }

    PackedI8Weights {
        data,
        col_sums,
        k,
        k4,
        n,
        n_pad,
    }
}

/// Accumulates `M` rows against one 64-column panel over the whole K range.
///
/// A is the broadcast memory operand of `vpdpbusd`, so each K step costs four
/// B loads and `M * 4` FMA-class ops and nothing else.
#[target_feature(
    enable = "avx512f",
    enable = "avx512bw",
    enable = "avx512vl",
    enable = "avx512vnni"
)]
#[inline]
unsafe fn accumulate_panel<const M: usize>(
    a: *const u8,
    lda: usize,
    bp: *const i8,
    kb_count: usize,
) -> [[__m512i; 4]; M] {
    unsafe {
        let mut acc = [[_mm512_setzero_si512(); 4]; M];
        for kb in 0..kb_count {
            let block = bp.add(kb * (N_PANEL * 4));
            let b0 = _mm512_loadu_si512(block as *const _);
            let b1 = _mm512_loadu_si512(block.add(64) as *const _);
            let b2 = _mm512_loadu_si512(block.add(128) as *const _);
            let b3 = _mm512_loadu_si512(block.add(192) as *const _);
            for (r, accr) in acc.iter_mut().enumerate() {
                let word = (a.add(r * lda + kb * 4) as *const i32).read_unaligned();
                let av = _mm512_set1_epi32(word);
                accr[0] = _mm512_dpbusd_epi32(accr[0], av, b0);
                accr[1] = _mm512_dpbusd_epi32(accr[1], av, b1);
                accr[2] = _mm512_dpbusd_epi32(accr[2], av, b2);
                accr[3] = _mm512_dpbusd_epi32(accr[3], av, b3);
            }
        }
        acc
    }
}

/// Applies the zero-point correction, the combined scale and the optional bias,
/// then stores. Columns past `n` are masked off rather than trimmed so the
/// panel loop never needs a narrow variant.
#[target_feature(
    enable = "avx512f",
    enable = "avx512bw",
    enable = "avx512vl",
    enable = "avx512vnni"
)]
#[inline]
unsafe fn store_panel<const M: usize>(
    acc: &[[__m512i; 4]; M],
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
        let zp = _mm512_set1_epi32(a_zero_point);
        let sa = _mm512_set1_ps(a_scale);
        for sub in 0..4 {
            let col = col_base + sub * 16;
            if col >= n {
                break;
            }
            let rem = n - col;
            let mask: __mmask16 = if rem >= 16 {
                0xffff
            } else {
                (1u16 << rem) - 1
            };
            let cs = _mm512_loadu_si512(col_sums.add(col) as *const _);
            let corr = _mm512_mullo_epi32(cs, zp);
            let scale = if w_scale_len == 1 {
                _mm512_mul_ps(sa, _mm512_set1_ps(*w_scale))
            } else {
                _mm512_mul_ps(sa, _mm512_maskz_loadu_ps(mask, w_scale.add(col)))
            };
            let bias_v = match bias {
                Some(p) => _mm512_maskz_loadu_ps(mask, p.add(col)),
                None => _mm512_setzero_ps(),
            };
            for (r, accr) in acc.iter().enumerate() {
                let v = _mm512_sub_epi32(accr[sub], corr);
                let f = _mm512_cvtepi32_ps(v);
                let y = _mm512_fmadd_ps(f, scale, bias_v);
                _mm512_mask_storeu_ps(out.add(r * ldc + col), mask, y);
            }
        }
    }
}

/// One `MR`-row-by-64-column tile: accumulate then store.
#[target_feature(
    enable = "avx512f",
    enable = "avx512bw",
    enable = "avx512vl",
    enable = "avx512vnni"
)]
#[inline]
unsafe fn tile<const M: usize>(
    a: *const u8,
    lda: usize,
    bp: *const i8,
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
/// `a` must have at least `pw.k4` readable bytes per row (i.e. rows padded to a
/// multiple of four when K is not); the padding bytes are multiplied by zeroed
/// weight bytes and so may hold anything.
///
/// # Safety
/// Caller guarantees the pointers are valid for the shapes described, `out` is
/// writable for `m * ldc` floats, and the CPU supports AVX-512 VNNI.
#[target_feature(
    enable = "avx512f",
    enable = "avx512bw",
    enable = "avx512vl",
    enable = "avx512vnni"
)]
pub unsafe fn qgemm_u8s8_f32(
    a: *const u8,
    m: usize,
    lda: usize,
    pw: &PackedI8Weights,
    a_zero_point: i32,
    a_scale: f32,
    w_scale: *const f32,
    w_scale_len: usize,
    bias: Option<*const f32>,
    out: *mut f32,
    ldc: usize,
) {
    unsafe {
        let kb_count = pw.k4 / 4;
        let panel_stride = kb_count * (N_PANEL * 4);
        let col_sums = pw.col_sums.as_ptr();

        // Panel-outer so the 64-column B slab stays resident while every row
        // streams past it.
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

/// True when this CPU can run [`qgemm_u8s8_f32`].
#[inline]
pub fn has_vnni() -> bool {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vl")
        && is_x86_feature_detected!("avx512vnni")
}
