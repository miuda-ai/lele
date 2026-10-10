//! The VNNI layout of [`crate::kernels::qgemm`]: `vpdpbusd` multiplies four `u8` by four
//! `i8` and adds them, exactly, into an `i32` lane, 64 products per 512-bit instruction:
//! four times the products of the 16-bit pairs and with no separate add. `fearless_simd`'s
//! AVX-512 level is the Ice Lake set, which includes VNNI, so weights are packed this way
//! whenever that level is detected (at run time, so statically built binaries get it too).
//! On other levels the instruction is emulated; that only runs in tests.
//!
//! A stays as its `u8` codes `q`, four depths to an `i32`. B is shifted into `i8`:
//! `b' = b - s`, with `s` 0 for `i8` codes and 128 for `u8` ones. The product the model
//! wants comes back exactly in the epilogue:
//!
//! ```text
//! sum (q - za)(b - zb) = sum q b' + (s - zb) * sum q - za * sum (b - zb)
//! ```
//!
//! with the row sums of `q` taken while quantizing A and the column terms when packing B.
//! So any zero points work, `u8` weights' included. Products are at most 255 * 128 and
//! summed in `i32`, which holds for depths up to 60 000.
//!
//! Tiles are 6 rows by 64 columns (24 accumulators, 4 vectors of B, the broadcast); a whole
//! panel's depth is one block, read from L2 (the previous AVX-512 kernel measured 420-610
//! GOP/s on an Ice Lake Xeon this way).

use crate::kernels::qgemm::{AlignedI32, Epilogue};
use fearless_simd::{Level, Simd, f32x16, i32x16};
use fearless_simd_macros::simd;

/// Rows of C per register tile.
const MR: usize = 6;
/// Columns of C per register tile and per panel of B: four vectors of 16 `i32` sums.
pub(crate) const NR: usize = 64;

/// B packed for `vpdpbusd`.
pub(crate) struct QuadWeights {
    /// Quads of depths: `k` rounded up to a multiple of 4, quartered.
    pub quads: usize,
    /// Panels of `NR` columns, each `quads` rows of `NR` quads of `i8` (`b - s`, the
    /// shallowest depth in the low byte); zero past `k` and `n`.
    pub data: AlignedI32,
    /// `s - zb` and `sum_k (b - zb)` of each column, zero past `n` to a whole panel.
    pub row_sum_factor: Vec<i32>,
    pub col_sums: Vec<i32>,
    /// Some `s - zb` is not zero. (Usually all are: `i8` weights have zero point 0, and
    /// `u8` ones mostly 128.)
    pub row_term: bool,
}

/// `code(i)` is element `i` of the row-major `k x n` codes (`i8` or `u8` values); `shift`
/// is `s` above, which must bring every code into `i8`.
pub(crate) fn pack(k: usize, n: usize, code: impl Fn(usize) -> i32, shift: i32, zero_point: impl Fn(usize) -> i32) -> QuadWeights {
    let quads = k.div_ceil(4);
    let panels = n.div_ceil(NR);
    let mut data = AlignedI32::zeroed(panels * quads * NR);
    let mut row_sum_factor = vec![0; panels * NR];
    let mut col_sums = vec![0; panels * NR];
    for kk in 0..k {
        for j in 0..n {
            let c = code(kk * n + j);
            let b = c - shift;
            assert!((-128..=127).contains(&b), "code {c} does not shift into i8 by {shift}");
            let slot = &mut data[((j / NR) * quads + kk / 4) * NR + j % NR];
            *slot |= ((b as i8 as u8) as i32) << (8 * (kk % 4));
            col_sums[j] += c - zero_point(j);
        }
    }
    for (j, f) in row_sum_factor[..n].iter_mut().enumerate() {
        *f = shift - zero_point(j);
    }
    let row_term = row_sum_factor.iter().any(|&f| f != 0);
    QuadWeights { quads, data, row_sum_factor, col_sums, row_term }
}

/// Rows of A for this layout: `m` rows of `quads` quads of `u8` codes
/// (`clamp(round_ties_even(x / scale) + zero_point, 0, 255)`), and the sum of each row.
pub(crate) fn quantize_rows(
    level: Level,
    src: &[f32],
    m: usize,
    k: usize,
    scale: f32,
    zero_point: i32,
    dst: &mut Vec<i32>,
    row_sums: &mut Vec<i32>,
) {
    assert!(src.len() >= m * k);
    // Every quad and sum is written, so the buffers are not filled first.
    crate::kernels::utils::ensure_capacity(dst, m * k.div_ceil(4));
    crate::kernels::utils::ensure_capacity(row_sums, m);
    crate::kernels::simd::simd_call!(level, quantize_rows_simd(src, m, k, scale, zero_point, dst, row_sums))
}

#[simd]
fn quantize_rows_simd<S: Simd>(
    simd: S,
    src: &[f32],
    m: usize,
    k: usize,
    scale: f32,
    zero_point: i32,
    dst: &mut [i32],
    row_sums: &mut [i32],
) {
    use fearless_simd::prelude::*;
    let quads = k.div_ceil(4);
    let code = |x: f32| {
        let q = (x / scale).round_ties_even().clamp(i32::MIN as f32, i32::MAX as f32) as i32;
        q.saturating_add(zero_point).clamp(0, 255)
    };
    let sv = f32x16::splat(simd, scale);
    let zv = i32x16::splat(simd, zero_point);
    let (lo, hi) = (i32x16::splat(simd, 0), i32x16::splat(simd, 255));
    let vcode = |x: &[f32]| {
        let q = simd.cvt_i32_f32x16((f32x16::from_slice(simd, x) / sv).round_ties_even()) + zv;
        simd.min_i32x16(simd.max_i32x16(q, lo), hi)
    };
    // Codes to bytes, in order, by two truncating narrows (they fit).
    let bytes = |q: [i32x16<S>; 4]| -> i32x16<S> {
        simd.narrow_i16x32(simd.narrow_i32x16(q[0], q[1]), simd.narrow_i32x16(q[2], q[3])).bitcast()
    };
    let zero = i32x16::splat(simd, 0);
    for r in 0..m {
        let row = &src[r * k..][..k];
        let out = &mut dst[r * quads..][..quads];
        let mut sum = zero;
        let (sixty_fours, rest) = row.as_chunks::<64>();
        let (out64, out_rest) = out.as_chunks_mut::<16>();
        for (x, o) in sixty_fours.iter().zip(out64) {
            let q: [i32x16<S>; 4] = core::array::from_fn(|v| vcode(&x[16 * v..][..16]));
            sum = sum + q[0] + q[1] + q[2] + q[3];
            bytes(q).store_slice(o);
        }
        let (sixteens, rest) = rest.as_chunks::<16>();
        for (x, o) in sixteens.iter().zip(out_rest.as_chunks_mut::<4>().0) {
            let q = vcode(x);
            sum = sum + q;
            o.copy_from_slice(&bytes([q, zero, zero, zero]).as_slice()[..4]);
        }
        let mut total: i32 = sum.as_slice().iter().sum();
        let done = sixteens.len() * 4;
        for (i, o) in out_rest[done..].iter_mut().enumerate() {
            *o = 0;
            for t in 0..4 {
                if let Some(&x) = rest.get(4 * i + t) {
                    let q = code(x);
                    total += q;
                    *o |= q << (8 * t);
                }
            }
        }
        row_sums[r] = total;
    }
}

/// `C = epilogue(A * B)`, A from [`quantize_rows`] (`lda` quads apart) with zero point
/// `za`, B packed by [`pack`] with column scales `col_scale` (`n` padded to a whole panel).
pub(crate) fn qgemm(
    level: Level,
    a: &[i32],
    row_sums: &[i32],
    m: usize,
    lda: usize,
    za: i32,
    w: &QuadWeights,
    n: usize,
    col_scale: &[f32],
    e: &Epilogue,
    c: &mut [f32],
) {
    assert!(lda >= w.quads && a.len() >= m.saturating_sub(1) * lda + w.quads && row_sums.len() >= m);
    assert!(c.len() >= m * n && col_scale.len() >= n.div_ceil(NR) * NR);
    assert!(e.bias.is_none_or(|b| b.len() >= n));
    crate::kernels::simd::simd_call!(level, qgemm_simd(a, row_sums, m, lda, za, w, n, col_scale, e, c))
}

#[simd]
fn qgemm_simd<S: Simd>(
    simd: S,
    a: &[i32],
    row_sums: &[i32],
    m: usize,
    lda: usize,
    za: i32,
    w: &QuadWeights,
    n: usize,
    col_scale: &[f32],
    e: &Epilogue,
    c: &mut [f32],
) {
    let kq = w.quads;
    for jp in 0..n.div_ceil(NR) {
        let j = jp * NR;
        let width = NR.min(n - j);
        // The panel's column terms, once rather than in every tile: with shallow layers
        // (depth 312) building them per tile cost several percent.
        let scale: [f32; NR] = core::array::from_fn(|v| e.scale * col_scale[j + v]);
        let col_term: [i32; NR] = core::array::from_fn(|v| za.wrapping_mul(w.col_sums[j + v]));
        let bias: [f32; NR] = core::array::from_fn(|v| e.bias.map_or(0.0, |b| if v < width { b[j + v] } else { 0.0 }));
        let t = Tile {
            kq,
            lda,
            width,
            ldc: n,
            scale: &scale,
            col_term: &col_term,
            row_sum_factor: w.row_term.then(|| w.row_sum_factor[j..][..NR].try_into().unwrap()),
            bias: &bias,
            relu: e.relu,
        };
        let b = &w.data[jp * kq * NR..][..kq * NR];
        let mut ir = 0;
        while ir < m {
            let (a, sums, c) = (&a[ir * lda..], &row_sums[ir..], &mut c[ir * n + j..]);
            match MR.min(m - ir) {
                6 => tile6(simd, &t, a, sums, b, c),
                5 => tile5(simd, &t, a, sums, b, c),
                4 => tile4(simd, &t, a, sums, b, c),
                3 => tile3(simd, &t, a, sums, b, c),
                2 => tile2(simd, &t, a, sums, b, c),
                _ => tile1(simd, &t, a, sums, b, c),
            }
            ir += MR;
        }
    }
}

/// What a tile does, besides its operands.
struct Tile<'a> {
    /// Quads of depths.
    kq: usize,
    /// Quads between the rows of A.
    lda: usize,
    /// Columns of C written (the panel is zero-padded past them).
    width: usize,
    ldc: usize,
    /// Per column: the scale of C, `za * sum_k (b - zb)`, `s - zb` (unless all are zero),
    /// and the bias (zero if none).
    scale: &'a [f32; NR],
    col_term: &'a [i32; NR],
    row_sum_factor: Option<&'a [i32; NR]>,
    bias: &'a [f32; NR],
    relu: bool,
}

/// `tileR(simd, t, a, row_sums, b, c)`: `R` rows of C from the rows of A in place and a
/// panel of B, `t.kq` rows of `NR` quads.
macro_rules! tile_fn {
    ($name:ident, $r:literal) => {
        #[simd]
        #[inline(never)]
        fn $name<S: Simd>(simd: S, t: &Tile, a: &[i32], row_sums: &[i32], b: &[i32], c: &mut [f32]) {
            use fearless_simd::prelude::*;
            const R: usize = $r;
            let a_rows: [&[i32]; R] = core::array::from_fn(|r| &a[r * t.lda..][..t.kq]);
            let mut acc = [[i32x16::splat(simd, 0); 4]; R];
            for (q, row) in b.as_chunks::<NR>().0[..t.kq].iter().enumerate() {
                let bv: [i32x16<S>; 4] = core::array::from_fn(|v| i32x16::from_slice(simd, &row[16 * v..][..16]));
                for r in 0..R {
                    let av = i32x16::splat(simd, a_rows[r][q]);
                    for v in 0..4 {
                        acc[r][v] = dpbusd(simd, acc[r][v], av, bv[v]);
                    }
                }
            }
            finish(simd, t, &row_sums[..R], acc, c);
        }
    };
}

tile_fn!(tile6, 6);
tile_fn!(tile5, 5);
tile_fn!(tile4, 4);
tile_fn!(tile3, 3);
tile_fn!(tile2, 2);
tile_fn!(tile1, 1);

/// Corrects `R` rows of sums for the zero points, applies the epilogue and writes them to
/// C, the first `t.width` columns.
#[inline(always)]
fn finish<S: Simd, const R: usize>(simd: S, t: &Tile, row_sums: &[i32], acc: [[i32x16<S>; 4]; R], c: &mut [f32]) {
    use fearless_simd::prelude::*;
    let vecs = |s: &[f32]| -> [f32x16<S>; 4] { core::array::from_fn(|v| f32x16::from_slice(simd, &s[16 * v..][..16])) };
    let ivecs = |s: &[i32]| -> [i32x16<S>; 4] { core::array::from_fn(|v| i32x16::from_slice(simd, &s[16 * v..][..16])) };
    let (scale, col_term, bias) = (vecs(t.scale), ivecs(t.col_term), vecs(t.bias));
    let factor = t.row_sum_factor.map(|f| ivecs(f));
    let zero = f32x16::splat(simd, 0.0);
    for r in 0..R {
        let rs = i32x16::splat(simd, row_sums[r]);
        let vals: [f32x16<S>; 4] = core::array::from_fn(|v| {
            let mut sum = acc[r][v] - col_term[v];
            if let Some(f) = &factor {
                sum = sum + simd.mul_i32x16(rs, f[v]);
            }
            let x = simd.cvt_f32_i32x16(sum) * scale[v] + bias[v];
            if t.relu { x.max(zero) } else { x }
        });
        let row = &mut c[r * t.ldc..][..t.width];
        if t.width == NR {
            for v in 0..4 {
                vals[v].store_slice(&mut row[16 * v..][..16]);
            }
        } else {
            let mut flat = [0.0f32; NR];
            for v in 0..4 {
                vals[v].store_slice(&mut flat[16 * v..][..16]);
            }
            row.copy_from_slice(&flat[..t.width]);
        }
    }
}

/// `acc[i] + sum_t a.byte[4i + t] * b.byte[4i + t]`, `a` unsigned and `b` signed: VNNI's
/// `vpdpbusd`, emulated off the AVX-512 level (which only tests run).
#[inline(always)]
fn dpbusd<S: Simd>(simd: S, acc: i32x16<S>, a: i32x16<S>, b: i32x16<S>) -> i32x16<S> {
    match simd.level() {
        #[cfg(target_arch = "x86_64")]
        Level::Avx512(_) => {
            use core::arch::x86_64::{__m512i, _mm512_dpbusd_epi32};
            use fearless_simd::SimdInto;
            let (acc, a, b): (__m512i, __m512i, __m512i) = (acc.into(), a.into(), b.into());
            // SAFETY: fearless_simd's AVX-512 level requires AVX512-VNNI.
            unsafe { _mm512_dpbusd_epi32(acc, a, b) }.simd_into(simd)
        }
        #[allow(unreachable_patterns)]
        _ => {
            use fearless_simd::prelude::*;
            let (acc, a, b) = (acc.as_slice(), a.as_slice(), b.as_slice());
            let out: [i32; 16] = core::array::from_fn(|i| {
                let (x, y) = (a[i].to_le_bytes(), b[i].to_le_bytes());
                acc[i].wrapping_add((0..4).map(|t| x[t] as i32 * y[t] as i8 as i32).sum())
            });
            i32x16::from_slice(simd, &out)
        }
    }
}
