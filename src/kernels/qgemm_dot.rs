//! The dot-product layouts of [`crate::kernels::qgemm`]: instructions that multiply four
//! bytes by four bytes and add the products, exactly, into an `i32` lane.
//!
//! - [`Dot::Vnni`]: AVX512-VNNI's `vpdpbusd`, `u8` by `i8`, 64 products per instruction.
//!   `fearless_simd`'s AVX-512 level is the Ice Lake set, which includes VNNI, so it is
//!   chosen whenever that level is detected (at run time, so statically built binaries get
//!   it too).
//! - [`Dot::Sdot`]: Arm's dot product extension (`sdot`), `i8` by `i8`, 16 products per
//!   instruction. Chosen on aarch64 builds that enable `dotprod` (Apple silicon, Neoverse,
//!   `-C target-cpu=...` for most cores since the Cortex-A55/A75).
//!
//! On other levels the instructions are emulated; that only runs in tests.
//!
//! A is quantized to its `u8` codes `q`, four depths to an `i32`. B is shifted into `i8`:
//! `b' = b - s`, with `s` 0 for `i8` codes and 128 for `u8` ones. The product the model
//! wants comes back exactly in the epilogue:
//!
//! ```text
//! sum (q - za)(b - zb) = sum q b' + (s - zb) * sum q - za * sum (b - zb)
//! ```
//!
//! with the row sums of `q` taken while quantizing A and the column terms when packing B.
//! `sdot` takes A signed too, so its bytes go in as `q - 128` (the sign bit flipped), and
//! `128 * sum b'` per column, known when packing, joins the column term. So any zero
//! points work, `u8` weights' included. Products are at most 255 * 128 and summed in `i32`,
//! which holds for depths up to 60 000.
//!
//! Tiles are 6 rows of C by a panel of B: 64 columns for VNNI (24 512-bit accumulators),
//! 16 for `sdot` (24 128-bit ones, of Neon's 32 registers). A whole panel's depth is one
//! block, read from L2.

use crate::kernels::qgemm::{AlignedI32, Epilogue};
use fearless_simd::{Level, Simd, f32x16, i32x16};
use fearless_simd_macros::simd;

/// Rows of C per register tile.
const MR: usize = 6;
/// The widest panel of B (VNNI's): four vectors of 16 `i32` sums.
pub(crate) const NR: usize = 64;

/// Which dot-product instruction B is packed for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Dot {
    Vnni,
    Sdot,
}

impl Dot {
    /// Columns per panel of B.
    pub(crate) fn nr(self) -> usize {
        match self {
            Dot::Vnni => 64,
            Dot::Sdot => 16,
        }
    }
}

/// B packed for a dot-product instruction.
pub(crate) struct QuadWeights {
    pub dot: Dot,
    /// Quads of depths: `k` rounded up to a multiple of 4, quartered.
    pub quads: usize,
    /// Panels of `dot.nr()` columns, each `quads` rows of `dot.nr()` quads of `i8`
    /// (`b - s`, the shallowest depth in the low byte); zero past `k` and `n`.
    pub data: AlignedI32,
    /// Per column, zero past `n` to a whole [`NR`]: `s - zb`, `sum_k (b - zb)`, and what the
    /// instruction adds to every sum (`-128 * sum_k (b - s)` for `sdot`, else 0).
    pub row_sum_factor: Vec<i32>,
    pub col_sums: Vec<i32>,
    pub dot_bias: Vec<i32>,
    /// Some `s - zb` is not zero. (Usually all are: `i8` weights have zero point 0, and
    /// `u8` ones mostly 128.)
    pub row_term: bool,
}

/// `code(i)` is element `i` of the row-major `k x n` codes (`i8` or `u8` values); `shift`
/// is `s` above, which must bring every code into `i8`.
pub(crate) fn pack(
    dot: Dot,
    k: usize,
    n: usize,
    code: impl Fn(usize) -> i32,
    shift: i32,
    zero_point: impl Fn(usize) -> i32,
) -> QuadWeights {
    let nr = dot.nr();
    let quads = k.div_ceil(4);
    let panels = n.div_ceil(nr);
    let mut data = AlignedI32::zeroed(panels * quads * nr);
    let padded = n.div_ceil(NR) * NR;
    let mut row_sum_factor = vec![0; padded];
    let mut col_sums = vec![0; padded];
    let mut dot_bias = vec![0; padded];
    for kk in 0..k {
        for j in 0..n {
            let c = code(kk * n + j);
            let b = c - shift;
            assert!((-128..=127).contains(&b), "code {c} does not shift into i8 by {shift}");
            let slot = &mut data[((j / nr) * quads + kk / 4) * nr + j % nr];
            *slot |= ((b as i8 as u8) as i32) << (8 * (kk % 4));
            col_sums[j] += c - zero_point(j);
            if dot == Dot::Sdot {
                dot_bias[j] -= 128 * b;
            }
        }
    }
    for (j, f) in row_sum_factor[..n].iter_mut().enumerate() {
        *f = shift - zero_point(j);
    }
    let row_term = row_sum_factor.iter().any(|&f| f != 0);
    QuadWeights { dot, quads, data, row_sum_factor, col_sums, dot_bias, row_term }
}

/// Rows of A for `dot`: `m` rows of `quads` quads of `u8` codes
/// (`clamp(round_ties_even(x / scale) + zero_point, 0, 255)`, with the sign bit flipped
/// for `sdot`), and the sum of each row's codes.
pub(crate) fn quantize_rows(
    level: Level,
    dot: Dot,
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
    let flip = if dot == Dot::Sdot { 0x8080_8080u32 as i32 } else { 0 };
    crate::kernels::simd::simd_call!(level, quantize_rows_simd(src, m, k, scale, zero_point, flip, dst, row_sums))
}

#[simd]
fn quantize_rows_simd<S: Simd>(
    simd: S,
    src: &[f32],
    m: usize,
    k: usize,
    scale: f32,
    zero_point: i32,
    flip: i32,
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
    let flipv = i32x16::splat(simd, flip);
    // Codes to bytes, in order, by two truncating narrows (they fit).
    let bytes = |q: [i32x16<S>; 4]| -> i32x16<S> {
        let b: i32x16<S> = simd.narrow_i16x32(simd.narrow_i32x16(q[0], q[1]), simd.narrow_i32x16(q[2], q[3])).bitcast();
        b ^ flipv
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
            *o ^= flip;
        }
        row_sums[r] = total;
    }
}

/// `C = epilogue(A * B)`, A from [`quantize_rows`] (`lda` quads apart) with zero point
/// `za`, B packed by [`pack`] with column scales `col_scale` (`n` padded to a whole [`NR`]).
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
    let (kq, nr) = (w.quads, w.dot.nr());
    for jp in 0..n.div_ceil(nr) {
        let j = jp * nr;
        let width = nr.min(n - j);
        // The panel's column terms, once rather than in every tile: with shallow layers
        // (depth 312) building them per tile cost several percent.
        // (Only the panel's `width` entries are set: with 16-column panels, building all of
        // `NR` doubled the time of a 1-row product.)
        let (mut scale, mut col_term, mut bias, mut factor) = ([0.0f32; NR], [0i32; NR], [0.0f32; NR], [0i32; NR]);
        for v in 0..width {
            scale[v] = e.scale * col_scale[j + v];
            col_term[v] = za.wrapping_mul(w.col_sums[j + v]).wrapping_add(w.dot_bias[j + v]);
            factor[v] = w.row_sum_factor[j + v];
        }
        if let Some(b) = e.bias {
            bias[..width].copy_from_slice(&b[j..][..width]);
        }
        let t = Tile {
            kq,
            lda,
            width,
            ldc: n,
            scale: &scale,
            col_term: &col_term,
            row_sum_factor: w.row_term.then_some(&factor),
            bias: &bias,
            relu: e.relu,
        };
        let b = &w.data[jp * kq * nr..][..kq * nr];
        let mut ir = 0;
        while ir < m {
            let (a, sums, c) = (&a[ir * lda..], &row_sums[ir..], &mut c[ir * n + j..]);
            match (w.dot, MR.min(m - ir)) {
                (Dot::Vnni, 6) => vnni6(simd, &t, a, sums, b, c),
                (Dot::Vnni, 5) => vnni5(simd, &t, a, sums, b, c),
                (Dot::Vnni, 4) => vnni4(simd, &t, a, sums, b, c),
                (Dot::Vnni, 3) => vnni3(simd, &t, a, sums, b, c),
                (Dot::Vnni, 2) => vnni2(simd, &t, a, sums, b, c),
                (Dot::Vnni, _) => vnni1(simd, &t, a, sums, b, c),
                (Dot::Sdot, 6) => sdot6(simd, &t, a, sums, b, c),
                (Dot::Sdot, 5) => sdot5(simd, &t, a, sums, b, c),
                (Dot::Sdot, 4) => sdot4(simd, &t, a, sums, b, c),
                (Dot::Sdot, 3) => sdot3(simd, &t, a, sums, b, c),
                (Dot::Sdot, 2) => sdot2(simd, &t, a, sums, b, c),
                (Dot::Sdot, _) => sdot1(simd, &t, a, sums, b, c),
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
    /// Per column: the scale of C, what to take from the sums (`za * sum_k (b - zb)`, less
    /// the instruction's own offset), `s - zb` (unless all are zero), and the bias (zero
    /// if none).
    scale: &'a [f32; NR],
    col_term: &'a [i32; NR],
    row_sum_factor: Option<&'a [i32; NR]>,
    bias: &'a [f32; NR],
    relu: bool,
}

/// `name(simd, t, a, row_sums, b, c)`: `R` rows of C from the rows of A in place and a
/// panel of B, `t.kq` rows of `16 * V` quads, multiplied with `dot`.
macro_rules! tile_fn {
    ($name:ident, $r:literal, $v:literal, $dot:ident) => {
        #[simd]
        #[inline(never)]
        fn $name<S: Simd>(simd: S, t: &Tile, a: &[i32], row_sums: &[i32], b: &[i32], c: &mut [f32]) {
            use fearless_simd::prelude::*;
            const R: usize = $r;
            const V: usize = $v;
            let a_rows: [&[i32]; R] = core::array::from_fn(|r| &a[r * t.lda..][..t.kq]);
            let mut acc = [[i32x16::splat(simd, 0); V]; R];
            for (q, row) in b.as_chunks::<{ 16 * V }>().0[..t.kq].iter().enumerate() {
                let bv: [i32x16<S>; V] = core::array::from_fn(|v| i32x16::from_slice(simd, &row[16 * v..][..16]));
                for r in 0..R {
                    let av = i32x16::splat(simd, a_rows[r][q]);
                    for v in 0..V {
                        acc[r][v] = $dot(simd, acc[r][v], av, bv[v]);
                    }
                }
            }
            finish::<S, R, V>(simd, t, &row_sums[..R], acc, c);
        }
    };
}

tile_fn!(vnni6, 6, 4, dpbusd);
tile_fn!(vnni5, 5, 4, dpbusd);
tile_fn!(vnni4, 4, 4, dpbusd);
tile_fn!(vnni3, 3, 4, dpbusd);
tile_fn!(vnni2, 2, 4, dpbusd);
tile_fn!(vnni1, 1, 4, dpbusd);
tile_fn!(sdot6, 6, 1, sdot);
tile_fn!(sdot5, 5, 1, sdot);
tile_fn!(sdot4, 4, 1, sdot);
tile_fn!(sdot3, 3, 1, sdot);
tile_fn!(sdot2, 2, 1, sdot);
tile_fn!(sdot1, 1, 1, sdot);

/// Corrects `R` rows of sums for the zero points, applies the epilogue and writes them to
/// C, the first `t.width` of its `16 * V` columns.
#[inline(always)]
fn finish<S: Simd, const R: usize, const V: usize>(
    simd: S,
    t: &Tile,
    row_sums: &[i32],
    acc: [[i32x16<S>; V]; R],
    c: &mut [f32],
) {
    use fearless_simd::prelude::*;
    let vecs = |s: &[f32]| -> [f32x16<S>; V] { core::array::from_fn(|v| f32x16::from_slice(simd, &s[16 * v..][..16])) };
    let ivecs = |s: &[i32]| -> [i32x16<S>; V] { core::array::from_fn(|v| i32x16::from_slice(simd, &s[16 * v..][..16])) };
    let (scale, col_term, bias) = (vecs(t.scale), ivecs(t.col_term), vecs(t.bias));
    let factor = t.row_sum_factor.map(|f| ivecs(f));
    let zero = f32x16::splat(simd, 0.0);
    for r in 0..R {
        let rs = i32x16::splat(simd, row_sums[r]);
        let vals: [f32x16<S>; V] = core::array::from_fn(|v| {
            let mut sum = acc[r][v] - col_term[v];
            if let Some(f) = &factor {
                sum = sum + simd.mul_i32x16(rs, f[v]);
            }
            let x = simd.cvt_f32_i32x16(sum) * scale[v] + bias[v];
            if t.relu { x.max(zero) } else { x }
        });
        let row = &mut c[r * t.ldc..][..t.width];
        if t.width == 16 * V {
            for v in 0..V {
                vals[v].store_slice(&mut row[16 * v..][..16]);
            }
        } else {
            let mut flat = [0.0f32; NR];
            for v in 0..V {
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
        _ => emulate(simd, acc, a, b, |x| x as i32),
    }
}

/// As [`dpbusd`] with both bytes signed: Arm's `sdot`, emulated where `dotprod` is not
/// enabled (which only tests run).
#[inline(always)]
fn sdot<S: Simd>(simd: S, acc: i32x16<S>, a: i32x16<S>, b: i32x16<S>) -> i32x16<S> {
    match simd.level() {
        #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
        Level::Neon(_) => {
            use core::arch::aarch64::{int32x4x4_t, vdotq_s32, vreinterpretq_s8_s32};
            use fearless_simd::SimdInto;
            let (acc, a, b): (int32x4x4_t, int32x4x4_t, int32x4x4_t) = (acc.into(), a.into(), b.into());
            // SAFETY: this arm is compiled only when the target has `dotprod`.
            let d = |acc, a, b| unsafe { vdotq_s32(acc, vreinterpretq_s8_s32(a), vreinterpretq_s8_s32(b)) };
            int32x4x4_t(d(acc.0, a.0, b.0), d(acc.1, a.1, b.1), d(acc.2, a.2, b.2), d(acc.3, a.3, b.3)).simd_into(simd)
        }
        #[allow(unreachable_patterns)]
        _ => emulate(simd, acc, a, b, |x| x as i8 as i32),
    }
}

/// Four-byte dot products lane by lane, `a`'s bytes read with `a_byte`, `b`'s as `i8`.
#[inline(always)]
fn emulate<S: Simd>(simd: S, acc: i32x16<S>, a: i32x16<S>, b: i32x16<S>, a_byte: impl Fn(u8) -> i32) -> i32x16<S> {
    use fearless_simd::prelude::*;
    let (acc, a, b) = (acc.as_slice(), a.as_slice(), b.as_slice());
    let out: [i32; 16] = core::array::from_fn(|i| {
        let (x, y) = (a[i].to_le_bytes(), b[i].to_le_bytes());
        acc[i].wrapping_add((0..4).map(|t| a_byte(x[t]) * y[t] as i8 as i32).sum())
    });
    i32x16::from_slice(simd, &out)
}
