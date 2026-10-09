//! `C (+)= alpha * A * B` for row-major `f32` matrices, on `fearless_simd`.
//!
//! Blocked as in BLIS. Panels of `B` (`KC` deep, `NR` wide) and of `A` (`KC` deep,
//! `MR` tall) are copied into contiguous buffers, and a `MR x NR` tile of C is kept in
//! registers across the whole depth of a pair of panels, so the inner loop streams two
//! buffers and does nothing but loads, broadcasts and multiply-adds.

use crate::kernels::simd::simd_call;
use fearless_simd::{Level, Simd, f32x8};
use fearless_simd_macros::simd;

/// Rows of C per register tile.
const MR: usize = 6;
/// Vectors of 8 columns of C per register tile.
const NV: usize = 2;
const NR: usize = 8 * NV;
/// Depth of a panel.
const KC: usize = 256;
/// Rows of A per block.
const MC: usize = 96;
/// Columns of B per block.
const NC: usize = 256;

/// Rows of C per tile when the rows of A are read in place: each needs its own address.
const MR_ROWS: usize = 4;

/// A is copied into panels only if B has more panels than this.
const PACK_A_MIN_PANELS: usize = 2;

/// A row-major problem: `A` is `m x k` with `lda` between rows, `B` is `k x n`, `C` is `m x n`.
#[derive(Clone, Copy)]
pub(crate) struct Dims {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lda: usize,
    pub ldb: usize,
    pub ldc: usize,
    pub alpha: f32,
    pub add: bool,
}

pub(crate) fn gemm_nn(level: Level, d: &Dims, a: &[f32], b: &[f32], c: &mut [f32]) {
    PACKED.with_borrow_mut(|(apack, bpack)| {
        // Every entry used is written first, so the buffers only have to be long enough.
        let kc = KC.min(d.k);
        let (a_len, b_len) = (MC.min(d.m).next_multiple_of(MR) * kc, NC.min(d.n).next_multiple_of(NR) * kc);
        if apack.len() < a_len {
            apack.resize(a_len, 0.0);
        }
        if bpack.len() < b_len {
            bpack.resize(b_len, 0.0);
        }
        simd_call!(level, gemm_nn_simd(d, a, b, c, apack, bpack))
    })
}

type Buffers = (Vec<f32>, Vec<f32>);

thread_local! {
    static PACKED: std::cell::RefCell<Buffers> = const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

#[simd]
fn gemm_nn_simd<S: Simd>(
    simd: S,
    d: &Dims,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    apack: &mut [f32],
    bpack: &mut [f32],
) {
    let pack_rows = d.n.div_ceil(NR) > PACK_A_MIN_PANELS;
    let mut jc = 0;
    while jc < d.n {
        let nc = NC.min(d.n - jc);
        let mut pc = 0;
        while pc < d.k {
            let kc = KC.min(d.k - pc);
            // The first depth overwrites C unless asked to add; later ones add.
            let accumulate = d.add || pc > 0;
            pack_b(b, d.ldb, pc, kc, jc, nc, bpack);
            let mut ic = 0;
            while ic < d.m {
                let mc = MC.min(d.m - ic);
                // A is copied when enough panels of B use it again to pay for that.
                if pack_rows {
                    pack_a(a, d.lda, ic, mc, pc, kc, apack);
                }
                for (jp, bp) in bpack[..nc.div_ceil(NR) * kc * NR].chunks_exact(kc * NR).enumerate() {
                    let j = jc + jp * NR;
                    let width = NR.min(d.n - j);
                    let mut ir = 0;
                    let mut ap = &apack[..];
                    while ir < mc {
                        let rows = if pack_rows { MR } else { MR_ROWS }.min(mc - ir);
                        let c = &mut c[(ic + ir) * d.ldc + j..];
                        if !pack_rows {
                            let a = &a[(ic + ir) * d.lda + pc..];
                            macro_rules! tile {
                                ($r:literal) => {
                                    tile_rows::<S, $r>(simd, d, a, bp, kc, c, width, accumulate)
                                };
                            }
                            match rows {
                                6 => tile!(6),
                                5 => tile!(5),
                                4 => tile!(4),
                                3 => tile!(3),
                                2 => tile!(2),
                                _ => tile!(1),
                            }
                            ir += MR_ROWS;
                            continue;
                        }
                        let (panel, rest) = ap.split_at(kc * rows);
                        ap = rest;
                        macro_rules! tile {
                            ($r:literal) => {
                                tile::<S, $r>(simd, d, panel, bp, c, width, accumulate)
                            };
                        }
                        match rows {
                            6 => tile!(6),
                            5 => tile!(5),
                            4 => tile!(4),
                            3 => tile!(3),
                            2 => tile!(2),
                            _ => tile!(1),
                        }
                        ir += MR;
                    }
                }
                ic += MC;
            }
            pc += KC;
        }
        jc += NC;
    }
}

/// Rows of C handled by `gemm_small_m`.
pub(crate) const SMALL_M: usize = 4;

/// `gemm_nn` for at most `SMALL_M` rows of C. B is used once per row of A at most, so
/// nothing is copied: tiles of C walk down the columns of B in place.
pub(crate) fn gemm_small_m(level: Level, d: &Dims, a: &[f32], b: &[f32], c: &mut [f32]) {
    simd_call!(level, gemm_small_m_simd(d, a, b, c))
}

#[simd]
fn gemm_small_m_simd<S: Simd>(simd: S, d: &Dims, a: &[f32], b: &[f32], c: &mut [f32]) {
    macro_rules! run {
        ($r:literal, $wide:literal) => {{
            // Wide tiles while they fit, then one vector at a time, then single columns.
            let mut j = 0;
            while j + 8 * $wide <= d.n {
                row_tile::<S, $r, $wide>(simd, d, a, b, c, j);
                j += 8 * $wide;
            }
            while j + 8 <= d.n {
                row_tile::<S, $r, 1>(simd, d, a, b, c, j);
                j += 8;
            }
            for j in j..d.n {
                for r in 0..$r {
                    let sum: f32 = (0..d.k).map(|p| a[r * d.lda + p] * b[p * d.ldb + j]).sum();
                    let out = &mut c[r * d.ldc + j];
                    *out = d.alpha * sum + if d.add { *out } else { 0.0 };
                }
            }
        }};
    }
    match d.m {
        1 => run!(1, 8),
        2 => run!(2, 6),
        3 => run!(3, 4),
        _ => run!(4, 3),
    }
}

/// `R` rows of C, columns `j..j + 8 * V`, over the whole depth.
#[inline(always)]
fn row_tile<S: Simd, const R: usize, const V: usize>(
    simd: S,
    d: &Dims,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    j: usize,
) {
    use fearless_simd::prelude::*;
    let a_rows: [&[f32]; R] = core::array::from_fn(|r| &a[r * d.lda..][..d.k]);
    let zero = f32x8::splat(simd, 0.0);
    let mut acc = [[zero; V]; R];
    for p in 0..d.k {
        let brow = &b[p * d.ldb + j..][..8 * V];
        let bv: [f32x8<S>; V] = core::array::from_fn(|v| f32x8::from_slice(simd, &brow[v * 8..][..8]));
        for r in 0..R {
            let av = f32x8::splat(simd, a_rows[r][p]);
            for v in 0..V {
                acc[r][v] = av.mul_add(bv[v], acc[r][v]);
            }
        }
    }
    let alpha = f32x8::splat(simd, d.alpha);
    for r in 0..R {
        let row = &mut c[r * d.ldc + j..][..8 * V];
        for v in 0..V {
            let dst = &mut row[v * 8..][..8];
            let val = acc[r][v] * alpha;
            if d.add {
                (val + f32x8::from_slice(simd, dst)).store_slice(dst);
            } else {
                val.store_slice(dst);
            }
        }
    }
}

/// Copies `B[pc.., jc..]` (`kc` rows, `nc` columns) into panels of `NR` columns, each
/// `kc` rows of `NR`, zero beyond `nc`.
fn pack_b(b: &[f32], ldb: usize, pc: usize, kc: usize, jc: usize, nc: usize, out: &mut [f32]) {
    for (jp, panel) in out.chunks_exact_mut(kc * NR).take(nc.div_ceil(NR)).enumerate() {
        let j = jc + jp * NR;
        let width = NR.min(jc + nc - j);
        for (p, row) in panel.chunks_exact_mut(NR).enumerate() {
            row[..width].copy_from_slice(&b[(pc + p) * ldb + j..][..width]);
            row[width..].fill(0.0);
        }
    }
}

/// Copies `A[ic.., pc..]` (`mc` rows, `kc` columns) into panels of `MR` rows (the last
/// may have fewer), each `kc` entries of one element per row of the panel.
fn pack_a(a: &[f32], lda: usize, ic: usize, mc: usize, pc: usize, kc: usize, out: &mut [f32]) {
    let mut out = out;
    let mut i = 0;
    while i < mc {
        let rows = MR.min(mc - i);
        let (panel, rest) = out.split_at_mut(kc * rows);
        out = rest;
        if rows == MR {
            // Read the rows side by side and write the panel in order.
            let src: [&[f32]; MR] = core::array::from_fn(|r| &a[(ic + i + r) * lda + pc..][..kc]);
            for (p, dst) in panel.as_chunks_mut::<MR>().0.iter_mut().enumerate() {
                for r in 0..MR {
                    dst[r] = src[r][p];
                }
            }
        } else {
            for r in 0..rows {
                let src = &a[(ic + i + r) * lda + pc..][..kc];
                for (p, &x) in src.iter().enumerate() {
                    panel[p * rows + r] = x;
                }
            }
        }
        i += MR;
    }
}

/// `R` rows of C from the rows of A in place (row `r` at `r * lda`, `kc` deep) and a
/// panel of B (`kc` entries of `NR`).
#[inline(always)]
fn tile_rows<S: Simd, const R: usize>(
    simd: S,
    d: &Dims,
    a: &[f32],
    bp: &[f32],
    kc: usize,
    c: &mut [f32],
    width: usize,
    accumulate: bool,
) {
    use fearless_simd::prelude::*;
    let a_rows: [&[f32]; R] = core::array::from_fn(|r| &a[r * d.lda..][..kc]);
    let b_rows = bp.as_chunks::<8>().0.as_chunks::<NV>().0;
    // One check here instead of one per access in the loop.
    assert_eq!(b_rows.len(), kc);
    let zero = f32x8::splat(simd, 0.0);
    let mut acc = [[zero; NV]; R];
    for (i, b) in b_rows.iter().enumerate() {
        let bv: [f32x8<S>; NV] = core::array::from_fn(|v| f32x8::load_array_ref(simd, &b[v]));
        for r in 0..R {
            let av = f32x8::splat(simd, a_rows[r][i]);
            for v in 0..NV {
                acc[r][v] = av.mul_add(bv[v], acc[r][v]);
            }
        }
    }
    store_tile(simd, d, acc, c, width, accumulate);
}

/// `R` rows of C from a panel of A (`kc` entries of `R`) and one of B (`kc` entries of `NR`).
#[inline(always)]
fn tile<S: Simd, const R: usize>(
    simd: S,
    d: &Dims,
    ap: &[f32],
    bp: &[f32],
    c: &mut [f32],
    width: usize,
    accumulate: bool,
) {
    use fearless_simd::prelude::*;
    let zero = f32x8::splat(simd, 0.0);
    let mut acc = [[zero; NV]; R];
    let a_cols = ap.as_chunks::<R>().0;
    let b_rows = bp.as_chunks::<8>().0.as_chunks::<NV>().0;
    for (a, b) in a_cols.iter().zip(b_rows) {
        let bv: [f32x8<S>; NV] = core::array::from_fn(|v| f32x8::load_array_ref(simd, &b[v]));
        for r in 0..R {
            let av = f32x8::splat(simd, a[r]);
            for v in 0..NV {
                acc[r][v] = av.mul_add(bv[v], acc[r][v]);
            }
        }
    }
    store_tile(simd, d, acc, c, width, accumulate);
}

/// Writes `alpha * acc` to the `R` rows of C, the first `width` columns, adding to what is there if asked.
#[inline(always)]
fn store_tile<S: Simd, const R: usize>(
    simd: S,
    d: &Dims,
    acc: [[f32x8<S>; NV]; R],
    c: &mut [f32],
    width: usize,
    accumulate: bool,
) {
    use fearless_simd::prelude::*;
    let alpha = f32x8::splat(simd, d.alpha);
    for r in 0..R {
        let row = &mut c[r * d.ldc..][..width];
        let vals = acc[r].map(|a| a * alpha);
        if width == NR {
            for (v, val) in vals.iter().enumerate() {
                let dst = &mut row[v * 8..][..8];
                if accumulate {
                    (*val + f32x8::from_slice(simd, dst)).store_slice(dst);
                } else {
                    val.store_slice(dst);
                }
            }
        } else {
            let mut flat = [0.0f32; NR];
            for (v, val) in vals.iter().enumerate() {
                val.store_slice(&mut flat[v * 8..][..8]);
            }
            for (x, &v) in row.iter_mut().zip(&flat) {
                *x = if accumulate { *x + v } else { v };
            }
        }
    }
}
