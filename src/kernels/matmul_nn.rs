//! `C (+)= alpha * A * B` for row-major `f32` matrices, on `fearless_simd`.
//!
//! Laid out as faer's x86 kernels (`private-gemm-x86`) are. A `MR x NR` tile of C
//! (6 rows, 2 vectors) stays in registers over a whole depth block: per step it loads a
//! row of a panel of B, broadcasts one element from each of the 6 rows of A, and does 12
//! multiply-adds.
//!
//! A is never copied: its rows are read in place, one broadcast per element. B is copied
//! into contiguous panels (rows `ldb` apart would share cache sets and pages) only when
//! C has enough rows to reuse them, and then by the first tile that reads each panel,
//! with the same loads it computes with, so copying costs no separate pass.
//!
//! Register budget: LLVM keeps two broadcasts in flight, so a tile needs its
//! accumulators, its B vectors and two more. 6x16 is 12 + 2 + 2, all 16 AVX2 registers;
//! faer's 4x24 would be 17, and spills an accumulator (to about half speed). Each tile is
//! its own target-feature function: inlined into the loops around it, the six row
//! addresses of A spill instead.

use crate::kernels::simd::simd_call;
use fearless_simd::{Level, Simd, f32x8};
use fearless_simd_macros::simd;

/// Rows of C per register tile.
const MR: usize = 6;
/// Vectors of 8 columns of C per register tile.
const NV: usize = 2;
const NR: usize = 8 * NV;
/// Most depth per block; the depth is split into equal blocks no deeper than this.
const KC: usize = 512;
/// Panels of B per column block: the packed block stays in L2 while the rows of A pass.
const PANELS: usize = 8;
/// B is read in place, not packed, when C has at most this many rows: one group of rows,
/// so each panel is read once. (Packed, it is read from L1 by later groups; in place, its
/// rows can be a power of two apart and all fall in the same cache sets.)
const PACK_MIN_ROWS: usize = MR;

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
    /// `b` is the transpose of B, row-major with `ldb` between rows (`gemm_nn` only).
    pub b_cols: bool,
}

/// The depth of each block: `k` in equal parts of at most `KC`, so no block is a sliver.
fn block_depth(k: usize) -> usize {
    k.div_ceil(k.div_ceil(KC))
}

pub(crate) fn gemm_nn(level: Level, d: &Dims, a: &[f32], b: &[f32], c: &mut [f32]) {
    PACKED.with_borrow_mut(|bpack| {
        // Every entry used is written first, so the buffer only has to be long enough.
        let len = PANELS * block_depth(d.k) * NR;
        if bpack.len() < len {
            bpack.resize(len, 0.0);
        }
        simd_call!(level, gemm_nn_simd(d, a, b, c, bpack))
    })
}

thread_local! {
    static PACKED: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// How a tile gets its panel of B.
const IN_PLACE: u8 = 0;
const PACKED_PANEL: u8 = 1;
/// In place, writing the packed panel as it goes.
const PACK_NOW: u8 = 2;

#[simd]
fn gemm_nn_simd<S: Simd>(simd: S, d: &Dims, a: &[f32], b: &[f32], c: &mut [f32], bpack: &mut [f32]) {
    let depth = block_depth(d.k);
    let pack = d.m > PACK_MIN_ROWS;
    let mut pc = 0;
    while pc < d.k {
        let kc = depth.min(d.k - pc);
        // The first depth block overwrites C unless asked to add; later ones add.
        let accumulate = d.add || pc > 0;
        let mut jc = 0;
        while jc < d.n {
            let nc = (PANELS * NR).min(d.n - jc);
            let panels = nc.div_ceil(NR);
            // Panels that cannot be loaded in place are packed up front: all of a
            // transposed B, and the narrow last panel. The rest load in place.
            let first_prepacked = if d.b_cols { 0 } else { nc / NR };
            pack_b(b, d.ldb, d.b_cols, pc, kc, jc, nc, first_prepacked, bpack);
            let mut ir = 0;
            while ir < d.m {
                let rows = MR.min(d.m - ir);
                let a = &a[ir * d.lda + pc..];
                for jp in 0..panels {
                    let j = jc + jp * NR;
                    let width = NR.min(d.n - j);
                    let c = &mut c[ir * d.ldc + j..];
                    let slot = &mut bpack[jp * kc * NR..][..kc * NR];
                    // (Only for a row-major B, in the branches below that read it.)
                    let in_place = || &b[pc * d.ldb + j..];
                    macro_rules! tile {
                        ($mode:expr, $src:expr, $lds:expr, $dst:expr) => {{
                            let t = Tile { kc, mode: $mode, lds: $lds, width, accumulate };
                            // A panel no wider than a vector takes the one-vector tiles.
                            match (rows, width > 8) {
                                (6, true) => tile6(simd, d, &t, a, $src, $dst, c),
                                (5, true) => tile5(simd, d, &t, a, $src, $dst, c),
                                (4, true) => tile4(simd, d, &t, a, $src, $dst, c),
                                (3, true) => tile3(simd, d, &t, a, $src, $dst, c),
                                (2, true) => tile2(simd, d, &t, a, $src, $dst, c),
                                (_, true) => tile1(simd, d, &t, a, $src, $dst, c),
                                (6, false) => tile6_narrow(simd, d, &t, a, $src, $dst, c),
                                (5, false) => tile5_narrow(simd, d, &t, a, $src, $dst, c),
                                (4, false) => tile4_narrow(simd, d, &t, a, $src, $dst, c),
                                (3, false) => tile3_narrow(simd, d, &t, a, $src, $dst, c),
                                (2, false) => tile2_narrow(simd, d, &t, a, $src, $dst, c),
                                (_, false) => tile1_narrow(simd, d, &t, a, $src, $dst, c),
                            }
                        }};
                    }
                    if jp >= first_prepacked || (pack && ir > 0) {
                        tile!(PACKED_PANEL, slot, NR, &mut [])
                    } else if pack {
                        tile!(PACK_NOW, in_place(), d.ldb, slot)
                    } else {
                        tile!(IN_PLACE, in_place(), d.ldb, &mut [])
                    }
                }
                ir += MR;
            }
            jc += nc;
        }
        pc += kc;
    }
}

/// Copies the panels `first..` of `B[pc.., jc..]` (`kc` rows, `nc` columns), each
/// `kc` rows of `NR` columns, into their slots of `out`, zero beyond `nc`.
///
/// With `b_cols`, `b` holds the transpose of B (column `j` of B is the contiguous row
/// `j` of `b`), and the rows of a panel are read side by side and written in order.
fn pack_b(
    b: &[f32],
    ldb: usize,
    b_cols: bool,
    pc: usize,
    kc: usize,
    jc: usize,
    nc: usize,
    first: usize,
    out: &mut [f32],
) {
    for jp in first..nc.div_ceil(NR) {
        let panel = &mut out[jp * kc * NR..][..kc * NR];
        let j = jc + jp * NR;
        let width = NR.min(jc + nc - j);
        if !b_cols {
            for (p, row) in panel.chunks_exact_mut(NR).enumerate() {
                row[..width].copy_from_slice(&b[(pc + p) * ldb + j..][..width]);
                row[width..].fill(0.0);
            }
        } else if width == NR {
            let src: [&[f32]; NR] = core::array::from_fn(|jj| &b[(j + jj) * ldb + pc..][..kc]);
            for (p, dst) in panel.as_chunks_mut::<NR>().0.iter_mut().enumerate() {
                for jj in 0..NR {
                    dst[jj] = src[jj][p];
                }
            }
        } else {
            panel.fill(0.0);
            for jj in 0..width {
                let src = &b[(j + jj) * ldb + pc..][..kc];
                for (p, &x) in src.iter().enumerate() {
                    panel[p * NR + jj] = x;
                }
            }
        }
    }
}

/// What a tile does, besides its operands.
struct Tile {
    /// Depth.
    kc: usize,
    /// `IN_PLACE`, `PACKED_PANEL` or `PACK_NOW`.
    mode: u8,
    /// Elements between the rows of the panel of B as given.
    lds: usize,
    /// Columns of C written (the panel may be zero-padded past them).
    width: usize,
    /// Add to C instead of overwriting it.
    accumulate: bool,
}

/// `tileR(simd, d, t, a, src, dst, c)`: `R` rows of C (columns `..t.width`) from the
/// rows of A in place (row `r` at `r * lda`, `t.kc` deep) and a panel of B, `src`, rows
/// `t.lds` apart, each of `NR` columns of which the first `V` vectors are used. With
/// `PACK_NOW` the panel is also copied into `dst`, `t.kc` contiguous rows.
macro_rules! tile_fn {
    ($name:ident, $r:literal, $v:expr) => {
        #[simd]
        #[inline(never)]
        fn $name<S: Simd>(simd: S, d: &Dims, t: &Tile, a: &[f32], src: &[f32], dst: &mut [f32], c: &mut [f32]) {
            use fearless_simd::prelude::*;
            const R: usize = $r;
            const V: usize = $v;
            let kc = t.kc;
            let a_rows: [&[f32]; R] = core::array::from_fn(|r| &a[r * d.lda..][..kc]);
            let zero = f32x8::splat(simd, 0.0);
            let mut acc = [[zero; V]; R];
            macro_rules! step {
                ($p:expr, $row:expr) => {{
                    let row: &[f32; NR] = $row;
                    let bv: [f32x8<S>; V] = core::array::from_fn(|v| f32x8::from_slice(simd, &row[v * 8..][..8]));
                    for r in 0..R {
                        let av = f32x8::splat(simd, a_rows[r][$p]);
                        for v in 0..V {
                            acc[r][v] = av.mul_add(bv[v], acc[r][v]);
                        }
                    }
                }};
            }
            match t.mode {
                PACKED_PANEL if V == 1 => {
                    // One vector per row is too few chains of multiply-adds to cover
                    // their latency: take two steps at a time into two sets of sums.
                    let rows = &src.as_chunks::<NR>().0[..kc];
                    let (pairs, rest) = rows.as_chunks::<2>();
                    let mut acc2 = [[zero; V]; R];
                    for (q, [row0, row1]) in pairs.iter().enumerate() {
                        let (b0, b1) = (f32x8::from_slice(simd, &row0[..8]), f32x8::from_slice(simd, &row1[..8]));
                        for r in 0..R {
                            acc[r][0] = f32x8::splat(simd, a_rows[r][2 * q]).mul_add(b0, acc[r][0]);
                            acc2[r][0] = f32x8::splat(simd, a_rows[r][2 * q + 1]).mul_add(b1, acc2[r][0]);
                        }
                    }
                    for row in rest {
                        step!(kc - 1, row);
                    }
                    for r in 0..R {
                        acc[r][0] = acc[r][0] + acc2[r][0];
                    }
                }
                PACKED_PANEL => {
                    let rows = &src.as_chunks::<NR>().0[..kc];
                    for (p, row) in rows.iter().enumerate() {
                        step!(p, row);
                    }
                }
                IN_PLACE => {
                    for (p, row) in src.chunks(t.lds).take(kc).enumerate() {
                        step!(p, row.first_chunk::<NR>().unwrap());
                    }
                }
                _ => {
                    let out = &mut dst.as_chunks_mut::<NR>().0[..kc];
                    for (p, (row, out)) in src.chunks(t.lds).zip(out).enumerate() {
                        let row = row.first_chunk::<NR>().unwrap();
                        *out = *row;
                        step!(p, row);
                    }
                }
            }
            store_tile(simd, d, acc, c, t.width, t.accumulate);
        }
    };
}

tile_fn!(tile6, 6, NV);
tile_fn!(tile5, 5, NV);
tile_fn!(tile4, 4, NV);
tile_fn!(tile3, 3, NV);
tile_fn!(tile2, 2, NV);
tile_fn!(tile1, 1, NV);
// For a panel no wider than one vector (the narrow last panel).
tile_fn!(tile6_narrow, 6, 1);
tile_fn!(tile5_narrow, 5, 1);
tile_fn!(tile4_narrow, 4, 1);
tile_fn!(tile3_narrow, 3, 1);
tile_fn!(tile2_narrow, 2, 1);
tile_fn!(tile1_narrow, 1, 1);

/// Writes `alpha * acc` to the `R` rows of C, the first `width` columns, adding to what is there if asked.
#[inline(always)]
fn store_tile<S: Simd, const R: usize, const V: usize>(
    simd: S,
    d: &Dims,
    acc: [[f32x8<S>; V]; R],
    c: &mut [f32],
    width: usize,
    accumulate: bool,
) {
    use fearless_simd::prelude::*;
    let alpha = f32x8::splat(simd, d.alpha);
    for r in 0..R {
        let row = &mut c[r * d.ldc..][..width];
        let vals = acc[r].map(|a| a * alpha);
        if width == 8 * V {
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
    for (p, brow) in b[j..].chunks(d.ldb).take(d.k).enumerate() {
        let brow = &brow[..8 * V];
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
