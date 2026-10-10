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
use fearless_simd::{Level, Simd, f32x4, f32x8};
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
            // transposed B, and a last panel narrower than the vectors its tiles load
            // (they load one vector up to 8 columns, two above). The rest load in place.
            let first_prepacked = if d.b_cols {
                0
            } else if nc % NR == 0 || nc % NR == 8 {
                panels
            } else {
                nc / NR
            };
            pack_b(simd, b, d.ldb, d.b_cols, pc, kc, jc, nc, first_prepacked, bpack);
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
#[inline(always)]
fn pack_b<S: Simd>(
    simd: S,
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
    use fearless_simd::prelude::*;
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
            // 4x4 blocks: four depths of four columns, transposed in registers.
            let quads = kc / 4;
            for jq in 0..NR / 4 {
                let src: [&[f32]; 4] = core::array::from_fn(|i| &b[(j + 4 * jq + i) * ldb + pc..][..kc]);
                for q in 0..quads {
                    let [r0, r1, r2, r3] = src.map(|s| f32x4::from_slice(simd, &s[4 * q..][..4]));
                    let (a0, a1) = simd.interleave_f32x4(r0, r2);
                    let (b0, b1) = simd.interleave_f32x4(r1, r3);
                    let (c0, c1) = simd.interleave_f32x4(a0, b0);
                    let (c2, c3) = simd.interleave_f32x4(a1, b1);
                    for (t, col) in [c0, c1, c2, c3].iter().enumerate() {
                        col.store_slice(&mut panel[(4 * q + t) * NR + 4 * jq..][..4]);
                    }
                }
                for p in 4 * quads..kc {
                    for i in 0..4 {
                        panel[p * NR + 4 * jq + i] = src[i][p];
                    }
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
                    let row: &[f32; 8 * V] = $row;
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
                        step!(kc - 1, row.first_chunk().unwrap());
                    }
                    for r in 0..R {
                        acc[r][0] = acc[r][0] + acc2[r][0];
                    }
                }
                PACKED_PANEL => {
                    let rows = &src.as_chunks::<NR>().0[..kc];
                    for (p, row) in rows.iter().enumerate() {
                        step!(p, row.first_chunk().unwrap());
                    }
                }
                // In place, only the `V` vectors used are read (and copied).
                IN_PLACE => {
                    for (p, row) in src.chunks(t.lds).take(kc).enumerate() {
                        step!(p, row.first_chunk().unwrap());
                    }
                }
                _ => {
                    let out = &mut dst.as_chunks_mut::<NR>().0[..kc];
                    for (p, (row, out)) in src.chunks(t.lds).zip(out).enumerate() {
                        let row: &[f32; 8 * V] = row.first_chunk().unwrap();
                        *out.first_chunk_mut().unwrap() = *row;
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

/// `gemm_nn` for at most `SMALL_M` rows of C. There is too little arithmetic per element
/// of B for it to be anything but a stream of B, so B is read in order, `P` rows at a
/// time, each pass adding them into C (which stays in L1). (Tiles of C walking down the
/// columns of B, as `gemm_nn` does, are 5-40% slower here: B arrives in short pieces
/// from rows far apart.) From 5 rows on, `gemm_nn` is faster.
pub(crate) fn gemm_small_m(level: Level, d: &Dims, a: &[f32], b: &[f32], c: &mut [f32]) {
    simd_call!(level, gemm_small_m_simd(d, a, b, c))
}

#[simd]
fn gemm_small_m_simd<S: Simd>(simd: S, d: &Dims, a: &[f32], b: &[f32], c: &mut [f32]) {
    // `R * P` broadcasts of A per pass; past 16 they are reloaded, which costs less than
    // passing over C more often.
    match d.m {
        1 => stream_rows::<S, 1, 8>(simd, d, a, b, c),
        2 => stream_rows::<S, 2, 6>(simd, d, a, b, c),
        3 => stream_rows::<S, 3, 4>(simd, d, a, b, c),
        _ => stream_rows::<S, 4, 4>(simd, d, a, b, c),
    }
}

/// `R` rows of C, `P` rows of B per pass.
#[inline(always)]
fn stream_rows<S: Simd, const R: usize, const P: usize>(
    simd: S,
    d: &Dims,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    let mut first = !d.add;
    let mut p = 0;
    while p + P <= d.k {
        stream_pass::<S, R, P>(simd, d, a, b, c, p, first);
        first = false;
        p += P;
    }
    while p < d.k {
        stream_pass::<S, R, 1>(simd, d, a, b, c, p, first);
        first = false;
        p += 1;
    }
    if first {
        for r in 0..R {
            c[r * d.ldc..][..d.n].fill(0.0);
        }
    }
}

/// Adds rows `p..p + P` of B, times their elements of A, into `R` rows of C (or
/// overwrites C with them if `first`).
#[inline(always)]
fn stream_pass<S: Simd, const R: usize, const P: usize>(
    simd: S,
    d: &Dims,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    p: usize,
    first: bool,
) {
    use fearless_simd::prelude::*;
    let n = d.n;
    let vecs = n / 8;
    let s: [[f32; P]; R] = core::array::from_fn(|r| core::array::from_fn(|i| d.alpha * a[r * d.lda + p + i]));
    let sv = s.map(|row| row.map(|x| f32x8::splat(simd, x)));
    let br: [&[f32]; P] = core::array::from_fn(|i| &b[(p + i) * d.ldb..][..n]);
    let mut c_rows = c.chunks_mut(d.ldc);
    let cr: [&mut [f32]; R] = core::array::from_fn(|_| &mut c_rows.next().unwrap()[..n]);
    // All cut to the same number of vectors, so indexing them needs no checks.
    let bv: [&[[f32; 8]]; P] = br.map(|row| &row.as_chunks::<8>().0[..vecs]);
    let cv = cr.map(|row| &mut row.as_chunks_mut::<8>().0[..vecs]);
    for j in 0..vecs {
        let mut acc: [f32x8<S>; R] =
            core::array::from_fn(|r| if first { f32x8::splat(simd, 0.0) } else { f32x8::from_slice(simd, &cv[r][j]) });
        for i in 0..P {
            let b = f32x8::from_slice(simd, &bv[i][j]);
            for r in 0..R {
                acc[r] = sv[r][i].mul_add(b, acc[r]);
            }
        }
        for r in 0..R {
            acc[r].store_slice(&mut cv[r][j]);
        }
    }
    for j in 8 * vecs..n {
        for r in 0..R {
            let sum: f32 = (0..P).map(|i| s[r][i] * br[i][j]).sum();
            let out = &mut c[r * d.ldc + j];
            *out = if first { sum } else { *out + sum };
        }
    }
}

