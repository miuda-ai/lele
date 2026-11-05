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
//!
//! On Neon the broadcast is a `ld1r`, which on Neoverse cores takes one of the two vector
//! pipes as well as a load pipe: 6 of them per 24 multiply-adds held the 6x16 tile to 80%
//! of the FMA rate. Neon tiles instead load four depths of each row of A as one vector and
//! multiply by its lanes (`fmla v.4s, v.4s, v.s[i]`), so the vector pipes only do
//! multiply-adds. That holds a row of A in a register for four steps, so they are 5 rows
//! high: 20 accumulators, 5 of A and 4 of B, of the 32 registers.

use crate::kernels::simd::simd_call;
use aligned_vec::AVec;
use fearless_simd::{Level, Simd, f32x4, f32x8};
use fearless_simd_macros::simd;

/// Rows of C per register tile.
const MR: usize = 6;
/// Rows of C per register tile multiplying by lanes (Neon).
const LANE_MR: usize = 5;
/// Vectors of 8 columns of C per register tile.
const NV: usize = 2;
/// Columns of C per register tile.
///
/// `NR` and `KC` fix the layout of `pack_f16`, in which generated weights files store
/// f16 matmul weights: changing either changes the format.
pub(crate) const NR: usize = 8 * NV;
/// Most depth per block; the depth is split into equal blocks no deeper than this.
const KC: usize = 512;
/// Panels of B per column block: the packed block stays in L2 while the rows of A pass.
const PANELS: usize = 8;
/// Most columns of B a [`Packer`] is asked for at a time.
pub(crate) const BLOCK_COLS: usize = PANELS * NR;

/// B as stored. f16 is widened to f32 as it is packed, or (with few rows of C) a block of
/// rows at a time, and from then on read exactly as f32 would be, so the product is the
/// same, bit for bit, as that of the widened matrix.
#[derive(Clone, Copy)]
pub(crate) enum BSrc<'a> {
    F32(&'a [f32]),
    F16(&'a [half::f16]),
    /// f16 in panel order ([`pack_f16`]), `n` a multiple of `NR`.
    F16Packed(&'a [half::f16]),
    /// f32 packed by the caller, a block at a time, as a convolution unfolds its input
    /// ([`Packer`]).
    Packer(&'a Packer<'a>),
    /// Integer codes of a quantized weight, row-major: element `c` is `c - zero_point`.
    /// Widened like f16, and exactly (the codes and their zero point are small integers),
    /// so the product is that of the widened matrix; the scale goes into `alpha`.
    U8 {
        codes: &'a [u8],
        zero_point: f32,
    },
    I8 {
        codes: &'a [i8],
        zero_point: f32,
    },
}

/// Writes the block of B `kc` deep from row `pc` and `nc` wide from column `jc`
/// (`pc, kc, jc, nc`) into the panels `gemm_nn` reads: `nc.div_ceil(NR)` panels in
/// turn, each `kc` rows of `NR`, so that row `p` of panel `jp` starts at
/// `(jp * kc + p) * NR`. Columns past `nc` are not read.
pub(crate) type Packer<'a> = dyn Fn((usize, usize, usize, usize), &mut [f32]) + 'a;

impl<'a> BSrc<'a> {
    fn f32(self) -> Option<&'a [f32]> {
        match self {
            BSrc::F32(b) => Some(b),
            _ => None,
        }
    }

    /// Widens elements `start..start + dst.len()` into `dst`.
    #[inline(always)]
    fn widen<S: Simd>(self, simd: S, start: usize, dst: &mut [f32]) {
        match self {
            BSrc::F32(b) => dst.copy_from_slice(&b[start..][..dst.len()]),
            BSrc::F16(b) => widen_f16(simd, &b[start..][..dst.len()], dst),
            BSrc::U8 { codes, zero_point } => {
                widen_codes(&codes[start..][..dst.len()], zero_point, dst)
            }
            BSrc::I8 { codes, zero_point } => {
                widen_codes(&codes[start..][..dst.len()], zero_point, dst)
            }
            BSrc::F16Packed(_) | BSrc::Packer(_) => {
                unreachable!("packed B is read by the panel")
            }
        }
    }
}

/// `dst[i] = src[i] - zero_point`: exact, whichever integer type the codes are.
#[inline(always)]
fn widen_codes<T: Copy + Into<f32>>(src: &[T], zero_point: f32, dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = s.into() - zero_point;
    }
}

/// A row-major `k x n` B, `n` a multiple of `NR`, in the order `gemm_nn` packs it: its
/// depth blocks in turn, each its panels of `NR` columns in turn, each `kc` rows of `NR`.
pub(crate) fn pack_f16(b: &[half::f16], k: usize, n: usize) -> Vec<half::f16> {
    assert!(n % NR == 0, "pack_f16: {n} columns");
    assert_eq!(b.len(), k * n, "pack_f16: B is not k x n");
    let depth = block_depth(k);
    let mut out = Vec::with_capacity(k * n);
    let mut pc = 0;
    while pc < k {
        let kc = depth.min(k - pc);
        for j in (0..n).step_by(NR) {
            for p in pc..pc + kc {
                out.extend_from_slice(&b[p * n + j..][..NR]);
            }
        }
        pc += kc;
    }
    out
}

/// `src` widened into `dst`, eight at a time with F16C's `vcvtph2ps` where the level has
/// it (fearless_simd's AVX2 level enables F16C), with `half` elsewhere. Widening is exact.
#[inline(always)]
fn widen_f16<S: Simd>(simd: S, src: &[half::f16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if matches!(simd.level(), Level::Avx2(_) | Level::Avx512(_)) {
        use core::arch::x86_64::{__m128i, _mm_loadu_si128, _mm256_cvtph_ps, _mm256_storeu_ps};
        let (eights, rest) = src.as_chunks::<8>();
        for (i, h) in eights.iter().enumerate() {
            // SAFETY: the level enables F16C and AVX; the load reads the 16 bytes of `h`
            // and the store writes 8 elements of `dst`, which is as long as `src`.
            unsafe {
                let h = _mm_loadu_si128(h.as_ptr() as *const __m128i);
                _mm256_storeu_ps(dst.as_mut_ptr().add(8 * i), _mm256_cvtph_ps(h));
            }
        }
        for (d, s) in dst[8 * eights.len()..].iter_mut().zip(rest) {
            *d = s.to_f32();
        }
        return;
    }
    let _ = simd;
    use half::slice::HalfFloatSliceExt;
    src.convert_to_f32_slice(dst);
}

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

pub(crate) fn gemm_nn(level: Level, d: &Dims, a: &[f32], b: BSrc, c: &mut [f32]) {
    debug_assert!(
        b.f32().is_some() || !d.b_cols,
        "f16 B is only taken row-major"
    );
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
    static PACKED: std::cell::RefCell<AVec<f32>> = std::cell::RefCell::new(AVec::new(0));
}

/// How a tile gets its panel of B.
const IN_PLACE: u8 = 0;
const PACKED_PANEL: u8 = 1;
/// In place, writing the packed panel as it goes.
const PACK_NOW: u8 = 2;
/// Widening the packed f16 panel `Tile::src16` (two vectors wide) as it goes, and
/// writing the widened panel if given somewhere to.
const WIDEN_NOW: u8 = 3;

#[simd]
fn gemm_nn_simd<S: Simd>(simd: S, d: &Dims, a: &[f32], b: BSrc, c: &mut [f32], bpack: &mut [f32]) {
    // f16 is packed whole, and the panels an f32 B would be read in place from are read
    // from the packed copy the same way (a narrow panel sums in another order packed).
    let b32 = b.f32();
    let depth = block_depth(d.k);
    let lanes = by_lanes(simd);
    let mr = if lanes { LANE_MR } else { MR };
    // B is read in place, not packed, when C has one group of rows, so each panel is read
    // once. (Packed, it is read from L1 by later groups; in place, its rows can be a power
    // of two apart and all fall in the same cache sets.)
    let pack = d.m > mr;
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
            let first_prepacked = if d.b_cols || matches!(b, BSrc::Packer(_)) {
                0
            } else if nc % NR == 0 || nc % NR == 8 {
                panels
            } else {
                nc / NR
            };
            // Packed f16 (whole panels only): its panels for this block lie in order from
            // `start`. The tiles of the first group of rows widen them as they read them,
            // into the packed panels for the next groups; tiles multiplying by lanes read
            // the block widened first.
            let mut block16: &[half::f16] = &[];
            if let BSrc::F16Packed(b) = b {
                let len = panels * kc * NR;
                block16 = &b[pc * d.n + jc * kc..][..len];
                if lanes {
                    widen_f16(simd, block16, &mut bpack[..len]);
                }
            } else if let BSrc::Packer(pack) = b {
                pack((pc, kc, jc, nc), &mut bpack[..panels * kc * NR]);
            } else {
                pack_b(
                    simd,
                    b,
                    d.ldb,
                    d.b_cols,
                    pc,
                    kc,
                    jc,
                    nc,
                    if b32.is_some() { first_prepacked } else { 0 },
                    bpack,
                );
            }
            let mut ir = 0;
            while ir < d.m {
                let rows = mr.min(d.m - ir);
                let a = &a[ir * d.lda + pc..];
                for jp in 0..panels {
                    let j = jc + jp * NR;
                    let width = NR.min(d.n - j);
                    let c = &mut c[ir * d.ldc + j..];
                    let slot = &mut bpack[jp * kc * NR..][..kc * NR];
                    // (Only for a row-major f32 B, in the branches below that read it.)
                    let in_place = || &b32.expect("f16 B is packed")[pc * d.ldb + j..];
                    let panel16 = if block16.is_empty() {
                        block16
                    } else {
                        &block16[jp * kc * NR..][..kc * NR]
                    };
                    macro_rules! tile {
                        ($mode:expr, $src:expr, $lds:expr, $dst:expr) => {{
                            let t = Tile {
                                kc,
                                mode: $mode,
                                lds: $lds,
                                width,
                                accumulate,
                                src16: panel16,
                            };
                            // A panel no wider than a vector takes the one-vector tiles.
                            match (rows, width > 8) {
                                _ if lanes => match (rows, width > 8) {
                                    (5, true) => lanes5(simd, d, &t, a, $src, $dst, c),
                                    (4, true) => lanes4(simd, d, &t, a, $src, $dst, c),
                                    (3, true) => lanes3(simd, d, &t, a, $src, $dst, c),
                                    (2, true) => lanes2(simd, d, &t, a, $src, $dst, c),
                                    (_, true) => lanes1(simd, d, &t, a, $src, $dst, c),
                                    (5, false) => lanes5_narrow(simd, d, &t, a, $src, $dst, c),
                                    (4, false) => lanes4_narrow(simd, d, &t, a, $src, $dst, c),
                                    (3, false) => lanes3_narrow(simd, d, &t, a, $src, $dst, c),
                                    (2, false) => lanes2_narrow(simd, d, &t, a, $src, $dst, c),
                                    (_, false) => lanes1_narrow(simd, d, &t, a, $src, $dst, c),
                                },
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
                    if !panel16.is_empty() && !lanes && ir == 0 {
                        // Reads the same values in the same order as the in-place tiles.
                        let dst: &mut [f32] = if pack { slot } else { &mut [] };
                        tile!(WIDEN_NOW, &[], NR, dst)
                    } else if jp >= first_prepacked || (pack && ir > 0) {
                        tile!(PACKED_PANEL, slot, NR, &mut [])
                    } else if b32.is_none() {
                        tile!(IN_PLACE, slot, NR, &mut [])
                    } else if pack {
                        tile!(PACK_NOW, in_place(), d.ldb, slot)
                    } else {
                        tile!(IN_PLACE, in_place(), d.ldb, &mut [])
                    }
                }
                ir += mr;
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
    b: BSrc,
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
            for (p, row) in panel.as_chunks_mut::<NR>().0.iter_mut().enumerate() {
                b.widen(simd, (pc + p) * ldb + j, &mut row[..width]);
                row[width..].fill(0.0);
            }
            continue;
        }
        let b = b.f32().expect("f16 B is only taken row-major");
        if width == NR {
            // 4x4 blocks: four depths of four columns, transposed in registers.
            let quads = kc / 4;
            for jq in 0..NR / 4 {
                let src: [&[f32]; 4] =
                    core::array::from_fn(|i| &b[(j + 4 * jq + i) * ldb + pc..][..kc]);
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
struct Tile<'a> {
    /// Depth.
    kc: usize,
    /// `IN_PLACE`, `PACKED_PANEL`, `PACK_NOW` or `WIDEN_NOW`.
    mode: u8,
    /// The packed f16 panel, for `WIDEN_NOW`.
    src16: &'a [half::f16],
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
        fn $name<S: Simd>(
            simd: S,
            d: &Dims,
            t: &Tile,
            a: &[f32],
            src: &[f32],
            dst: &mut [f32],
            c: &mut [f32],
        ) {
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
                    let bv: [f32x8<S>; V] =
                        core::array::from_fn(|v| f32x8::from_slice(simd, &row[v * 8..][..8]));
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
                        let (b0, b1) = (
                            f32x8::from_slice(simd, &row0[..8]),
                            f32x8::from_slice(simd, &row1[..8]),
                        );
                        for r in 0..R {
                            acc[r][0] = f32x8::splat(simd, a_rows[r][2 * q]).mul_add(b0, acc[r][0]);
                            acc2[r][0] =
                                f32x8::splat(simd, a_rows[r][2 * q + 1]).mul_add(b1, acc2[r][0]);
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
                WIDEN_NOW => {
                    let mut out = dst.as_chunks_mut::<NR>().0.iter_mut();
                    for (p, row16) in t.src16.as_chunks::<NR>().0[..kc].iter().enumerate() {
                        let mut row = [0.0f32; NR];
                        widen_f16(simd, row16, &mut row);
                        if let Some(out) = out.next() {
                            *out = row;
                        }
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

/// Whether the tiles multiply by lanes ([`lane_tile_fn`]): on Neon.
#[inline(always)]
fn by_lanes<S: Simd>(simd: S) -> bool {
    #[cfg(target_arch = "aarch64")]
    if let Level::Neon(_) = simd.level() {
        return true;
    }
    let _ = simd;
    false
}

/// `acc + b * a[L]`: on Neon, `fmla` by lane, twice.
#[inline(always)]
fn fma_lane<S: Simd, const L: i32>(simd: S, acc: f32x8<S>, b: f32x8<S>, a: f32x4<S>) -> f32x8<S> {
    use fearless_simd::prelude::*;
    match simd.level() {
        #[cfg(target_arch = "aarch64")]
        Level::Neon(_) => {
            use core::arch::aarch64::{float32x4_t, float32x4x2_t, vfmaq_laneq_f32};
            use fearless_simd::SimdInto;
            let (acc, b): (float32x4x2_t, float32x4x2_t) = (acc.into(), b.into());
            let a: float32x4_t = a.into();
            // SAFETY: Neon is part of every aarch64 target.
            let fma = |acc, b| unsafe { vfmaq_laneq_f32::<L>(acc, b, a) };
            float32x4x2_t(fma(acc.0, b.0), fma(acc.1, b.1)).simd_into(simd)
        }
        #[allow(unreachable_patterns)]
        _ => f32x8::splat(simd, a.as_slice()[L as usize]).mul_add(b, acc),
    }
}

/// As [`tile_fn`], multiplying by the lanes of four depths of each row of A at a time.
macro_rules! lane_tile_fn {
    ($name:ident, $r:literal, $v:expr) => {
        #[simd]
        #[inline(never)]
        fn $name<S: Simd>(
            simd: S,
            d: &Dims,
            t: &Tile,
            a: &[f32],
            src: &[f32],
            dst: &mut [f32],
            c: &mut [f32],
        ) {
            use fearless_simd::prelude::*;
            const R: usize = $r;
            const V: usize = $v;
            let kc = t.kc;
            // The panel as the steps read it: copied first when it is to be packed.
            let (b, ldb): (&[f32], usize) = match t.mode {
                IN_PLACE => (src, t.lds),
                PACK_NOW => {
                    for (row, out) in src
                        .chunks(t.lds)
                        .zip(&mut dst.as_chunks_mut::<NR>().0[..kc])
                    {
                        out[..8 * V].copy_from_slice(&row[..8 * V]);
                    }
                    (&*dst, NR)
                }
                _ => (src, NR),
            };
            let a_rows: [&[f32]; R] = core::array::from_fn(|r| &a[r * d.lda..][..kc]);
            let mut acc = [[f32x8::splat(simd, 0.0); V]; R];
            let brow = |p: usize| -> [f32x8<S>; V] {
                let row = &b[p * ldb..][..8 * V];
                core::array::from_fn(|v| f32x8::from_slice(simd, &row[8 * v..][..8]))
            };
            let quads = kc / 4;
            for q in 0..quads {
                let av: [f32x4<S>; R] =
                    core::array::from_fn(|r| f32x4::from_slice(simd, &a_rows[r][4 * q..][..4]));
                macro_rules! lane {
                    ($l:literal) => {{
                        let bv = brow(4 * q + $l);
                        for r in 0..R {
                            for v in 0..V {
                                acc[r][v] = fma_lane::<S, $l>(simd, acc[r][v], bv[v], av[r]);
                            }
                        }
                    }};
                }
                lane!(0);
                lane!(1);
                lane!(2);
                lane!(3);
            }
            for p in 4 * quads..kc {
                let bv = brow(p);
                for r in 0..R {
                    let s = f32x8::splat(simd, a_rows[r][p]);
                    for v in 0..V {
                        acc[r][v] = s.mul_add(bv[v], acc[r][v]);
                    }
                }
            }
            store_tile(simd, d, acc, c, t.width, t.accumulate);
        }
    };
}

lane_tile_fn!(lanes5, 5, NV);
lane_tile_fn!(lanes4, 4, NV);
lane_tile_fn!(lanes3, 3, NV);
lane_tile_fn!(lanes2, 2, NV);
lane_tile_fn!(lanes1, 1, NV);
lane_tile_fn!(lanes5_narrow, 5, 1);
lane_tile_fn!(lanes4_narrow, 4, 1);
lane_tile_fn!(lanes3_narrow, 3, 1);
lane_tile_fn!(lanes2_narrow, 2, 1);
lane_tile_fn!(lanes1_narrow, 1, 1);

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
pub(crate) fn gemm_small_m(level: Level, d: &Dims, a: &[f32], b: BSrc, c: &mut [f32]) {
    if let BSrc::F32(b) = b {
        return simd_call!(level, gemm_small_m_simd(d, a, b, c));
    }
    if matches!(b, BSrc::F16Packed(_) | BSrc::Packer(_)) {
        unreachable!("packed B goes to gemm_nn");
    }
    // f16 and integer codes: blocks of `WIDEN_ROWS` rows of B, widened, each a product of
    // that depth adding into C. The blocks are a whole number of passes of every height,
    // so the passes are those of the widened B.
    WIDENED.with_borrow_mut(|wide| {
        wide.resize(WIDEN_ROWS.min(d.k) * d.n, 0.0);
        simd_call!(level, gemm_small_m_wide_simd(d, a, b, c, wide))
    })
}

#[simd]
fn gemm_small_m_wide_simd<S: Simd>(
    simd: S,
    d: &Dims,
    a: &[f32],
    b: BSrc,
    c: &mut [f32],
    wide: &mut [f32],
) {
    let mut p = 0;
    while p < d.k {
        let rows = WIDEN_ROWS.min(d.k - p);
        let wide = &mut wide[..rows * d.n];
        for (i, row) in wide.chunks_exact_mut(d.n).enumerate() {
            b.widen(simd, (p + i) * d.ldb, row);
        }
        let block = Dims {
            k: rows,
            ldb: d.n,
            add: d.add || p > 0,
            ..*d
        };
        gemm_small_m_simd(simd, &block, &a[p..], wide, c);
        p += rows;
    }
}

/// Rows of an f16 B `gemm_small_m` widens at a time: a multiple of every pass height.
const WIDEN_ROWS: usize = 48;

thread_local! {
    static WIDENED: std::cell::RefCell<AVec<f32>> = std::cell::RefCell::new(AVec::new(0));
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
    let s: [[f32; P]; R] =
        core::array::from_fn(|r| core::array::from_fn(|i| d.alpha * a[r * d.lda + p + i]));
    let sv = s.map(|row| row.map(|x| f32x8::splat(simd, x)));
    let br: [&[f32]; P] = core::array::from_fn(|i| &b[(p + i) * d.ldb..][..n]);
    let mut c_rows = c.chunks_mut(d.ldc);
    let cr: [&mut [f32]; R] = core::array::from_fn(|_| &mut c_rows.next().unwrap()[..n]);
    // All cut to the same number of vectors, so indexing them needs no checks.
    let bv: [&[[f32; 8]]; P] = br.map(|row| &row.as_chunks::<8>().0[..vecs]);
    let cv = cr.map(|row| &mut row.as_chunks_mut::<8>().0[..vecs]);
    for j in 0..vecs {
        let mut acc: [f32x8<S>; R] = core::array::from_fn(|r| {
            if first {
                f32x8::splat(simd, 0.0)
            } else {
                f32x8::from_slice(simd, &cv[r][j])
            }
        });
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
