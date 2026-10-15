//! Integer GEMM for quantized models, on `fearless_simd`: `C = epilogue(A * B)` with A and
//! B 8-bit codes and C `f32`.
//!
//! Both operands come centered on their zero points and widened to 16 bits (`a - za` and
//! `b - zb`, each in `[-255, 255]`), with two depths packed into each `i32`. A pair of
//! depths is then one 16-bit multiply-add into an `i32` lane (`vpmaddwd` on x86,
//! `i32x4.dot_i16x8_s` on wasm), and it is exact: a pair sums to at most `2 * 255^2`, so
//! the `i32` sums hold for depths up to 33 000 and no zero-point correction is left for the
//! end.
//!
//! The tiles are `matmul_nn`'s: `MR` rows by 16 columns of sums in registers over a depth
//! block. A is read in place, one broadcast pair per row and step. B is packed once, when
//! the weight is prepared, into panels of 16 columns. Depth blocks before the last keep
//! their sums in memory; the last applies the epilogue (scales, bias, ReLU) and writes C.
//!
//! On Zen 3 this reaches about 200 GOP/s on the models' shapes, 1.5 times the f32 GEMM.
//! Without VNNI each multiply-add needs a separate add, so a step of a 6x16 tile is 12
//! `vpmaddwd`, 12 `vpaddd` and 8 loads, and the loop is bound by issuing them. (The
//! `vpmaddubsw` route does 32 products per instruction but saturates at `i16`, so it is
//! exact only for 7-bit weights.)
//!
//! Where a 4-byte dot-product instruction is available (VNNI with `fearless_simd`'s AVX-512
//! level, `sdot` on aarch64 builds with `dotprod`), weights are packed for
//! [`crate::kernels::qgemm_dot`] instead, which is about four times faster; [`QWeights`]
//! picks the layout when it packs, and [`run_at`] the matching kernel.

use crate::kernels::qgemm_dot::{self, Dot, QuadWeights};
use crate::kernels::simd::simd_call;
use crate::tensor::TensorView;
use aligned_vec::AVec;
use fearless_simd::{Level, Simd, f32x8, i16x16, i32x8};
use fearless_simd_macros::simd;

/// Rows of C per register tile.
const MR: usize = 6;
/// Columns of C per register tile and per panel of B: two vectors of `i32` sums.
const NR: usize = 16;
/// Most pairs of depths per block; the depth is split into equal blocks no deeper than
/// this. A block of a panel is then up to 64 KB, read from L2 by every row tile; blocks of
/// 256 pairs, which stay in L1, measured 5% slower at depth 1536, from passing the sums
/// through memory.
const KC: usize = 1024;

/// Zeroed `i32`s that start on a 64-byte boundary (a cache line, and one 512-bit vector):
/// in a panel that does not, every load of B straddles two lines, which measured up to
/// 40% slower on an Ice Lake Xeon.
pub(crate) struct AlignedI32 {
    buf: Vec<i32>,
    start: usize,
    len: usize,
}

impl AlignedI32 {
    pub(crate) fn zeroed(len: usize) -> Self {
        let buf = vec![0; len + 15];
        let start = buf.as_ptr().align_offset(64);
        assert!(start < 16);
        Self { buf, start, len }
    }
}

impl std::ops::Deref for AlignedI32 {
    type Target = [i32];
    fn deref(&self) -> &[i32] {
        &self.buf[self.start..][..self.len]
    }
}

impl std::ops::DerefMut for AlignedI32 {
    fn deref_mut(&mut self) -> &mut [i32] {
        &mut self.buf[self.start..][..self.len]
    }
}

/// Two centered codes in one `i32`: the even depth in the low half, the odd one in the
/// high half, as the lanes of a 16-bit vector bitcast from it are laid out.
#[inline]
fn pack_pair(lo: i32, hi: i32) -> i32 {
    (lo as i16 as u16 as i32) | (hi << 16)
}

/// A `k x n` weight of 8-bit codes, packed for the integer GEMM of the machine it runs on.
pub struct QWeights {
    k: usize,
    n: usize,
    packed: Packed,
    /// The scale of each column, zero past `n` to a whole panel of either layout.
    scale: Vec<f32>,
}

/// How B is packed, which decides how A is quantized for it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Layout {
    /// Centered 16-bit pairs, for `vpmaddwd` and its equivalents ([`qgemm_at`]).
    Pairs,
    /// `i8` quads for VNNI's `vpdpbusd` or Arm's `sdot` ([`crate::kernels::qgemm_dot`]).
    /// (Each chosen only where its instruction is; elsewhere only the tests pack them,
    /// emulating it.)
    #[cfg_attr(
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
        allow(dead_code)
    )]
    Dot(Dot),
    /// Centered codes in `f32`, multiplied by the f32 GEMM: where no integer instruction
    /// beats it (aarch64 without `dotprod`, whose widening multiplies do no more per
    /// instruction than f32 multiply-adds), and on macOS, whose Accelerate runs f32 on the
    /// matrix units. Sums are exact while they stay under 2^24 (depths up to 258), and
    /// rounded as an f32 GEMM's are beyond.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    Float,
}

impl Layout {
    /// The layout that runs fastest at `level`.
    pub(crate) fn for_level(level: Level) -> Self {
        match level {
            #[cfg(target_arch = "x86_64")]
            Level::Avx512(_) => Layout::Dot(Dot::Vnni),
            #[cfg(target_arch = "aarch64")]
            Level::Neon(_) if cfg!(target_os = "macos") || !cfg!(target_feature = "dotprod") => {
                Layout::Float
            }
            #[cfg(target_arch = "aarch64")]
            Level::Neon(_) => Layout::Dot(Dot::Sdot),
            _ => Layout::Pairs,
        }
    }

    /// The layouts the tests run at every level.
    #[cfg(test)]
    pub(crate) const ALL: [Layout; 4] = [
        Layout::Pairs,
        Layout::Dot(Dot::Vnni),
        Layout::Dot(Dot::Sdot),
        Layout::Float,
    ];
}

enum Packed {
    /// Panels of `NR` columns, each `pairs` rows of `NR` packed pairs (`b - zb`); zero past
    /// `k` and `n`.
    Pairs {
        pairs: usize,
        data: AlignedI32,
    },
    Dot(QuadWeights),
    /// Row-major `k x n` centered codes (`b - zb`).
    Float(Vec<f32>),
}

impl QWeights {
    /// Packs row-major `k x n` codes, `i8` if `signed` and `u8` otherwise, with one zero
    /// point and one scale for all columns or one of each per column.
    pub fn new(
        codes: &[u8],
        k: usize,
        n: usize,
        signed: bool,
        zero_point: &[i32],
        scale: &[f32],
    ) -> Self {
        Self::with_layout(
            Layout::for_level(Level::new()),
            codes,
            k,
            n,
            signed,
            zero_point,
            scale,
        )
    }

    pub(crate) fn with_layout(
        layout: Layout,
        codes: &[u8],
        k: usize,
        n: usize,
        signed: bool,
        zero_point: &[i32],
        scale: &[f32],
    ) -> Self {
        assert_eq!(codes.len(), k * n);
        let code = |i: usize| {
            if signed {
                codes[i] as i8 as i32
            } else {
                codes[i] as i32
            }
        };
        Self::pack(
            layout,
            k,
            n,
            code,
            if signed { 0 } else { 128 },
            zero_point,
            scale,
        )
    }

    /// As [`QWeights::new`], from codes held in `f32` (as lele keeps integer tensors); they
    /// are taken as `i8` if any is negative and `u8` otherwise.
    pub fn from_f32_codes(
        codes: &[f32],
        k: usize,
        n: usize,
        zero_point: &[i32],
        scale: &[f32],
    ) -> Self {
        Self::from_f32_codes_with_layout(
            Layout::for_level(Level::new()),
            codes,
            k,
            n,
            zero_point,
            scale,
        )
    }

    pub(crate) fn from_f32_codes_with_layout(
        layout: Layout,
        codes: &[f32],
        k: usize,
        n: usize,
        zero_point: &[i32],
        scale: &[f32],
    ) -> Self {
        assert_eq!(codes.len(), k * n);
        let shift = if codes.iter().any(|&c| c < 0.0) {
            0
        } else {
            128
        };
        Self::pack(layout, k, n, |i| codes[i] as i32, shift, zero_point, scale)
    }

    /// As [`QWeights::new`], from a row-major `k x n` activation on the `u8` grid of `scale`
    /// and `zero_point` (it came out of a `QuantizeLinear` -> `DequantizeLinear` pair): each
    /// value is quantized back to its code, as [`quantize_rows_at`] quantizes A. Packed in
    /// `layout`, at `level`.
    pub(crate) fn from_grid(
        level: Level,
        layout: Layout,
        b: &[f32],
        k: usize,
        n: usize,
        scale: f32,
        zero_point: i32,
    ) -> Self {
        assert_eq!(b.len(), k * n);
        let packed = match layout {
            Layout::Pairs => {
                let pairs = k.div_ceil(2);
                let mut data = AlignedI32::zeroed(n.div_ceil(NR) * pairs * NR);
                simd_call!(
                    level,
                    pack_grid_pairs_simd(b, k, n, scale, zero_point, &mut data)
                );
                Packed::Pairs { pairs, data }
            }
            Layout::Dot(dot) => Packed::Dot(simd_call!(
                level,
                pack_grid_quads_simd(dot, b, k, n, scale, zero_point)
            )),
            Layout::Float => {
                let mut codes = vec![0.0; k * n];
                simd_call!(
                    level,
                    quantize_rows_f32_simd(b, k * n, scale, zero_point, 0, &mut codes)
                );
                let code = |i: usize| codes[i] as i32;
                return Self::pack(layout, k, n, code, 128, &[zero_point], &[scale]);
            }
        };
        let mut col_scale = vec![0.0; n.div_ceil(qgemm_dot::NR) * qgemm_dot::NR];
        col_scale[..n].fill(scale);
        Self {
            k,
            n,
            packed,
            scale: col_scale,
        }
    }

    /// `code(i)` is element `i` of the row-major codes; `shift` brings them into `i8` (0
    /// for `i8` codes, 128 for `u8`), which the quads need.
    fn pack(
        layout: Layout,
        k: usize,
        n: usize,
        code: impl Fn(usize) -> i32,
        shift: i32,
        zero_point: &[i32],
        scale: &[f32],
    ) -> Self {
        assert!(zero_point.len() == 1 || zero_point.len() == n);
        assert!(scale.len() == 1 || scale.len() == n);
        let zp = |j: usize| zero_point[if zero_point.len() == 1 { 0 } else { j }];
        let packed = match layout {
            Layout::Pairs => {
                let pairs = k.div_ceil(2);
                let centered =
                    |kk: usize, j: usize| if kk < k { code(kk * n + j) - zp(j) } else { 0 };
                let mut data = AlignedI32::zeroed(n.div_ceil(NR) * pairs * NR);
                for p in 0..pairs {
                    for j in 0..n {
                        let pair = pack_pair(centered(2 * p, j), centered(2 * p + 1, j));
                        data[((j / NR) * pairs + p) * NR + j % NR] = pair;
                    }
                }
                Packed::Pairs { pairs, data }
            }
            Layout::Dot(dot) => Packed::Dot(qgemm_dot::pack(dot, k, n, code, shift, zp)),
            Layout::Float => {
                Packed::Float((0..k * n).map(|i| (code(i) - zp(i % n)) as f32).collect())
            }
        };
        let mut col_scale = vec![0.0; n.div_ceil(qgemm_dot::NR) * qgemm_dot::NR];
        for (j, s) in col_scale[..n].iter_mut().enumerate() {
            *s = scale[if scale.len() == 1 { 0 } else { j }];
        }
        Self {
            k,
            n,
            packed,
            scale: col_scale,
        }
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn n(&self) -> usize {
        self.n
    }

    #[cfg(test)]
    pub(crate) fn layout(&self) -> Layout {
        match &self.packed {
            Packed::Pairs { .. } => Layout::Pairs,
            Packed::Dot(q) => Layout::Dot(q.dot),
            Packed::Float(_) => Layout::Float,
        }
    }

    /// Pairs of depths per row of A, and the panels, for the pairs layout.
    fn pairs(&self) -> (usize, &[i32]) {
        match &self.packed {
            Packed::Pairs { pairs, data } => (*pairs, data),
            _ => panic!("weights not packed as pairs used as pairs"),
        }
    }
}

/// What happens to the integer sums on the way out:
/// `C[i][j] = relu?(requant?(sum * (scale * column scale of B)) + bias[j])`.
pub(crate) struct Epilogue<'a> {
    /// The scale of A.
    pub scale: f32,
    /// Snap the products onto a `u8` grid, if any.
    pub requant: Option<Requant>,
    /// `n` values to add, if any.
    pub bias: Option<&'a [f32]>,
    pub relu: bool,
}

/// A `u8` `QuantizeLinear` -> `DequantizeLinear` round trip of a product, as static-QDQ
/// graphs follow every product with:
/// `(clamp(round_ties_even(x / scale) + zero_point, 0, 255) - zero_point) * scale`, each
/// step rounded as `fake_quantize_linear` rounds it.
#[derive(Clone, Copy, Debug)]
pub struct Requant {
    pub scale: f32,
    pub zero_point: i32,
}

impl Requant {
    #[inline(always)]
    pub(crate) fn apply(self, x: f32) -> f32 {
        let z = self.zero_point as f32;
        (((x / self.scale).round_ties_even() + z).clamp(0.0, 255.0) - z) * self.scale
    }
}

/// [`Requant::apply`] on a vector (`f32x8` or `f32x16`), or `x` as it is without one.
macro_rules! requant {
    ($simd:expr, $vec:ident, $requant:expr, $x:expr) => {
        match $requant {
            Some(r) => {
                let (s, z) = (
                    $vec::splat($simd, r.scale),
                    $vec::splat($simd, r.zero_point as f32),
                );
                let (lo, hi) = ($vec::splat($simd, 0.0), $vec::splat($simd, 255.0));
                ((($x / s).round_ties_even() + z).max(lo).min(hi) - z) * s
            }
            None => $x,
        }
    };
}
pub(crate) use requant;

/// `C = epilogue(A * B)` for B in the pairs layout. A is `m` rows of `w.pairs()` packed
/// pairs, `lda` apart, as [`quantize_rows_at`] writes them; C is `m x n`, row-major.
pub(crate) fn qgemm_at(
    level: Level,
    a: &[i32],
    m: usize,
    lda: usize,
    w: &QWeights,
    e: &Epilogue,
    c: &mut [f32],
) {
    let (pairs, _) = w.pairs();
    assert!(lda >= pairs && a.len() >= m.saturating_sub(1) * lda + pairs);
    assert!(c.len() >= m * w.n);
    assert!(e.bias.is_none_or(|b| b.len() >= w.n));
    SUMS.with_borrow_mut(|sums| {
        let sums = if pairs > KC {
            sums.resize(m * w.n.div_ceil(NR) * NR, 0);
            &mut sums[..]
        } else {
            &mut []
        };
        simd_call!(level, qgemm_simd(a, m, lda, w, e, c, sums))
    })
}

thread_local! {
    /// The sums of C between depth blocks.
    static SUMS: std::cell::RefCell<AVec<i32>> = std::cell::RefCell::new(AVec::new(0));
}

#[simd]
fn qgemm_simd<S: Simd>(
    simd: S,
    a: &[i32],
    m: usize,
    lda: usize,
    w: &QWeights,
    e: &Epilogue,
    c: &mut [f32],
    sums: &mut [i32],
) {
    let (pairs, data) = w.pairs();
    let depth = pairs.div_ceil(pairs.div_ceil(KC).max(1));
    let panels = w.n.div_ceil(NR);
    let lds = panels * NR;
    let mut pc = 0;
    loop {
        let kc = depth.min(pairs - pc);
        let last = pc + kc == pairs;
        for jp in 0..panels {
            let j = jp * NR;
            let b = &data[(jp * pairs + pc) * NR..][..kc * NR];
            let t = Tile {
                kc,
                lda,
                width: NR.min(w.n - j),
                first: pc == 0,
                last,
                lds,
                ldc: w.n,
                scale: e.scale,
                col_scale: w.scale[j..][..NR].try_into().unwrap(),
                requant: e.requant,
                bias: e.bias.map(|b| &b[j..w.n.min(j + NR)]),
                relu: e.relu,
            };
            let mut ir = 0;
            while ir < m {
                let a = &a[ir * lda + pc..];
                // (Empty with a single depth block.)
                let sums = if sums.is_empty() {
                    &mut [][..]
                } else {
                    &mut sums[ir * lds + j..]
                };
                let c = &mut c[ir * w.n + j..];
                match MR.min(m - ir) {
                    6 => tile6(simd, &t, a, b, sums, c),
                    5 => tile5(simd, &t, a, b, sums, c),
                    4 => tile4(simd, &t, a, b, sums, c),
                    3 => tile3(simd, &t, a, b, sums, c),
                    2 => tile2(simd, &t, a, b, sums, c),
                    _ => tile1(simd, &t, a, b, sums, c),
                }
                ir += MR;
            }
        }
        pc += kc;
        if last {
            break;
        }
    }
}

/// What a tile does, besides its operands.
struct Tile<'a> {
    /// Pairs of depths.
    kc: usize,
    /// Pairs between the rows of A.
    lda: usize,
    /// Columns of C written (the panel is zero-padded past them).
    width: usize,
    /// Start from zero rather than from the sums so far.
    first: bool,
    /// Finish: apply the epilogue and write C, rather than the sums.
    last: bool,
    /// Elements between the rows of the sums and of C.
    lds: usize,
    ldc: usize,
    scale: f32,
    col_scale: &'a [f32; NR],
    requant: Option<Requant>,
    /// `width` values.
    bias: Option<&'a [f32]>,
    relu: bool,
}

/// `tileR(simd, t, a, b, sums, c)`: `R` rows of sums from the rows of A in place (row
/// `r` at `r * t.lda`, `t.kc` pairs) and a block of a panel of B, `t.kc` rows of `NR`
/// pairs.
macro_rules! tile_fn {
    ($name:ident, $r:literal) => {
        #[simd]
        #[inline(never)]
        fn $name<S: Simd>(
            simd: S,
            t: &Tile,
            a: &[i32],
            b: &[i32],
            sums: &mut [i32],
            c: &mut [f32],
        ) {
            use fearless_simd::prelude::*;
            const R: usize = $r;
            let kc = t.kc;
            let a_rows: [&[i32]; R] = core::array::from_fn(|r| &a[r * t.lda..][..kc]);
            let mut acc: [[i32x8<S>; 2]; R] = core::array::from_fn(|r| {
                core::array::from_fn(|v| {
                    if t.first {
                        i32x8::splat(simd, 0)
                    } else {
                        i32x8::from_slice(simd, &sums[r * t.lds + 8 * v..][..8])
                    }
                })
            });
            for (p, row) in b.as_chunks::<NR>().0[..kc].iter().enumerate() {
                let b0: i16x16<S> = i32x8::from_slice(simd, &row[..8]).bitcast();
                let b1: i16x16<S> = i32x8::from_slice(simd, &row[8..]).bitcast();
                for r in 0..R {
                    // A float broadcast: from memory, it takes no vector ALU slot on Zen 3
                    // (an integer one does), which is worth 4%.
                    let av: i16x16<S> =
                        f32x8::splat(simd, f32::from_bits(a_rows[r][p] as u32)).bitcast();
                    acc[r][0] = acc[r][0] + madd(simd, av, b0);
                    acc[r][1] = acc[r][1] + madd(simd, av, b1);
                }
            }
            if t.last {
                finish(simd, t, acc, c);
            } else {
                for r in 0..R {
                    for v in 0..2 {
                        acc[r][v].store_slice(&mut sums[r * t.lds + 8 * v..][..8]);
                    }
                }
            }
        }
    };
}

tile_fn!(tile6, 6);
tile_fn!(tile5, 5);
tile_fn!(tile4, 4);
tile_fn!(tile3, 3);
tile_fn!(tile2, 2);
tile_fn!(tile1, 1);

/// Applies the epilogue to `R` rows of sums and writes them to C, the first `t.width` columns.
#[inline(always)]
fn finish<S: Simd, const R: usize>(simd: S, t: &Tile, acc: [[i32x8<S>; 2]; R], c: &mut [f32]) {
    use fearless_simd::prelude::*;
    let scale = f32x8::splat(simd, t.scale);
    let col_scale: [f32x8<S>; 2] =
        core::array::from_fn(|v| scale * f32x8::from_slice(simd, &t.col_scale[8 * v..][..8]));
    let mut bias = [0.0f32; NR];
    if let Some(b) = t.bias {
        bias[..t.width].copy_from_slice(b);
    }
    let bias: [f32x8<S>; 2] =
        core::array::from_fn(|v| f32x8::from_slice(simd, &bias[8 * v..][..8]));
    let zero = f32x8::splat(simd, 0.0);
    for r in 0..R {
        let vals: [f32x8<S>; 2] = core::array::from_fn(|v| {
            let x = simd.cvt_f32_i32x8(acc[r][v]) * col_scale[v];
            let x = requant!(simd, f32x8, t.requant, x) + bias[v];
            if t.relu { x.max(zero) } else { x }
        });
        let row = &mut c[r * t.ldc..][..t.width];
        if t.width == NR {
            for v in 0..2 {
                vals[v].store_slice(&mut row[8 * v..][..8]);
            }
        } else {
            let mut flat = [0.0f32; NR];
            for v in 0..2 {
                vals[v].store_slice(&mut flat[8 * v..][..8]);
            }
            row.copy_from_slice(&flat[..t.width]);
        }
    }
}

/// Multiplies 16-bit lanes and adds adjacent products into 32-bit lanes:
/// `out[i] = a[2i] * b[2i] + a[2i + 1] * b[2i + 1]`, exactly.
///
/// `fearless_simd` has no such operation, so each level supplies its own instruction.
#[inline(always)]
fn madd<S: Simd>(simd: S, a: i16x16<S>, b: i16x16<S>) -> i32x8<S> {
    match simd.level() {
        #[cfg(target_arch = "x86_64")]
        Level::Avx512(_) | Level::Avx2(_) => {
            use core::arch::x86_64::{__m256i, _mm256_madd_epi16};
            use fearless_simd::SimdInto;
            let (a, b): (__m256i, __m256i) = (a.into(), b.into());
            // SAFETY: both levels guarantee AVX2.
            unsafe { _mm256_madd_epi16(a, b) }.simd_into(simd)
        }
        #[cfg(target_arch = "x86_64")]
        Level::Sse4_2(_) | Level::Sse2(_) => {
            use core::arch::x86_64::{__m128i, _mm_madd_epi16};
            use fearless_simd::SimdInto;
            let (a0, a1) = simd.split_i16x16(a);
            let (b0, b1) = simd.split_i16x16(b);
            let half = |a: __m128i, b: __m128i| {
                // SAFETY: both levels guarantee SSE2.
                unsafe { _mm_madd_epi16(a, b) }.simd_into(simd)
            };
            simd.combine_i32x4(half(a0.into(), b0.into()), half(a1.into(), b1.into()))
        }
        #[cfg(target_arch = "aarch64")]
        Level::Neon(_) => {
            use core::arch::aarch64::{
                int16x8_t, int32x4_t, vget_low_s16, vmull_high_s16, vmull_s16, vpaddq_s32,
            };
            use fearless_simd::SimdInto;
            let (a0, a1) = simd.split_i16x16(a);
            let (b0, b1) = simd.split_i16x16(b);
            let half = |a: int16x8_t, b: int16x8_t| -> int32x4_t {
                // SAFETY: the level guarantees Neon.
                unsafe {
                    let lo = vmull_s16(vget_low_s16(a), vget_low_s16(b));
                    vpaddq_s32(lo, vmull_high_s16(a, b))
                }
            };
            simd.combine_i32x4(
                half(a0.into(), b0.into()).simd_into(simd),
                half(a1.into(), b1.into()).simd_into(simd),
            )
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Level::WasmSimd128(_) => {
            use core::arch::wasm32::i32x4_dot_i16x8;
            use fearless_simd::SimdInto;
            let (a0, a1) = simd.split_i16x16(a);
            let (b0, b1) = simd.split_i16x16(b);
            let half = |a, b| i32x4_dot_i16x8(a, b).simd_into(simd);
            simd.combine_i32x4(half(a0.into(), b0.into()), half(a1.into(), b1.into()))
        }
        #[allow(unreachable_patterns)]
        _ => {
            use fearless_simd::prelude::*;
            let (a, b) = (a.as_slice(), b.as_slice());
            let out: [i32; 8] = core::array::from_fn(|i| {
                a[2 * i] as i32 * b[2 * i] as i32 + a[2 * i + 1] as i32 * b[2 * i + 1] as i32
            });
            i32x8::from_slice(simd, &out)
        }
    }
}

/// `clamp(round_ties_even(x / scale) + zero_point, 0, 255)`: ONNX `QuantizeLinear` to `u8`.
#[inline(always)]
fn quantize_u8(x: f32, scale: f32, zero_point: i32) -> i32 {
    let q = (x / scale)
        .round_ties_even()
        .clamp(i32::MIN as f32, i32::MAX as f32) as i32;
    q.saturating_add(zero_point).clamp(0, 255)
}

/// Packs the row-major `k x n` activation `b`, quantized to `u8` and centered on
/// `zero_point`, into the pairs layout ([`Packed::Pairs`]): rows `2p` and `2p + 1` of B
/// become pair row `p` of every panel. `data` must be zeroed (the padding past `k` and `n`).
#[simd]
fn pack_grid_pairs_simd<S: Simd>(
    simd: S,
    b: &[f32],
    k: usize,
    n: usize,
    scale: f32,
    zero_point: i32,
    data: &mut [i32],
) {
    use fearless_simd::prelude::*;
    let pairs = k.div_ceil(2);
    let sv = f32x8::splat(simd, scale);
    let zv = i32x8::splat(simd, zero_point);
    let (lo, hi) = (i32x8::splat(simd, 0), i32x8::splat(simd, 255));
    let low_half = i32x8::splat(simd, 0xffff);
    // Centered codes of 8 values of B.
    macro_rules! codes {
        ($x:expr) => {{
            let q = simd.cvt_i32_f32x8((f32x8::from_slice(simd, $x) / sv).round_ties_even()) + zv;
            simd.min_i32x8(simd.max_i32x8(q, lo), hi) - zv
        }};
    }
    let whole = n / 8 * 8;
    for p in 0..pairs {
        let even = &b[2 * p * n..][..n];
        let odd = (2 * p + 1 < k).then(|| &b[(2 * p + 1) * n..][..n]);
        for j in (0..whole).step_by(8) {
            let q0 = codes!(&even[j..][..8]);
            let q1 = match odd {
                Some(odd) => codes!(&odd[j..][..8]),
                None => lo,
            };
            let pair = simd.or_i32x8(simd.and_i32x8(q0, low_half), simd.shl_i32x8(q1, 16));
            pair.store_slice(&mut data[((j / NR) * pairs + p) * NR + j % NR..][..8]);
        }
        for j in whole..n {
            let code = |x: f32| quantize_u8(x, scale, zero_point) - zero_point;
            let hi = odd.map_or(0, |odd| code(odd[j]));
            data[((j / NR) * pairs + p) * NR + j % NR] = pack_pair(code(even[j]), hi);
        }
    }
}

/// Packs the row-major `k x n` activation `b`, quantized to `u8`, for `dot`, as
/// [`qgemm_dot::pack`] packs codes shifted by 128 into `i8`: rows `4q..4q + 4` of B become
/// quad row `q` of every panel.
#[simd]
fn pack_grid_quads_simd<S: Simd>(
    simd: S,
    dot: Dot,
    b: &[f32],
    k: usize,
    n: usize,
    scale: f32,
    zero_point: i32,
) -> QuadWeights {
    use fearless_simd::prelude::*;
    let (nr, quads) = (dot.nr(), k.div_ceil(4));
    let mut data = AlignedI32::zeroed(n.div_ceil(nr) * quads * nr);
    let padded = n.div_ceil(qgemm_dot::NR) * qgemm_dot::NR;
    let (mut col_sums, mut dot_bias) = (vec![0; padded], vec![0; padded]);
    let sv = f32x8::splat(simd, scale);
    let zv = i32x8::splat(simd, zero_point);
    let (lo, hi) = (i32x8::splat(simd, 0), i32x8::splat(simd, 255));
    let (shift, byte) = (i32x8::splat(simd, 128), i32x8::splat(simd, 0xff));
    // Codes of 8 values of B.
    macro_rules! codes {
        ($x:expr) => {{
            let q = simd.cvt_i32_f32x8((f32x8::from_slice(simd, $x) / sv).round_ties_even()) + zv;
            simd.min_i32x8(simd.max_i32x8(q, lo), hi)
        }};
    }
    let whole = n / 8 * 8;
    for q in 0..quads {
        let rows: [Option<&[f32]>; 4] =
            core::array::from_fn(|t| (4 * q + t < k).then(|| &b[(4 * q + t) * n..][..n]));
        for j in (0..whole).step_by(8) {
            let (mut word, mut sum, mut shifted) = (lo, lo, lo);
            for (t, row) in rows.iter().enumerate() {
                if let Some(row) = row {
                    let c = codes!(&row[j..][..8]);
                    let s = c - shift;
                    word =
                        simd.or_i32x8(word, simd.shl_i32x8(simd.and_i32x8(s, byte), 8 * t as u32));
                    sum = sum + c - zv;
                    shifted += s;
                }
            }
            word.store_slice(&mut data[((j / nr) * quads + q) * nr + j % nr..][..8]);
            let sums = &mut col_sums[j..][..8];
            (i32x8::from_slice(simd, sums) + sum).store_slice(sums);
            if dot == Dot::Sdot {
                let bias = &mut dot_bias[j..][..8];
                // `128 * shifted`.
                (i32x8::from_slice(simd, bias) - simd.shl_i32x8(shifted, 7)).store_slice(bias);
            }
        }
        for j in whole..n {
            let mut word = 0;
            for (t, row) in rows.iter().enumerate() {
                if let Some(row) = row {
                    let c = quantize_u8(row[j], scale, zero_point);
                    word |= ((c - 128) as i8 as u8 as i32) << (8 * t);
                    col_sums[j] += c - zero_point;
                    if dot == Dot::Sdot {
                        dot_bias[j] -= 128 * (c - 128);
                    }
                }
            }
            data[((j / nr) * quads + q) * nr + j % nr] = word;
        }
    }
    let mut row_sum_factor = vec![0; padded];
    row_sum_factor[..n].fill(128 - zero_point);
    QuadWeights {
        dot,
        quads,
        data,
        row_term: n > 0 && zero_point != 128,
        row_sum_factor,
        col_sums,
        dot_bias,
    }
}

/// Writes `m` rows of `k` activations, row-major in `src`, as rows of A for [`qgemm_at`]:
/// each value quantized as `ONNX QuantizeLinear` does to `u8`
/// (`clamp(round_ties_even(x / scale) + zero_point, 0, 255)`), less `center` (the zero
/// point of the codes), two depths to a pair. `dst` gets `m * k.div_ceil(2)` pairs; an odd
/// last depth is paired with zero.
///
/// Activations that are codes already take `scale` 1 and `zero_point` 0.
pub(crate) fn quantize_rows_at(
    level: Level,
    src: &[f32],
    m: usize,
    k: usize,
    scale: f32,
    zero_point: i32,
    center: i32,
    dst: &mut AVec<i32>,
) {
    assert!(src.len() >= m * k);
    // SAFETY: every pair is written, so the buffer is not filled first.
    unsafe { crate::kernels::utils::ensure_scratch(dst, m * k.div_ceil(2)) };
    simd_call!(
        level,
        quantize_rows_simd(src, m, k, scale, zero_point, center, dst)
    )
}

#[simd]
fn quantize_rows_simd<S: Simd>(
    simd: S,
    src: &[f32],
    m: usize,
    k: usize,
    scale: f32,
    zero_point: i32,
    center: i32,
    dst: &mut [i32],
) {
    use fearless_simd::prelude::*;
    let pairs = k.div_ceil(2);
    let code = |x: f32| {
        let q = (x / scale)
            .round_ties_even()
            .clamp(i32::MIN as f32, i32::MAX as f32) as i32;
        q.saturating_add(zero_point).clamp(0, 255) - center
    };
    let sv = f32x8::splat(simd, scale);
    let (zv, cv) = (i32x8::splat(simd, zero_point), i32x8::splat(simd, center));
    let (lo, hi) = (i32x8::splat(simd, 0), i32x8::splat(simd, 255));
    let vcode = |x: f32x8<S>| {
        let q = simd.cvt_i32_f32x8((x / sv).round_ties_even()) + zv;
        simd.min_i32x8(simd.max_i32x8(q, lo), hi) - cv
    };
    for r in 0..m {
        let row = &src[r * k..][..k];
        let out = &mut dst[r * pairs..][..pairs];
        let (chunks, rest) = row.as_chunks::<16>();
        for (x, out) in chunks.iter().zip(out.as_chunks_mut::<8>().0) {
            let q0 = vcode(f32x8::from_slice(simd, &x[..8]));
            let q1 = vcode(f32x8::from_slice(simd, &x[8..]));
            let packed: i32x8<S> = simd.narrow_i32x8(q0, q1).bitcast();
            packed.store_slice(out);
        }
        let done = chunks.len() * 8;
        for (i, out) in out[done..].iter_mut().enumerate() {
            let lo = code(rest[2 * i]);
            let hi = rest.get(2 * i + 1).map_or(0, |&x| code(x));
            *out = pack_pair(lo, hi);
        }
    }
}

/// The scale and zero point `ONNX DynamicQuantizeLinear` picks for `x`: the range of
/// `x`, widened to include zero, spread over the 256 codes of `u8`.
pub fn dynamic_quant_params(x: &[f32]) -> (f32, i32) {
    dynamic_quant_params_at(Level::new(), x)
}

pub(crate) fn dynamic_quant_params_at(level: Level, x: &[f32]) -> (f32, i32) {
    let (min, max) = simd_call!(level, min_max_simd(x));
    let (min, max) = (min.min(0.0), max.max(0.0));
    let scale = (max - min).max(1e-5) / 255.0;
    let zero_point = (-min / scale).round_ties_even().clamp(0.0, 255.0) as i32;
    (scale, zero_point)
}

/// `relu?(dequantize(DynamicQuantizeLinear(x)) * dequantize(w) + bias)`: ONNX's
/// dynamically quantized linear layer. The last axis of `x` is the depth; the output keeps
/// the other axes and has `n` for the last.
pub fn qlinear_dynamic<'a>(
    x: &TensorView<'_, f32>,
    w: &QWeights,
    bias: Option<&[f32]>,
    relu: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    let (scale, zero_point) = dynamic_quant_params(&x.data);
    let e = Epilogue {
        scale,
        requant: None,
        bias,
        relu,
    };
    quantized_matmul(x, zero_point, zero_point, w, &e, out)
}

/// `requant?(x * dequantize(w)) + bias`, a static-QDQ linear layer, for an `x` that goes
/// through a `QuantizeLinear` -> `DequantizeLinear` pair of `scale` and `zero_point` (or
/// came out of one: it is quantized either way). With `requant`, the product goes through
/// the pair that follows it, before the bias is added.
pub fn qlinear_static<'a>(
    x: &TensorView<'_, f32>,
    scale: f32,
    zero_point: i32,
    w: &QWeights,
    requant: Option<Requant>,
    bias: Option<&[f32]>,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    let e = Epilogue {
        scale,
        requant,
        bias,
        relu: false,
    };
    quantized_matmul(x, zero_point, zero_point, w, &e, out)
}

/// ONNX `MatMulInteger` against a prepared B (whose scale should be 1): `a` holds `u8`
/// codes, as `f32`.
pub fn mat_mul_integer_qweights<'a>(
    a: &TensorView<'_, f32>,
    a_zero_point: i32,
    w: &QWeights,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    let e = Epilogue {
        scale: 1.0,
        requant: None,
        bias: None,
        relu: false,
    };
    quantized_matmul(a, 0, a_zero_point, w, &e, out)
}

/// ONNX `MatMulInteger` with both operands given at run time as `f32` holding codes (`a`
/// `u8`, `b` `u8` or `i8`), then an optional scale (one, or one per column), bias and
/// ReLU. B is packed on every call, once per batch of it.
pub fn mat_mul_integer_f32<'a>(
    a: &TensorView<'_, f32>,
    b: &TensorView<'_, f32>,
    a_zero_point: i32,
    b_zero_point: &[i32],
    scale: Option<&[f32]>,
    bias: Option<&[f32]>,
    relu: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    let (ad, bd) = (a.shape.len(), b.shape.len());
    assert!(
        ad >= 2 && bd >= 2,
        "MatMulInteger of {:?} and {:?}",
        a.shape,
        b.shape
    );
    let (m, k, n) = (a.shape[ad - 2], a.shape[ad - 1], b.shape[bd - 1]);
    assert_eq!(
        b.shape[bd - 2],
        k,
        "MatMulInteger of {:?} and {:?}",
        a.shape,
        b.shape
    );
    let batch_a: usize = a.shape[..ad - 2].iter().product();
    let batch_b: usize = b.shape[..bd - 2].iter().product();
    let batch = batch_a.max(batch_b);
    // SAFETY: every batch of C is written whole below before any is read.
    unsafe { crate::kernels::utils::ensure_capacity(out, batch * m * n) };
    let scale = scale.unwrap_or(&[1.0]);
    let e = Epilogue {
        scale: 1.0,
        requant: None,
        bias,
        relu,
    };
    let level = Level::new();
    if batch_b == 1 {
        // One B for every batch of A: all their rows in one product.
        let w = QWeights::from_f32_codes(&b.data[..k * n], k, n, b_zero_point, scale);
        run_at(
            level,
            &w,
            &a.data,
            batch_a * m,
            1.0,
            0,
            a_zero_point,
            &e,
            out,
        );
    } else {
        for i in 0..batch {
            let w =
                QWeights::from_f32_codes(&b.data[i * k * n..][..k * n], k, n, b_zero_point, scale);
            let a = &a.data[if batch_a == 1 { 0 } else { i * m * k }..][..m * k];
            run_at(
                level,
                &w,
                a,
                m,
                1.0,
                0,
                a_zero_point,
                &e,
                &mut out[i * m * n..][..m * n],
            );
        }
    }
    let mut shape = if batch_a >= batch_b {
        a.shape[..ad - 2].to_vec()
    } else {
        b.shape[..bd - 2].to_vec()
    };
    shape.extend([m, n]);
    TensorView::from_slice(out, shape)
}

/// `a * b` for two activations that each came out of a `u8` `QuantizeLinear` ->
/// `DequantizeLinear` pair (scalar scales and zero points), as static-QDQ attention
/// multiplies its queries by its keys and its probabilities by its values: both are
/// quantized back to their codes and multiplied in integers, then scaled once by
/// `a_scale * b_scale`.
///
/// `a` and `b` need not be on their grids: they are quantized as their `QuantizeLinear`s
/// would. With `requant`, the product goes through the `QuantizeLinear` ->
/// `DequantizeLinear` pair that follows it.
///
/// `a` is `[..., m, k]` and `b` `[..., k, n]`, their batch axes equal or either one
/// batch; B is packed once per batch. Other broadcasts multiply the dequantized values in
/// `f32`.
pub fn qmatmul_acts<'a>(
    a: &TensorView<'_, f32>,
    a_scale: f32,
    a_zero_point: i32,
    b: &TensorView<'_, f32>,
    b_scale: f32,
    b_zero_point: i32,
    requant: Option<Requant>,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    let (ad, bd) = (a.shape.len(), b.shape.len());
    let (batch_a, batch_b) = (
        &a.shape[..ad.saturating_sub(2)],
        &b.shape[..bd.saturating_sub(2)],
    );
    let (count_a, count_b) = (
        batch_a.iter().product::<usize>(),
        batch_b.iter().product::<usize>(),
    );
    let simple = batch_a == batch_b || count_a == 1 || count_b == 1;
    if ad < 2 || bd < 2 || count_a == 0 || count_b == 0 || !simple {
        let on_grid = |x: &TensorView<'_, f32>, scale, zero_point| {
            let r = Requant { scale, zero_point };
            let data: Vec<f32> = x.data.iter().map(|&v| r.apply(v)).collect();
            TensorView::from_owned(data, x.shape.to_vec())
        };
        let (a, b) = (
            on_grid(a, a_scale, a_zero_point),
            on_grid(b, b_scale, b_zero_point),
        );
        let shape = crate::kernels::matmul(&a, &b, out).shape.to_vec();
        if let Some(r) = requant {
            let len = shape.iter().product();
            out[..len].iter_mut().for_each(|v| *v = r.apply(*v));
        }
        return TensorView::from_slice(out, shape);
    }
    let (m, k, n) = (a.shape[ad - 2], a.shape[ad - 1], b.shape[bd - 1]);
    assert_eq!(
        b.shape[bd - 2],
        k,
        "MatMul of {:?} and {:?}",
        a.shape,
        b.shape
    );
    let batch = count_a.max(count_b);
    // SAFETY: every batch of C is written whole below before any is read.
    unsafe { crate::kernels::utils::ensure_capacity(out, batch * m * n) };
    let level = Level::new();
    let layout = Layout::for_level(level);
    let e = Epilogue {
        scale: a_scale,
        requant,
        bias: None,
        relu: false,
    };
    let pack_b = |b: &[f32]| QWeights::from_grid(level, layout, b, k, n, b_scale, b_zero_point);
    let product = |w: &QWeights, a: &[f32], rows: usize, c: &mut [f32]| {
        run_at(
            level,
            w,
            a,
            rows,
            a_scale,
            a_zero_point,
            a_zero_point,
            &e,
            c,
        )
    };
    if count_b == 1 {
        // One B for every batch of A: all their rows in one product.
        let w = pack_b(&b.data[..k * n]);
        product(&w, &a.data, count_a * m, &mut out[..batch * m * n]);
    } else {
        for i in 0..batch {
            let w = pack_b(&b.data[i * k * n..][..k * n]);
            let a = &a.data[if count_a == 1 { 0 } else { i * m * k }..][..m * k];
            product(&w, a, m, &mut out[i * m * n..][..m * n]);
        }
    }
    // The batch axes broadcast: one side's are all 1 or both are the same.
    let rank = batch_a.len().max(batch_b.len());
    let dim = |s: &[usize], i: usize| (i + s.len()).checked_sub(rank).map_or(1, |i| s[i]);
    let mut shape: Vec<usize> = (0..rank)
        .map(|i| dim(batch_a, i).max(dim(batch_b, i)))
        .collect();
    shape.extend([m, n]);
    TensorView::from_slice(out, shape)
}

thread_local! {
    /// Rows of A, quantized, and (for the quads) their sums.
    static A_ROWS: std::cell::RefCell<AVec<i32>> = std::cell::RefCell::new(AVec::new(0));
    static ROW_SUMS: std::cell::RefCell<AVec<i32>> = std::cell::RefCell::new(AVec::new(0));
    /// Rows of A, quantized and centered, in `f32`.
    static A_FLOAT: std::cell::RefCell<AVec<f32>> = std::cell::RefCell::new(AVec::new(0));
}

/// `C = epilogue(A * w)` for the `m x k` activations `src`, quantized with `scale` and
/// `zero_point` and taken less `center` (as [`quantize_rows_at`] does), in whichever layout
/// `w` is packed.
pub(crate) fn run_at(
    level: Level,
    w: &QWeights,
    src: &[f32],
    m: usize,
    scale: f32,
    zero_point: i32,
    center: i32,
    e: &Epilogue,
    c: &mut [f32],
) {
    // A sequence of length 0, as the first step of a model fed its prompt in two parts
    // is, has no rows to multiply.
    if m == 0 {
        return;
    }
    A_ROWS.with_borrow_mut(|a| match &w.packed {
        Packed::Pairs { pairs, .. } => {
            quantize_rows_at(level, src, m, w.k, scale, zero_point, center, a);
            qgemm_at(level, a, m, *pairs, w, e, c);
        }
        Packed::Dot(q) => ROW_SUMS.with_borrow_mut(|sums| {
            qgemm_dot::quantize_rows(level, q.dot, src, m, w.k, scale, zero_point, a, sums);
            qgemm_dot::qgemm(level, a, sums, m, q.quads, center, q, w.n, &w.scale, e, c);
        }),
        Packed::Float(b) => A_FLOAT.with_borrow_mut(|a| {
            let (k, n) = (w.k, w.n);
            // SAFETY: every element is written below before any is read.
            unsafe { crate::kernels::utils::ensure_scratch(a, m * k) };
            simd_call!(
                level,
                quantize_rows_f32_simd(src, m * k, scale, zero_point, center, a)
            );
            sgemm(&a[..m * k], b, m, k, n, &mut c[..m * n]);
            simd_call!(level, epilogue_simd(m, n, &w.scale, e, c));
        }),
    })
}

/// `dst = clamp(round_ties_even(x / scale) + zero_point, 0, 255) - center`, in `f32`.
#[simd]
fn quantize_rows_f32_simd<S: Simd>(
    simd: S,
    src: &[f32],
    len: usize,
    scale: f32,
    zero_point: i32,
    center: i32,
    dst: &mut [f32],
) {
    use fearless_simd::prelude::*;
    let (src, dst) = (&src[..len], &mut dst[..len]);
    let code = |x: f32| {
        let q = (x / scale)
            .round_ties_even()
            .clamp(i32::MIN as f32, i32::MAX as f32) as i32;
        (q.saturating_add(zero_point).clamp(0, 255) - center) as f32
    };
    let sv = f32x8::splat(simd, scale);
    let (lo, hi) = (
        f32x8::splat(simd, -zero_point as f32),
        f32x8::splat(simd, (255 - zero_point) as f32),
    );
    let shift = f32x8::splat(simd, (zero_point - center) as f32);
    let (chunks, rest) = src.as_chunks::<8>();
    let (out, out_rest) = dst.as_chunks_mut::<8>();
    for (x, o) in chunks.iter().zip(out) {
        // Clamping the rounded quotient before adding the zero point keeps it exact in f32.
        let q = (f32x8::from_slice(simd, x) / sv)
            .round_ties_even()
            .max(lo)
            .min(hi);
        (q + shift).store_slice(o);
    }
    for (x, o) in rest.iter().zip(out_rest) {
        *o = code(*x);
    }
}

/// `c = a * b` for row-major `m x k` and `k x n` operands.
fn sgemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, c: &mut [f32]) {
    use crate::kernels::matmul::{Accum, MatMut, MatRef, Par, matmul};
    assert!(a.len() >= m * k && b.len() >= k * n && c.len() >= m * n);
    // SAFETY: the slices hold the row-major operands of the sizes given.
    unsafe {
        let a = MatRef::<f32>::from_raw_parts(a.as_ptr(), m, k, k as isize, 1);
        let b = MatRef::<f32>::from_raw_parts(b.as_ptr(), k, n, n as isize, 1);
        let c = MatMut::<f32>::from_raw_parts_mut(c.as_mut_ptr(), m, n, n as isize, 1);
        matmul(c, Accum::Replace, a, b, 1.0, Par::Seq);
    }
}

/// `c[i][j] = relu?(requant?(c[i][j] * e.scale * col_scale[j]) + bias[j])`, in place.
#[simd]
fn epilogue_simd<S: Simd>(
    simd: S,
    m: usize,
    n: usize,
    col_scale: &[f32],
    e: &Epilogue,
    c: &mut [f32],
) {
    use fearless_simd::prelude::*;
    let zero = f32x8::splat(simd, 0.0);
    let sa = f32x8::splat(simd, e.scale);
    for row in c[..m * n].chunks_exact_mut(n) {
        let (chunks, rest) = row.as_chunks_mut::<8>();
        for (i, o) in chunks.iter_mut().enumerate() {
            let s = f32x8::from_slice(simd, &col_scale[8 * i..][..8]) * sa;
            let b = e
                .bias
                .map_or(zero, |b| f32x8::from_slice(simd, &b[8 * i..][..8]));
            let x = f32x8::from_slice(simd, o) * s;
            let x = requant!(simd, f32x8, e.requant, x) + b;
            (if e.relu { x.max(zero) } else { x }).store_slice(o);
        }
        let done = chunks.len() * 8;
        for (j, o) in rest.iter_mut().enumerate() {
            let x = *o * (e.scale * col_scale[done + j]);
            let x = e.requant.map_or(x, |r| r.apply(x)) + e.bias.map_or(0.0, |b| b[done + j]);
            *o = if e.relu { x.max(0.0) } else { x };
        }
    }
}

/// Quantizes `x` with `e.scale`, `zero_point` and `center` (as [`quantize_rows_at`]) and
/// multiplies it by `w`, with `e.scale` the scale of the result's A.
fn quantized_matmul<'a>(
    x: &TensorView<'_, f32>,
    zero_point: i32,
    center: i32,
    w: &QWeights,
    e: &Epilogue,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    let k = x.shape.last().copied().unwrap_or(1);
    assert_eq!(
        k, w.k,
        "depth of {:?} against a {}x{} weight",
        x.shape, w.k, w.n
    );
    let m: usize = x.shape[..x.shape.len().saturating_sub(1)].iter().product();
    // SAFETY: C is written whole, so the buffer is not filled first.
    unsafe { crate::kernels::utils::ensure_capacity(out, m * w.n) };
    run_at(
        Level::new(),
        w,
        &x.data,
        m,
        e.scale,
        zero_point,
        center,
        e,
        out,
    );
    let mut shape = x.shape.to_vec();
    match shape.last_mut() {
        Some(last) => *last = w.n,
        None => shape.push(w.n),
    }
    TensorView::from_slice(out, shape)
}

#[simd]
fn min_max_simd<S: Simd>(simd: S, x: &[f32]) -> (f32, f32) {
    use fearless_simd::prelude::*;
    let (chunks, rest) = x.as_chunks::<8>();
    let mut lo = f32x8::splat(simd, f32::MAX);
    let mut hi = f32x8::splat(simd, f32::MIN);
    for c in chunks {
        let v = f32x8::from_slice(simd, c);
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let lo = rest.iter().fold(
        lo.as_slice().iter().copied().fold(f32::MAX, f32::min),
        |a, &b| a.min(b),
    );
    let hi = rest.iter().fold(
        hi.as_slice().iter().copied().fold(f32::MIN, f32::max),
        |a, &b| a.max(b),
    );
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::test_util::{Rng, assert_close, assert_same_bits, levels};

    fn codes(rng: &mut Rng, len: usize) -> Vec<u8> {
        (0..len).map(|_| rng.f32(0.0, 256.0) as u8).collect()
    }

    /// Unpacked activation codes, centered: `m x k`.
    fn unpack_rows(a: &[i32], m: usize, k: usize) -> Vec<i32> {
        let pairs = k.div_ceil(2);
        let mut out = vec![0; m * k];
        for r in 0..m {
            for kk in 0..k {
                let p = a[r * pairs + kk / 2];
                out[r * k + kk] = if kk % 2 == 0 {
                    p as i16 as i32
                } else {
                    p >> 16
                };
            }
        }
        out
    }

    #[test]
    fn test_qgemm_matches_reference_at_every_level() {
        let mut rng = Rng::new(7);
        let sizes = [
            (1, 1, 1),
            (1, 2, 16),
            (2, 3, 17),
            (3, 7, 5),
            (4, 64, 33),
            (5, 31, 40),
            (6, 16, 16),
            (7, 40, 48),
            (13, 9, 100),
            (1, 600, 70),
            (9, 513, 20),
            (8, 1100, 37),
            (17, 0, 9),
            // Several depth blocks.
            (7, 2100, 20),
            (2, 4500, 33),
        ];
        // Every layout at every level: the dot layouts emulate their instructions elsewhere.
        for (level, layout) in levels()
            .into_iter()
            .flat_map(|l| Layout::ALL.map(|layout| (l, layout)))
        {
            for &(m, k, n) in &sizes {
                for case in 0..4 {
                    let signed = case % 2 == 0;
                    let per_col = case >= 2;
                    let raw = codes(&mut rng, k * n);
                    // Case 0 is QDQ's symmetric i8 (zero point 0: the quads skip their row term).
                    let zp: Vec<i32> = (0..if per_col { n } else { 1 })
                        .map(|_| match case {
                            0 => 0,
                            _ if signed => rng.f32(-20.0, 20.0) as i32,
                            _ => rng.f32(0.0, 256.0) as i32,
                        })
                        .collect();
                    let ws: Vec<f32> = (0..if per_col { n } else { 1 })
                        .map(|_| rng.f32(0.001, 0.1))
                        .collect();
                    let w = QWeights::with_layout(layout, &raw, k, n, signed, &zp, &ws);
                    assert_eq!(w.layout(), layout);
                    let x = rng.vec(m * k, -3.0, 3.0);
                    let (sa, za) = (rng.f32(0.01, 0.05), rng.f32(0.0, 256.0) as i32);
                    let bias = rng.vec(n, -1.0, 1.0);
                    let relu = case == 1;
                    // Case 2 snaps the products onto a grid about as fine as their spread
                    // (but not where the f32 GEMM rounds the sums: a rounding could cross a
                    // step).
                    let requant =
                        (case == 2 && !(layout == Layout::Float && k > 258)).then(|| Requant {
                            scale: sa * 0.05 * 128.0 * (k as f32).sqrt().max(1.0) / 40.0,
                            zero_point: 120,
                        });
                    let e = Epilogue {
                        scale: sa,
                        requant,
                        bias: (case != 3).then_some(&bias[..]),
                        relu,
                    };
                    let mut c = vec![f32::NAN; m * n];
                    run_at(level, &w, &x, m, sa, za, za, &e, &mut c);

                    let ac: Vec<i32> = x
                        .iter()
                        .map(|&v| ((v / sa).round_ties_even() as i32 + za).clamp(0, 255) - za)
                        .collect();
                    let mut want = vec![0.0f32; m * n];
                    for i in 0..m {
                        for j in 0..n {
                            let zb = zp[if per_col { j } else { 0 }];
                            let sum: i64 = (0..k)
                                .map(|kk| {
                                    let b = raw[kk * n + j];
                                    let b = if signed { b as i8 as i32 } else { b as i32 } - zb;
                                    ac[i * k + kk] as i64 * b as i64
                                })
                                .sum();
                            let s = sa * ws[if per_col { j } else { 0 }];
                            let mut v = sum as i32 as f32 * s;
                            if let Some(r) = requant {
                                let z = r.zero_point as f32;
                                let q = ((v / r.scale).round_ties_even() + z).clamp(0.0, 255.0);
                                v = (q - z) * r.scale;
                            }
                            v += e.bias.map_or(0.0, |b| b[j]);
                            if relu {
                                v = v.max(0.0);
                            }
                            want[i * n + j] = v;
                        }
                    }
                    let what = format!("{level:?} {layout:?} {m}x{k}x{n} case {case}");
                    if layout == Layout::Float && k > 258 {
                        // Sums past 2^24 are rounded by the f32 GEMM.
                        let scale = want.iter().fold(1.0f32, |a, v| a.max(v.abs()));
                        for (g, w) in c.iter().zip(&want) {
                            assert!((g - w).abs() <= 1e-5 * scale, "{what}: {g} vs {w}");
                        }
                    } else {
                        assert_same_bits(&c, &want, &what);
                    }
                }
            }
        }
    }

    #[test]
    fn test_quantize_rows_matches_reference_at_every_level() {
        let mut rng = Rng::new(11);
        for level in levels() {
            for &(m, k) in &[(1, 1), (1, 16), (2, 17), (3, 33), (4, 100)] {
                // The last: codes already, less their zero point.
                for &(scale, zp, center) in &[
                    (0.02f32, 128, 128),
                    (0.01, 0, 0),
                    (0.05, 255, 255),
                    (1.0, 3, 3),
                    (1.0, 0, 179),
                ] {
                    // On the grid (as QDQ activations are), off it, and out of range.
                    let mut x = rng.vec(m * k, -8.0, 8.0);
                    for (i, v) in x.iter_mut().enumerate().step_by(3) {
                        *v = scale * ((i % 300) as f32 - 20.0 - zp as f32);
                    }
                    let mut got = AVec::new(0);
                    quantize_rows_at(level, &x, m, k, scale, zp, center, &mut got);
                    let got = unpack_rows(&got, m, k);
                    for (i, (&g, &v)) in got.iter().zip(&x).enumerate() {
                        let want =
                            ((v / scale).round_ties_even() as i32 + zp).clamp(0, 255) - center;
                        assert_eq!(
                            g, want,
                            "{level:?} {m}x{k} scale {scale} zp {zp} center {center} at {i}: {v}"
                        );
                    }
                }
            }
        }
    }

    /// An input of no rows gives an output of no rows.
    #[test]
    fn test_quantized_matmul_of_no_rows_is_empty() {
        let mut rng = Rng::new(3);
        let (k, n) = (37, 21);
        let raw = codes(&mut rng, k * n);
        let w = QWeights::new(&raw, k, n, false, &[131], &[1.0]);
        let mut out = Vec::new();
        let x = TensorView::from_slice(&[], vec![1, 0, k]);
        let got = mat_mul_integer_qweights(&x, 179, &w, &mut out);
        assert_eq!(got.shape.as_ref(), &[1, 0, n]);
        assert!(got.data.is_empty());
        let ws = vec![0.01f32; n];
        let w = QWeights::new(&raw, k, n, true, &[0], &ws);
        let got = qlinear_static(&x, 0.03, 100, &w, None, None, &mut out);
        assert_eq!(got.shape.as_ref(), &[1, 0, n]);
    }

    /// The ONNX-level entry points, against the ONNX definitions in f64: what each one
    /// quantizes, and with which zero point, is exactly what the kernel tests take as given.
    #[test]
    fn test_onnx_entry_points_match_their_definitions() {
        let mut rng = Rng::new(13);
        let (batch, m, k, n) = (2, 5, 37, 21);
        let raw = codes(&mut rng, k * n);
        let w_i8 = |kk: usize, j: usize| raw[kk * n + j] as i8 as f64;
        let ws: Vec<f32> = (0..n).map(|_| rng.f32(0.001, 0.01)).collect();
        let bias = rng.vec(n, -1.0, 1.0);
        let matmul =
            |a: &dyn Fn(usize, usize) -> f64, b: &dyn Fn(usize, usize) -> f64| -> Vec<f64> {
                let mut out = vec![0.0; batch * m * n];
                for i in 0..batch * m {
                    for j in 0..n {
                        out[i * n + j] = (0..k).map(|kk| a(i, kk) * b(kk, j)).sum();
                    }
                }
                out
            };
        let rows = batch * m;

        // MatMulInteger: codes in, integer sums out; u8 B with a zero point.
        let a_codes: Vec<f32> = (0..rows * k).map(|_| rng.f32(0.0, 256.0).floor()).collect();
        let w = QWeights::new(&raw, k, n, false, &[131], &[1.0]);
        let mut out = Vec::new();
        let got = mat_mul_integer_qweights(
            &TensorView::from_slice(&a_codes, vec![batch, m, k]),
            179,
            &w,
            &mut out,
        );
        assert_eq!(got.shape.as_ref(), &[batch, m, n]);
        let want = matmul(&|i, kk| a_codes[i * k + kk] as f64 - 179.0, &|kk, j| {
            raw[kk * n + j] as f64 - 131.0
        });
        assert_close(&got.data, &want, 0.0, "MatMulInteger");

        // Static QDQ: activations on the grid, i8 B, per-column scales.
        let w = QWeights::new(&raw, k, n, true, &[0], &ws);
        let (sa, za) = (0.03f32, 100);
        let x: Vec<f32> = (0..rows * k)
            .map(|_| sa * (rng.f32(0.0, 256.0).floor() - za as f32))
            .collect();
        let got = qlinear_static(
            &TensorView::from_slice(&x, vec![batch, m, k]),
            sa,
            za,
            &w,
            None,
            None,
            &mut out,
        );
        let want = matmul(&|i, kk| x[i * k + kk] as f64, &|kk, j| {
            w_i8(kk, j) * ws[j] as f64
        });
        assert_close(&got.data, &want, 1e-5, "QDQ");

        // The same, fused with the activation's QuantizeLinear -> DequantizeLinear pair
        // before it, and the product's and a bias after: bit for bit the unfused chain.
        let unfused = got.data.to_vec();
        let r = Requant {
            scale: 0.07,
            zero_point: 131,
        };
        let raw_x: Vec<f32> = x.iter().map(|&v| v + rng.f32(-0.4, 0.4) * sa).collect();
        let got = qlinear_static(
            &TensorView::from_slice(&raw_x, vec![batch, m, k]),
            sa,
            za,
            &w,
            Some(r),
            Some(&bias),
            &mut out,
        );
        let chain: Vec<f32> = unfused
            .iter()
            .enumerate()
            .map(|(i, &v)| r.apply(v) + bias[i % n])
            .collect();
        assert_same_bits(&got.data, &chain, "fused QDQ");

        // Dynamic: anything in, quantized to its own range; bias and ReLU.
        let x = rng.vec(rows * k, -2.0, 3.0);
        let got = qlinear_dynamic(
            &TensorView::from_slice(&x, vec![batch, m, k]),
            &w,
            Some(&bias),
            true,
            &mut out,
        );
        let (s, z) = dynamic_quant_params(&x);
        let dq =
            |v: f32| s as f64 * (((v / s).round_ties_even() as i32 + z).clamp(0, 255) - z) as f64;
        let want: Vec<f64> = matmul(&|i, kk| dq(x[i * k + kk]), &|kk, j| {
            w_i8(kk, j) * ws[j] as f64
        })
        .iter()
        .enumerate()
        .map(|(i, &v)| (v + bias[i % n] as f64).max(0.0))
        .collect();
        assert_close(&got.data, &want, 1e-5, "dynamic");
        // And close to the unquantized product, so the quantization itself is sane.
        let exact: Vec<f64> = matmul(&|i, kk| x[i * k + kk] as f64, &|kk, j| {
            w_i8(kk, j) * ws[j] as f64
        })
        .iter()
        .enumerate()
        .map(|(i, &v)| (v + bias[i % n] as f64).max(0.0))
        .collect();
        assert_close(&got.data, &exact, 0.05, "dynamic vs f32");
    }

    /// B packed from values on (and off) a `u8` grid multiplies exactly as B packed from
    /// their codes, in every layout at every level.
    #[test]
    fn test_weights_from_grid_match_weights_from_codes() {
        let mut rng = Rng::new(17);
        for (level, layout) in levels()
            .into_iter()
            .flat_map(|l| Layout::ALL.map(|layout| (l, layout)))
        {
            for &(m, k, n) in &[
                (1, 1, 1),
                (3, 2, 8),
                (7, 5, 17),
                (6, 64, 400),
                (13, 33, 23),
                (2, 400, 64),
            ] {
                for &(scale, zp) in &[(0.02f32, 128), (0.003, 0), (0.05, 255), (0.01, 37)] {
                    // Mostly on the grid, as QDQ activations are; some off it and out of range.
                    let b: Vec<f32> = (0..k * n)
                        .map(|i| match i % 5 {
                            4 => rng.f32(-8.0, 8.0),
                            _ => scale * (rng.f32(0.0, 256.0).floor() - zp as f32),
                        })
                        .collect();
                    let codes: Vec<u8> =
                        b.iter().map(|&v| quantize_u8(v, scale, zp) as u8).collect();
                    let w = QWeights::from_grid(level, layout, &b, k, n, scale, zp);
                    assert_eq!(w.layout(), layout);
                    let want_w =
                        QWeights::with_layout(layout, &codes, k, n, false, &[zp], &[scale]);
                    let (sa, za) = (rng.f32(0.01, 0.05), rng.f32(0.0, 256.0) as i32);
                    let x: Vec<f32> = (0..m * k)
                        .map(|_| sa * (rng.f32(0.0, 256.0).floor() - za as f32))
                        .collect();
                    let e = Epilogue {
                        scale: sa,
                        requant: None,
                        bias: None,
                        relu: false,
                    };
                    let (mut got, mut want) = (vec![f32::NAN; m * n], vec![f32::NAN; m * n]);
                    run_at(level, &w, &x, m, sa, za, za, &e, &mut got);
                    run_at(level, &want_w, &x, m, sa, za, za, &e, &mut want);
                    let what = format!("{level:?} {layout:?} {m}x{k}x{n} scale {scale} zp {zp}");
                    assert_same_bits(&got, &want, &what);
                }
            }
        }
    }

    /// `qmatmul_acts` against the product of the dequantized operands in f64, over the
    /// batch shapes it multiplies in integers.
    #[test]
    fn test_qmatmul_acts_matches_the_dequantized_product() {
        let mut rng = Rng::new(19);
        let (m, k, n) = (5, 37, 21);
        let ((sa, za), (sb, zb)) = ((0.03f32, 100), (0.02f32, 131));
        let cases: [(&[usize], &[usize], &[usize]); 4] = [
            (&[2, 3], &[2, 3], &[2, 3]),
            (&[1, 3], &[], &[1, 3]),
            (&[4], &[1, 1], &[1, 4]),
            (&[], &[2], &[2]),
        ];
        for (batch_a, batch_b, batch_out) in cases {
            let on_grid = |rng: &mut Rng, len: usize, s: f32, z: i32| -> Vec<f32> {
                (0..len)
                    .map(|_| s * (rng.f32(0.0, 256.0).floor() - z as f32))
                    .collect()
            };
            let (count_a, count_b) = (
                batch_a.iter().product::<usize>(),
                batch_b.iter().product::<usize>(),
            );
            let a = on_grid(&mut rng, count_a * m * k, sa, za);
            let b = on_grid(&mut rng, count_b * k * n, sb, zb);
            let shape = |batch: &[usize], r: usize, c: usize| [batch, &[r, c]].concat();
            let mut out = Vec::new();
            let got = qmatmul_acts(
                &TensorView::from_slice(&a, shape(batch_a, m, k)),
                sa,
                za,
                &TensorView::from_slice(&b, shape(batch_b, k, n)),
                sb,
                zb,
                None,
                &mut out,
            );
            assert_eq!(
                got.shape.as_ref(),
                shape(batch_out, m, n),
                "{batch_a:?} x {batch_b:?}"
            );
            let unfused = got.data.to_vec();
            // Off the grids by less than half a step, and with the product's own pair:
            // bit for bit the chain of the pairs around the product.
            let off = |x: &[f32], s: f32, rng: &mut Rng| -> Vec<f32> {
                x.iter().map(|&v| v + rng.f32(-0.4, 0.4) * s).collect()
            };
            let (raw_a, raw_b) = (off(&a, sa, &mut rng), off(&b, sb, &mut rng));
            let r = Requant {
                scale: 0.3,
                zero_point: 140,
            };
            let mut fused_out = Vec::new();
            let fused = qmatmul_acts(
                &TensorView::from_slice(&raw_a, shape(batch_a, m, k)),
                sa,
                za,
                &TensorView::from_slice(&raw_b, shape(batch_b, k, n)),
                sb,
                zb,
                Some(r),
                &mut fused_out,
            );
            let chain: Vec<f32> = unfused.iter().map(|&v| r.apply(v)).collect();
            assert_same_bits(
                &fused.data,
                &chain,
                &format!("fused {batch_a:?} x {batch_b:?}"),
            );
            // Batch `i` of the output reads batch `i` of a side with as many, or its only one.
            let (rank, count) = (batch_out.len(), batch_out.iter().product::<usize>());
            let index = |batch: &[usize], i: usize| {
                let (mut i, mut idx, mut stride) = (i, 0, 1);
                for d in (0..rank).rev() {
                    let (o, dim) = (
                        batch_out[d],
                        (d + batch.len()).checked_sub(rank).map_or(1, |d| batch[d]),
                    );
                    if dim != 1 {
                        idx += (i % o) * stride;
                    }
                    stride *= dim;
                    i /= o;
                }
                idx
            };
            let mut want = vec![0.0f64; count * m * n];
            for bi in 0..count {
                let (ia, ib) = (index(batch_a, bi), index(batch_b, bi));
                for i in 0..m {
                    for j in 0..n {
                        want[(bi * m + i) * n + j] = (0..k)
                            .map(|kk| {
                                a[(ia * m + i) * k + kk] as f64 * b[(ib * k + kk) * n + j] as f64
                            })
                            .sum();
                    }
                }
            }
            assert_close(
                &got.data,
                &want,
                1e-5,
                &format!("{batch_a:?} x {batch_b:?}"),
            );
        }
    }

    /// The epilogues' round trip is `fake_quantize_linear`'s, bit for bit, so folding the
    /// pair after a product into it changes nothing.
    #[test]
    fn test_requant_is_fake_quantize_linear() {
        let mut rng = Rng::new(23);
        let x: Vec<f32> = (0..1000)
            .map(|i| rng.f32(-30.0, 30.0) * (i % 7) as f32)
            .collect();
        for &(scale, zero_point) in &[(0.07f32, 131), (0.3, 0), (0.011, 255), (1.0, 128)] {
            let r = Requant { scale, zero_point };
            let mut fq = Vec::new();
            let want = crate::kernels::fake_quantize_linear(
                &TensorView::from_slice(&x, vec![x.len()]),
                &TensorView::from_slice(&[scale], vec![]),
                Some(&TensorView::from_slice(&[zero_point as f32], vec![])),
                1,
                0,
                0.0,
                255.0,
                &mut fq,
            );
            let got: Vec<f32> = x.iter().map(|&v| r.apply(v)).collect();
            assert_same_bits(
                &got,
                &want.data,
                &format!("scale {scale} zero point {zero_point}"),
            );
            for level in levels() {
                let mut c = x.clone();
                let e = Epilogue {
                    scale: 1.0,
                    requant: Some(r),
                    bias: None,
                    relu: false,
                };
                simd_call!(
                    level,
                    epilogue_simd(1, x.len(), &vec![1.0; x.len()], &e, &mut c)
                );
                assert_same_bits(&c, &want.data, &format!("{level:?} scale {scale}"));
            }
        }
    }

    #[test]
    fn test_dynamic_quant_params() {
        for level in levels() {
            let x: Vec<f32> = (0..37).map(|i| i as f32 * 0.25 - 2.0).collect();
            let (s, z) = dynamic_quant_params_at(level, &x);
            assert_eq!(s, 9.0 / 255.0);
            assert_eq!(z, (2.0f32 / s).round_ties_even() as i32);
            // All positive: the range is widened down to zero.
            let (s, z) = dynamic_quant_params_at(level, &[1.0, 2.0, 3.0]);
            assert_eq!((s, z), (3.0 / 255.0, 0));
        }
    }
}
