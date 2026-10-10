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

use crate::kernels::simd::simd_call;
use crate::tensor::TensorView;
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

/// Two centered codes in one `i32`: the even depth in the low half, the odd one in the
/// high half, as the lanes of a 16-bit vector bitcast from it are laid out.
#[inline]
fn pack_pair(lo: i32, hi: i32) -> i32 {
    (lo as i16 as u16 as i32) | (hi << 16)
}

/// A `k x n` weight of 8-bit codes, centered and packed for [`qgemm`].
pub struct QWeights {
    k: usize,
    n: usize,
    /// Pairs of depths: `k`, rounded up to even, halved.
    pairs: usize,
    /// Panels of `NR` columns, each `pairs` rows of `NR` packed pairs; zero past `k` and `n`.
    data: Vec<i32>,
    /// The scale of each column, zero past `n` to a whole panel.
    scale: Vec<f32>,
}

impl QWeights {
    /// Packs row-major `k x n` codes, `i8` if `signed` and `u8` otherwise, with one zero
    /// point and one scale for all columns or one of each per column.
    pub fn new(codes: &[u8], k: usize, n: usize, signed: bool, zero_point: &[i32], scale: &[f32]) -> Self {
        assert_eq!(codes.len(), k * n);
        Self::pack(k, n, |i| if signed { codes[i] as i8 as i32 } else { codes[i] as i32 }, zero_point, scale)
    }

    /// As [`QWeights::new`], from codes held in `f32` (as lele keeps integer tensors).
    pub fn from_f32_codes(codes: &[f32], k: usize, n: usize, zero_point: &[i32], scale: &[f32]) -> Self {
        assert_eq!(codes.len(), k * n);
        Self::pack(k, n, |i| codes[i] as i32, zero_point, scale)
    }

    /// `code(i)` is element `i` of the row-major codes.
    fn pack(k: usize, n: usize, code: impl Fn(usize) -> i32, zero_point: &[i32], scale: &[f32]) -> Self {
        assert!(zero_point.len() == 1 || zero_point.len() == n);
        assert!(scale.len() == 1 || scale.len() == n);
        let pairs = k.div_ceil(2);
        let panels = n.div_ceil(NR);
        let zp = |j: usize| zero_point[if zero_point.len() == 1 { 0 } else { j }];
        let code = |kk: usize, j: usize| if kk < k { code(kk * n + j) - zp(j) } else { 0 };
        let mut data = vec![0; panels * pairs * NR];
        for p in 0..pairs {
            for j in 0..n {
                data[((j / NR) * pairs + p) * NR + j % NR] = pack_pair(code(2 * p, j), code(2 * p + 1, j));
            }
        }
        let mut col_scale = vec![0.0; panels * NR];
        for (j, s) in col_scale[..n].iter_mut().enumerate() {
            *s = scale[if scale.len() == 1 { 0 } else { j }];
        }
        Self { k, n, pairs, data, scale: col_scale }
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// Pairs of depths per row of A.
    pub fn pairs(&self) -> usize {
        self.pairs
    }
}

/// What happens to the integer sums on the way out:
/// `C[i][j] = relu?(sum * (scale * column scale of B) + bias[j])`.
pub struct Epilogue<'a> {
    /// The scale of A.
    pub scale: f32,
    /// `n` values to add, if any.
    pub bias: Option<&'a [f32]>,
    pub relu: bool,
}

/// `C = epilogue(A * B)`. A is `m` rows of `w.pairs()` packed pairs, `lda` apart, as
/// [`quantize_rows`] writes them; C is `m x n`, row-major.
pub fn qgemm(a: &[i32], m: usize, lda: usize, w: &QWeights, e: &Epilogue, c: &mut [f32]) {
    qgemm_at(Level::new(), a, m, lda, w, e, c)
}

pub(crate) fn qgemm_at(level: Level, a: &[i32], m: usize, lda: usize, w: &QWeights, e: &Epilogue, c: &mut [f32]) {
    assert!(lda >= w.pairs && a.len() >= m.saturating_sub(1) * lda + w.pairs);
    assert!(c.len() >= m * w.n);
    assert!(e.bias.is_none_or(|b| b.len() >= w.n));
    SUMS.with_borrow_mut(|sums| {
        let sums = if w.pairs > KC {
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
    static SUMS: std::cell::RefCell<Vec<i32>> = const { std::cell::RefCell::new(Vec::new()) };
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
    let depth = w.pairs.div_ceil(w.pairs.div_ceil(KC).max(1));
    let panels = w.n.div_ceil(NR);
    let lds = panels * NR;
    let mut pc = 0;
    loop {
        let kc = depth.min(w.pairs - pc);
        let last = pc + kc == w.pairs;
        for jp in 0..panels {
            let j = jp * NR;
            let b = &w.data[(jp * w.pairs + pc) * NR..][..kc * NR];
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
                bias: e.bias.map(|b| &b[j..w.n.min(j + NR)]),
                relu: e.relu,
            };
            let mut ir = 0;
            while ir < m {
                let a = &a[ir * lda + pc..];
                // (Empty with a single depth block.)
                let sums = if sums.is_empty() { &mut [][..] } else { &mut sums[ir * lds + j..] };
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
        fn $name<S: Simd>(simd: S, t: &Tile, a: &[i32], b: &[i32], sums: &mut [i32], c: &mut [f32]) {
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
                    let av: i16x16<S> = f32x8::splat(simd, f32::from_bits(a_rows[r][p] as u32)).bitcast();
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
    let bias: [f32x8<S>; 2] = core::array::from_fn(|v| f32x8::from_slice(simd, &bias[8 * v..][..8]));
    let zero = f32x8::splat(simd, 0.0);
    for r in 0..R {
        let vals: [f32x8<S>; 2] = core::array::from_fn(|v| {
            let x = simd.cvt_f32_i32x8(acc[r][v]) * col_scale[v] + bias[v];
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
            use core::arch::aarch64::{int16x8_t, int32x4_t, vget_low_s16, vmull_high_s16, vmull_s16, vpaddq_s32};
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
            simd.combine_i32x4(half(a0.into(), b0.into()).simd_into(simd), half(a1.into(), b1.into()).simd_into(simd))
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
            let out: [i32; 8] =
                core::array::from_fn(|i| a[2 * i] as i32 * b[2 * i] as i32 + a[2 * i + 1] as i32 * b[2 * i + 1] as i32);
            i32x8::from_slice(simd, &out)
        }
    }
}

/// Writes `m` rows of `k` activations, row-major in `src`, as rows of A for [`qgemm`]:
/// each value quantized as `ONNX QuantizeLinear` does to `u8`
/// (`clamp(round_ties_even(x / scale) + zero_point, 0, 255)`), less `center` (the zero
/// point of the codes), two depths to a pair. `dst` gets `m * k.div_ceil(2)` pairs; an odd
/// last depth is paired with zero.
///
/// Activations that are codes already take `scale` 1 and `zero_point` 0.
pub fn quantize_rows(src: &[f32], m: usize, k: usize, scale: f32, zero_point: i32, center: i32, dst: &mut Vec<i32>) {
    quantize_rows_at(Level::new(), src, m, k, scale, zero_point, center, dst)
}

pub(crate) fn quantize_rows_at(
    level: Level,
    src: &[f32],
    m: usize,
    k: usize,
    scale: f32,
    zero_point: i32,
    center: i32,
    dst: &mut Vec<i32>,
) {
    assert!(src.len() >= m * k);
    dst.clear();
    dst.resize(m * k.div_ceil(2), 0);
    simd_call!(level, quantize_rows_simd(src, m, k, scale, zero_point, center, dst))
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
        let q = (x / scale).round_ties_even().clamp(i32::MIN as f32, i32::MAX as f32) as i32;
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
    let zero_point = (-min / scale).round().clamp(0.0, 255.0) as i32;
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
    quantized_matmul(x, scale, zero_point, zero_point, w, bias, relu, out)
}

/// `x * dequantize(w)` for an `x` already on the grid of `scale` and `zero_point` (it came
/// out of a `QuantizeLinear` -> `DequantizeLinear` pair): a static-QDQ linear layer.
pub fn qlinear_static<'a>(
    x: &TensorView<'_, f32>,
    scale: f32,
    zero_point: i32,
    w: &QWeights,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    quantized_matmul(x, scale, zero_point, zero_point, w, None, false, out)
}

/// ONNX `MatMulInteger` against a prepared B (whose scale should be 1): `a` holds `u8`
/// codes, as `f32`.
pub fn mat_mul_integer_qweights<'a>(
    a: &TensorView<'_, f32>,
    a_zero_point: i32,
    w: &QWeights,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    quantized_matmul(a, 1.0, 0, a_zero_point, w, None, false, out)
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
    assert!(ad >= 2 && bd >= 2, "MatMulInteger of {:?} and {:?}", a.shape, b.shape);
    let (m, k, n) = (a.shape[ad - 2], a.shape[ad - 1], b.shape[bd - 1]);
    assert_eq!(b.shape[bd - 2], k, "MatMulInteger of {:?} and {:?}", a.shape, b.shape);
    let batch_a: usize = a.shape[..ad - 2].iter().product();
    let batch_b: usize = b.shape[..bd - 2].iter().product();
    let batch = batch_a.max(batch_b);
    out.clear();
    out.resize(batch * m * n, 0.0);
    let scale = scale.unwrap_or(&[1.0]);
    let e = Epilogue { scale: 1.0, bias, relu };
    A_ROWS.with_borrow_mut(|rows| {
        if batch_b == 1 {
            // One B for every batch of A: all their rows in one product.
            let w = QWeights::from_f32_codes(&b.data[..k * n], k, n, b_zero_point, scale);
            quantize_rows(&a.data, batch_a * m, k, 1.0, 0, a_zero_point, rows);
            qgemm(rows, batch_a * m, w.pairs, &w, &e, out);
            return;
        }
        for i in 0..batch {
            let w = QWeights::from_f32_codes(&b.data[i * k * n..][..k * n], k, n, b_zero_point, scale);
            let ai = if batch_a == 1 { 0 } else { i };
            quantize_rows(&a.data[ai * m * k..][..m * k], m, k, 1.0, 0, a_zero_point, rows);
            qgemm(rows, m, w.pairs, &w, &e, &mut out[i * m * n..][..m * n]);
        }
    });
    let mut shape = if batch_a >= batch_b { a.shape[..ad - 2].to_vec() } else { b.shape[..bd - 2].to_vec() };
    shape.extend([m, n]);
    TensorView::from_slice(out, shape)
}

thread_local! {
    /// Rows of A, quantized.
    static A_ROWS: std::cell::RefCell<Vec<i32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Quantizes `x` with `scale`, `zero_point` and `center` (as [`quantize_rows`]) and
/// multiplies it by `w`, with `scale` the scale of the result's A.
fn quantized_matmul<'a>(
    x: &TensorView<'_, f32>,
    scale: f32,
    zero_point: i32,
    center: i32,
    w: &QWeights,
    bias: Option<&[f32]>,
    relu: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    let k = x.shape.last().copied().unwrap_or(1);
    assert_eq!(k, w.k, "depth of {:?} against a {}x{} weight", x.shape, w.k, w.n);
    let m: usize = x.shape[..x.shape.len().saturating_sub(1)].iter().product();
    out.clear();
    out.resize(m * w.n, 0.0);
    A_ROWS.with_borrow_mut(|a| {
        quantize_rows(&x.data, m, k, scale, zero_point, center, a);
        qgemm(a, m, w.pairs, w, &Epilogue { scale, bias, relu }, out);
    });
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
    let lo = rest.iter().fold(lo.as_slice().iter().copied().fold(f32::MAX, f32::min), |a, &b| a.min(b));
    let hi = rest.iter().fold(hi.as_slice().iter().copied().fold(f32::MIN, f32::max), |a, &b| a.max(b));
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
                out[r * k + kk] = if kk % 2 == 0 { p as i16 as i32 } else { p >> 16 };
            }
        }
        out
    }

    #[test]
    fn test_qgemm_matches_reference_at_every_level() {
        let mut rng = Rng::new(7);
        let sizes = [
            (1, 1, 1), (1, 2, 16), (2, 3, 17), (3, 7, 5), (4, 64, 33), (5, 31, 40), (6, 16, 16),
            (7, 40, 48), (13, 9, 100), (1, 600, 70), (9, 513, 20), (8, 1100, 37), (17, 0, 9),
            // Several depth blocks.
            (7, 2100, 20), (2, 4500, 33),
        ];
        for level in levels() {
            for &(m, k, n) in &sizes {
                for case in 0..4 {
                    let signed = case % 2 == 0;
                    let per_col = case >= 2;
                    let raw = codes(&mut rng, k * n);
                    let zp: Vec<i32> = (0..if per_col { n } else { 1 })
                        .map(|_| if signed { rng.f32(-20.0, 20.0) as i32 } else { rng.f32(0.0, 256.0) as i32 })
                        .collect();
                    let ws: Vec<f32> = (0..if per_col { n } else { 1 }).map(|_| rng.f32(0.001, 0.1)).collect();
                    let w = QWeights::new(&raw, k, n, signed, &zp, &ws);
                    let x = rng.vec(m * k, -3.0, 3.0);
                    let (sa, za) = (rng.f32(0.01, 0.05), rng.f32(0.0, 256.0) as i32);
                    let mut a = Vec::new();
                    quantize_rows_at(level, &x, m, k, sa, za, za, &mut a);
                    let bias = rng.vec(n, -1.0, 1.0);
                    let relu = case == 1;
                    let e = Epilogue { scale: sa, bias: (case != 3).then_some(&bias[..]), relu };
                    let mut c = vec![f32::NAN; m * n];
                    qgemm_at(level, &a, m, w.pairs(), &w, &e, &mut c);

                    let ac = unpack_rows(&a, m, k);
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
                            let mut v = sum as i32 as f32 * s + e.bias.map_or(0.0, |b| b[j]);
                            if relu {
                                v = v.max(0.0);
                            }
                            want[i * n + j] = v;
                        }
                    }
                    assert_same_bits(&c, &want, &format!("{level:?} {m}x{k}x{n} case {case}"));
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
                for &(scale, zp, center) in &[(0.02f32, 128, 128), (0.01, 0, 0), (0.05, 255, 255), (1.0, 3, 3), (1.0, 0, 179)] {
                    // On the grid (as QDQ activations are), off it, and out of range.
                    let mut x = rng.vec(m * k, -8.0, 8.0);
                    for (i, v) in x.iter_mut().enumerate().step_by(3) {
                        *v = scale * ((i % 300) as f32 - 20.0 - zp as f32);
                    }
                    let mut got = Vec::new();
                    quantize_rows_at(level, &x, m, k, scale, zp, center, &mut got);
                    let got = unpack_rows(&got, m, k);
                    for (i, (&g, &v)) in got.iter().zip(&x).enumerate() {
                        let want = ((v / scale).round_ties_even() as i32 + zp).clamp(0, 255) - center;
                        assert_eq!(g, want, "{level:?} {m}x{k} scale {scale} zp {zp} center {center} at {i}: {v}");
                    }
                }
            }
        }
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
        let matmul = |a: &dyn Fn(usize, usize) -> f64, b: &dyn Fn(usize, usize) -> f64| -> Vec<f64> {
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
        let got = mat_mul_integer_qweights(&TensorView::from_slice(&a_codes, vec![batch, m, k]), 179, &w, &mut out);
        assert_eq!(got.shape.as_ref(), &[batch, m, n]);
        let want = matmul(&|i, kk| a_codes[i * k + kk] as f64 - 179.0, &|kk, j| raw[kk * n + j] as f64 - 131.0);
        assert_close(&got.data, &want, 0.0, "MatMulInteger");

        // Static QDQ: activations on the grid, i8 B, per-column scales.
        let w = QWeights::new(&raw, k, n, true, &[0], &ws);
        let (sa, za) = (0.03f32, 100);
        let x: Vec<f32> = (0..rows * k).map(|_| sa * (rng.f32(0.0, 256.0).floor() - za as f32)).collect();
        let got = qlinear_static(&TensorView::from_slice(&x, vec![batch, m, k]), sa, za, &w, &mut out);
        let want = matmul(&|i, kk| x[i * k + kk] as f64, &|kk, j| w_i8(kk, j) * ws[j] as f64);
        assert_close(&got.data, &want, 1e-5, "QDQ");

        // Dynamic: anything in, quantized to its own range; bias and ReLU.
        let x = rng.vec(rows * k, -2.0, 3.0);
        let got = qlinear_dynamic(&TensorView::from_slice(&x, vec![batch, m, k]), &w, Some(&bias), true, &mut out);
        let (s, z) = dynamic_quant_params(&x);
        let dq = |v: f32| s as f64 * (((v / s).round_ties_even() as i32 + z).clamp(0, 255) - z) as f64;
        let want: Vec<f64> = matmul(&|i, kk| dq(x[i * k + kk]), &|kk, j| w_i8(kk, j) * ws[j] as f64)
            .iter()
            .enumerate()
            .map(|(i, &v)| (v + bias[i % n] as f64).max(0.0))
            .collect();
        assert_close(&got.data, &want, 1e-5, "dynamic");
        // And close to the unquantized product, so the quantization itself is sane.
        let exact: Vec<f64> = matmul(&|i, kk| x[i * k + kk] as f64, &|kk, j| w_i8(kk, j) * ws[j] as f64)
            .iter()
            .enumerate()
            .map(|(i, &v)| (v + bias[i % n] as f64).max(0.0))
            .collect();
        assert_close(&got.data, &exact, 0.05, "dynamic vs f32");
    }

    #[test]
    fn test_dynamic_quant_params() {
        for level in levels() {
            let x: Vec<f32> = (0..37).map(|i| i as f32 * 0.25 - 2.0).collect();
            let (s, z) = dynamic_quant_params_at(level, &x);
            assert_eq!(s, 9.0 / 255.0);
            assert_eq!(z, (2.0f32 / s).round() as i32);
            // All positive: the range is widened down to zero.
            let (s, z) = dynamic_quant_params_at(level, &[1.0, 2.0, 3.0]);
            assert_eq!((s, z), (3.0 / 255.0, 0));
        }
    }
}
