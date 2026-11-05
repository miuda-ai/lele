//! Transcendental functions for `fearless_simd` kernels, which has none of its
//! own.

use fearless_simd::{Simd, f32x16, i32x16, prelude::*};

/// `exp(x)` for each lane, to a few ULP.
///
/// Splits `x = n·ln2 + r` with `|r| <= ln2/2` (`ln2` in two parts, so `n·ln2`
/// loses nothing), evaluates `exp(r)` with a degree-7 polynomial and scales it
/// by `2^n` through the exponent bits.
///
/// Inputs are clamped to [-87.3, 88.0]: below, the result is the smallest
/// normal float rather than 0 (callers that sum many such lanes should treat
/// that as negligible); above, it is `exp(88)` rather than infinity. Both
/// keep `2^n` a normal float, which the exponent trick requires.
#[inline(always)]
pub(crate) fn exp<S: Simd>(x: f32x16<S>) -> f32x16<S> {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693359375;
    const LN2_LO: f32 = -2.12194440e-4;
    // Taylor-like coefficients of exp(r) - 1 - r, from r².
    const C: [f32; 6] = [
        0.5,
        0.166666671633720398,
        0.0416657844442129135,
        0.00833345670066840443,
        0.00139712726883569741,
        0.000198712018891638893,
    ];

    let simd = x.simd;
    let x = x
        .max(f32x16::splat(simd, -87.3))
        .min(f32x16::splat(simd, 88.0));
    let n = (x * f32x16::splat(simd, LOG2E)).round_ties_even();
    let r = (-n).mul_add(f32x16::splat(simd, LN2_HI), x);
    let r = (-n).mul_add(f32x16::splat(simd, LN2_LO), r);

    let one = f32x16::splat(simd, 1.0);
    let mut p = f32x16::splat(simd, C[5]);
    for &c in C[..5].iter().rev() {
        p = p.mul_add(r, f32x16::splat(simd, c));
    }
    // 1 + r + r²·p
    let r2 = r * r;
    let y = p.mul_add(r2, r + one);

    // n is integral and within [-126, 127], so n + 127 is a normal exponent.
    let bits = (i32x16::truncate_from(n) + i32x16::splat(simd, 127)) << 23;
    y * bits.bitcast::<f32x16<S>>()
}

/// `sin(x)` for each lane, to about an ulp of the largest of it and 1e-7.
///
/// Splits `x = n·π/2 + r` with `|r| <= π/4` (`π/2` in three parts, exact for
/// `|x| <= 8192`), evaluates the sine or cosine of `r` by Cephes' polynomials
/// as `n` says, and negates them in the lower half-turn. A lane past 8192,
/// infinite or NaN is computed in f64 instead.
#[inline(always)]
pub(crate) fn sin<S: Simd>(x: f32x16<S>) -> f32x16<S> {
    sin_quadrants(x, 0)
}

/// `cos(x)`, as `sin(x + π/2)`: [`sin`] a quadrant on.
#[inline(always)]
pub(crate) fn cos<S: Simd>(x: f32x16<S>) -> f32x16<S> {
    sin_quadrants(x, 1)
}

/// `sin(x + quarters·π/2)`.
#[inline(always)]
fn sin_quadrants<S: Simd>(x: f32x16<S>, quarters: i32) -> f32x16<S> {
    const LIMIT: f32 = 8192.0;
    let simd = x.simd;
    let in_range = x.abs().simd_le(f32x16::splat(simd, LIMIT));
    if in_range.all_true() {
        return sin_reduced(x, quarters);
    }
    // Lane by lane, so that each result is the same whatever else shares its
    // vector.
    let mut lanes = [0.0f32; 16];
    let mut ys = [0.0f32; 16];
    x.store_array(&mut lanes);
    sin_reduced(in_range.select(x, f32x16::splat(simd, 0.0)), quarters).store_array(&mut ys);
    for (y, &v) in ys.iter_mut().zip(&lanes) {
        if v.abs() > LIMIT || v.is_nan() {
            *y = (v as f64 + quarters as f64 * std::f64::consts::FRAC_PI_2).sin() as f32;
        }
    }
    f32x16::from_slice(simd, &ys)
}

/// [`sin_quadrants`] for lanes no larger than its `LIMIT`.
#[inline(always)]
fn sin_reduced<S: Simd>(x: f32x16<S>, quarters: i32) -> f32x16<S> {
    const FRAC_2_PI: f32 = std::f32::consts::FRAC_2_PI;
    // π/2 = DP1 + DP2 + DP3; DP1 and DP2 have few enough bits that n·DPi is
    // exact for the n of |x| <= 8192.
    const DP1: f32 = 1.5703125;
    const DP2: f32 = 4.837513e-4;
    const DP3: f32 = 7.54979e-8;
    // Cephes `sinf` and `cosf` coefficients on [-π/4, π/4].
    const S: [f32; 3] = [-1.6666655e-1, 8.332161e-3, -1.9515296e-4];
    const C: [f32; 3] = [4.1666646e-2, -1.3887316e-3, 2.4433157e-5];

    let simd = x.simd;
    // Below 2^-12, sin(r) rounds to r and cos(r) to 1. Such lanes are zeroed
    // before any arithmetic and their results set at the end: their powers
    // (and a subnormal x itself) would be subnormal, which x86 multiplies by
    // microcode, many times slower.
    let tiny = f32x16::splat(simd, 1.0 / 4096.0);
    let zero = f32x16::splat(simd, 0.0);
    let small_x = x.abs().simd_lt(tiny);
    let x0 = small_x.select(zero, x);
    let n = (x0 * f32x16::splat(simd, FRAC_2_PI)).round_ties_even();
    let r = (-n).mul_add(f32x16::splat(simd, DP1), x0);
    let r = (-n).mul_add(f32x16::splat(simd, DP2), r);
    let r = (-n).mul_add(f32x16::splat(simd, DP3), r);
    // (Near a multiple of π/2, r is small but normal; its fourth power need not be.)
    let small_r = r.abs().simd_lt(tiny);
    let r_poly = small_r.select(zero, r);
    let r2 = r_poly * r_poly;
    // r + r³·(S0 + r²·(S1 + r²·S2))
    let ps = f32x16::splat(simd, S[2])
        .mul_add(r2, f32x16::splat(simd, S[1]))
        .mul_add(r2, f32x16::splat(simd, S[0]));
    let sin_r = small_r.select(r, (ps * r2).mul_add(r_poly, r_poly));
    // 1 - r²/2 + r⁴·(C0 + r²·(C1 + r²·C2))
    let pc = f32x16::splat(simd, C[2])
        .mul_add(r2, f32x16::splat(simd, C[1]))
        .mul_add(r2, f32x16::splat(simd, C[0]));
    let cos_r = (pc * r2).mul_add(r2, f32x16::splat(simd, -0.5).mul_add(r2, f32x16::splat(simd, 1.0)));
    let q = i32x16::truncate_from(n) + i32x16::splat(simd, quarters);
    let odd = (q & i32x16::splat(simd, 1)).simd_eq(i32x16::splat(simd, 1));
    let y = odd.select(cos_r, sin_r);
    // Quadrants 2 and 3 negate: bit 1 of q moved to the sign bit.
    let sign = (q & i32x16::splat(simd, 2)) << 30;
    let y = (y.bitcast::<i32x16<S>>() ^ sign).bitcast::<f32x16<S>>();
    // A small x was taken as 0: cos(0) = 1 came out above; the sine is x itself
    // (keeping its sign, and exact for a subnormal x).
    if quarters == 0 { small_x.select(x, y) } else { y }
}

/// `1 / (1 + exp(-x))`. Saturates to 0 and 1 where `exp(-x)` is clamped.
#[inline(always)]
pub(crate) fn sigmoid<S: Simd>(x: f32x16<S>) -> f32x16<S> {
    let one = f32x16::splat(x.simd, 1.0);
    one / (one + exp(-x))
}

/// `x * sigmoid(x)`, as `x / (1 + exp(-x))`.
#[inline(always)]
pub(crate) fn silu<S: Simd>(x: f32x16<S>) -> f32x16<S> {
    let one = f32x16::splat(x.simd, 1.0);
    x / (one + exp(-x))
}

/// `tanh(x) = sign(x) * (1 - e) / (1 + e)` with `e = exp(-2|x|)`: the
/// exponent never overflows, so large inputs saturate to ±1.
#[inline(always)]
pub(crate) fn tanh<S: Simd>(x: f32x16<S>) -> f32x16<S> {
    let simd = x.simd;
    let one = f32x16::splat(simd, 1.0);
    let e = exp(-(x.abs() * f32x16::splat(simd, 2.0)));
    ((one - e) / (one + e)).copysign(x)
}

/// `erf(x)` by Abramowitz & Stegun 7.1.26, to 1.5e-7 absolute:
/// `1 - (a1·t + … + a5·t⁵)·exp(-x²)` with `t = 1 / (1 + 0.3275911·|x|)`.
#[inline(always)]
pub(crate) fn erf<S: Simd>(x: f32x16<S>) -> f32x16<S> {
    const P: f32 = 0.3275911;
    const A: [f32; 5] = [
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
    ];

    let simd = x.simd;
    let one = f32x16::splat(simd, 1.0);
    let ax = x.abs();
    let t = one / ax.mul_add(f32x16::splat(simd, P), one);
    let mut poly = f32x16::splat(simd, A[4]);
    for &a in A[..4].iter().rev() {
        poly = poly.mul_add(t, f32x16::splat(simd, a));
    }
    // 1 - poly·t·exp(-x²)
    let tail = exp(-(ax * ax));
    (-(poly * t)).mul_add(tail, one).copysign(x)
}

/// `x * (0.5 * (1 + erf(x / √2)))`, in exactly the operations and order of
/// the ONNX `Div -> Erf -> Add -> Mul -> Mul` subgraph the compiler fuses into
/// it, so that fusing changes no bits: a true division rather than a
/// reciprocal multiply, and the same `erf` as the standalone kernel.
#[inline(always)]
pub(crate) fn gelu_erf<S: Simd>(x: f32x16<S>) -> f32x16<S> {
    let simd = x.simd;
    let half = f32x16::splat(simd, 0.5);
    let e = erf(x / f32x16::splat(simd, std::f32::consts::SQRT_2));
    x * (half * (e + f32x16::splat(simd, 1.0)))
}

/// `x * (0.5 * (1 + erf(x / √2)))` with the division folded into a multiply:
/// [`gelu_erf`] without the bit-for-bit contract, a fraction of an ulp off it.
#[inline(always)]
pub(crate) fn gelu<S: Simd>(x: f32x16<S>) -> f32x16<S> {
    let simd = x.simd;
    let e = erf(x * f32x16::splat(simd, std::f32::consts::FRAC_1_SQRT_2));
    (x * f32x16::splat(simd, 0.5)) * (f32x16::splat(simd, 1.0) + e)
}

/// `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`.
#[inline(always)]
pub(crate) fn fast_gelu<S: Simd>(x: f32x16<S>) -> f32x16<S> {
    let simd = x.simd;
    let x3 = (x * x) * x;
    let inner =
        f32x16::splat(simd, 0.7978845608028654) * f32x16::splat(simd, 0.044715).mul_add(x3, x);
    (x * f32x16::splat(simd, 0.5)) * (f32x16::splat(simd, 1.0) + tanh(inner))
}

/// `exp(x)` as the ONNX operator: [`exp`] clamps its input to stay finite,
/// this restores infinity for inputs past `ln(f32::MAX)`.
#[inline(always)]
pub(crate) fn exp_saturating<S: Simd>(x: f32x16<S>) -> f32x16<S> {
    let simd = x.simd;
    let overflow = x.simd_gt(f32x16::splat(simd, 88.72284));
    overflow.select(f32x16::splat(simd, f32::INFINITY), exp(x))
}

#[inline(always)]
pub(crate) fn relu<S: Simd>(x: f32x16<S>) -> f32x16<S> {
    x.max(f32x16::splat(x.simd, 0.0))
}

/// `x` for `x >= 0`, else `alpha · x`.
#[inline(always)]
pub(crate) fn leaky_relu<S: Simd>(x: f32x16<S>, alpha: f32x16<S>) -> f32x16<S> {
    x.simd_ge(f32x16::splat(x.simd, 0.0)).select(x, x * alpha)
}

/// `x` for `x >= 0`, else `alpha · (eˣ - 1)`.
#[inline(always)]
pub(crate) fn elu<S: Simd>(x: f32x16<S>, alpha: f32x16<S>) -> f32x16<S> {
    let simd = x.simd;
    x.simd_ge(f32x16::splat(simd, 0.0))
        .select(x, alpha * (exp_saturating(x) - f32x16::splat(simd, 1.0)))
}

#[inline(always)]
pub(crate) fn sqrt<S: Simd>(x: f32x16<S>) -> f32x16<S> {
    x.sqrt()
}

/// Writes `$f(src[i])` to `out[i]` for every `i`, where `$f` is one of the
/// vector functions above and `$simd` the token of the enclosing `#[simd]`
/// function. Needs `src.len() >= 16`.
///
/// A macro, not a function taking `f`: a function item or closure passed to a
/// function is compiled on its own, without the caller's target features, and
/// every vector operation in it becomes an out-of-line call (25x slower on
/// AVX2). Expanded in the `#[simd]` body, the calls inline.
///
/// A length that is not a multiple of 16 ends with the last 16 elements,
/// overlapping the previous vector: each output depends only on its own
/// input and `src` and `out` are distinct, so recomputing the overlap writes
/// the values already there.
macro_rules! map {
    ($simd:expr, $src:expr, $out:expr, $f:path $(, $arg:expr)* $(,)?) => {{
        use fearless_simd::prelude::*;
        let (simd, src, out): (_, &[f32], &mut [f32]) = ($simd, $src, $out);
        let out = &mut out[..src.len()];
        let (x, x_tail) = src.as_chunks::<16>();
        let (o, _) = out.as_chunks_mut::<16>();
        let (x2, x1) = x.as_chunks::<2>();
        let (o2, o1) = o.as_chunks_mut::<2>();
        // Two vectors per step: the polynomials are long dependency chains,
        // and a second independent one fills the gaps.
        for ([a, b], [oa, ob]) in x2.iter().zip(o2) {
            let ya = $f(fearless_simd::f32x16::load_array_ref(simd, a) $(, $arg)*);
            let yb = $f(fearless_simd::f32x16::load_array_ref(simd, b) $(, $arg)*);
            ya.store_array(oa);
            yb.store_array(ob);
        }
        for (a, oa) in x1.iter().zip(o1) {
            $f(fearless_simd::f32x16::load_array_ref(simd, a) $(, $arg)*).store_array(oa);
        }
        if !x_tail.is_empty() {
            let last = src.last_chunk::<16>().expect("at least a vector");
            let out_last = out.last_chunk_mut::<16>().expect("at least a vector");
            $f(fearless_simd::f32x16::load_array_ref(simd, last) $(, $arg)*).store_array(out_last);
        }
    }};
}

pub(crate) use map;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::simd::simd_call;
    use crate::kernels::test_util::levels;
    use fearless_simd::Level;
    use fearless_simd_macros::simd;

    #[simd]
    fn exp_all<S: Simd>(simd: S, x: &[f32; 16], out: &mut [f32; 16]) {
        exp(f32x16::load_array_ref(simd, x)).store_array(out);
    }

    fn exp_of(level: Level, x: &[f32; 16]) -> [f32; 16] {
        let mut out = [0.0; 16];
        simd_call!(level, exp_all(x, &mut out));
        out
    }

    #[test]
    fn test_exp_is_accurate_over_its_range_at_every_level() {
        for level in levels() {
            let mut worst = 0.0f64;
            // 0.01 steps over [-87, 88] cover every exponent, and the
            // fractional parts drift across the polynomial's interval.
            let xs: Vec<f32> = (-8700..=8800).map(|i| i as f32 * 0.01).collect();
            for chunk in xs.chunks(16) {
                let mut x = [0.0f32; 16];
                x[..chunk.len()].copy_from_slice(chunk);
                let got = exp_of(level, &x);
                for (&x, &g) in x.iter().zip(&got).take(chunk.len()) {
                    let want = (x as f64).exp();
                    let err = ((g as f64 - want) / want).abs();
                    worst = worst.max(err);
                }
            }
            // A few ULP: f32 epsilon is 1.2e-7.
            assert!(worst < 5e-7, "{level:?}: relative error {worst:.3e}");
        }
    }

    #[test]
    fn test_exp_saturates_outside_its_range() {
        for level in levels() {
            let x = [
                -1000.0, -200.0, -88.0, -87.3, 88.0, 89.0, 500.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            ];
            let got = exp_of(level, &x);
            for (i, &g) in got.iter().enumerate().take(7) {
                assert!(g.is_finite() && g > 0.0, "{level:?} x={}: got {g}", x[i]);
            }
            assert!(got[0] < 1e-37, "{level:?}: exp(-1000) = {}", got[0]);
            assert!(got[6] > 1e37, "{level:?}: exp(500) = {}", got[6]);
            assert_eq!(got[7], 1.0, "{level:?}: exp(0)");
        }
    }

    #[simd]
    fn sin_cos_all<S: Simd>(simd: S, x: &[f32; 16], out: &mut [[f32; 16]; 2]) {
        let x = f32x16::load_array_ref(simd, x);
        sin(x).store_array(&mut out[0]);
        cos(x).store_array(&mut out[1]);
    }

    /// `sin` and `cos` against f64 over [-8192, 8192] (finely near 0, where the
    /// reduction does nothing, and coarsely out to where it is exact), within
    /// two ulp of the larger of the result and 1/2; and past that, and at
    /// infinity and NaN, as `f32::sin` and `cos` (in f64) give.
    #[test]
    fn test_sin_cos_are_accurate_at_every_level() {
        let fine = (-20000..=20000).map(|i| i as f32 * 1e-3);
        let coarse = (-81920..=81920).map(|i| i as f32 * 0.1 + 0.0123);
        let xs: Vec<f32> = fine.chain(coarse).chain([8192.0, -8192.0]).collect();
        for level in levels() {
            let mut worst = [0.0f64; 2];
            for chunk in xs.chunks(16) {
                let mut x = [0.0f32; 16];
                x[..chunk.len()].copy_from_slice(chunk);
                let mut got = [[0.0; 16]; 2];
                simd_call!(level, sin_cos_all(&x, &mut got));
                for (i, &x) in x.iter().enumerate().take(chunk.len()) {
                    for (k, want) in [(x as f64).sin(), (x as f64).cos()].into_iter().enumerate() {
                        let err = (got[k][i] as f64 - want).abs() / want.abs().max(0.5);
                        worst[k] = worst[k].max(err / f32::EPSILON as f64);
                    }
                }
            }
            assert!(worst[0] <= 2.0 && worst[1] <= 2.0, "{level:?}: {worst:?} ulp");

            // Below 2^-12, down to subnormals: sin is x (bit for bit, -0 too)
            // and cos is 1, as rounding the true values gives.
            let x = [
                0.0, -0.0, 1e-40, -1e-40, f32::MIN_POSITIVE, -1e-30, 1e-20, 1e-5, -2e-4,
                1.0 / 4097.0, -1.0 / 4097.0, 3e-38, -5e-39, 1e-10, 2.4e-4, -1e-7,
            ];
            let mut got = [[0.0; 16]; 2];
            simd_call!(level, sin_cos_all(&x, &mut got));
            for (i, &x) in x.iter().enumerate() {
                assert_eq!(got[0][i].to_bits(), x.to_bits(), "{level:?}: sin({x:e})");
                assert_eq!(got[1][i], 1.0, "{level:?}: cos({x:e})");
            }

            // Lanes out of range take f64; the others in their vector are what
            // they are in a vector all in range.
            let x = [
                1e5, -3e6, f32::INFINITY, f32::NAN, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                10.0, 11.0, 12.0,
            ];
            let mut in_range = x;
            in_range[..4].fill(0.5);
            let (mut got, mut alone) = ([[0.0; 16]; 2], [[0.0; 16]; 2]);
            simd_call!(level, sin_cos_all(&x, &mut got));
            simd_call!(level, sin_cos_all(&in_range, &mut alone));
            for (i, &x) in x.iter().enumerate() {
                let want = if i < 4 {
                    [(x as f64).sin() as f32, (x as f64).cos() as f32]
                } else {
                    [alone[0][i], alone[1][i]]
                };
                for k in 0..2 {
                    assert!(
                        got[k][i].to_bits() == want[k].to_bits() || got[k][i].is_nan() && want[k].is_nan(),
                        "{level:?} x={x}: {} vs {}",
                        got[k][i],
                        want[k]
                    );
                }
            }
        }
    }
}
