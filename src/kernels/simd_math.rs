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
    let x = x.max(f32x16::splat(simd, -87.3)).min(f32x16::splat(simd, 88.0));
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
    const A: [f32; 5] = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429];

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
    let inner = f32x16::splat(simd, 0.7978845608028654)
        * f32x16::splat(simd, 0.044715).mul_add(x3, x);
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
            let x = [-1000.0, -200.0, -88.0, -87.3, 88.0, 89.0, 500.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            let got = exp_of(level, &x);
            for (i, &g) in got.iter().enumerate().take(7) {
                assert!(g.is_finite() && g > 0.0, "{level:?} x={}: got {g}", x[i]);
            }
            assert!(got[0] < 1e-37, "{level:?}: exp(-1000) = {}", got[0]);
            assert!(got[6] > 1e37, "{level:?}: exp(500) = {}", got[6]);
            assert_eq!(got[7], 1.0, "{level:?}: exp(0)");
        }
    }
}
