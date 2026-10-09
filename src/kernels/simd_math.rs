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
