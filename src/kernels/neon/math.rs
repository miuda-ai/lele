#[cfg(nightly_build)]
use std::simd::StdFloat;
#[cfg(nightly_build)]
use std::simd::prelude::*;

#[cfg(nightly_build)]
#[inline(always)]
pub(crate) fn simd_exp(x: f32x4) -> f32x4 {
    x.exp()
}

#[cfg(nightly_build)]
#[inline(always)]
pub(crate) fn simd_tanh(x: f32x4) -> f32x4 {
    let one = f32x4::splat(1.0);
    let zero = f32x4::splat(0.0);
    let two = f32x4::splat(2.0);

    let abs_x = x.abs();
    let neg_two_abs_x = zero - (two * abs_x);
    let e = simd_exp(neg_two_abs_x);

    let num = one - e;
    let den = one + e;
    let res_abs = num / den;

    // Restore sign: if x < 0, result is -res_abs
    let is_negative = x.simd_lt(zero);
    is_negative.select(zero - res_abs, res_abs)
}

#[cfg(nightly_build)]
#[inline(always)]
pub(crate) fn simd_sigmoid(x: f32x4) -> f32x4 {
    let one = f32x4::splat(1.0);
    let neg_x = f32x4::splat(0.0) - x;
    let e = simd_exp(neg_x);
    one / (one + e)
}
