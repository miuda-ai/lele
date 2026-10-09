//! Bias and activation applied in place to one row of a convolution output.

use crate::kernels::conv2d::Activation;
use crate::kernels::simd::simd_call;
use crate::kernels::simd_math as vm;
use fearless_simd::{Level, Simd};
use fearless_simd_macros::simd;

/// `data[i] = act(data[i] + bias)`.
pub(crate) fn bias_act_inplace(data: &mut [f32], bias: f32, act: Activation) {
    bias_act_inplace_at(Level::new(), data, bias, act)
}

fn bias_act_inplace_at(level: Level, data: &mut [f32], bias: f32, act: Activation) {
    // The kernels need a whole vector. A shorter row goes through a padded
    // copy made here rather than in them: its `memcpy` calls would give
    // every call, long rows included, a stack frame and saved registers.
    if data.len() < 16 {
        let mut padded = [0.0f32; 16];
        padded[..data.len()].copy_from_slice(data);
        bias_act_vectors(level, &mut padded, bias, act);
        data.copy_from_slice(&padded[..data.len()]);
    } else {
        bias_act_vectors(level, data, bias, act);
    }
}

/// Needs `data.len() >= 16`.
fn bias_act_vectors(level: Level, data: &mut [f32], bias: f32, act: Activation) {
    match act {
        Activation::None => simd_call!(level, max = Avx2, bias_add_simd(data, bias)),
        Activation::Relu => simd_call!(level, max = Avx2, bias_relu_simd(data, bias)),
        Activation::SiLU => simd_call!(level, max = Avx2, bias_silu_simd(data, bias)),
    }
}

/// Applies `$f(x + bias)` to every element of `$data`.
///
/// A macro for the reason `vm::map!` is: `$f` has to be expanded inside the
/// `#[simd]` body to be compiled with its target features.
///
/// A length that is not a multiple of 16 ends with the last 16 elements, as
/// in `map!`, but the data is overwritten in place, so the last vector is
/// loaded before the main loop touches the overlap. Recomputing it from those
/// original values then stores what the overlap already holds. Needs
/// `data.len() >= 16`.
macro_rules! map_inplace {
    ($simd:expr, $data:expr, $bias:expr, $f:path) => {{
        use fearless_simd::prelude::*;
        let (simd, data, bias): (_, &mut [f32], f32) = ($simd, $data, $bias);
        let vbias = fearless_simd::f32x16::splat(simd, bias);
        let last = fearless_simd::f32x16::load_array_ref(simd, data.last_chunk::<16>().unwrap());
        let (chunks, tail) = data.as_chunks_mut::<16>();
        let (pairs, single) = chunks.as_chunks_mut::<2>();
        // Two vectors per step, like `map!`.
        for [a, b] in pairs {
            let ya = $f(fearless_simd::f32x16::load_array_ref(simd, a) + vbias);
            let yb = $f(fearless_simd::f32x16::load_array_ref(simd, b) + vbias);
            ya.store_array(a);
            yb.store_array(b);
        }
        for c in single {
            $f(fearless_simd::f32x16::load_array_ref(simd, c) + vbias).store_array(c);
        }
        if !tail.is_empty() {
            let out_last = data.last_chunk_mut::<16>().unwrap();
            $f(last + vbias).store_array(out_last);
        }
    }};
}

#[inline(always)]
fn identity<S: Simd>(x: fearless_simd::f32x16<S>) -> fearless_simd::f32x16<S> {
    x
}

#[simd]
fn bias_add_simd<S: Simd>(simd: S, data: &mut [f32], bias: f32) {
    map_inplace!(simd, data, bias, identity)
}

#[simd]
fn bias_relu_simd<S: Simd>(simd: S, data: &mut [f32], bias: f32) {
    map_inplace!(simd, data, bias, vm::relu)
}

#[simd]
fn bias_silu_simd<S: Simd>(simd: S, data: &mut [f32], bias: f32) {
    map_inplace!(simd, data, bias, vm::silu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::test_util::{assert_close, levels, Rng, AWKWARD_LENS};

    fn reference(x: f32, bias: f32, act: Activation) -> f64 {
        let v = x as f64 + bias as f64;
        match act {
            Activation::None => v,
            Activation::Relu => v.max(0.0),
            Activation::SiLU => v / (1.0 + (-v).exp()),
        }
    }

    #[test]
    fn test_bias_act_matches_reference_at_every_level_and_length() {
        let mut rng = Rng::new(7);
        for level in levels() {
            for act in [Activation::None, Activation::Relu, Activation::SiLU] {
                for &len in AWKWARD_LENS {
                    let src = rng.vec(len, -12.0, 12.0);
                    let bias = rng.f32(-2.0, 2.0);
                    let mut got = src.clone();
                    bias_act_inplace_at(level, &mut got, bias, act);
                    let want: Vec<f64> = src.iter().map(|&x| reference(x, bias, act)).collect();
                    assert_close(&got, &want, 1e-5, &format!("{level:?} len {len}"));
                }
            }
        }
    }

    #[test]
    fn test_bias_is_added_exactly_once_to_every_element() {
        // The tail must not be revisited: a bias of 1 on zeros gives exactly 1.
        for level in levels() {
            for &len in AWKWARD_LENS {
                let mut data = vec![0.0f32; len];
                bias_act_inplace_at(level, &mut data, 1.0, Activation::None);
                assert!(data.iter().all(|&v| v == 1.0), "{level:?} len {len}");
            }
        }
    }

    #[test]
    fn test_bias_act_leaves_neighbours_alone() {
        let mut buf = vec![5.0f32; 40];
        bias_act_inplace(&mut buf[10..23], 1.0, Activation::Relu);
        assert!(buf[..10].iter().all(|&v| v == 5.0));
        assert!(buf[10..23].iter().all(|&v| v == 6.0));
        assert!(buf[23..].iter().all(|&v| v == 5.0));
    }
}
