//! Bias and activation applied in place to a convolution or product output.

use crate::kernels::simd::simd_call;
use crate::kernels::simd_math as vm;
use crate::tensor::TensorView;
use fearless_simd::{Level, Simd, f32x8, f32x16};
use fearless_simd_macros::simd;

/// `data[i] = act?(data[i] + bias)`.
pub(crate) fn bias_act_inplace(data: &mut [f32], bias: f32, act: Option<Act>) {
    bias_act_inplace_at(Level::new(), data, bias, act)
}

pub(crate) fn bias_act_inplace_at(level: Level, data: &mut [f32], bias: f32, act: Option<Act>) {
    // The kernels need a whole vector. A shorter row goes through a padded
    // copy made here rather than in them: its `memcpy` calls would give
    // every call, long rows included, a stack frame and saved registers.
    let run = |data: &mut [f32]| match act {
        None => simd_call!(level, max = Avx2, bias_add_simd(data, bias)),
        Some(Act::Relu) => simd_call!(level, max = Avx2, bias_relu_simd(data, bias)),
        Some(Act::Silu) => simd_call!(level, max = Avx2, bias_silu_simd(data, bias)),
        Some(Act::Sigmoid) => simd_call!(level, max = Avx2, bias_sigmoid_simd(data, bias)),
        Some(Act::GeluErf) => simd_call!(level, max = Avx2, bias_gelu_erf_simd(data, bias)),
    };
    if data.len() < 16 {
        let mut padded = [0.0f32; 16];
        padded[..data.len()].copy_from_slice(data);
        run(&mut padded);
        data.copy_from_slice(&padded[..data.len()]);
    } else {
        run(data);
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
    // `$f(x)`, with no bias added: adding a zero would turn -0 into +0.
    ($simd:expr, $data:expr, $f:path) => {{
        use fearless_simd::prelude::*;
        let (simd, data): (_, &mut [f32]) = ($simd, $data);
        let last = fearless_simd::f32x16::load_array_ref(simd, data.last_chunk::<16>().unwrap());
        let (chunks, tail) = data.as_chunks_mut::<16>();
        let (pairs, single) = chunks.as_chunks_mut::<2>();
        for [a, b] in pairs {
            let ya = $f(fearless_simd::f32x16::load_array_ref(simd, a));
            let yb = $f(fearless_simd::f32x16::load_array_ref(simd, b));
            ya.store_array(a);
            yb.store_array(b);
        }
        for c in single {
            $f(fearless_simd::f32x16::load_array_ref(simd, c)).store_array(c);
        }
        if !tail.is_empty() {
            let out_last = data.last_chunk_mut::<16>().unwrap();
            $f(last).store_array(out_last);
        }
    }};
}

/// An activation applied in place to the output of a product or a
/// convolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    Relu,
    Silu,
    Sigmoid,
    /// GELU as [`gelu_erf`](crate::kernels::gelu_erf) computes it.
    GeluErf,
}

/// `act` of every lane, as [`act_inplace`] computes it: kernels that end in an
/// activation apply this to their results before storing them.
#[inline(always)]
pub(crate) fn act_f32x16<S: Simd>(act: Act, x: f32x16<S>) -> f32x16<S> {
    match act {
        Act::Relu => vm::relu(x),
        Act::Silu => vm::silu(x),
        Act::Sigmoid => vm::sigmoid(x),
        Act::GeluErf => vm::gelu_erf(x),
    }
}

/// [`act_f32x16`] of `V` (1 or 2) vectors of 8 lanes, which go through one
/// vector of 16 (padded with zeros for one).
#[inline(always)]
pub(crate) fn act_f32x8<S: Simd, const V: usize>(
    simd: S,
    act: Act,
    x: [f32x8<S>; V],
) -> [f32x8<S>; V] {
    use fearless_simd::prelude::*;
    let zero = f32x8::splat(simd, 0.0);
    let (lo, hi) = act_f32x16(act, x[0].combine(if V > 1 { x[V - 1] } else { zero })).split();
    core::array::from_fn(|v| if v == 0 { lo } else { hi })
}

/// `data[i] = act(data[i])`, element for element as the activation's own
/// kernel ([`relu`](crate::kernels::relu), [`silu`](crate::kernels::silu),
/// [`sigmoid`](crate::kernels::sigmoid), [`gelu_erf`](crate::kernels::gelu_erf))
/// computes it.
pub fn act_inplace(act: Act, data: &mut [f32]) {
    act_inplace_at(Level::new(), act, data)
}

/// [`act_inplace`] at `level`. Where it has no fused multiply-add, the activations round
/// differently: kernels applying one as they store must run at the level it would.
pub(crate) fn act_inplace_at(level: Level, act: Act, data: &mut [f32]) {
    let run = |data: &mut [f32]| match act {
        Act::Relu => simd_call!(level, max = Avx2, relu_simd(data)),
        Act::Silu => simd_call!(level, max = Avx2, silu_simd(data)),
        Act::Sigmoid => simd_call!(level, max = Avx2, sigmoid_simd(data)),
        Act::GeluErf => simd_call!(level, max = Avx2, gelu_erf_simd(data)),
    };
    // Shorter than a vector: through a padded copy, as those kernels do.
    if data.len() < 16 {
        let mut padded = [0.0f32; 16];
        padded[..data.len()].copy_from_slice(data);
        run(&mut padded);
        data.copy_from_slice(&padded[..data.len()]);
    } else {
        run(data);
    }
}

/// The output of `product`, which writes it to the start of `out`, with
/// `act` applied in place ([`act_inplace`]): what the activation's kernel
/// would make of it, without a buffer for the product alone.
pub fn activated<'a, F>(act: Act, out: &'a mut Vec<f32>, product: F) -> TensorView<'a>
where
    F: for<'b> FnOnce(&'b mut Vec<f32>) -> TensorView<'b>,
{
    let (shape, len, at) = {
        let y = product(out);
        let at = y.data.as_ptr();
        (y.shape.into_owned(), y.data.len(), at)
    };
    assert!(
        std::ptr::eq(at, out.as_ptr()),
        "activated: the product is not at the start of its buffer"
    );
    act_inplace(act, &mut out[..len]);
    TensorView::from_slice(&out[..len], shape)
}

/// `act_simd!(name, f)` and `bias_act_simd!(name, f)` define the in-place
/// kernels of one activation: one function per activation, so that no body
/// (or, in debug builds, stack frame) holds all of them.
macro_rules! act_simd {
    ($name:ident, $f:path) => {
        /// Needs `data.len() >= 16`.
        #[simd]
        fn $name<S: Simd>(simd: S, data: &mut [f32]) {
            map_inplace!(simd, data, $f)
        }
    };
}

macro_rules! bias_act_simd {
    ($name:ident, $f:path) => {
        /// Needs `data.len() >= 16`.
        #[simd]
        fn $name<S: Simd>(simd: S, data: &mut [f32], bias: f32) {
            map_inplace!(simd, data, bias, $f)
        }
    };
}

act_simd!(relu_simd, vm::relu);
act_simd!(silu_simd, vm::silu);
act_simd!(sigmoid_simd, vm::sigmoid);
act_simd!(gelu_erf_simd, vm::gelu_erf);
bias_act_simd!(bias_add_simd, identity);
bias_act_simd!(bias_relu_simd, vm::relu);
bias_act_simd!(bias_silu_simd, vm::silu);
bias_act_simd!(bias_sigmoid_simd, vm::sigmoid);
bias_act_simd!(bias_gelu_erf_simd, vm::gelu_erf);

#[inline(always)]
fn identity<S: Simd>(x: f32x16<S>) -> f32x16<S> {
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::test_util::{AWKWARD_LENS, Rng, assert_same_bits, levels};

    /// `bias_act_inplace` is the bias added, then `act_inplace`, bit for bit.
    #[test]
    fn test_bias_act_is_bias_then_the_activation_at_every_level_and_length() {
        let mut rng = Rng::new(7);
        for level in levels() {
            for act in [
                None,
                Some(Act::Relu),
                Some(Act::Silu),
                Some(Act::Sigmoid),
                Some(Act::GeluErf),
            ] {
                for &len in AWKWARD_LENS {
                    let src = rng.vec(len, -12.0, 12.0);
                    let bias = rng.f32(-2.0, 2.0);
                    let mut got = src.clone();
                    bias_act_inplace_at(level, &mut got, bias, act);
                    let mut want: Vec<f32> = src.iter().map(|&x| x + bias).collect();
                    if let Some(act) = act {
                        act_inplace_at(level, act, &mut want);
                    }
                    assert_same_bits(&got, &want, &format!("{level:?} {act:?} len {len}"));
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
                bias_act_inplace_at(level, &mut data, 1.0, None);
                assert!(data.iter().all(|&v| v == 1.0), "{level:?} len {len}");
            }
        }
    }

    /// `act_inplace` is each activation's own kernel, bit for bit, at every
    /// length, signed zeros and saturating values included.
    #[test]
    fn test_act_inplace_is_the_activation_kernel() {
        let mut rng = Rng::new(3);
        for &len in AWKWARD_LENS {
            let mut src = rng.vec(len, -12.0, 12.0);
            for (x, special) in src.iter_mut().zip([-0.0, 0.0, 100.0, -100.0]) {
                *x = special;
            }
            let x = TensorView::from_slice(&src, vec![len]);
            for act in [Act::Relu, Act::Silu, Act::Sigmoid, Act::GeluErf] {
                let mut buf = Vec::new();
                let want = match act {
                    Act::Relu => crate::kernels::relu(&x, &mut buf),
                    Act::Silu => crate::kernels::silu(&x, &mut buf),
                    Act::Sigmoid => crate::kernels::sigmoid(&x, &mut buf),
                    Act::GeluErf => crate::kernels::gelu_erf(&x, &mut buf),
                };
                let mut got = src.clone();
                act_inplace(act, &mut got);
                assert_same_bits(&got, &want.data, &format!("{act:?} len {len}"));
            }
        }
    }

    /// `activated` hands the product its buffer and applies the activation to
    /// what it wrote, keeping its shape.
    #[test]
    fn test_activated_is_the_product_then_the_activation() {
        let mut rng = Rng::new(5);
        let (a, b) = (rng.vec(3 * 20, -1.0, 1.0), rng.vec(20 * 24, -1.0, 1.0));
        let a = TensorView::from_slice(&a, vec![3, 20]);
        let b = TensorView::from_slice(&b, vec![20, 24]);
        let (mut p, mut r, mut out) = (Vec::new(), Vec::new(), Vec::new());
        let product = crate::kernels::matmul(&a, &b, &mut p);
        let want = crate::kernels::gelu_erf(&product, &mut r);
        let got = activated(Act::GeluErf, &mut out, |out| {
            crate::kernels::matmul(&a, &b, out)
        });
        assert_eq!(got.shape, want.shape);
        assert_same_bits(&got.data, &want.data, "matmul then gelu");
    }

    #[test]
    fn test_bias_act_leaves_neighbours_alone() {
        let mut buf = vec![5.0f32; 40];
        bias_act_inplace(&mut buf[10..23], 1.0, Some(Act::Relu));
        assert!(buf[..10].iter().all(|&v| v == 5.0));
        assert!(buf[10..23].iter().all(|&v| v == 6.0));
        assert!(buf[23..].iter().all(|&v| v == 5.0));
    }
}
