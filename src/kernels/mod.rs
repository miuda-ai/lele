pub mod activations;
#[cfg(target_arch = "x86_64")]
pub mod avx;

/// Denormals are the smallest `f32`s (below ~1.2e-38) and show up naturally in
/// long-tail data like softmax probabilities. On x86 a multiply-add with a
/// denormal operand takes a microcode assist, which costs ~100 cycles: the same
/// GEMM runs 100x+ slower. ARM evaluates them in hardware, so this is an x86-only
/// cliff. Every BLAS and ONNX Runtime flushes them; `FTZ` (results) and `DAZ`
/// (operands) in `MXCSR` do the same here.
///
/// `MXCSR` is per-thread, so the flag is checked once per thread. The numerical
/// effect is confined to values below the normal `f32` range, which are zeroed;
/// set `LELE_FLUSH_DENORMALS=0` to keep them.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn enable_flush_denormals() {
    use std::cell::Cell;
    use std::sync::OnceLock;

    static WANTED: OnceLock<bool> = OnceLock::new();
    let wanted = *WANTED.get_or_init(|| {
        std::env::var_os("LELE_FLUSH_DENORMALS").is_none_or(|v| v != "0")
    });
    if !wanted {
        return;
    }

    thread_local! {
        static DONE: Cell<bool> = const { Cell::new(false) };
    }
    DONE.with(|done| {
        if !done.get() {
            use core::arch::x86_64::{_mm_getcsr, _mm_setcsr};
            // SAFETY: reading and writing MXCSR needs no memory safety.
            unsafe { _mm_setcsr(_mm_getcsr() | (1 << 15) | (1 << 6)) };
            done.set(true);
        }
    });
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn enable_flush_denormals() {}
pub(crate) mod bias_act;
pub(crate) mod binary;
pub mod conv1d;
pub(crate) mod conv1d_direct;
pub mod conv2d;
pub mod conv_integer;
pub mod fft;
pub mod gemm;
pub mod manipulation;
pub mod math;
pub mod matmul;
pub(crate) mod matmul_dot;
pub(crate) mod matmul_nn;
#[cfg(target_arch = "aarch64")]
pub mod neon;
pub mod norm;
pub mod pooling;
pub mod qgemm;
pub(crate) mod qgemm_vnni;
pub mod qmatmul_i8;
pub mod quantization;
pub mod rnn;
pub mod shape;
pub(crate) mod simd;
pub(crate) mod simd_math;
#[cfg(test)]
pub(crate) mod test_util;
pub mod timing;
pub mod utils;
pub use conv1d::conv1d;
pub use conv1d::conv1d_fused;
pub use conv2d::{
    conv_transpose, conv2d, conv2d_fused, conv2d_silu, gather_elements, max_pool2d,
    print_conv_stats, reset_conv_stats, resize_nearest, topk,
};
pub use conv_integer::{ConvWeights, conv_integer, conv_integer_packed};
pub use gemm::{gemm, matmul, matmul_fused_add};
pub use manipulation::*;
pub use manipulation::{split, where_op};
pub use math::*;
pub use math::{cos, cumsum, einsum_bs_d_bsd, exp, expand, greater_or_equal, hard_sigmoid, less, leaky_relu, logical_and, logical_or, logical_xor, min_max, neg, range, sin, tile};
pub use norm::*;
pub use pooling::*;
pub use pooling::{average_pool2d, global_average_pool};
pub use qgemm::{QWeights, mat_mul_integer_qweights, qlinear_dynamic, qlinear_static};
pub use qmatmul_i8::{QuantizedWeights, prepare_quantized_weights, qmatmul_i8};
pub use quantization::*;
pub use rnn::*;
pub use shape::*;
