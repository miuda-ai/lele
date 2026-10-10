//! Static-QDQ matmul against int8 weights.
//!
//! ONNX QDQ graphs express a quantized linear layer as a dequantized activation
//! multiplied by a dequantized weight. Folding the weight's `DequantizeLinear`
//! into an f32 constant, as constant folding otherwise would, throws away the
//! only thing that makes an integer kernel possible. The compiler instead keeps
//! the int8 weight and emits a call here.
//!
//! Which kernel actually runs is a runtime decision made once per weight, when
//! it is prepared:
//!
//! * Everywhere but on aarch64, [`crate::kernels::qgemm`] runs it in integers:
//!   with AVX-512 VNNI through `vpdpbusd` (four times the AVX2 rate), elsewhere
//!   through 16-bit multiply-adds (`vpmaddwd` with AVX2), at 1.5 times the f32
//!   GEMM's rate.
//! * On aarch64 the weight is dequantized to f32 once and the ordinary
//!   [`matmul`] runs, which is bit-identical to not having taken this path at
//!   all. (Neon's widening multiplies do 8 MAC per instruction, no more than
//!   f32 multiply-adds.)

use crate::kernels::qgemm::{QWeights, qlinear_static};
use crate::tensor::TensorView;

/// A weight tensor prepared for whichever quantized path this machine can run.
pub enum QuantizedWeights {
    /// Packed for [`crate::kernels::qgemm`].
    Int(QWeights),
    /// Dequantized to f32 `[k, n]`; the ordinary f32 GEMM handles it.
    Float { data: Vec<f32>, k: usize, n: usize },
}

impl QuantizedWeights {
    /// True when an integer kernel will run, rather than the f32 fallback.
    pub fn is_integer(&self) -> bool {
        !matches!(self, QuantizedWeights::Float { .. })
    }
}

/// Prepares a row-major `[k, n]` int8 weight with a per-output-channel scale.
///
/// `raw` is the weight's bytes straight out of the model blob, reinterpreted as
/// i8. `w_scale` holds either one scale for the whole tensor or one per output
/// channel. Symmetric quantization (zero point 0) is assumed; the compiler only
/// routes weights here after checking that.
pub fn prepare_quantized_weights(
    raw: &[u8],
    k: usize,
    n: usize,
    w_scale: &[f32],
) -> QuantizedWeights {
    debug_assert_eq!(raw.len(), k * n);
    debug_assert!(w_scale.len() == 1 || w_scale.len() == n);

    if cfg!(not(target_arch = "aarch64")) {
        return QuantizedWeights::Int(QWeights::new(raw, k, n, true, &[0], w_scale));
    }

    let mut data = vec![0f32; k * n];
    for kk in 0..k {
        for j in 0..n {
            let s = if w_scale.len() == 1 { w_scale[0] } else { w_scale[j] };
            data[kk * n + j] = (raw[kk * n + j] as i8) as f32 * s;
        }
    }
    QuantizedWeights::Float { data, k, n }
}

/// `out = a @ dequantize(w)`, computed in integers where the hardware allows.
///
/// `a` is the already-dequantized activation, still on the quantization grid,
/// with per-tensor `a_scale` and `a_zero_point`.
pub fn qmatmul_i8<'a>(
    a: &TensorView<'_, f32>,
    a_scale: &TensorView<'_, f32>,
    a_zero_point: &TensorView<'_, f32>,
    qw: &QuantizedWeights,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    // ONNX MatMul semantics: a 1-D activation is a vector @ matrix product.
    // View the vector as a single row, compute `[1, n]`, and drop the leading
    // dimension.
    if a.shape.len() == 1 {
        let k = a.shape[0];
        let row = TensorView::from_slice(&a.data, vec![1, k]);
        let n = {
            let r = qmatmul_i8(&row, a_scale, a_zero_point, qw, out);
            r.shape[r.shape.len() - 1]
        };
        return TensorView::from_slice(out, vec![n]);
    }
    match qw {
        QuantizedWeights::Float { data, k, n } => {
            let w = TensorView::from_slice(data.as_slice(), vec![*k, *n]);
            crate::kernels::matmul(a, &w, out)
        }
        QuantizedWeights::Int(w) => {
            let sa = a_scale.data.first().copied().unwrap_or(1.0);
            let za = a_zero_point.data.first().copied().unwrap_or(0.0) as i32;
            qlinear_static(a, sa, za, w, out)
        }
    }
}
