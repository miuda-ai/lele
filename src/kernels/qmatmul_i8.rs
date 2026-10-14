//! Static-QDQ matmul against int8 weights.
//!
//! ONNX QDQ graphs express a quantized linear layer as a dequantized activation
//! multiplied by a dequantized weight. Folding the weight's `DequantizeLinear`
//! into an f32 constant, as constant folding otherwise would, throws away the
//! only thing that makes an integer kernel possible. The compiler instead keeps
//! the int8 weight and emits a call here, which [`crate::kernels::qgemm`] runs in
//! whichever layout suits the machine (see [`QWeights`]).

use crate::kernels::qgemm::{QWeights, Requant, qlinear_static};
use crate::tensor::TensorView;

/// Prepares a row-major `[k, n]` int8 weight with a per-output-channel scale.
///
/// `raw` is the weight's bytes straight out of the model blob, reinterpreted as
/// i8. `w_scale` holds either one scale for the whole tensor or one per output
/// channel. Symmetric quantization (zero point 0) is assumed; the compiler only
/// routes weights here after checking that.
pub fn prepare_quantized_weights(raw: &[u8], k: usize, n: usize, w_scale: &[f32]) -> QWeights {
    debug_assert_eq!(raw.len(), k * n);
    debug_assert!(w_scale.len() == 1 || w_scale.len() == n);
    QWeights::new(raw, k, n, true, &[0], w_scale)
}

/// `out = requant?(a @ dequantize(w)) + bias?`, computed in integers where the hardware
/// allows.
///
/// `a` is the activation of a `QuantizeLinear` -> `DequantizeLinear` pair with per-tensor
/// `a_scale` and `a_zero_point`, before or after the pair: it is quantized either way.
/// `requant` is the pair that follows the product, if any, and `bias` is added after it;
/// both are applied as the product is written. A 1-D `a` is one row, as ONNX MatMul
/// takes it, and gives a 1-D result.
pub fn qmatmul_i8<'a>(
    a: &TensorView<'_, f32>,
    a_scale: &TensorView<'_, f32>,
    a_zero_point: &TensorView<'_, f32>,
    w: &QWeights,
    requant: Option<Requant>,
    bias: Option<&TensorView<'_, f32>>,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    let sa = a_scale.data.first().copied().unwrap_or(1.0);
    let za = a_zero_point.data.first().copied().unwrap_or(0.0) as i32;
    qlinear_static(a, sa, za, w, requant, bias.map(|b| &b.data[..]), out)
}
