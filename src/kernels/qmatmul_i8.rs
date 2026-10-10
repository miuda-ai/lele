//! Static-QDQ matmul against int8 weights.
//!
//! ONNX QDQ graphs express a quantized linear layer as a dequantized activation
//! multiplied by a dequantized weight. Folding the weight's `DequantizeLinear`
//! into an f32 constant, as constant folding otherwise would, throws away the
//! only thing that makes an integer kernel possible. The compiler instead keeps
//! the int8 weight and emits a call here, which [`crate::kernels::qgemm`] runs in
//! whichever layout suits the machine (see [`QWeights`]).

use crate::kernels::qgemm::{QWeights, qlinear_static};
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

/// `out = a @ dequantize(w)`, computed in integers where the hardware allows.
///
/// `a` is the already-dequantized activation, still on the quantization grid,
/// with per-tensor `a_scale` and `a_zero_point`.
pub fn qmatmul_i8<'a>(
    a: &TensorView<'_, f32>,
    a_scale: &TensorView<'_, f32>,
    a_zero_point: &TensorView<'_, f32>,
    w: &QWeights,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    // ONNX MatMul semantics: a 1-D activation is a vector @ matrix product.
    // View the vector as a single row, compute `[1, n]`, and drop the leading
    // dimension.
    if a.shape.len() == 1 {
        let k = a.shape[0];
        let row = TensorView::from_slice(&a.data, vec![1, k]);
        let n = {
            let r = qmatmul_i8(&row, a_scale, a_zero_point, w, out);
            r.shape[r.shape.len() - 1]
        };
        return TensorView::from_slice(out, vec![n]);
    }
    let sa = a_scale.data.first().copied().unwrap_or(1.0);
    let za = a_zero_point.data.first().copied().unwrap_or(0.0) as i32;
    qlinear_static(a, sa, za, w, out)
}
