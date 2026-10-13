use crate::kernels::simd::simd_call;
use crate::tensor::TensorView;
use fearless_simd::{Level, Simd, f32x8};
use fearless_simd_macros::simd;

// MatMulInteger operation: accepts f32 tensors and converts internally to u8
pub fn mat_mul_integer<'a, 'b, 'c>(
    a: &TensorView<'b, f32>,
    b: &TensorView<'c, f32>,
    a_zero_point: Option<&TensorView<'b, f32>>,
    b_zero_point: Option<&TensorView<'c, f32>>,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    mat_mul_integer_with_scale_bias(a, b, a_zero_point, b_zero_point, None, None, out)
}

// MatMulInteger with optional bias fusion (backward compatibility)
pub fn mat_mul_integer_with_bias<'a, 'b, 'c>(
    a: &TensorView<'b, f32>,
    b: &TensorView<'c, f32>,
    a_zero_point: Option<&TensorView<'b, f32>>,
    b_zero_point: Option<&TensorView<'c, f32>>,
    bias: Option<&TensorView<'b, f32>>,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    mat_mul_integer_with_scale_bias(a, b, a_zero_point, b_zero_point, None, bias, out)
}

// MatMulInteger with optional scale and bias fusion (full fusion)
pub fn mat_mul_integer_with_scale_bias<'a, 'b, 'c>(
    a: &TensorView<'b, f32>,
    b: &TensorView<'c, f32>,
    a_zero_point: Option<&TensorView<'b, f32>>,
    b_zero_point: Option<&TensorView<'c, f32>>,
    scale: Option<&TensorView<'b, f32>>,
    bias: Option<&TensorView<'b, f32>>,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    mat_mul_integer_with_scale_bias_activation(
        a,
        b,
        a_zero_point,
        b_zero_point,
        scale,
        bias,
        false,
        out,
    )
}

// MatMulInteger with optional scale, bias, and ReLU fusion
pub fn mat_mul_integer_with_scale_bias_relu<'a, 'b, 'c>(
    a: &TensorView<'b, f32>,
    b: &TensorView<'c, f32>,
    a_zero_point: Option<&TensorView<'b, f32>>,
    b_zero_point: Option<&TensorView<'c, f32>>,
    scale: Option<&TensorView<'b, f32>>,
    bias: Option<&TensorView<'b, f32>>,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    mat_mul_integer_with_scale_bias_activation(
        a,
        b,
        a_zero_point,
        b_zero_point,
        scale,
        bias,
        true,
        out,
    )
}

/// Fully-fused quantized linear: DynamicQuantizeLinear + MatMulInteger + Scale + Bias [+ ReLU].
///
/// The weight arrives as a tensor here, so it is packed for [`crate::kernels::qgemm`] on
/// every call; generated code packs the weights it knows once instead
/// ([`crate::kernels::QWeights`]).
pub fn fused_quantized_linear<'a>(
    input: &TensorView<'_, f32>,
    weight_int8: &TensorView<'_, f32>,
    weight_scale: &TensorView<'_, f32>,
    weight_zero: &TensorView<'_, f32>,
    bias: &TensorView<'_, f32>,
    apply_relu: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    use crate::kernels::qgemm::{QWeights, qlinear_dynamic};
    let dims = weight_int8.shape.len();
    let (k, n) = (weight_int8.shape[dims - 2], weight_int8.shape[dims - 1]);
    let mut zero_point: Vec<i32> = weight_zero.data.iter().map(|&z| z as i32).collect();
    if zero_point.is_empty() {
        zero_point.push(0);
    }
    let w = QWeights::from_f32_codes(&weight_int8.data[..k * n], k, n, &zero_point, &weight_scale.data);
    let bias = (!bias.data.is_empty()).then_some(&bias.data[..]);
    qlinear_dynamic(input, &w, bias, apply_relu, out)
}

/// `MatMulInteger` with B given at run time, through [`crate::kernels::qgemm`]: it takes
/// `u8` and `i8` weights alike.
fn mat_mul_integer_with_scale_bias_activation<'a, 'b, 'c>(
    a: &TensorView<'b, f32>,
    b: &TensorView<'c, f32>,
    a_zero_point: Option<&TensorView<'b, f32>>,
    b_zero_point: Option<&TensorView<'c, f32>>,
    scale: Option<&TensorView<'b, f32>>,
    bias: Option<&TensorView<'b, f32>>,
    apply_relu: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    let a_zp = a_zero_point.and_then(|z| z.data.first()).map_or(0, |&z| z as i32);
    let b_zp: Vec<i32> = match b_zero_point {
        Some(z) if !z.data.is_empty() => z.data.iter().map(|&z| z as i32).collect(),
        _ => vec![0],
    };
    crate::kernels::qgemm::mat_mul_integer_f32(
        a,
        b,
        a_zp,
        &b_zp,
        scale.map(|s| &s.data[..]),
        bias.map(|b| &b.data[..]),
        apply_relu,
        out,
    )
}

/// ONNX `DynamicQuantizeLinear`: `x` quantized to `u8` over its own range (widened to
/// include zero), with the scale and zero point it picked. The codes are the same ones the
/// fused path ([`crate::kernels::qgemm::qlinear_dynamic`]) multiplies with.
pub fn dynamic_quantize_linear<'a, 'b>(
    x: &TensorView<'b, f32>,
    out_y_storage: &'a mut Vec<f32>,
    out_scale: &'a mut Vec<f32>,
    out_zp: &'a mut Vec<f32>,
) -> (
    TensorView<'a, f32>,
    TensorView<'a, f32>,
    TensorView<'a, f32>,
) {
    let (scale, zero_point) = crate::kernels::qgemm::dynamic_quant_params(&x.data);
    out_scale.clear();
    out_scale.push(scale);
    out_zp.clear();
    out_zp.push(zero_point as f32);
    let p = PerTensor { scale, zero_point: zero_point as f32, qmin: 0.0, qmax: 255.0, round_trip: false };
    quantize_per_tensor(Level::new(), &p, &x.data, out_y_storage);
    (
        TensorView::from_slice(out_y_storage, x.shape.to_vec()),
        TensorView::from_slice(out_scale, vec![1]),
        TensorView::from_slice(out_zp, vec![1]),
    )
}

/// `QuantizeLinear` with one scale and zero point for the whole tensor
/// (`clamp(round_ties_even(x / scale) + zero_point, qmin, qmax)`), or with `round_trip` the
/// `DequantizeLinear` after it as well (`(q - zero_point) * scale`).
#[derive(Clone, Copy)]
pub(crate) struct PerTensor {
    pub scale: f32,
    pub zero_point: f32,
    pub qmin: f32,
    pub qmax: f32,
    pub round_trip: bool,
}

impl PerTensor {
    #[inline(always)]
    fn apply(&self, x: f32) -> f32 {
        let q = ((x / self.scale).round_ties_even() + self.zero_point).clamp(self.qmin, self.qmax);
        if self.round_trip { (q - self.zero_point) * self.scale } else { q }
    }
}

/// Writes `p` applied to each element of `src` to `out` (resized to fit). The vectors do
/// the scalar definition's operations in its order (an exact division, not a reciprocal),
/// so every element matches it.
pub(crate) fn quantize_per_tensor(level: Level, p: &PerTensor, src: &[f32], out: &mut Vec<f32>) {
    // Every element is written, so the buffer is not filled first: workspace buffers
    // alternate between tensor sizes, and zeroing them was 4-5% of Smart Turn.
    crate::kernels::utils::ensure_capacity(out, src.len());
    simd_call!(level, quantize_per_tensor_simd(p, src, out))
}

#[simd]
fn quantize_per_tensor_simd<S: Simd>(simd: S, p: &PerTensor, src: &[f32], dst: &mut [f32]) {
    use fearless_simd::prelude::*;
    let (scale, zp) = (f32x8::splat(simd, p.scale), f32x8::splat(simd, p.zero_point));
    let (lo, hi) = (f32x8::splat(simd, p.qmin), f32x8::splat(simd, p.qmax));
    let (src_chunks, src_rest) = src.as_chunks::<8>();
    let (dst_chunks, dst_rest) = dst.as_chunks_mut::<8>();
    for (x, y) in src_chunks.iter().zip(dst_chunks) {
        let q = ((f32x8::from_slice(simd, x) / scale).round_ties_even() + zp).max(lo).min(hi);
        let r = if p.round_trip { (q - zp) * scale } else { q };
        r.store_slice(y);
    }
    for (x, y) in src_rest.iter().zip(dst_rest) {
        *y = p.apply(*x);
    }
}

// ---------------------------------------------------------------------------
// QuantizeLinear / DequantizeLinear (static QDQ models)
//
// Integer tensors are carried in `TensorView<f32>` holding the integer codes as
// f32 values, matching the convention already used by `dynamic_quantize_linear`
// and `mat_mul_integer`.
// ---------------------------------------------------------------------------

/// Layout of the scale/zero-point operands relative to the data tensor.
enum QParamLayout {
    /// One scale for the whole tensor.
    PerTensor,
    /// One scale per slice along `axis`.
    PerAxis { outer: usize, dim: usize, inner: usize },
    /// One scale per block of `block_size` elements along `axis`.
    Blocked {
        outer: usize,
        dim: usize,
        inner: usize,
        block_size: usize,
        blocks: usize,
    },
}

fn qparam_layout(
    shape: &[usize],
    scale_len: usize,
    scale_rank: usize,
    axis: i64,
    block_size: usize,
) -> QParamLayout {
    if scale_len <= 1 {
        return QParamLayout::PerTensor;
    }
    let rank = shape.len();
    // ONNX: "When the rank of the input is 1, per-tensor quantization is
    // applied", even when block_size is set — a rank-1 input with a blocked
    // scale is the shape the blocked formula itself cannot produce, so
    // classifying it as per-axis would index the scale out of bounds.
    if rank <= 1 && block_size > 0 {
        return QParamLayout::PerTensor;
    }
    let axis = if axis < 0 { axis + rank as i64 } else { axis };
    let axis = (axis.max(0) as usize).min(rank.saturating_sub(1));
    let outer: usize = shape[..axis].iter().product();
    let dim = shape.get(axis).copied().unwrap_or(1);
    let inner: usize = shape[axis + 1..].iter().product();
    if block_size > 0 && scale_rank > 1 {
        let blocks = dim.div_ceil(block_size);
        QParamLayout::Blocked {
            outer,
            dim,
            inner,
            block_size,
            blocks,
        }
    } else {
        assert!(
            scale_len == dim,
            "per-axis quantization needs one scale per element of axis {axis} \
             (axis dim {dim}, got {scale_len} scales)"
        );
        QParamLayout::PerAxis { outer, dim, inner }
    }
}

/// Applies `f(value, scale, zero_point)` over `x`, broadcasting the quantization
/// parameters according to `axis` / `block_size`.
fn qdq_apply<F>(
    x: &TensorView<'_, f32>,
    scale: &TensorView<'_, f32>,
    zero_point: Option<&TensorView<'_, f32>>,
    axis: i64,
    block_size: usize,
    out: &mut Vec<f32>,
    f: F,
) where
    F: Fn(f32, f32, f32) -> f32,
{
    let zp_data: &[f32] = zero_point.map(|z| z.data.as_ref()).unwrap_or(&[]);
    let scale_data = scale.data.as_ref();
    let zp_at = |i: usize| -> f32 {
        match zp_data.len() {
            0 => 0.0,
            1 => zp_data[0],
            _ => zp_data[i],
        }
    };

    out.clear();
    out.reserve(x.data.len());

    match qparam_layout(
        &x.shape,
        scale_data.len(),
        scale.shape.len(),
        axis,
        block_size,
    ) {
        QParamLayout::PerTensor => {
            let s = scale_data.first().copied().unwrap_or(1.0);
            let z = zp_at(0);
            out.extend(x.data.iter().map(|&v| f(v, s, z)));
        }
        QParamLayout::PerAxis { outer, dim, inner } => {
            for _o in 0..outer {
                for d in 0..dim {
                    let s = scale_data[d];
                    let z = zp_at(d);
                    let base = out.len();
                    out.extend(x.data[base..base + inner].iter().map(|&v| f(v, s, z)));
                }
            }
        }
        QParamLayout::Blocked {
            outer,
            dim,
            inner,
            block_size,
            blocks,
        } => {
            for o in 0..outer {
                for d in 0..dim {
                    let block = d / block_size;
                    let p_base = (o * blocks + block) * inner;
                    let base = out.len();
                    for i in 0..inner {
                        let s = scale_data[p_base + i];
                        let z = zp_at(p_base + i);
                        out.push(f(x.data[base + i], s, z));
                    }
                }
            }
        }
    }
}

/// ONNX `QuantizeLinear`: `y = saturate(round_half_even(x / scale) + zero_point)`.
///
/// `qmin`/`qmax` are the saturation bounds of the target integer type and are
/// resolved by the compiler from the zero-point dtype (or the `output_dtype`
/// attribute).
pub fn quantize_linear<'a>(
    x: &TensorView<'_, f32>,
    y_scale: &TensorView<'_, f32>,
    y_zero_point: Option<&TensorView<'_, f32>>,
    axis: i64,
    block_size: usize,
    qmin: f32,
    qmax: f32,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    if let Some(p) = per_tensor(x, y_scale, y_zero_point, axis, block_size, qmin, qmax, false) {
        quantize_per_tensor(Level::new(), &p, &x.data, out);
    } else {
        qdq_apply(x, y_scale, y_zero_point, axis, block_size, out, |v, s, z| {
            ((v / s).round_ties_even() + z).clamp(qmin, qmax)
        });
    }
    TensorView::from_slice(out.as_slice(), x.shape.to_vec())
}

/// The parameters as one [`PerTensor`], when they are per tensor.
fn per_tensor(
    x: &TensorView<'_, f32>,
    scale: &TensorView<'_, f32>,
    zero_point: Option<&TensorView<'_, f32>>,
    axis: i64,
    block_size: usize,
    qmin: f32,
    qmax: f32,
    round_trip: bool,
) -> Option<PerTensor> {
    let layout = qparam_layout(&x.shape, scale.data.len(), scale.shape.len(), axis, block_size);
    matches!(layout, QParamLayout::PerTensor).then(|| PerTensor {
        scale: scale.data.first().copied().unwrap_or(1.0),
        zero_point: zero_point.and_then(|z| z.data.first().copied()).unwrap_or(0.0),
        qmin,
        qmax,
        round_trip,
    })
}

/// ONNX `DequantizeLinear`: `y = (x - zero_point) * scale`.
pub fn dequantize_linear<'a>(
    x: &TensorView<'_, f32>,
    x_scale: &TensorView<'_, f32>,
    x_zero_point: Option<&TensorView<'_, f32>>,
    axis: i64,
    block_size: usize,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    qdq_apply(x, x_scale, x_zero_point, axis, block_size, out, |v, s, z| {
        (v - z) * s
    });
    TensorView::from_slice(out.as_slice(), x.shape.to_vec())
}

/// Fused `QuantizeLinear` → `DequantizeLinear` round trip ("fake quantize").
///
/// QDQ graphs express activation quantization as a `QuantizeLinear` immediately
/// followed by a `DequantizeLinear` sharing the same scale and zero point, so
/// the pair is a no-op on the tensor type and only exists to clamp the value to
/// the quantization grid. Computing both steps in a single pass gives a
/// bit-identical result while halving the memory traffic and removing the
/// intermediate buffer.
///
/// A per-tensor scale (what activation quantization almost always uses) takes the SIMD
/// path. It divides exactly rather than multiplying by a reciprocal: that was measured
/// once and bought only 3-7% of a memory-bound kernel, while landing within half an ulp of
/// every rounding tie.
pub fn fake_quantize_linear<'a>(
    x: &TensorView<'_, f32>,
    scale: &TensorView<'_, f32>,
    zero_point: Option<&TensorView<'_, f32>>,
    axis: i64,
    block_size: usize,
    qmin: f32,
    qmax: f32,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    if let Some(p) = per_tensor(x, scale, zero_point, axis, block_size, qmin, qmax, true) {
        quantize_per_tensor(Level::new(), &p, &x.data, out);
        return TensorView::from_slice(out.as_slice(), x.shape.to_vec());
    }

    qdq_apply(x, scale, zero_point, axis, block_size, out, |v, s, z| {
        (((v / s).round_ties_even() + z).clamp(qmin, qmax) - z) * s
    });
    TensorView::from_slice(out.as_slice(), x.shape.to_vec())
}

/// Saturation bounds for an ONNX integer tensor element type.
/// Returns `None` for types that are not valid quantization targets.
pub fn quant_range(onnx_dtype: i32) -> Option<(f32, f32)> {
    match onnx_dtype {
        2 => Some((0.0, 255.0)),                            // UINT8
        3 => Some((-128.0, 127.0)),                         // INT8
        4 => Some((0.0, 65535.0)),                          // UINT16
        5 => Some((-32768.0, 32767.0)),                     // INT16
        6 => Some((i32::MIN as f32, i32::MAX as f32)),      // INT32
        12 => Some((0.0, u32::MAX as f32)),                 // UINT32
        21 => Some((0.0, 15.0)),                            // UINT4
        22 => Some((-8.0, 7.0)),                            // INT4
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::test_util::{AWKWARD_LENS, Rng, levels};

    #[test]
    fn test_per_tensor_quantize_matches_its_definition_at_every_level() {
        let mut rng = Rng::new(17);
        for level in levels() {
            for &len in AWKWARD_LENS {
                for (scale, zero_point, qmin, qmax) in [(0.02, 128.0, 0.0, 255.0), (0.05, 0.0, -128.0, 127.0), (0.3, -3.0, -8.0, 7.0)] {
                    // Values on the grid, at its rounding ties, and past both ends.
                    let mut x = rng.vec(len, -40.0, 40.0);
                    for (i, v) in x.iter_mut().enumerate().step_by(3) {
                        *v = scale * ((i % 300) as f32 * 0.5 - 75.0);
                    }
                    for round_trip in [false, true] {
                        let p = PerTensor { scale, zero_point, qmin, qmax, round_trip };
                        let mut got = Vec::new();
                        quantize_per_tensor(level, &p, &x, &mut got);
                        for (i, (&g, &v)) in got.iter().zip(&x).enumerate() {
                            let q = ((v / scale).round_ties_even() + zero_point).clamp(qmin, qmax);
                            let want = if round_trip { (q - zero_point) * scale } else { q };
                            // `==`, not bits: a clamp may keep either zero's sign.
                            assert!(g == want, "{level:?} len {len} round_trip {round_trip} at {i}: {v} -> {g}, want {want}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_dynamic_quantize_linear_follows_onnx() {
        let x: Vec<f32> = (0..37).map(|i| i as f32 * 0.37 - 4.1).collect();
        let (mut y, mut s, mut z) = (Vec::new(), Vec::new(), Vec::new());
        let (q, scale, zp) = dynamic_quantize_linear(&TensorView::from_slice(&x, vec![37]), &mut y, &mut s, &mut z);
        let (min, max) = (-4.1f32, 36.0 * 0.37 - 4.1);
        let want_scale = (max - min) / 255.0;
        let want_zp = (-min / want_scale).round_ties_even().clamp(0.0, 255.0);
        assert_eq!((scale.data[0], zp.data[0]), (want_scale, want_zp));
        for (&g, &v) in q.data.iter().zip(&x) {
            assert_eq!(g, ((v / want_scale).round_ties_even() + want_zp).clamp(0.0, 255.0));
        }
    }
}
