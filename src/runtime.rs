//! Runtime support for generated models.
//!
//! Every generated model struct wraps a [`Weights`] and derefs to it, so the
//! generated code reads weights and calls the fused ops below as `self.*`.

use aligned_vec::AVec;

use crate::kernels::{self, ConvWeights, QWeights};
use crate::tensor::TensorView;

/// A model's weight blob plus the weights prepared from it, when the model is
/// created, for kernels that cannot read them as they are stored.
pub struct Weights<'a> {
    data: &'a [u8],
    /// The weights [`Self::with_f16`], [`Self::with_u8`] and [`Self::with_i8`]
    /// widened to f32, one after another, each starting on a cache line.
    widened: AVec<f32>,
    /// Byte offset of each widened weight and where it starts in `widened`,
    /// by offset, for each type stored.
    f16_index: Vec<(usize, usize)>,
    u8_index: Vec<(usize, usize)>,
    i8_index: Vec<(usize, usize)>,
    /// The weights [`Self::with_i32_i64`] and [`Self::with_bool_i64`] widened
    /// to i64, laid out as `widened` is.
    ints: AVec<i64>,
    i32_index: Vec<(usize, usize)>,
    bool_index: Vec<(usize, usize)>,
    /// Weights of the integer products, packed by [`Self::with_qweights`] and
    /// [`Self::with_conv_weights`], by the number generated code reads them by.
    qweights: Vec<QWeights>,
    conv_weights: Vec<ConvWeights>,
}

impl<'a> Weights<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            widened: AVec::new(0),
            f16_index: Vec::new(),
            u8_index: Vec::new(),
            i8_index: Vec::new(),
            ints: AVec::new(0),
            i32_index: Vec::new(),
            bool_index: Vec::new(),
            qweights: Vec::new(),
            conv_weights: Vec::new(),
        }
    }

    /// Converts the f16 weights at `(byte offset, byte length)` to f32 once,
    /// so that [`Self::weight_f16`] reads them without converting on every
    /// call.
    pub fn with_f16(mut self, regions: &[(usize, usize)]) -> Self {
        use half::slice::HalfFloatSliceExt;
        self.f16_index = widen(self.data, &mut self.widened, regions, 2, |bytes, out| {
            let halves: Vec<half::f16> = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&b| half::f16::from_le_bytes(b))
                .collect();
            halves.convert_to_f32_slice(out);
        });
        self
    }

    /// Widens the u8 weights at `(byte offset, byte length)` to f32 once, for
    /// [`Self::weight_u8`].
    pub fn with_u8(mut self, regions: &[(usize, usize)]) -> Self {
        self.u8_index = widen(self.data, &mut self.widened, regions, 1, |bytes, out| {
            for (o, &b) in out.iter_mut().zip(bytes) {
                *o = b as f32;
            }
        });
        self
    }

    /// Widens the i8 weights at `(byte offset, byte length)` to f32 once, for
    /// [`Self::weight_i8`].
    pub fn with_i8(mut self, regions: &[(usize, usize)]) -> Self {
        self.i8_index = widen(self.data, &mut self.widened, regions, 1, |bytes, out| {
            for (o, &b) in out.iter_mut().zip(bytes) {
                *o = b as i8 as f32;
            }
        });
        self
    }

    /// Widens the i32 weights at `(byte offset, byte length)` to i64 once,
    /// for [`Self::weight_i32_i64`].
    pub fn with_i32_i64(mut self, regions: &[(usize, usize)]) -> Self {
        self.i32_index = widen(self.data, &mut self.ints, regions, 4, |bytes, out| {
            for (o, &b) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
                *o = i32::from_le_bytes(b) as i64;
            }
        });
        self
    }

    /// Widens the bool weights at `(byte offset, byte length)` to i64 once,
    /// for [`Self::weight_bool_i64`].
    pub fn with_bool_i64(mut self, regions: &[(usize, usize)]) -> Self {
        self.bool_index = widen(self.data, &mut self.ints, regions, 1, |bytes, out| {
            for (o, &b) in out.iter_mut().zip(bytes) {
                *o = (b != 0) as i64;
            }
        });
        self
    }

    pub fn weight_f32(&self, offset: usize, len: usize, shape: &'a [usize]) -> TensorView<'a, f32> {
        TensorView::from_bytes_f32(&self.data[offset..offset + len], shape)
    }

    /// An i64 weight, borrowed where aligned (else copied).
    pub fn weight_i64(&self, offset: usize, len: usize, shape: &'a [usize]) -> TensorView<'a, i64> {
        let bytes = &self.data[offset..offset + len];
        // SAFETY: every bit pattern is a valid `i64`.
        match unsafe { bytes.align_to::<i64>() } {
            ([], ints, []) => TensorView::new(ints, shape),
            _ => TensorView::from_bytes_i64(bytes, shape.to_vec()),
        }
    }

    /// An i32 weight as i64: borrowed if [`Self::with_i32_i64`] widened it,
    /// widened now otherwise.
    pub fn weight_i32_i64(&self, offset: usize, len: usize, shape: &'a [usize]) -> TensorView<'_, i64> {
        match widened(&self.ints, &self.i32_index, offset, len / 4) {
            Some(data) => TensorView::new(data, shape),
            None => TensorView::from_bytes_i32_as_i64(&self.data[offset..offset + len], shape.to_vec()),
        }
    }

    /// A bool weight as i64: borrowed if [`Self::with_bool_i64`] widened it,
    /// widened now otherwise.
    pub fn weight_bool_i64(&self, offset: usize, len: usize, shape: &'a [usize]) -> TensorView<'_, i64> {
        match widened(&self.ints, &self.bool_index, offset, len) {
            Some(data) => TensorView::new(data, shape),
            None => TensorView::from_bytes_bool_as_i64(&self.data[offset..offset + len], shape.to_vec()),
        }
    }

    pub fn weight_i32(&self, offset: usize, len: usize, shape: &'a [usize]) -> TensorView<'static, i32> {
        TensorView::from_bytes_i32(&self.data[offset..offset + len], shape.to_vec())
    }

    pub fn weight_i64_f32(&self, offset: usize, len: usize, shape: &'a [usize]) -> TensorView<'static, f32> {
        TensorView::from_bytes_i64_as_f32(&self.data[offset..offset + len], shape.to_vec())
    }

    pub fn weight_i32_f32(&self, offset: usize, len: usize, shape: &'a [usize]) -> TensorView<'static, f32> {
        TensorView::from_bytes_i32_as_f32(&self.data[offset..offset + len], shape.to_vec())
    }

    /// A u8 weight as f32: borrowed if [`Self::with_u8`] widened it, widened
    /// now otherwise.
    pub fn weight_u8(&self, offset: usize, len: usize, shape: &'a [usize]) -> TensorView<'_, f32> {
        match widened(&self.widened, &self.u8_index, offset, len) {
            Some(data) => TensorView::new(data, shape),
            None => TensorView::from_bytes_u8(&self.data[offset..offset + len], shape.to_vec()),
        }
    }

    /// An i8 weight as f32: borrowed if [`Self::with_i8`] widened it, widened
    /// now otherwise.
    pub fn weight_i8(&self, offset: usize, len: usize, shape: &'a [usize]) -> TensorView<'_, f32> {
        match widened(&self.widened, &self.i8_index, offset, len) {
            Some(data) => TensorView::new(data, shape),
            None => TensorView::from_bytes_i8(&self.data[offset..offset + len], shape.to_vec()),
        }
    }

    /// An f16 weight as f32: borrowed if [`Self::with_f16`] converted it,
    /// converted now otherwise.
    pub fn weight_f16(&self, offset: usize, len: usize, shape: &'a [usize]) -> TensorView<'_, f32> {
        match widened(&self.widened, &self.f16_index, offset, len / 2) {
            Some(data) => TensorView::new(data, shape),
            None => TensorView::from_bytes_f16(&self.data[offset..offset + len], shape.to_vec()),
        }
    }

    /// An f16 weight as stored, for kernels that widen it as they read it
    /// (the B operand of [`Self::linear`] and `kernels::matmul`).
    pub fn weight_f16_raw(&self, offset: usize, len: usize, shape: &'a [usize]) -> TensorView<'a, half::f16> {
        let bytes = &self.data[offset..offset + len];
        // SAFETY: `f16` is two bytes, and every bit pattern is a valid one.
        let (head, halves, tail) = unsafe { bytes.align_to::<half::f16>() };
        let data = if head.is_empty() && tail.is_empty() {
            std::borrow::Cow::Borrowed(halves)
        } else {
            let halves = bytes.as_chunks::<2>().0.iter().map(|&b| half::f16::from_le_bytes(b));
            std::borrow::Cow::Owned(halves.collect())
        };
        TensorView {
            data,
            shape: std::borrow::Cow::Borrowed(shape),
        }
    }

    /// An f16 weight stored as [`kernels::matmul::pack_f16`] lays it out, borrowed
    /// where aligned (else copied).
    pub fn weight_f16_packed(&self, offset: usize, len: usize, shape: &'a [usize]) -> TensorView<'a, kernels::PackedF16> {
        let halves = self.weight_f16_raw(offset, len, shape).data;
        let data = match halves {
            // SAFETY: `PackedF16` is a transparent `f16`.
            std::borrow::Cow::Borrowed(h) => std::borrow::Cow::Borrowed(unsafe {
                std::slice::from_raw_parts(h.as_ptr() as *const kernels::PackedF16, h.len())
            }),
            std::borrow::Cow::Owned(h) => std::borrow::Cow::Owned(h.into_iter().map(kernels::PackedF16).collect()),
        };
        TensorView {
            data,
            shape: std::borrow::Cow::Borrowed(shape),
        }
    }

    /// Raw bytes of a weight, for prepared weights that skip the f32 conversion.
    pub fn weight_u8_raw(&self, offset: usize, len: usize) -> &'a [u8] {
        &self.data[offset..offset + len]
    }

    /// A u8 weight as stored, for kernels that widen what they read of it
    /// ([`kernels::gather_widened`]).
    pub fn weight_u8_stored(&self, offset: usize, len: usize, shape: &'a [usize]) -> TensorView<'a, u8> {
        TensorView::new(&self.data[offset..offset + len], shape)
    }

    /// An i8 weight as stored, for kernels that widen what they read of it
    /// ([`kernels::gather_widened`]).
    pub fn weight_i8_stored(&self, offset: usize, len: usize, shape: &'a [usize]) -> TensorView<'a, i8> {
        let bytes = &self.data[offset..offset + len];
        // SAFETY: `i8` and `u8` have the same size and alignment, and every
        // byte is a valid `i8`.
        let signed = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i8, bytes.len()) };
        TensorView::new(signed, shape)
    }

    /// The `[k, n]` 8-bit weight at `offset` packed for the integer GEMM, with
    /// its zero points subtracted and its scales applied, for
    /// [`Self::with_qweights`].
    #[allow(clippy::too_many_arguments)]
    pub fn pack_qweights(
        &self,
        offset: usize,
        len: usize,
        k: usize,
        n: usize,
        signed: bool,
        zero_point: &TensorView<f32>,
        scale: &TensorView<f32>,
    ) -> QWeights {
        let zp: Vec<i32> = zero_point.data.iter().map(|&z| z as i32).collect();
        QWeights::new(&self.data[offset..offset + len], k, n, signed, &zp, &scale.data)
    }

    /// The 8-bit `ConvInteger` weight at `offset` prepared with its zero
    /// points, for [`Self::with_conv_weights`].
    pub fn pack_conv_weights(
        &self,
        offset: usize,
        len: usize,
        shape: [usize; 4],
        groups: usize,
        signed: bool,
        zero_point: &TensorView<f32>,
    ) -> ConvWeights {
        let zp: Vec<i32> = zero_point.data.iter().map(|&z| z as i32).collect();
        ConvWeights::new(&self.data[offset..offset + len], &shape, groups, signed, &zp)
    }

    /// Keeps the weights of the integer products, which
    /// [`Self::pack_qweights`] packed when the model was created, for
    /// [`Self::qweights`] to read by number.
    pub fn with_qweights(mut self, qweights: Vec<QWeights>) -> Self {
        self.qweights = qweights;
        self
    }

    /// Keeps the weights of `ConvInteger`s, which
    /// [`Self::pack_conv_weights`] prepared when the model was created, for
    /// [`Self::conv_weights`] to read by number.
    pub fn with_conv_weights(mut self, conv_weights: Vec<ConvWeights>) -> Self {
        self.conv_weights = conv_weights;
        self
    }

    /// Packed weights number `i` of [`Self::with_qweights`].
    pub fn qweights(&self, i: usize) -> &QWeights {
        &self.qweights[i]
    }

    /// Prepared weights number `i` of [`Self::with_conv_weights`].
    pub fn conv_weights(&self, i: usize) -> &ConvWeights {
        &self.conv_weights[i]
    }
}

/// Appends the weights of `data` at `regions`, of `size` bytes an element, to
/// `out` through `widen`, each on a cache line; returns where each starts, by
/// byte offset.
fn widen<T: Copy + Default>(
    data: &[u8],
    out: &mut AVec<T>,
    regions: &[(usize, usize)],
    size: usize,
    widen: impl Fn(&[u8], &mut [T]),
) -> Vec<(usize, usize)> {
    let line = aligned_vec::CACHELINE_ALIGN / size_of::<T>();
    let mut regions = regions.to_vec();
    regions.sort_unstable();
    regions.dedup();
    let mut start = out.len();
    let total = regions
        .iter()
        .map(|&(_, len)| (len / size).next_multiple_of(line))
        .sum::<usize>();
    out.resize(start + total, T::default());
    let mut index = Vec::with_capacity(regions.len());
    for (offset, len) in regions {
        widen(&data[offset..offset + len], &mut out[start..][..len / size]);
        index.push((offset, start));
        start += (len / size).next_multiple_of(line);
    }
    index
}

/// The `len` elements of `widened` the weight at `offset` in `index` widened
/// to, if widened.
fn widened<'s, T>(widened: &'s [T], index: &[(usize, usize)], offset: usize, len: usize) -> Option<&'s [T]> {
    let i = index.binary_search_by_key(&offset, |&(o, _)| o).ok()?;
    Some(&widened[index[i].1..][..len])
}

/// Fused ops the compiler's patterns emit as `self.<op>(...)`.
#[allow(clippy::too_many_arguments)]
impl Weights<'_> {
    pub fn layer_norm<'c, 'd>(
        &self,
        input: &TensorView<'c>,
        scale: TensorView<'c>,
        bias: TensorView<'c>,
        epsilon: TensorView<'c>,
        _two: TensorView<'c>,
        output_buf: &'d mut Vec<f32>,
    ) -> TensorView<'d> {
        let eps = epsilon.data.first().cloned().unwrap_or(1e-5);
        kernels::layer_norm(input, &scale, &bias, -1, eps, output_buf)
    }

    /// Dynamically quantized linear layer (+ activation), its weight packed on every call.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_quantized<'c, 'd>(
        &self,
        input: &TensorView<'c>,
        weight_int8: TensorView<'c>,
        weight_scale: TensorView<'c>,
        weight_zero: TensorView<'c>,
        bias: TensorView<'c>,
        act: Option<kernels::Act>,
        output_buf: &'d mut Vec<f32>,
    ) -> TensorView<'d> {
        kernels::fused_quantized_linear(
            input,
            &weight_int8,
            &weight_scale,
            &weight_zero,
            &bias,
            act,
            output_buf,
        )
    }

    /// Dynamically quantized linear layer (+ activation) against an int8 weight packed
    /// once for the integer GEMM.
    pub fn linear_quantized_packed<'c, 'd>(
        &self,
        input: &TensorView<'c>,
        weight: &QWeights,
        bias: TensorView<'c>,
        act: Option<kernels::Act>,
        output_buf: &'d mut Vec<f32>,
    ) -> TensorView<'d> {
        let bias = (!bias.data.is_empty()).then_some(&bias.data[..]);
        kernels::qlinear_dynamic(input, weight, bias, act, output_buf)
    }

    /// MatMulInteger against a static weight packed for the integer GEMM.
    pub fn mat_mul_integer_packed<'c, 'd>(
        &self,
        a: &TensorView<'c>,
        weight: &QWeights,
        a_zero_point: Option<&TensorView<'c>>,
        output_buf: &'d mut Vec<f32>,
    ) -> TensorView<'d> {
        let za = a_zero_point.and_then(|z| z.data.first().copied()).unwrap_or(0.0) as i32;
        kernels::mat_mul_integer_qweights(a, za, weight, output_buf)
    }

    /// ConvInteger against a static weight prepared for it.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_integer_packed<'c, 'd>(
        &self,
        x: &TensorView<'c>,
        weight: &ConvWeights,
        x_zero_point: Option<&TensorView<'c>>,
        dilations: &[i64],
        pads: &[i64],
        strides: &[i64],
        output_buf: &'d mut Vec<f32>,
    ) -> TensorView<'d> {
        let zx = x_zero_point.and_then(|z| z.data.first().copied()).unwrap_or(0.0) as i32;
        kernels::conv_integer_packed(x, weight, zx, dilations, pads, strides, output_buf)
    }

    /// `a` times a quantized `[k, n]` weight of integer codes, whose values are
    /// `scale * (code - zero_point)`, read from the weights as they are stored.
    pub fn weight_only_matmul<'c, 'd>(
        &self,
        a: &TensorView<'c>,
        weight_offset: usize,
        weight_len: usize,
        weight_signed: bool,
        weight_shape: (usize, usize),
        scale: &TensorView<'c>,
        zero_point: Option<&TensorView<'c>>,
        output_buf: &'d mut Vec<f32>,
    ) -> TensorView<'d> {
        let bytes = self.weight_u8_raw(weight_offset, weight_len);
        let codes = if weight_signed {
            // SAFETY: `i8` and `u8` have the same size and alignment, and every
            // byte is a valid `i8`.
            let signed = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i8, bytes.len()) };
            kernels::Codes::I8(signed)
        } else {
            kernels::Codes::U8(bytes)
        };
        let zp = zero_point.and_then(|z| z.data.first().copied()).unwrap_or(0.0);
        kernels::weight_only_matmul(a, codes, weight_shape, zp, scale.data[0], output_buf)
    }

    /// Linear layer on an input that is already quantized (attention).
    pub fn linear_quantized_prequant<'c, 'd>(
        &self,
        input_quantized: &TensorView<'c>,
        input_scale: &TensorView<'c>,
        input_zero_point: &TensorView<'c>,
        weight_int8: TensorView<'c>,
        weight_scale: TensorView<'c>,
        weight_zero: TensorView<'c>,
        bias: TensorView<'c>,
        output_buf: &'d mut Vec<f32>,
        scale_buf: &'d mut Vec<f32>,
    ) -> TensorView<'d> {
        let combined_scale = kernels::mul(input_scale, &weight_scale, scale_buf);
        // MatMul + scale + bias in one pass.
        kernels::mat_mul_integer_with_scale_bias(
            input_quantized,
            &weight_int8,
            Some(input_zero_point),
            Some(&weight_zero),
            Some(&combined_scale),
            Some(&bias),
            output_buf,
        )
    }

    /// Static-QDQ matmul: weight kept in int8 so an integer GEMM can run where
    /// the hardware supports one, with the product's own `QuantizeLinear` ->
    /// `DequantizeLinear` pair and bias, if any, applied as it is written
    /// ([`kernels::qmatmul_i8`]).
    pub fn qmatmul_i8<'c, 'd>(
        &self,
        input: &TensorView<'c>,
        input_scale: &TensorView<'c>,
        input_zero_point: &TensorView<'c>,
        weight: &QWeights,
        requant: Option<kernels::Requant>,
        bias: Option<&TensorView<'c>>,
        output_buf: &'d mut Vec<f32>,
    ) -> TensorView<'d> {
        kernels::qmatmul_i8(input, input_scale, input_zero_point, weight, requant, bias, output_buf)
    }

    pub fn linear<'c, 'd, B: kernels::WeightElem>(
        &self,
        input: &TensorView<'c>,
        weight: &TensorView<'c, B>,
        bias: &TensorView<'c>,
        output_buf: &'d mut Vec<f32>,
    ) -> TensorView<'d> {
        kernels::matmul_fused_add(input, weight, bias, output_buf)
    }

    /// [`linear`](Self::linear), then `act`, applied as the output is written.
    pub fn linear_act<'c, 'd, B: kernels::WeightElem>(
        &self,
        input: &TensorView<'c>,
        weight: &TensorView<'c, B>,
        bias: &TensorView<'c>,
        act: Option<kernels::Act>,
        output_buf: &'d mut Vec<f32>,
    ) -> TensorView<'d> {
        kernels::matmul_fused_add_act(input, weight, bias, act, output_buf)
    }

    /// ConstantOfShape + Concat: `weight` followed along axis 0 by a tensor of
    /// shape `shape` filled with `value`.
    pub fn embedding_concat<'c, 'd>(
        &self,
        shape: &TensorView<'c, i64>,
        value: f32,
        weight: TensorView<'c>,
        output_buf: &'d mut Vec<f32>,
    ) -> TensorView<'d> {
        embedding_concat(shape, value, weight, output_buf)
    }

    /// [`Self::embedding_concat`] for i64 tensors.
    pub fn embedding_concat_i64<'c, 'd>(
        &self,
        shape: &TensorView<'c, i64>,
        value: i64,
        weight: TensorView<'c, i64>,
        output_buf: &'d mut Vec<i64>,
    ) -> TensorView<'d, i64> {
        embedding_concat(shape, value, weight, output_buf)
    }
}

fn embedding_concat<'d, T: Clone + std::fmt::Debug>(shape: &TensorView<'_, i64>, value: T, weight: TensorView<'_, T>, output_buf: &'d mut Vec<T>) -> TensorView<'d, T> {
    let const_len: usize = shape.data.iter().map(|&x| x as usize).product();
    output_buf.clear();
    output_buf.reserve(weight.data.len() + const_len);
    output_buf.extend_from_slice(&weight.data);
    output_buf.resize(weight.data.len() + const_len, value);

    let mut out_shape = weight.shape.to_vec();
    out_shape[0] += shape.data[0] as usize;
    TensorView {
        data: std::borrow::Cow::Borrowed(output_buf),
        shape: std::borrow::Cow::Owned(out_shape),
    }
}

/// Panics unless `shape`, that of input `input` of `model`, has the
/// dimensions `expected` gives, a `None` matching any. The model was compiled
/// for those.
pub fn check_input_shape(model: &str, input: &str, shape: &[usize], expected: &[Option<usize>]) {
    let matches = shape.len() == expected.len()
        && shape
            .iter()
            .zip(expected)
            .all(|(&d, e)| e.is_none_or(|e| e == d));
    if !matches {
        let dims: Vec<String> = expected
            .iter()
            .map(|e| e.map_or("_".to_string(), |d| d.to_string()))
            .collect();
        panic!(
            "{model} was compiled for input {input} of shape [{}], not {shape:?}",
            dims.join(", ")
        );
    }
}

/// Panics unless `data`, that of input `input` of `model`, is `expected`,
/// which the model was compiled with.
pub fn check_input_value<T: PartialEq + std::fmt::Debug>(
    model: &str,
    input: &str,
    data: &[T],
    expected: &[T],
) {
    if data != expected {
        panic!("{model} was compiled for input {input} = {expected:?}, not {data:?}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn input_checks_pass_what_the_model_was_compiled_for() {
        check_input_shape("M", "x", &[1, 512], &[Some(1), Some(512)]);
        check_input_shape("M", "x", &[1, 7, 80], &[Some(1), None, Some(80)]);
        check_input_shape("M", "sr", &[], &[]);
        check_input_value("M", "sr", &[16000i64], &[16000]);
        let shape =
            std::panic::catch_unwind(|| check_input_shape("M", "x", &[1, 7], &[None, Some(8)]));
        let msg = *shape.unwrap_err().downcast::<String>().unwrap();
        assert_eq!(
            msg,
            "M was compiled for input x of shape [_, 8], not [1, 7]"
        );
        assert!(
            std::panic::catch_unwind(|| check_input_shape("M", "x", &[1], &[Some(1), None]))
                .is_err()
        );
        let value = std::panic::catch_unwind(|| check_input_value("M", "sr", &[8000i64], &[16000]));
        let msg = *value.unwrap_err().downcast::<String>().unwrap();
        assert_eq!(msg, "M was compiled for input sr = [16000], not [8000]");
    }

    /// f16 weights converted at load read the same as converted per call,
    /// and those not listed are still converted per call.
    #[test]
    fn f16_weights_converted_at_load_match() {
        let values = [0.5f32, -1.25, 3.0, 65504.0, 6.0e-8, -0.0, 1.0e-3, 7.5];
        let mut data = vec![0u8; 16];
        data.extend(values.iter().flat_map(|&x| f16::from_f32(x).to_le_bytes()));
        let (a, b) = ((16, 8), (24, 8));
        let shape = [4];
        let lazy = Weights::new(&data);
        let loaded = Weights::new(&data).with_f16(&[b, a, a]);
        let partial = Weights::new(&data).with_f16(&[a]);
        for (offset, len) in [a, b] {
            let want = lazy.weight_f16(offset, len, &shape);
            let got = loaded.weight_f16(offset, len, &shape);
            assert!(matches!(got.data, std::borrow::Cow::Borrowed(_)));
            assert_eq!(got.shape.as_ref(), &shape);
            let bits = |v: &TensorView| v.data.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&want));
            assert_eq!(bits(&partial.weight_f16(offset, len, &shape)), bits(&want));
        }
        assert!(matches!(partial.weight_f16(b.0, b.1, &shape).data, std::borrow::Cow::Owned(_)));
    }

    /// Integer and bool weights widened at load read the same as widened per
    /// call, borrowed; aligned i64 weights are borrowed as they are stored.
    #[test]
    fn int_weights_widened_at_load_match() {
        let mut bytes = vec![0u8; 16];
        bytes.extend([-3i64, 7].iter().flat_map(|x| x.to_le_bytes()));
        bytes.extend([5i32, -2, i32::MAX, 0].iter().flat_map(|x| x.to_le_bytes()));
        bytes.extend([1u8, 0, 2, 1]);
        let data = AVec::<u8>::from_slice(16, &bytes);
        let (i64s, i32s, bools) = ((16, 16), (32, 16), (48, 4));
        let lazy = Weights::new(&data);
        let loaded = Weights::new(&data).with_i32_i64(&[i32s]).with_bool_i64(&[bools]);
        let borrowed = |v: &TensorView<i64>| matches!(v.data, std::borrow::Cow::Borrowed(_));

        let got = loaded.weight_i64(i64s.0, i64s.1, &[2]);
        assert!(borrowed(&got));
        assert_eq!(got.data.as_ref(), &[-3, 7]);
        let got = loaded.weight_i32_i64(i32s.0, i32s.1, &[4]);
        assert!(borrowed(&got));
        assert_eq!(got.data, lazy.weight_i32_i64(i32s.0, i32s.1, &[4]).data);
        assert_eq!(got.data.as_ref(), &[5, -2, i32::MAX as i64, 0]);
        let got = loaded.weight_bool_i64(bools.0, bools.1, &[4]);
        assert!(borrowed(&got));
        assert_eq!(got.data, lazy.weight_bool_i64(bools.0, bools.1, &[4]).data);
        assert_eq!(got.data.as_ref(), &[1, 0, 1, 1]);
        assert!(!borrowed(&lazy.weight_bool_i64(bools.0, bools.1, &[4])));
    }
}
