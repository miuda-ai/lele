//! Runtime support for generated models.
//!
//! Every generated model struct wraps a [`Weights`] and derefs to it, so the
//! generated code reads weights and calls the fused ops below as `self.*`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use crate::kernels::{self, ConvWeights, QWeights};
use crate::tensor::TensorView;

type QuantizedWeightCache = HashMap<(usize, usize, usize, usize), (Vec<f32>, Arc<QWeights>)>;
type QWeightCache = HashMap<(usize, usize, usize, usize, bool), (Vec<f32>, Vec<f32>, Arc<QWeights>)>;
type ConvWeightCache = HashMap<(usize, usize, usize, bool), (Vec<f32>, Arc<ConvWeights>)>;

/// A model's weight blob plus the weights prepared from it for the integer
/// kernels. Preparing is architecture dependent and costs a full pass over the
/// weight, so it happens once per tensor and is cached for the life of the model.
pub struct Weights<'a> {
    data: &'a [u8],
    /// The f16 weights [`Self::with_f16`] converted, one after another.
    f16: Vec<f32>,
    /// Byte offset of each converted f16 weight and where it starts in
    /// `f16`, by offset.
    f16_index: Vec<(usize, usize)>,
    quantized_weights_cache: RefCell<QuantizedWeightCache>,
    qweights_cache: RefCell<QWeightCache>,
    conv_weights_cache: RefCell<ConvWeightCache>,
}

impl<'a> Weights<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            f16: Vec::new(),
            f16_index: Vec::new(),
            quantized_weights_cache: RefCell::default(),
            qweights_cache: RefCell::default(),
            conv_weights_cache: RefCell::default(),
        }
    }

    /// Converts the f16 weights at `(byte offset, byte length)` to f32 once,
    /// so that [`Self::weight_f16`] reads them without converting on every
    /// call.
    pub fn with_f16(mut self, regions: &[(usize, usize)]) -> Self {
        use half::slice::HalfFloatSliceExt;
        let mut regions = regions.to_vec();
        regions.sort_unstable();
        regions.dedup();
        self.f16 = vec![0.0; regions.iter().map(|&(_, len)| len / 2).sum()];
        self.f16_index.reserve(regions.len());
        let mut start = 0;
        for (offset, len) in regions {
            let halves: Vec<half::f16> = self.data[offset..offset + len]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&b| half::f16::from_le_bytes(b))
                .collect();
            halves.convert_to_f32_slice(&mut self.f16[start..start + halves.len()]);
            self.f16_index.push((offset, start));
            start += halves.len();
        }
        self
    }

    pub fn weight_f32(&self, offset: usize, len: usize, shape: &'a [usize]) -> TensorView<'a, f32> {
        TensorView::from_bytes_f32(&self.data[offset..offset + len], shape)
    }

    pub fn weight_i64(&self, offset: usize, len: usize, shape: &'a [usize]) -> TensorView<'static, i64> {
        TensorView::from_bytes_i64(&self.data[offset..offset + len], shape.to_vec())
    }

    pub fn weight_i32_i64(&self, offset: usize, len: usize, shape: &'a [usize]) -> TensorView<'static, i64> {
        TensorView::from_bytes_i32_as_i64(&self.data[offset..offset + len], shape.to_vec())
    }

    pub fn weight_bool_i64(&self, offset: usize, len: usize, shape: &'a [usize]) -> TensorView<'static, i64> {
        TensorView::from_bytes_bool_as_i64(&self.data[offset..offset + len], shape.to_vec())
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

    pub fn weight_u8(&self, offset: usize, len: usize, shape: &'a [usize]) -> TensorView<'static, f32> {
        TensorView::from_bytes_u8(&self.data[offset..offset + len], shape.to_vec())
    }

    pub fn weight_i8(&self, offset: usize, len: usize, shape: &'a [usize]) -> TensorView<'static, f32> {
        TensorView::from_bytes_i8(&self.data[offset..offset + len], shape.to_vec())
    }

    /// An f16 weight as f32: borrowed if [`Self::with_f16`] converted it,
    /// converted now otherwise.
    pub fn weight_f16(&self, offset: usize, len: usize, shape: &'a [usize]) -> TensorView<'_, f32> {
        match self.f16_index.binary_search_by_key(&offset, |&(o, _)| o) {
            Ok(i) => {
                let start = self.f16_index[i].1;
                TensorView {
                    data: std::borrow::Cow::Borrowed(&self.f16[start..start + len / 2]),
                    shape: std::borrow::Cow::Borrowed(shape),
                }
            }
            Err(_) => TensorView::from_bytes_f16(&self.data[offset..offset + len], shape.to_vec()),
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

    /// Int8 weights for QLinearMatMulI8.
    fn get_quantized_weight(&self, offset: usize, len: usize, k: usize, n: usize, scale: &TensorView<f32>) -> Arc<QWeights> {
        // Distinct initializers can share identical bytes, and one initializer
        // can sit behind several DequantizeLinear nodes carrying different
        // scales, so the key must include the shape and the cached entry must
        // match the exact scale.
        let key = (offset, len, k, n);
        let same_bits = |s: &[f32]| s.len() == scale.data.len() && s.iter().zip(scale.data.iter()).all(|(a, b)| a.to_bits() == b.to_bits());
        if let Some((_, qw)) = self.quantized_weights_cache.borrow().get(&key).filter(|(s, _)| same_bits(s)) {
            return qw.clone();
        }
        let raw = &self.data[offset..offset + len];
        let qw = Arc::new(kernels::prepare_quantized_weights(raw, k, n, &scale.data));
        self.quantized_weights_cache
            .borrow_mut()
            .insert(key, (scale.data.to_vec(), qw.clone()));
        qw
    }

    /// 8-bit weights packed for the integer GEMM, with their zero points
    /// subtracted, so the cached entry must match those and the scales.
    #[allow(clippy::too_many_arguments)]
    fn get_qweights(&self, offset: usize, len: usize, k: usize, n: usize, signed: bool, zero_point: &[f32], scale: &[f32]) -> Arc<QWeights> {
        let key = (offset, len, k, n, signed);
        if let Some((_, _, w)) = self
            .qweights_cache
            .borrow()
            .get(&key)
            .filter(|(z, s, _)| z.as_slice() == zero_point && s.as_slice() == scale)
        {
            return w.clone();
        }
        let zp: Vec<i32> = zero_point.iter().map(|&z| z as i32).collect();
        let w = Arc::new(QWeights::new(&self.data[offset..offset + len], k, n, signed, &zp, scale));
        self.qweights_cache
            .borrow_mut()
            .insert(key, (zero_point.to_vec(), scale.to_vec(), w.clone()));
        w
    }

    /// ConvInteger weights, prepared with their zero points, which the entry must match.
    fn get_conv_weights(&self, offset: usize, len: usize, shape: [usize; 4], groups: usize, signed: bool, zero_point: &[f32]) -> Arc<ConvWeights> {
        let key = (offset, len, groups, signed);
        if let Some((_, w)) = self
            .conv_weights_cache
            .borrow()
            .get(&key)
            .filter(|(z, _)| z.as_slice() == zero_point)
        {
            return w.clone();
        }
        let zp: Vec<i32> = zero_point.iter().map(|&z| z as i32).collect();
        let w = Arc::new(ConvWeights::new(&self.data[offset..offset + len], &shape, groups, signed, &zp));
        self.conv_weights_cache
            .borrow_mut()
            .insert(key, (zero_point.to_vec(), w.clone()));
        w
    }
}

/// Fused ops the compiler's patterns emit as `self.<op>(...)`.
#[allow(clippy::too_many_arguments)]
impl Weights<'_> {
    pub fn conv1d_relu<'c, 'd>(
        &self,
        input: TensorView<'c>,
        weight: TensorView<'c>,
        bias: Option<&TensorView<'c>>,
        stride: usize,
        dilation: usize,
        groups: usize,
        padding: usize,
        output_buf: &'d mut Vec<f32>,
    ) -> TensorView<'d> {
        kernels::conv1d_fused(
            &input,
            &weight,
            bias,
            &[dilation as i64],
            groups as i64,
            &[padding as i64, padding as i64],
            &[stride as i64],
            true,
            output_buf,
        )
    }

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

    pub fn linear_quantized<'c, 'd>(
        &self,
        input: &TensorView<'c>,
        weight_int8: TensorView<'c>,
        weight_scale: TensorView<'c>,
        weight_zero: TensorView<'c>,
        bias: TensorView<'c>,
        output_buf: &'d mut Vec<f32>,
    ) -> TensorView<'d> {
        kernels::fused_quantized_linear(input, &weight_int8, &weight_scale, &weight_zero, &bias, false, output_buf)
    }

    pub fn linear_quantized_relu<'c, 'd>(
        &self,
        input: &TensorView<'c>,
        weight_int8: TensorView<'c>,
        weight_scale: TensorView<'c>,
        weight_zero: TensorView<'c>,
        bias: TensorView<'c>,
        output_buf: &'d mut Vec<f32>,
    ) -> TensorView<'d> {
        kernels::fused_quantized_linear(input, &weight_int8, &weight_scale, &weight_zero, &bias, true, output_buf)
    }

    /// Dynamically quantized linear layer (+ ReLU) against an int8 weight packed
    /// once for the integer GEMM.
    pub fn linear_quantized_packed<'c, 'd>(
        &self,
        input: &TensorView<'c>,
        weight_offset: usize,
        weight_len: usize,
        weight_k: usize,
        weight_n: usize,
        weight_signed: bool,
        weight_scale: TensorView<'c>,
        weight_zero: TensorView<'c>,
        bias: TensorView<'c>,
        relu: bool,
        output_buf: &'d mut Vec<f32>,
    ) -> TensorView<'d> {
        let w = self.get_qweights(weight_offset, weight_len, weight_k, weight_n, weight_signed, &weight_zero.data, &weight_scale.data);
        let bias = (!bias.data.is_empty()).then_some(&bias.data[..]);
        kernels::qlinear_dynamic(input, &w, bias, relu, output_buf)
    }

    /// MatMulInteger against a static weight packed once for the integer GEMM.
    pub fn mat_mul_integer_packed<'c, 'd>(
        &self,
        a: &TensorView<'c>,
        weight_offset: usize,
        weight_len: usize,
        weight_k: usize,
        weight_n: usize,
        weight_signed: bool,
        a_zero_point: Option<&TensorView<'c>>,
        b_zero_point: Option<&TensorView<'c>>,
        output_buf: &'d mut Vec<f32>,
    ) -> TensorView<'d> {
        let zb = b_zero_point.map_or(&[0.0f32][..], |z| &z.data[..]);
        let w = self.get_qweights(weight_offset, weight_len, weight_k, weight_n, weight_signed, zb, &[1.0]);
        let za = a_zero_point.and_then(|z| z.data.first().copied()).unwrap_or(0.0) as i32;
        kernels::mat_mul_integer_qweights(a, za, &w, output_buf)
    }

    /// ConvInteger against a static weight prepared once.
    pub fn conv_integer_packed<'c, 'd>(
        &self,
        x: &TensorView<'c>,
        weight_offset: usize,
        weight_len: usize,
        weight_shape: [usize; 4],
        weight_signed: bool,
        group: i64,
        x_zero_point: Option<&TensorView<'c>>,
        w_zero_point: Option<&TensorView<'c>>,
        dilations: &[i64],
        pads: &[i64],
        strides: &[i64],
        output_buf: &'d mut Vec<f32>,
    ) -> TensorView<'d> {
        let zw = w_zero_point.filter(|z| !z.data.is_empty()).map_or(&[0.0f32][..], |z| &z.data[..]);
        let w = self.get_conv_weights(weight_offset, weight_len, weight_shape, group as usize, weight_signed, zw);
        let zx = x_zero_point.and_then(|z| z.data.first().copied()).unwrap_or(0.0) as i32;
        kernels::conv_integer_packed(x, &w, zx, dilations, pads, strides, output_buf)
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
        weight_offset: usize,
        weight_len: usize,
        weight_k: usize,
        weight_n: usize,
        weight_scale: &TensorView<'c>,
        requant: Option<kernels::Requant>,
        bias: Option<&TensorView<'c>>,
        output_buf: &'d mut Vec<f32>,
    ) -> TensorView<'d> {
        let qw = self.get_quantized_weight(weight_offset, weight_len, weight_k, weight_n, weight_scale);
        kernels::qmatmul_i8(input, input_scale, input_zero_point, &qw, requant, bias, output_buf)
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
}
