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
    quantized_weights_cache: RefCell<QuantizedWeightCache>,
    qweights_cache: RefCell<QWeightCache>,
    conv_weights_cache: RefCell<ConvWeightCache>,
}

impl<'a> Weights<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            quantized_weights_cache: RefCell::default(),
            qweights_cache: RefCell::default(),
            conv_weights_cache: RefCell::default(),
        }
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

    pub fn weight_f16(&self, offset: usize, len: usize, shape: &'a [usize]) -> TensorView<'static, f32> {
        TensorView::from_bytes_f16(&self.data[offset..offset + len], shape.to_vec())
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

    /// Static-QDQ matmul: activation already on the quantization grid, weight
    /// kept in int8 so an integer GEMM can run where the hardware supports one.
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
        output_buf: &'d mut Vec<f32>,
    ) -> TensorView<'d> {
        let qw = self.get_quantized_weight(weight_offset, weight_len, weight_k, weight_n, weight_scale);
        kernels::qmatmul_i8(input, input_scale, input_zero_point, &qw, output_buf)
    }

    pub fn linear<'c, 'd>(
        &self,
        input: &TensorView<'c>,
        weight: &TensorView<'c>,
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
