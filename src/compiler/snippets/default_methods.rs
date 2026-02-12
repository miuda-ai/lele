fn conv1d_relu<'c, 'd>(
    &self,
    input: lele::tensor::TensorView<'c>,
    weight: lele::tensor::TensorView<'c>,
    bias: Option<&lele::tensor::TensorView<'c>>,
    stride: usize,
    dilation: usize,
    groups: usize,
    padding: usize,
    output_buf: &'d mut Vec<f32>,
) -> lele::tensor::TensorView<'d> {
    lele::kernels::conv1d_fused(
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
fn layer_norm<'c, 'd>(
    &self,
    input: &lele::tensor::TensorView<'c>,
    scale: lele::tensor::TensorView<'c>,
    bias: lele::tensor::TensorView<'c>,
    epsilon: lele::tensor::TensorView<'c>,
    _two: lele::tensor::TensorView<'c>,
    output_buf: &'d mut Vec<f32>,
) -> lele::tensor::TensorView<'d> {
    let eps = epsilon.data.first().cloned().unwrap_or(1e-5);
    lele::kernels::layer_norm(input, &scale, &bias, -1, eps, output_buf)
}
fn linear_quantized<'c, 'd>(
    &self,
    input: &lele::tensor::TensorView<'c, f32>,
    weight_int8: lele::tensor::TensorView<'c, f32>,
    weight_scale: lele::tensor::TensorView<'c, f32>,
    weight_zero: lele::tensor::TensorView<'c, f32>,
    bias: lele::tensor::TensorView<'c, f32>,
    output_buf: &'d mut Vec<f32>,
) -> lele::tensor::TensorView<'d, f32> {
    let mut buf_q = Vec::<f32>::new();
    let mut buf_s = Vec::<f32>::new();
    let mut buf_z = Vec::<f32>::new();
    let (q, s, z) =
        lele::kernels::dynamic_quantize_linear(input, &mut buf_q, &mut buf_s, &mut buf_z);

    let mut buf_sm = Vec::<f32>::new();
    let combined_scale = lele::kernels::mul(&s, &weight_scale, &mut buf_sm);

    // FUSED: MatMul + Scale + Bias in one operation
    lele::kernels::mat_mul_integer_with_scale_bias(
        &q,
        &weight_int8,
        Some(&z),
        Some(&weight_zero),
        Some(&combined_scale),
        Some(&bias),
        output_buf,
    )
}

fn linear_quantized_relu<'c, 'd>(
    &self,
    input: &lele::tensor::TensorView<'c, f32>,
    weight_int8: lele::tensor::TensorView<'c, f32>,
    weight_scale: lele::tensor::TensorView<'c, f32>,
    weight_zero: lele::tensor::TensorView<'c, f32>,
    bias: lele::tensor::TensorView<'c, f32>,
    output_buf: &'d mut Vec<f32>,
) -> lele::tensor::TensorView<'d, f32> {
    let mut buf_q = Vec::<f32>::new();
    let mut buf_s = Vec::<f32>::new();
    let mut buf_z = Vec::<f32>::new();
    let (q, s, z) =
        lele::kernels::dynamic_quantize_linear(input, &mut buf_q, &mut buf_s, &mut buf_z);

    let mut buf_sm = Vec::<f32>::new();
    let combined_scale = lele::kernels::mul(&s, &weight_scale, &mut buf_sm);

    // FUSED: MatMul + Scale + Bias + ReLU in one operation
    lele::kernels::mat_mul_integer_with_scale_bias_relu(
        &q,
        &weight_int8,
        Some(&z),
        Some(&weight_zero),
        Some(&combined_scale),
        Some(&bias),
        output_buf,
    )
}

#[cfg(target_arch = "aarch64")]
fn linear_quantized_arm<'c, 'd>(
    &self,
    input: &lele::tensor::TensorView<'c, f32>,
    weight_offset: usize,
    weight_len: usize,
    weight_k: usize,
    weight_n: usize,
    weight_scale: lele::tensor::TensorView<'c, f32>,
    weight_zero: lele::tensor::TensorView<'c, f32>,
    bias: lele::tensor::TensorView<'c, f32>,
    output_buf: &'d mut Vec<f32>,
) -> lele::tensor::TensorView<'d, f32> {
    let pw = self.get_prepared_weight(weight_offset, weight_len, weight_k, weight_n);
    let zp_b = weight_zero.data.first().map(|&v| v as u8);

    lele::kernels::fused_dq_gemm_prepared_arm(
        input,
        &pw,
        zp_b,
        &weight_scale,
        Some(&bias),
        false,
        output_buf,
    )
}

/// ARM-optimized quantized linear + ReLU with pre-packed weights.
/// Uses fused DynQuant+GEMM: eliminates f32 intermediate buffer, per-call u8
/// allocation, and separate f32→u8 conversion pass.
#[cfg(target_arch = "aarch64")]
fn linear_quantized_relu_arm<'c, 'd>(
    &self,
    input: &lele::tensor::TensorView<'c, f32>,
    weight_offset: usize,
    weight_len: usize,
    weight_k: usize,
    weight_n: usize,
    weight_scale: lele::tensor::TensorView<'c, f32>,
    weight_zero: lele::tensor::TensorView<'c, f32>,
    bias: lele::tensor::TensorView<'c, f32>,
    output_buf: &'d mut Vec<f32>,
) -> lele::tensor::TensorView<'d, f32> {
    let pw = self.get_prepared_weight(weight_offset, weight_len, weight_k, weight_n);
    let zp_b = weight_zero.data.first().map(|&v| v as u8);

    lele::kernels::fused_dq_gemm_prepared_arm(
        input,
        &pw,
        zp_b,
        &weight_scale,
        Some(&bias),
        true,
        output_buf,
    )
}

/// ARM-optimized MatMulInteger with pre-packed weight cache.
/// Used for unfused MatMulInteger nodes where B is a static model weight.
/// Eliminates: per-call B packing, B u8→f32→u8 roundtrip, heap alloc.
#[cfg(target_arch = "aarch64")]
fn mat_mul_integer_arm<'c, 'd>(
    &self,
    a: &lele::tensor::TensorView<'c, f32>,
    weight_offset: usize,
    weight_len: usize,
    weight_k: usize,
    weight_n: usize,
    a_zero_point: Option<&lele::tensor::TensorView<'c, f32>>,
    b_zero_point: Option<&lele::tensor::TensorView<'c, f32>>,
    output_buf: &'d mut Vec<f32>,
) -> lele::tensor::TensorView<'d, f32> {
    let pw = self.get_prepared_weight(weight_offset, weight_len, weight_k, weight_n);
    let zp_a = a_zero_point.and_then(|z| z.data.first().cloned());
    let zp_b = b_zero_point.and_then(|z| z.data.first()).map(|&v| v as u8);

    lele::kernels::mat_mul_integer_prepared_arm(a, &pw, zp_a, zp_b, None, None, false, output_buf)
}

// Helper for pre-quantized inputs (used in attention where input is already quantized)
#[inline]
fn linear_quantized_prequant<'c, 'd>(
    &self,
    input_quantized: &lele::tensor::TensorView<'c, f32>,
    input_scale: &lele::tensor::TensorView<'c, f32>,
    input_zero_point: &lele::tensor::TensorView<'c, f32>,
    weight_int8: lele::tensor::TensorView<'c, f32>,
    weight_scale: lele::tensor::TensorView<'c, f32>,
    weight_zero: lele::tensor::TensorView<'c, f32>,
    bias: lele::tensor::TensorView<'c, f32>,
    output_buf: &'d mut Vec<f32>,
    scale_buf: &'d mut Vec<f32>,
) -> lele::tensor::TensorView<'d, f32> {
    let combined_scale = lele::kernels::mul(input_scale, &weight_scale, scale_buf);

    // FUSED: MatMul + Scale + Bias in one operation
    lele::kernels::mat_mul_integer_with_scale_bias(
        input_quantized,
        &weight_int8,
        Some(input_zero_point),
        Some(&weight_zero),
        Some(&combined_scale),
        Some(&bias),
        output_buf,
    )
}
fn linear<'c, 'd>(
    &self,
    input: &lele::tensor::TensorView<'c>,
    weight: &lele::tensor::TensorView<'c>,
    bias: &lele::tensor::TensorView<'c>,
    output_buf: &'d mut Vec<f32>,
) -> lele::tensor::TensorView<'d> {
    lele::kernels::matmul_fused_add(input, weight, bias, output_buf)
}

fn embedding_concat<'c, 'd>(
    &self,
    shape: &lele::tensor::TensorView<'c, i64>,
    value: f32,
    weight: lele::tensor::TensorView<'c>,
    output_buf: &'d mut Vec<f32>,
) -> lele::tensor::TensorView<'d> {
    // ConstantOfShape + Concat pattern
    // shape defines the shape of the constant tensor filled with `value`
    // Then concatenate weight and the constant along axis 0
    let const_shape: Vec<usize> = shape.data.iter().map(|&x| x as usize).collect();
    let const_len: usize = const_shape.iter().product();

    output_buf.clear();
    output_buf.reserve(weight.data.len() + const_len);
    output_buf.extend_from_slice(&weight.data);
    output_buf.resize(weight.data.len() + const_len, value);

    let mut out_shape = weight.shape.to_vec();
    out_shape[0] += const_shape[0];

    lele::tensor::TensorView {
        data: std::borrow::Cow::Borrowed(output_buf),
        shape: std::borrow::Cow::Owned(out_shape),
    }
}

fn embedding_concat_i64<'c, 'd>(
    &self,
    shape: &lele::tensor::TensorView<'c, i64>,
    value: i64,
    weight: lele::tensor::TensorView<'c, i64>,
    output_buf: &'d mut Vec<i64>,
) -> lele::tensor::TensorView<'d, i64> {
    // ConstantOfShape + Concat pattern (i64)
    // shape defines the shape of the constant tensor filled with `value`
    // Then concatenate weight and the constant along axis 0
    let const_shape: Vec<usize> = shape.data.iter().map(|&x| x as usize).collect();
    let const_len: usize = const_shape.iter().product();

    output_buf.clear();
    output_buf.reserve(weight.data.len() + const_len);
    output_buf.extend_from_slice(&weight.data);
    output_buf.resize(weight.data.len() + const_len, value);

    let mut out_shape = weight.shape.to_vec();
    out_shape[0] += const_shape[0];

    lele::tensor::TensorView {
        data: std::borrow::Cow::Borrowed(output_buf),
        shape: std::borrow::Cow::Owned(out_shape),
    }
}
