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
    lele::kernels::fused_quantized_linear(
        input, &weight_int8, &weight_scale, &weight_zero, &bias, false, output_buf,
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
    lele::kernels::fused_quantized_linear(
        input, &weight_int8, &weight_scale, &weight_zero, &bias, true, output_buf,
    )
}

/// Zero point of a weight for the ARM kernels, which take u8 codes: `get_prepared_weight`
/// shifts i8 weights by 128, so their zero point moves with them.
#[cfg(target_arch = "aarch64")]
fn arm_zero_point(z: f32, signed: bool) -> u8 {
    if signed { (z as i32 + 128) as u8 } else { z as u8 }
}

#[cfg(target_arch = "aarch64")]
fn linear_quantized_arm<'c, 'd>(
    &self,
    input: &lele::tensor::TensorView<'c, f32>,
    weight_offset: usize,
    weight_len: usize,
    weight_k: usize,
    weight_n: usize,
    weight_signed: bool,
    weight_scale: lele::tensor::TensorView<'c, f32>,
    weight_zero: lele::tensor::TensorView<'c, f32>,
    bias: lele::tensor::TensorView<'c, f32>,
    output_buf: &'d mut Vec<f32>,
) -> lele::tensor::TensorView<'d, f32> {
    let pw = self.get_prepared_weight(weight_offset, weight_len, weight_k, weight_n, weight_signed);
    let zp_b = Some(Self::arm_zero_point(weight_zero.data.first().copied().unwrap_or(0.0), weight_signed));

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
    weight_signed: bool,
    weight_scale: lele::tensor::TensorView<'c, f32>,
    weight_zero: lele::tensor::TensorView<'c, f32>,
    bias: lele::tensor::TensorView<'c, f32>,
    output_buf: &'d mut Vec<f32>,
) -> lele::tensor::TensorView<'d, f32> {
    let pw = self.get_prepared_weight(weight_offset, weight_len, weight_k, weight_n, weight_signed);
    let zp_b = Some(Self::arm_zero_point(weight_zero.data.first().copied().unwrap_or(0.0), weight_signed));

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
    weight_signed: bool,
    a_zero_point: Option<&lele::tensor::TensorView<'c, f32>>,
    b_zero_point: Option<&lele::tensor::TensorView<'c, f32>>,
    output_buf: &'d mut Vec<f32>,
) -> lele::tensor::TensorView<'d, f32> {
    let pw = self.get_prepared_weight(weight_offset, weight_len, weight_k, weight_n, weight_signed);
    let zp_a = a_zero_point.and_then(|z| z.data.first().cloned());
    let zb = b_zero_point.and_then(|z| z.data.first().copied()).unwrap_or(0.0);
    let zp_b = Some(Self::arm_zero_point(zb, weight_signed));

    lele::kernels::mat_mul_integer_prepared_arm(a, &pw, zp_a, zp_b, None, None, false, output_buf)
}

/// Dynamically quantized linear layer (+ ReLU) against an int8 weight packed
/// once for the integer GEMM.
#[cfg(not(target_arch = "aarch64"))]
fn linear_quantized_packed<'c, 'd>(
    &self,
    input: &lele::tensor::TensorView<'c, f32>,
    weight_offset: usize,
    weight_len: usize,
    weight_k: usize,
    weight_n: usize,
    weight_signed: bool,
    weight_scale: lele::tensor::TensorView<'c, f32>,
    weight_zero: lele::tensor::TensorView<'c, f32>,
    bias: lele::tensor::TensorView<'c, f32>,
    relu: bool,
    output_buf: &'d mut Vec<f32>,
) -> lele::tensor::TensorView<'d, f32> {
    let w = self.get_qweights(weight_offset, weight_len, weight_k, weight_n, weight_signed, &weight_zero.data, &weight_scale.data);
    let bias = (!bias.data.is_empty()).then_some(&bias.data[..]);
    lele::kernels::qlinear_dynamic(input, &w, bias, relu, output_buf)
}

/// MatMulInteger against a static weight packed once for the integer GEMM.
#[cfg(not(target_arch = "aarch64"))]
fn mat_mul_integer_packed<'c, 'd>(
    &self,
    a: &lele::tensor::TensorView<'c, f32>,
    weight_offset: usize,
    weight_len: usize,
    weight_k: usize,
    weight_n: usize,
    weight_signed: bool,
    a_zero_point: Option<&lele::tensor::TensorView<'c, f32>>,
    b_zero_point: Option<&lele::tensor::TensorView<'c, f32>>,
    output_buf: &'d mut Vec<f32>,
) -> lele::tensor::TensorView<'d, f32> {
    let zb = b_zero_point.map_or(&[0.0f32][..], |z| &z.data[..]);
    let w = self.get_qweights(weight_offset, weight_len, weight_k, weight_n, weight_signed, zb, &[1.0]);
    let za = a_zero_point.and_then(|z| z.data.first().copied()).unwrap_or(0.0) as i32;
    lele::kernels::mat_mul_integer_qweights(a, za, &w, output_buf)
}

/// ConvInteger against a static weight prepared once.
fn conv_integer_packed<'c, 'd>(
    &self,
    x: &lele::tensor::TensorView<'c, f32>,
    weight_offset: usize,
    weight_len: usize,
    weight_shape: [usize; 4],
    weight_signed: bool,
    group: i64,
    x_zero_point: Option<&lele::tensor::TensorView<'c, f32>>,
    w_zero_point: Option<&lele::tensor::TensorView<'c, f32>>,
    dilations: &[i64],
    pads: &[i64],
    strides: &[i64],
    output_buf: &'d mut Vec<f32>,
) -> lele::tensor::TensorView<'d, f32> {
    let zw = w_zero_point.filter(|z| !z.data.is_empty()).map_or(&[0.0f32][..], |z| &z.data[..]);
    let w = self.get_conv_weights(weight_offset, weight_len, weight_shape, group as usize, weight_signed, zw);
    let zx = x_zero_point.and_then(|z| z.data.first().copied()).unwrap_or(0.0) as i32;
    lele::kernels::conv_integer_packed(x, &w, zx, dilations, pads, strides, output_buf)
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
/// Static-QDQ matmul: activation already on the quantization grid, weight kept
/// in int8 so an integer GEMM can run where the hardware supports one.
fn qmatmul_i8<'c, 'd>(
    &self,
    input: &lele::tensor::TensorView<'c, f32>,
    input_scale: &lele::tensor::TensorView<'c, f32>,
    input_zero_point: &lele::tensor::TensorView<'c, f32>,
    weight_offset: usize,
    weight_len: usize,
    weight_k: usize,
    weight_n: usize,
    weight_scale: &lele::tensor::TensorView<'c, f32>,
    output_buf: &'d mut Vec<f32>,
) -> lele::tensor::TensorView<'d, f32> {
    let qw = self.get_quantized_weight(weight_offset, weight_len, weight_k, weight_n, weight_scale);
    lele::kernels::qmatmul_i8(input, input_scale, input_zero_point, &qw, output_buf)
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
