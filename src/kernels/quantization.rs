use crate::tensor::TensorView;

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

// Internal function with activation parameter
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
    // Convert f32 tensors to u8 for actual computation
    let a_u8: Vec<u8> = a.data.iter().map(|&x| x as u8).collect();
    let b_u8: Vec<u8> = b.data.iter().map(|&x| x as u8).collect();

    let a_u8_view = TensorView::from_slice(&a_u8, a.shape.to_vec());
    let b_u8_view = TensorView::from_slice(&b_u8, b.shape.to_vec());

    let a_zp_u8 = a_zero_point.map(|z| {
        let data: Vec<u8> = z.data.iter().map(|&x| x as u8).collect();
        TensorView::from_owned(data, z.shape.to_vec())
    });
    let b_zp_u8 = b_zero_point.map(|z| {
        let data: Vec<u8> = z.data.iter().map(|&x| x as u8).collect();
        TensorView::from_owned(data, z.shape.to_vec())
    });

    mat_mul_integer_u8(
        &a_u8_view,
        &b_u8_view,
        a_zp_u8.as_ref(),
        b_zp_u8.as_ref(),
        scale,
        bias,
        apply_relu,
        out,
    )
}

// True quantization version (u8 x u8 -> f32 output)
fn mat_mul_integer_u8<'a, 'b, 'c>(
    a: &TensorView<'b, u8>,
    b: &TensorView<'c, u8>,
    a_zero_point: Option<&TensorView<'b, u8>>,
    b_zero_point: Option<&TensorView<'c, u8>>,
    scale: Option<&TensorView<'b, f32>>,
    bias: Option<&TensorView<'b, f32>>,
    apply_relu: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    {
        crate::kernels::neon::quantization::mat_mul_integer_u8(
            a,
            b,
            a_zero_point,
            b_zero_point,
            scale,
            bias,
            apply_relu,
            out,
        )
    }
    #[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
    {
        use crate::kernels::utils;
        let zp_a_ref: &[u8] = a_zero_point.map(|z| z.data.as_ref()).unwrap_or(&[]);
        let zp_b_ref: &[u8] = b_zero_point.map(|z| z.data.as_ref()).unwrap_or(&[]);
        let zp_a_scalar = if zp_a_ref.len() == 1 { zp_a_ref[0] as f32 } else { 0.0 };
        let zp_b_scalar = if zp_b_ref.len() == 1 { zp_b_ref[0] as f32 } else { 0.0 };

        let a_dims = a.shape.len();
        let b_dims = b.shape.len();
        let m = a.shape[a_dims - 2];
        let k = a.shape[a_dims - 1];
        let n = b.shape[b_dims - 1];

        // Batch handling
        let batch_a: usize = a.shape[..a_dims - 2].iter().product();
        let batch_b: usize = b.shape[..b_dims - 2].iter().product();
        let final_batch = batch_a.max(batch_b);

        let output_len = final_batch * m * n;
        utils::ensure_capacity(out, output_len);

        // Ensure exact size for safety
        out.resize(output_len, 0.0);

        let stride_a = m * k;
        let stride_b = k * n;
        let stride_out = m * n;

        // Naive loop with f32 accumulation (slow) - Fallback
        for b_i in 0..final_batch {
            let a_offset = if batch_a == 1 { 0 } else { b_i * stride_a };
            let b_offset = if batch_b == 1 { 0 } else { b_i * stride_b };
            let out_offset = b_i * stride_out;

            let a_data = &a.data[a_offset..a_offset + stride_a];
            let b_data = &b.data[b_offset..b_offset + stride_b];
            let out_data = &mut out[out_offset..out_offset + stride_out];

            for i in 0..m {
                let global_row = b_i * m + i;
                let zp_a = if zp_a_ref.len() > 1 { zp_a_ref[global_row % zp_a_ref.len()] as f32 } else { zp_a_scalar };
                
                for j in 0..n {
                    let zp_b = if zp_b_ref.len() > 1 { zp_b_ref[j] as f32 } else { zp_b_scalar };
                
                    let mut sum = 0.0;
                    for l in 0..k {
                        let val_a = a_data[i * k + l] as f32 - zp_a;
                        let val_b = b_data[l * n + j] as f32 - zp_b;
                        sum += val_a * val_b;
                    }

                    // Apply scale if provided
                    if let Some(scale_data) = scale {
                        if scale_data.data.len() == 1 {
                            sum *= scale_data.data[0];
                        } else {
                            sum *= scale_data.data[j];
                        }
                    }

                    // Apply bias if provided (per-column)
                    if let Some(bias_data) = bias {
                        sum += bias_data.data[j];
                    }

                    // Apply ReLU if requested
                    if apply_relu && sum < 0.0 {
                        sum = 0.0;
                    }

                    out_data[i * n + j] = sum;
                }
            }
        }

        let mut output_shape = if batch_a >= batch_b {
            a.shape[..a_dims - 2].to_vec()
        } else {
            b.shape[..b_dims - 2].to_vec()
        };
        output_shape.push(m);
        output_shape.push(n);

        TensorView::from_slice(out, output_shape)
    }
}

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
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    {
        crate::kernels::neon::quantization::dynamic_quantize_linear(
            x,
            out_y_storage,
            out_scale,
            out_zp,
        )
    }
    #[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
    {
        let len = x.data.len();
        
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        for &v in x.data.iter() {
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }

        let adjusted_max = max_val.max(0.0);
        let adjusted_min = min_val.min(0.0);
        let range = (adjusted_max - adjusted_min).max(1e-5);
        let scale = range / 255.0;
        let zp = (-adjusted_min / scale).round().clamp(0.0, 255.0);
        let inv_scale = 1.0 / scale;

        out_scale.clear();
        out_scale.push(scale);

        out_zp.clear();
        out_zp.push(zp);

        // Calculate and write directly to output
        out_y_storage.clear();
        out_y_storage.reserve(len);
        for i in 0..len {
            let q = (x.data[i] * inv_scale + zp).round().clamp(0.0, 255.0);
            out_y_storage.push(q);
        }

        (
            TensorView::from_slice(out_y_storage, x.shape.to_vec()),
            TensorView::from_slice(out_scale, vec![1]),
            TensorView::from_slice(out_zp, vec![1]),
        )
    }
}