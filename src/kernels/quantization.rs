use crate::tensor::TensorView;

// Re-export ARM prepared weights
#[cfg(target_arch = "aarch64")]
pub use crate::kernels::neon::quantization::{PreparedWeightsArm, prepare_weights_arm};

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

/// Optimized version that takes raw u8 weight data directly, avoiding f32->u8 conversion.
/// `a` is quantized input (f32 values 0-255), `b_u8_data`/`b_shape` is the raw u8 weight.
pub fn mat_mul_integer_u8_weights<'a, 'b>(
    a: &TensorView<'b, f32>,
    b_u8_data: &[u8],
    b_shape: &[usize],
    a_zero_point: Option<f32>,
    b_zero_point: Option<u8>,
    scale: Option<&TensorView<'b, f32>>,
    bias: Option<&TensorView<'b, f32>>,
    apply_relu: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    // Convert only A from f32->u8 (small: M×K, typically 93×512)
    let a_u8: Vec<u8> = a.data.iter().map(|&x| x as u8).collect();
    let a_u8_view = TensorView::from_slice(&a_u8, a.shape.to_vec());
    let b_u8_view = TensorView::from_slice(b_u8_data, b_shape.to_vec());

    let a_zp_u8 = a_zero_point.map(|z| TensorView::from_owned(vec![z as u8], vec![1]));
    let b_zp_u8 = b_zero_point.map(|z| TensorView::from_owned(vec![z], vec![1]));

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

/// Pre-processed weight data for fast quantized GEMM.
/// Stores the transposed+XOR'd weight matrix and column sums.
pub struct PreparedWeights {
    /// B transposed [N, K_padded] with XOR 0x80 applied (i8 reinterpretation)
    pub b_t: Vec<u8>,
    /// Column sums of original B u8 values (for zero-point correction)
    pub col_sums_b_u8: Vec<i32>,
    /// Original K dimension
    pub k: usize,
    /// K padded to multiple of 16
    pub k_padded: usize,
    /// N dimension (number of output columns)
    pub n: usize,
}

/// Pre-process weight matrix B [K, N] for fast quantized GEMM.
/// Transposes to [N, K_padded] and XORs with 0x80 for VPMADDWD compatibility.
pub fn prepare_weights(b_data: &[u8], k: usize, n: usize) -> PreparedWeights {
    let k_padded = (k + 15) & !15;
    let mut b_t = vec![0u8; n * k_padded];
    let mut col_sums_b_u8 = vec![0i32; n];

    for jj in 0..n {
        let mut csum: i32 = 0;
        for kk in 0..k {
            let b_val = b_data[kk * n + jj];
            b_t[jj * k_padded + kk] = b_val ^ 0x80;
            csum += b_val as i32;
        }
        col_sums_b_u8[jj] = csum;
    }

    PreparedWeights {
        b_t,
        col_sums_b_u8,
        k,
        k_padded,
        n,
    }
}

/// Fast GEMM with pre-processed weights. Avoids per-call transpose.
/// Uses multi-threaded row parallelism for large M dimensions.
/// `a` is f32 quantized input (values 0-255), `pw` is pre-processed weight data.
pub fn mat_mul_integer_prepared<'a, 'b>(
    a: &TensorView<'b, f32>,
    pw: &PreparedWeights,
    a_zero_point: Option<f32>,
    b_zero_point: Option<u8>,
    scale: Option<&TensorView<'b, f32>>,
    bias: Option<&TensorView<'b, f32>>,
    apply_relu: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    // Convert A from f32 to u8
    let len = a.data.len();
    let mut a_u8: Vec<u8> = Vec::with_capacity(len);
    unsafe {
        a_u8.set_len(len);
        let src = a.data.as_ptr();
        let dst = a_u8.as_mut_ptr();
        for i in 0..len {
            *dst.add(i) = *src.add(i) as u8;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        let a_dims = a.shape.len();
        let m = a.shape[a_dims - 2];
        let batch: usize = a.shape[..a_dims - 2].iter().product();
        let batch_shape = &a.shape[..a_dims - 2];

        let total_batch = batch.max(1);
        let output_len = total_batch * m * pw.n;
        crate::kernels::utils::ensure_capacity(out, output_len);
        out.resize(output_len, 0.0);

        let zp_a = a_zero_point.unwrap_or(0.0) as i32;
        let zp_b = b_zero_point.unwrap_or(0) as i32;
        let k = pw.k;
        let n = pw.n;
        let k_padded = pw.k_padded;
        let k_zp_b = k as i32 * zp_b;
        let corr_128_minus_zpb = 128 - zp_b;
        let k_aligned = k == k_padded;
        let stride_a = m * k;
        let stride_out = m * n;

        // Extract scale/bias data for gemm_row_avx2
        let scale_data_ptr = scale.map(|s| s.data.as_ptr());
        let scale_len = scale.map(|s| s.data.len()).unwrap_or(0);
        let bias_data_ptr = bias.map(|b| b.data.as_ptr());

        for b_i in 0..total_batch {
            let a_batch = &a_u8[b_i * stride_a..(b_i + 1) * stride_a];
            let out_batch = &mut out[b_i * stride_out..(b_i + 1) * stride_out];

            for i in 0..m {
                unsafe {
                    crate::kernels::avx::quantization::gemm_row_avx2(
                        &a_batch[i * k..i * k + k],
                        pw.b_t.as_ptr(),
                        k,
                        n,
                        k_padded,
                        k_aligned,
                        pw.col_sums_b_u8.as_ptr(),
                        zp_a,
                        zp_b,
                        k_zp_b,
                        corr_128_minus_zpb,
                        scale_data_ptr,
                        scale_len,
                        bias_data_ptr,
                        apply_relu,
                        &mut out_batch[i * n..(i + 1) * n],
                    );
                }
            }
        }

        let mut output_shape = batch_shape.to_vec();
        output_shape.push(m);
        output_shape.push(n);
        TensorView::from_slice(out, output_shape)
    }
    #[cfg(target_arch = "aarch64")]
    {
        // Use the optimized NEON prepared kernel directly
        let a_dims = a.shape.len();
        let m = a.shape[a_dims - 2];
        let batch: usize = a.shape[..a_dims - 2].iter().product();
        let batch_shape = &a.shape[..a_dims - 2];

        let total_batch = batch.max(1);
        let output_len = total_batch * m * pw.n;
        crate::kernels::utils::ensure_capacity(out, output_len);
        out.resize(output_len, 0.0);

        let zp_a = a_zero_point.unwrap_or(0.0) as i32;
        let zp_b = b_zero_point.unwrap_or(0) as i32;
        let k = pw.k;
        let n = pw.n;
        let stride_a = m * k;
        let stride_out = m * n;

        // Reconstruct B u8 from the x86-format PreparedWeights (transpose + XOR)
        let mut b_u8 = vec![0u8; pw.k * pw.n];
        for jj in 0..pw.n {
            for kk in 0..pw.k {
                b_u8[kk * pw.n + jj] = pw.b_t[jj * pw.k_padded + kk] ^ 0x80;
            }
        }

        // Create ARM prepared weights from reconstructed B
        let pw_arm = crate::kernels::neon::quantization::prepare_weights_arm(&b_u8, k, n);

        for b_i in 0..total_batch {
            let a_batch = &a_u8[b_i * stride_a..(b_i + 1) * stride_a];

            if total_batch == 1 {
                crate::kernels::neon::quantization::mat_mul_integer_prepared_neon(
                    a_batch, m, k, &pw_arm, zp_a, zp_b, scale, bias, apply_relu, out,
                );
            } else {
                let mut batch_out = vec![0f32; stride_out];
                crate::kernels::neon::quantization::mat_mul_integer_prepared_neon(
                    a_batch,
                    m,
                    k,
                    &pw_arm,
                    zp_a,
                    zp_b,
                    scale,
                    bias,
                    apply_relu,
                    &mut batch_out,
                );
                let out_offset = b_i * stride_out;
                out[out_offset..out_offset + stride_out].copy_from_slice(&batch_out);
            }
        }

        let mut output_shape = batch_shape.to_vec();
        output_shape.push(m);
        output_shape.push(n);
        TensorView::from_slice(out, output_shape)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        // Generic fallback: reconstruct original B layout and use existing path
        let a_u8_view = TensorView::from_slice(&a_u8, a.shape.to_vec());
        let mut b_u8 = vec![0u8; pw.k * pw.n];
        for jj in 0..pw.n {
            for kk in 0..pw.k {
                b_u8[kk * pw.n + jj] = pw.b_t[jj * pw.k_padded + kk] ^ 0x80;
            }
        }
        let b_view = TensorView::from_slice(&b_u8, vec![pw.k, pw.n]);
        let a_zp = a_zero_point.map(|z| TensorView::from_owned(vec![z as u8], vec![1]));
        let b_zp = b_zero_point.map(|z| TensorView::from_owned(vec![z], vec![1]));
        mat_mul_integer_u8(
            &a_u8_view,
            &b_view,
            a_zp.as_ref(),
            b_zp.as_ref(),
            scale,
            bias,
            apply_relu,
            out,
        )
    }
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
    #[cfg(target_arch = "aarch64")]
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
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            crate::kernels::avx::quantization::mat_mul_integer_u8_avx2(
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
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        use crate::kernels::utils;
        let zp_a_ref: &[u8] = a_zero_point.map(|z| z.data.as_ref()).unwrap_or(&[]);
        let zp_b_ref: &[u8] = b_zero_point.map(|z| z.data.as_ref()).unwrap_or(&[]);
        let zp_a_scalar = if zp_a_ref.len() == 1 {
            zp_a_ref[0] as f32
        } else {
            0.0
        };
        let zp_b_scalar = if zp_b_ref.len() == 1 {
            zp_b_ref[0] as f32
        } else {
            0.0
        };

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
                let zp_a = if zp_a_ref.len() > 1 {
                    zp_a_ref[global_row % zp_a_ref.len()] as f32
                } else {
                    zp_a_scalar
                };

                for j in 0..n {
                    let zp_b = if zp_b_ref.len() > 1 {
                        zp_b_ref[j] as f32
                    } else {
                        zp_b_scalar
                    };

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

#[cfg(target_arch = "aarch64")]
pub fn fused_dq_gemm_prepared_arm<'a>(
    input: &TensorView<'_, f32>,
    pw_arm: &PreparedWeightsArm,
    b_zero_point: Option<u8>,
    weight_scale: &TensorView<'_, f32>,
    bias: Option<&TensorView<'_, f32>>,
    apply_relu: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    use std::cell::RefCell;
    thread_local! {
        static SCRATCH: RefCell<(Vec<u8>, Vec<f32>)> = RefCell::new((Vec::new(), Vec::new()));
    }

    let a_dims = input.shape.len();
    let m = input.shape[a_dims - 2];
    let k = input.shape[a_dims - 1];
    let batch_shape = &input.shape[..a_dims.saturating_sub(2)];
    let batch: usize = batch_shape.iter().product();
    let total_batch = batch.max(1);
    let zp_b = b_zero_point.unwrap_or(0) as i32;
    let n = pw_arm.n;

    if total_batch <= 1 {
        // Fast path: no batching
        SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let (a_u8, scale_buf) = &mut *scratch;

            crate::kernels::neon::quantization::fused_dq_gemm_neon(
                &input.data,
                m,
                k,
                pw_arm,
                zp_b,
                &weight_scale.data,
                bias.map(|b| &*b.data),
                apply_relu,
                a_u8,
                scale_buf,
                out,
            );
        });
        // Preserve batch dimensions in output shape
        if batch_shape.is_empty() {
            TensorView::from_slice(out, vec![m, n])
        } else {
            let mut output_shape = batch_shape.to_vec();
            output_shape.push(m);
            output_shape.push(n);
            TensorView::from_slice(out, output_shape)
        }
    } else {
        // Batch path
        let stride_in = m * k;
        let stride_out = m * n;
        let output_len = total_batch * stride_out;
        crate::kernels::utils::ensure_capacity(out, output_len);
        out.resize(output_len, 0.0);

        SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let (a_u8, scale_buf) = &mut *scratch;

            for b_i in 0..total_batch {
                let batch_start = b_i * stride_in;
                let batch_end = batch_start + stride_in;
                let batch_data = &input.data[batch_start..batch_end];

                let mut batch_out = vec![0f32; stride_out];
                // Create a temporary TensorView-like slice for this batch
                crate::kernels::neon::quantization::fused_dq_gemm_neon(
                    batch_data,
                    m,
                    k,
                    pw_arm,
                    zp_b,
                    &weight_scale.data,
                    bias.map(|b| &*b.data),
                    apply_relu,
                    a_u8,
                    scale_buf,
                    &mut batch_out,
                );
                let out_offset = b_i * stride_out;
                out[out_offset..out_offset + stride_out].copy_from_slice(&batch_out);
            }
        });

        let mut output_shape = batch_shape.to_vec();
        output_shape.push(m);
        output_shape.push(n);
        TensorView::from_slice(out, output_shape)
    }
}

#[cfg(target_arch = "aarch64")]
pub fn mat_mul_integer_prepared_arm<'a, 'b>(
    a: &TensorView<'b, f32>,
    pw_arm: &PreparedWeightsArm,
    a_zero_point: Option<f32>,
    b_zero_point: Option<u8>,
    scale: Option<&TensorView<'b, f32>>,
    bias: Option<&TensorView<'b, f32>>,
    apply_relu: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    // Convert A from f32 to u8
    let len = a.data.len();
    let mut a_u8: Vec<u8> = Vec::with_capacity(len);
    unsafe {
        a_u8.set_len(len);
        let src = a.data.as_ptr();
        let dst = a_u8.as_mut_ptr();
        // NEON vectorized f32→u8 conversion
        let mut i = 0;
        while i + 16 <= len {
            use core::arch::aarch64::*;
            let v0 = vld1q_f32(src.add(i));
            let v1 = vld1q_f32(src.add(i + 4));
            let v2 = vld1q_f32(src.add(i + 8));
            let v3 = vld1q_f32(src.add(i + 12));
            let u0 = vcvtq_u32_f32(v0);
            let u1 = vcvtq_u32_f32(v1);
            let u2 = vcvtq_u32_f32(v2);
            let u3 = vcvtq_u32_f32(v3);
            let n0 = vqmovn_u32(u0);
            let n1 = vqmovn_u32(u1);
            let n2 = vqmovn_u32(u2);
            let n3 = vqmovn_u32(u3);
            let nn0 = vcombine_u16(n0, n1);
            let nn1 = vcombine_u16(n2, n3);
            let b0 = vqmovn_u16(nn0);
            let b1 = vqmovn_u16(nn1);
            let res = vcombine_u8(b0, b1);
            vst1q_u8(dst.add(i), res);
            i += 16;
        }
        while i < len {
            *dst.add(i) = *src.add(i) as u8;
            i += 1;
        }
    }

    let a_dims = a.shape.len();
    let m = a.shape[a_dims - 2];
    let k = a.shape[a_dims - 1];
    let batch: usize = a.shape[..a_dims - 2].iter().product();
    let batch_shape = &a.shape[..a_dims - 2];

    let total_batch = batch.max(1);
    let zp_a = a_zero_point.unwrap_or(0.0) as i32;
    let zp_b = b_zero_point.unwrap_or(0) as i32;
    let n = pw_arm.n;
    let stride_a = m * k;
    let stride_out = m * n;
    let output_len = total_batch * stride_out;
    crate::kernels::utils::ensure_capacity(out, output_len);
    out.resize(output_len, 0.0);

    for b_i in 0..total_batch {
        let a_batch = &a_u8[b_i * stride_a..(b_i + 1) * stride_a];

        if total_batch == 1 {
            crate::kernels::neon::quantization::mat_mul_integer_prepared_neon(
                a_batch, m, k, pw_arm, zp_a, zp_b, scale, bias, apply_relu, out,
            );
        } else {
            let mut batch_out = vec![0f32; stride_out];
            crate::kernels::neon::quantization::mat_mul_integer_prepared_neon(
                a_batch,
                m,
                k,
                pw_arm,
                zp_a,
                zp_b,
                scale,
                bias,
                apply_relu,
                &mut batch_out,
            );
            let out_offset = b_i * stride_out;
            out[out_offset..out_offset + stride_out].copy_from_slice(&batch_out);
        }
    }

    let mut output_shape = batch_shape.to_vec();
    output_shape.push(m);
    output_shape.push(n);
    TensorView::from_slice(out, output_shape)
}

#[cfg(target_arch = "aarch64")]
pub fn prepare_weights_arm_from_i8(b_i8_bytes: &[u8], k: usize, n: usize) -> PreparedWeightsArm {
    prepare_weights_arm(b_i8_bytes, k, n)
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
    dynamic_quantize_linear_inner(x, out_y_storage, out_scale, out_zp)
}

fn dynamic_quantize_linear_inner<'a, 'b>(
    x: &TensorView<'b, f32>,
    out_y_storage: &'a mut Vec<f32>,
    out_scale: &'a mut Vec<f32>,
    out_zp: &'a mut Vec<f32>,
) -> (
    TensorView<'a, f32>,
    TensorView<'a, f32>,
    TensorView<'a, f32>,
) {
    #[cfg(target_arch = "aarch64")]
    {
        crate::kernels::neon::quantization::dynamic_quantize_linear(
            x,
            out_y_storage,
            out_scale,
            out_zp,
        )
    }
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            crate::kernels::avx::quantization::dynamic_quantize_linear_avx2(
                x,
                out_y_storage,
                out_scale,
                out_zp,
            )
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
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
