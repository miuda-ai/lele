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

/// Fully-fused quantized linear: DynamicQuantizeLinear + MatMulInteger + Scale + Bias [+ ReLU].
///
/// The weight arrives as a tensor here, so outside aarch64 it is packed for
/// [`crate::kernels::qgemm`] on every call; generated code packs the weights it knows once
/// instead (`QWeights`).
pub fn fused_quantized_linear<'a>(
    input: &TensorView<'_, f32>,
    weight_int8: &TensorView<'_, f32>,
    weight_scale: &TensorView<'_, f32>,
    weight_zero: &TensorView<'_, f32>,
    bias: &TensorView<'_, f32>,
    apply_relu: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    #[cfg(not(target_arch = "aarch64"))]
    {
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

    #[cfg(target_arch = "aarch64")]
    {
        use std::cell::RefCell;
        thread_local! {
            static BUF_Q: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
            static BUF_S: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
            static BUF_Z: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
            static BUF_SM: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
        }

        let mut buf_q = BUF_Q.with(|c| std::mem::take(&mut *c.borrow_mut()));
        let mut buf_s = BUF_S.with(|c| std::mem::take(&mut *c.borrow_mut()));
        let mut buf_z = BUF_Z.with(|c| std::mem::take(&mut *c.borrow_mut()));
        let mut buf_sm = BUF_SM.with(|c| std::mem::take(&mut *c.borrow_mut()));

        let (q, s, z) = dynamic_quantize_linear(input, &mut buf_q, &mut buf_s, &mut buf_z);
        let combined_scale = crate::kernels::mul(&s, weight_scale, &mut buf_sm);
        let result = mat_mul_integer_with_scale_bias_activation(
            &q,
            weight_int8,
            Some(&z),
            Some(weight_zero),
            Some(&combined_scale),
            Some(bias),
            apply_relu,
            out,
        );
        BUF_Q.with(|c| *c.borrow_mut() = buf_q);
        BUF_S.with(|c| *c.borrow_mut() = buf_s);
        BUF_Z.with(|c| *c.borrow_mut() = buf_z);
        BUF_SM.with(|c| *c.borrow_mut() = buf_sm);
        result
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
    #[cfg(not(target_arch = "aarch64"))]
    {
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

    #[cfg(target_arch = "aarch64")]
    {
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

        crate::kernels::neon::quantization::mat_mul_integer_u8(
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
    // On macOS aarch64, use Apple Accelerate cblas_sgemm with dequantized fp32 weights
    // This leverages the AMX hardware which is much faster than NEON UDOT for these sizes
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        return fused_dq_gemm_accelerate(
            input,
            pw_arm,
            b_zero_point,
            weight_scale,
            bias,
            apply_relu,
            out,
        );
    }

    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    {
        fused_dq_gemm_prepared_arm_neon(
            input,
            pw_arm,
            b_zero_point,
            weight_scale,
            bias,
            apply_relu,
            out,
        )
    }
}

/// Apple Accelerate AMX-based implementation: dequantize weights lazily, then use cblas_sgemm
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn fused_dq_gemm_accelerate<'a>(
    input: &TensorView<'_, f32>,
    pw_arm: &PreparedWeightsArm,
    b_zero_point: Option<u8>,
    weight_scale: &TensorView<'_, f32>,
    bias: Option<&TensorView<'_, f32>>,
    apply_relu: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    crate::kernels::gemm::accelerate_init();

    let a_dims = input.shape.len();
    let m = input.shape[a_dims - 2];
    let k = input.shape[a_dims - 1];
    let batch_shape = &input.shape[..a_dims.saturating_sub(2)];
    let batch: usize = batch_shape.iter().product();
    let total_batch = batch.max(1);
    let n = pw_arm.n;

    // Get or compute the dequantized fp32 weight matrix [K, N]
    let weight_f32 = pw_arm.get_dequantized_weights(b_zero_point, weight_scale);

    let output_len = total_batch * m * n;
    crate::kernels::utils::ensure_capacity(out, output_len);
    unsafe {
        out.set_len(output_len);
    }

    let stride_in = m * k;
    let stride_out = m * n;
    let has_bias = bias.is_some();

    for b_i in 0..total_batch {
        let a_offset = b_i * stride_in;
        let out_offset = b_i * stride_out;

        // C = A * B  (beta=0, no pre-fill needed)
        unsafe {
            crate::kernels::gemm::accelerate_sgemm(
                m as i32,
                n as i32,
                k as i32,
                1.0f32,
                input.data.as_ptr().add(a_offset),
                k as i32,
                weight_f32.as_ptr(),
                n as i32,
                0.0f32,
                out.as_mut_ptr().add(out_offset),
                n as i32,
            );
        }

        // Fused bias add + ReLU in a single vectorized pass
        if has_bias || apply_relu {
            let bias_data = bias.map(|b| b.data.as_ptr());
            let out_ptr = out.as_mut_ptr();
            unsafe {
                use core::arch::aarch64::*;
                let zero = vdupq_n_f32(0.0);
                for row in 0..m {
                    let row_offset = out_offset + row * n;
                    let mut j = 0;
                    if has_bias && apply_relu {
                        let bp = bias_data.unwrap();
                        while j + 16 <= n {
                            let mut v0 = vld1q_f32(out_ptr.add(row_offset + j));
                            let mut v1 = vld1q_f32(out_ptr.add(row_offset + j + 4));
                            let mut v2 = vld1q_f32(out_ptr.add(row_offset + j + 8));
                            let mut v3 = vld1q_f32(out_ptr.add(row_offset + j + 12));
                            v0 = vaddq_f32(v0, vld1q_f32(bp.add(j)));
                            v1 = vaddq_f32(v1, vld1q_f32(bp.add(j + 4)));
                            v2 = vaddq_f32(v2, vld1q_f32(bp.add(j + 8)));
                            v3 = vaddq_f32(v3, vld1q_f32(bp.add(j + 12)));
                            vst1q_f32(out_ptr.add(row_offset + j), vmaxq_f32(v0, zero));
                            vst1q_f32(out_ptr.add(row_offset + j + 4), vmaxq_f32(v1, zero));
                            vst1q_f32(out_ptr.add(row_offset + j + 8), vmaxq_f32(v2, zero));
                            vst1q_f32(out_ptr.add(row_offset + j + 12), vmaxq_f32(v3, zero));
                            j += 16;
                        }
                        while j < n {
                            let val = *out_ptr.add(row_offset + j) + *bp.add(j);
                            *out_ptr.add(row_offset + j) = if val > 0.0 { val } else { 0.0 };
                            j += 1;
                        }
                    } else if has_bias {
                        let bp = bias_data.unwrap();
                        while j + 16 <= n {
                            let v0 = vaddq_f32(
                                vld1q_f32(out_ptr.add(row_offset + j)),
                                vld1q_f32(bp.add(j)),
                            );
                            let v1 = vaddq_f32(
                                vld1q_f32(out_ptr.add(row_offset + j + 4)),
                                vld1q_f32(bp.add(j + 4)),
                            );
                            let v2 = vaddq_f32(
                                vld1q_f32(out_ptr.add(row_offset + j + 8)),
                                vld1q_f32(bp.add(j + 8)),
                            );
                            let v3 = vaddq_f32(
                                vld1q_f32(out_ptr.add(row_offset + j + 12)),
                                vld1q_f32(bp.add(j + 12)),
                            );
                            vst1q_f32(out_ptr.add(row_offset + j), v0);
                            vst1q_f32(out_ptr.add(row_offset + j + 4), v1);
                            vst1q_f32(out_ptr.add(row_offset + j + 8), v2);
                            vst1q_f32(out_ptr.add(row_offset + j + 12), v3);
                            j += 16;
                        }
                        while j < n {
                            *out_ptr.add(row_offset + j) += *bp.add(j);
                            j += 1;
                        }
                    } else {
                        // relu only
                        while j + 16 <= n {
                            let v0 = vld1q_f32(out_ptr.add(row_offset + j));
                            let v1 = vld1q_f32(out_ptr.add(row_offset + j + 4));
                            let v2 = vld1q_f32(out_ptr.add(row_offset + j + 8));
                            let v3 = vld1q_f32(out_ptr.add(row_offset + j + 12));
                            vst1q_f32(out_ptr.add(row_offset + j), vmaxq_f32(v0, zero));
                            vst1q_f32(out_ptr.add(row_offset + j + 4), vmaxq_f32(v1, zero));
                            vst1q_f32(out_ptr.add(row_offset + j + 8), vmaxq_f32(v2, zero));
                            vst1q_f32(out_ptr.add(row_offset + j + 12), vmaxq_f32(v3, zero));
                            j += 16;
                        }
                        while j < n {
                            let val = *out_ptr.add(row_offset + j);
                            *out_ptr.add(row_offset + j) = if val > 0.0 { val } else { 0.0 };
                            j += 1;
                        }
                    }
                }
            }
        }
    }

    let output_shape = if batch_shape.is_empty() {
        vec![m, n]
    } else {
        let mut s = batch_shape.to_vec();
        s.push(m);
        s.push(n);
        s
    };
    TensorView::from_slice(out, output_shape)
}

/// NEON UDOT-based implementation (non-macOS or fallback)
#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
fn fused_dq_gemm_prepared_arm_neon<'a>(
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
    #[cfg(target_arch = "wasm32")]
    {
        use std::arch::wasm32::*;
        let len = x.data.len();
        let ptr = x.data.as_ptr();

        // Phase 1: SIMD min/max scan
        let mut min_v = f32x4_splat(f32::MAX);
        let mut max_v = f32x4_splat(f32::MIN);
        let mut i = 0usize;
        unsafe {
            while i + 16 <= len {
                let v0 = v128_load(ptr.add(i) as *const v128);
                let v1 = v128_load(ptr.add(i + 4) as *const v128);
                let v2 = v128_load(ptr.add(i + 8) as *const v128);
                let v3 = v128_load(ptr.add(i + 12) as *const v128);
                min_v = f32x4_min(min_v, f32x4_min(v0, f32x4_min(v1, f32x4_min(v2, v3))));
                max_v = f32x4_max(max_v, f32x4_max(v0, f32x4_max(v1, f32x4_max(v2, v3))));
                i += 16;
            }
            while i + 4 <= len {
                let v = v128_load(ptr.add(i) as *const v128);
                min_v = f32x4_min(min_v, v);
                max_v = f32x4_max(max_v, v);
                i += 4;
            }
        }
        // Horizontal reduce
        let mut min_val = f32x4_extract_lane::<0>(min_v)
            .min(f32x4_extract_lane::<1>(min_v))
            .min(f32x4_extract_lane::<2>(min_v))
            .min(f32x4_extract_lane::<3>(min_v));
        let mut max_val = f32x4_extract_lane::<0>(max_v)
            .max(f32x4_extract_lane::<1>(max_v))
            .max(f32x4_extract_lane::<2>(max_v))
            .max(f32x4_extract_lane::<3>(max_v));
        // Handle remainder
        for j in i..len {
            let v = x.data[j];
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

        // Phase 2: SIMD vectorized quantization
        out_y_storage.clear();
        out_y_storage.resize(len, 0.0);
        let dst = out_y_storage.as_mut_ptr();
        let inv_scale_v = f32x4_splat(inv_scale);
        let zp_v = f32x4_splat(zp);
        let zero_v = f32x4_splat(0.0);
        let max255_v = f32x4_splat(255.0);
        let half_v = f32x4_splat(0.5);
        let mut i = 0usize;
        unsafe {
            while i + 4 <= len {
                let v = v128_load(ptr.add(i) as *const v128);
                // round via floor(x + 0.5) to match .round() semantics
                let q = f32x4_floor(f32x4_add(
                    f32x4_add(f32x4_mul(v, inv_scale_v), zp_v),
                    half_v,
                ));
                let q = f32x4_max(f32x4_min(q, max255_v), zero_v);
                v128_store(dst.add(i) as *mut v128, q);
                i += 4;
            }
        }
        for j in i..len {
            out_y_storage[j] = (x.data[j] * inv_scale + zp).round().clamp(0.0, 255.0);
        }

        return (
            TensorView::from_slice(out_y_storage, x.shape.to_vec()),
            TensorView::from_slice(out_scale, vec![1]),
            TensorView::from_slice(out_zp, vec![1]),
        );
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        target_arch = "wasm32"
    )))]
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
    qdq_apply(
        x,
        y_scale,
        y_zero_point,
        axis,
        block_size,
        out,
        |v, s, z| ((v / s).round_ties_even() + z).clamp(qmin, qmax),
    );
    TensorView::from_slice(out.as_slice(), x.shape.to_vec())
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
/// A per-tensor scale — what activation quantization almost always uses — takes
/// a division-free SIMD path; see
/// [`fake_quantize_per_tensor_avx2`](crate::kernels::avx::quantization::fake_quantize_per_tensor_avx2)
/// for why, and for the one way its result can differ.
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
    #[cfg(target_arch = "x86_64")]
    if matches!(
        qparam_layout(
            &x.shape,
            scale.data.len(),
            scale.shape.len(),
            axis,
            block_size
        ),
        QParamLayout::PerTensor
    ) && is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
    {
        let s = scale.data.first().copied().unwrap_or(1.0);
        let z = zero_point
            .and_then(|z| z.data.first().copied())
            .unwrap_or(0.0);
        let len = x.data.len();
        crate::kernels::utils::ensure_capacity(out, len);
        unsafe {
            out.set_len(len);
            crate::kernels::avx::quantization::fake_quantize_per_tensor_avx2(
                x.data.as_ptr(),
                out.as_mut_ptr(),
                len,
                s,
                z,
                qmin,
                qmax,
            );
        }
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
