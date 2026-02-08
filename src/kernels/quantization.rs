use crate::tensor::TensorView;
use std::simd::prelude::*;
use std::simd::StdFloat;
use rayon::prelude::*;

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
        let zp_a_scalar = if zp_a_ref.len() == 1 { zp_a_ref[0] as i32 } else { 0 };
        let zp_b_scalar = if zp_b_ref.len() == 1 { zp_b_ref[0] as i32 } else { 0 };
        let zp_b_is_vector = zp_b_ref.len() > 1;

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
        out.resize(output_len, 0.0);

        let stride_a = m * k;
        let stride_b = k * n;

        // Parallel execution over output rows
        out.par_chunks_mut(n)
            .enumerate()
            .for_each(|(global_row_idx, out_row)| {
                let b_i = global_row_idx / m;
                let i = global_row_idx % m;

                let a_idx = if batch_a == 1 { 0 } else { b_i };
                let b_idx = if batch_b == 1 { 0 } else { b_i };
                
                let a_row = &a.data[a_idx * stride_a + i * k .. a_idx * stride_a + (i + 1) * k];
                let b_base = &b.data[b_idx * stride_b .. b_idx * stride_b + stride_b];

                let zp_a = if zp_a_ref.len() > 1 { zp_a_ref[global_row_idx % zp_a_ref.len()] as i32 } else { zp_a_scalar };

                let mut j = 0;
                while j + 8 <= n {
                    let mut acc_v = i32x8::splat(0);

                    let zp_b_v = if zp_b_is_vector {
                        Simd::<u8, 8>::from_slice(&zp_b_ref[j..j+8]).cast::<i32>()
                    } else {
                        i32x8::splat(zp_b_scalar)
                    };

                    for l in 0..k {
                        let val_a = a_row[l] as i32 - zp_a;
                        if val_a == 0 { continue; }

                        let val_a_v = i32x8::splat(val_a);
                        let b_slice = &b_base[l * n + j .. l * n + j + 8];
                        let b_v = Simd::<u8, 8>::from_slice(b_slice).cast::<i32>();
                        
                        acc_v += val_a_v * (b_v - zp_b_v);
                    }

                    let mut acc_f32 = acc_v.cast::<f32>();

                    if let Some(s) = scale {
                        let s_vec = if s.data.len() > 1 {
                             f32x8::from_slice(&s.data[j..j+8])
                        } else {
                             f32x8::splat(s.data[0])
                        };
                        acc_f32 *= s_vec;
                    }
                    if let Some(b) = bias {
                         let b_vec = f32x8::from_slice(&b.data[j..j+8]);
                         acc_f32 += b_vec;
                    }
                    if apply_relu {
                        acc_f32 = acc_f32.simd_max(f32x8::splat(0.0));
                    }

                    acc_f32.copy_to_slice(&mut out_row[j..j+8]);
                    j += 8;
                }

                // Remainder Handling
                for jj in j..n {
                    let mut acc = 0;
                    let curr_zp_b = if zp_b_is_vector { zp_b_ref[jj] as i32 } else { zp_b_scalar };

                    for l in 0..k {
                        let val_a = a_row[l] as i32 - zp_a;
                        if val_a == 0 { continue; }
                        let val_b = b_base[l * n + jj] as i32 - curr_zp_b;
                        acc += val_a * val_b;
                    }

                    let mut res = acc as f32;
                    if let Some(s) = scale { res *= if s.data.len() > jj { s.data[jj] } else { s.data[0] }; }
                    if let Some(b) = bias { res += b.data[jj]; }
                    if apply_relu && res < 0.0 { res = 0.0; }
                    out_row[jj] = res;
                }
            });

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

        // Vectorized Min/Max
        let (prefix, middle, suffix) = x.data.as_simd::<8>();
        for &v in prefix {
            if v < min_val { min_val = v; }
            if v > max_val { max_val = v; }
        }
        if !middle.is_empty() {
            let mut v_min = f32x8::splat(f32::MAX);
            let mut v_max = f32x8::splat(f32::MIN);
            for &v in middle {
                v_min = v_min.simd_min(v);
                v_max = v_max.simd_max(v);
            }
            min_val = min_val.min(v_min.reduce_min());
            max_val = max_val.max(v_max.reduce_max());
        }
        for &v in suffix {
            if v < min_val { min_val = v; }
            if v > max_val { max_val = v; }
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

        // Vectorized Quantization
        out_y_storage.clear();
        out_y_storage.resize(len, 0.0);
        
        let v_inv_scale = f32x8::splat(inv_scale);
        let v_zp = f32x8::splat(zp);
        let v_min_q = f32x8::splat(0.0);
        let v_max_q = f32x8::splat(255.0);

        let (prefix, middle, suffix) = x.data.as_simd::<8>();
        let mut offset = 0;
        for &v in prefix {
            out_y_storage[offset] = (v * inv_scale + zp).round().clamp(0.0, 255.0);
            offset += 1;
        }
        for &v in middle {
            let q = (v * v_inv_scale + v_zp).round().simd_clamp(v_min_q, v_max_q);
            q.copy_to_slice(&mut out_y_storage[offset..offset+8]);
            offset += 8;
        }
        for &v in suffix {
            out_y_storage[offset] = (v * inv_scale + zp).round().clamp(0.0, 255.0);
            offset += 1;
        }

        (
            TensorView::from_slice(out_y_storage, x.shape.to_vec()),
            TensorView::from_slice(out_scale, vec![1]),
            TensorView::from_slice(out_zp, vec![1]),
        )
    }
}