use crate::tensor::TensorView;
use crate::kernels::utils;
use std::borrow::Cow;
pub fn softmax<'b, 'a>(
    input: &TensorView<'b>,
    axis: i32,
    out_buf: &'a mut Vec<f32>
) -> TensorView<'a> {
    let ndim = input.shape.len();
    let axis = if axis < 0 { ndim as i32 + axis } else { axis } as usize;
    assert!(axis < ndim);
    let numel = input.data.len();
    utils::ensure_capacity(out_buf, numel);
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_buf.as_mut_ptr(), numel) };
    let inner_size: usize = input.shape[axis+1..].iter().product();
    let axis_size = input.shape[axis];
    let outer_size: usize = input.shape[..axis].iter().product();
    let data = &input.data;
    if inner_size == 1 {
        for i in 0..outer_size {
            let start = i * axis_size;
            let end = start + axis_size;
            let src = &data[start..end];
            let dst = &mut out_slice[start..end];
            let max_val = src.iter().fold(f32::MIN, |a, &b| a.max(b));
            let mut sum = 0.0;
            for (j, &val) in src.iter().enumerate() {
                let e = (val - max_val).exp();
                dst[j] = e;
                sum += e;
            }
            let inv_sum = 1.0 / sum;
            for x in dst.iter_mut() {
                *x *= inv_sum;
            }
        }
    } else {
        unimplemented!("Softmax only supported on last dimension for now");
    }
    TensorView {
        data: Cow::Borrowed(out_slice),
        shape: std::borrow::Cow::Owned(input.shape.to_vec())
    }
}
pub fn layer_norm<'b, 'a>(
    input: &TensorView<'b>,
    scale: &TensorView<'b>,
    bias: &TensorView<'b>,
    axis: i32, 
    epsilon: f32,
    out_buf: &'a mut Vec<f32>
) -> TensorView<'a> {
    let ndim = input.shape.len();
    let axis = if axis < 0 { ndim as i32 + axis } else { axis } as usize;
    let outer_size: usize = input.shape[..axis].iter().product();
    let norm_size: usize = input.shape[axis..].iter().product();
    
    utils::ensure_capacity(out_buf, input.data.len());
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_buf.as_mut_ptr(), input.data.len()) };
    
    let src = &input.data;
    let gamma = &scale.data;
    let beta = &bias.data;

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        for i in 0..outer_size {
            let offset = i * norm_size;
            let chunk = &src[offset .. offset + norm_size];
            let out_chunk = &mut out_slice[offset .. offset + norm_size];
            
            // Mean
            let mut sum_v = unsafe { vdupq_n_f32(0.0) };
            let mut j = 0;
            while j + 4 <= norm_size {
                unsafe {
                    let v = vld1q_f32(chunk.as_ptr().add(j));
                    sum_v = vaddq_f32(sum_v, v);
                }
                j += 4;
            }
            let mut sum = unsafe { vaddvq_f32(sum_v) };
            while j < norm_size {
                sum += chunk[j];
                j += 1;
            }
            let mean = sum / norm_size as f32;
            let mean_v = unsafe { vdupq_n_f32(mean) };

            // Variance
            let mut var_sum_v = unsafe { vdupq_n_f32(0.0) };
            j = 0;
            while j + 4 <= norm_size {
                unsafe {
                    let v = vld1q_f32(chunk.as_ptr().add(j));
                    let diff = vsubq_f32(v, mean_v);
                    var_sum_v = vfmaq_f32(var_sum_v, diff, diff);
                }
                j += 4;
            }
            let mut var_sum = unsafe { vaddvq_f32(var_sum_v) };
            while j < norm_size {
                let diff = chunk[j] - mean;
                var_sum += diff * diff;
                j += 1;
            }
            let var = var_sum / norm_size as f32;
            let inv_std = 1.0 / (var + epsilon).sqrt();
            let inv_std_v = unsafe { vdupq_n_f32(inv_std) };

            // Normalize
            j = 0;
            while j + 4 <= norm_size {
                unsafe {
                    let v = vld1q_f32(chunk.as_ptr().add(j));
                    let g = vld1q_f32(gamma.as_ptr().add(j));
                    let b = vld1q_f32(beta.as_ptr().add(j));
                    
                    let diff = vsubq_f32(v, mean_v);
                    let norm = vmulq_f32(diff, inv_std_v);
                    let res = vfmaq_f32(b, norm, g);
                    vst1q_f32(out_chunk.as_mut_ptr().add(j), res);
                }
                j += 4;
            }
            while j < norm_size {
                out_chunk[j] = (chunk[j] - mean) * inv_std * gamma[j] + beta[j];
                j += 1;
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..outer_size {
            let offset = i * norm_size;
            let chunk = &src[offset .. offset + norm_size];
            let out_chunk = &mut out_slice[offset .. offset + norm_size];
            let sum: f32 = chunk.iter().sum();
            let mean = sum / norm_size as f32;
            let var_sum: f32 = chunk.iter().map(|&x| (x - mean) * (x - mean)).sum();
            let var = var_sum / norm_size as f32;
            let inv_std = 1.0 / (var + epsilon).sqrt();
            for j in 0..norm_size {
                out_chunk[j] = (chunk[j] - mean) * inv_std * gamma[j] + beta[j];
            }
        }
    }
    TensorView {
        data: Cow::Borrowed(out_slice),
        shape: std::borrow::Cow::Owned(input.shape.to_vec())
    }
}
pub fn batch_norm<'b, 'a>(
    input: &TensorView<'b>,
    scale: &TensorView<'b>,
    bias: &TensorView<'b>,
    mean: &TensorView<'b>,
    var: &TensorView<'b>,
    epsilon: f32,
    out_buf: &'a mut Vec<f32>
) -> TensorView<'a> {
    let shape = &input.shape;
    let numel = input.data.len();
    utils::ensure_capacity(out_buf, numel);
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_buf.as_mut_ptr(), numel) };
    if shape.len() == 4 {
        let n = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];
        let spatial_size = h * w;
        let src = &input.data;
        let s = &scale.data;
        let b = &bias.data;
        let m = &mean.data;
        let v = &var.data;
        for ni in 0..n {
            for ci in 0..c {
                let offset = (ni * c + ci) * spatial_size;
                let scale_val = s[ci] / (v[ci] + epsilon).sqrt();
                let bias_val = b[ci] - m[ci] * scale_val;
                for i in 0..spatial_size {
                    out_slice[offset + i] = src[offset + i] * scale_val + bias_val;
                }
            }
        }
    } else {
        let c = if shape.len() > 1 { shape[1] } else { shape[0] };
        let outer_size = shape[0];
        let inner_size: usize = if shape.len() > 2 { shape[2..].iter().product() } else { 1 };
        let src = &input.data;
        let s = &scale.data;
        let b = &bias.data;
        let m = &mean.data;
        let v = &var.data;
        for i in 0..outer_size {
            for j in 0..c {
                let scale_val = s[j] / (v[j] + epsilon).sqrt();
                let bias_val = b[j] - m[j] * scale_val;
                for k in 0..inner_size {
                    let idx = (i * c + j) * inner_size + k;
                    out_slice[idx] = src[idx] * scale_val + bias_val;
                }
            }
        }
    }
    TensorView {
        data: Cow::Borrowed(out_slice),
        shape: Cow::Owned(shape.to_vec())
    }
}
