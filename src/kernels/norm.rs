use crate::kernels::utils;
use crate::tensor::TensorView;
use std::borrow::Cow;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

pub fn softmax<'b, 'a>(
    input: &TensorView<'b>,
    axis: i32,
    out_buf: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let ndim = input.shape.len();
    let axis = if axis < 0 { ndim as i32 + axis } else { axis } as usize;
    assert!(axis < ndim);
    let numel = input.data.len();
    utils::ensure_capacity(out_buf, numel);
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_buf.as_mut_ptr(), numel) };
    let inner_size: usize = input.shape[axis + 1..].iter().product();
    let axis_size = input.shape[axis];
    let outer_size: usize = input.shape[..axis].iter().product();
    let data = &input.data;
    if inner_size == 1 {
        #[cfg(target_arch = "aarch64")]
        {
            for i in 0..outer_size {
                let start = i * axis_size;
                let end = start + axis_size;
                let src = &data[start..end];
                let dst = &mut out_slice[start..end];

                unsafe {
                    // SIMD max finding
                    let mut max_vec = vdupq_n_f32(f32::MIN);
                    let mut j = 0;
                    let simd_end = (axis_size / 4) * 4;

                    while j < simd_end {
                        let v = vld1q_f32(src.as_ptr().add(j));
                        max_vec = vmaxq_f32(max_vec, v);
                        j += 4;
                    }

                    // Horizontal max
                    let mut max_val = vgetq_lane_f32(max_vec, 0)
                        .max(vgetq_lane_f32(max_vec, 1))
                        .max(vgetq_lane_f32(max_vec, 2))
                        .max(vgetq_lane_f32(max_vec, 3));

                    for &v in &src[simd_end..] {
                        max_val = max_val.max(v);
                    }

                    // SIMD exp and sum using accurate exp
                    let max_broadcast = vdupq_n_f32(max_val);
                    let mut sum_vec = vdupq_n_f32(0.0);

                    j = 0;
                    while j < simd_end {
                        let v = vld1q_f32(src.as_ptr().add(j));
                        let shifted = vsubq_f32(v, max_broadcast);

                        // Use accurate exp for numerical stability
                        let exp_val = crate::kernels::neon::math::neon_exp_f32x4(shifted);
                        vst1q_f32(dst.as_mut_ptr().add(j), exp_val);
                        sum_vec = vaddq_f32(sum_vec, exp_val);
                        j += 4;
                    }

                    // Horizontal sum + remaining elements
                    let mut sum = vgetq_lane_f32(sum_vec, 0)
                        + vgetq_lane_f32(sum_vec, 1)
                        + vgetq_lane_f32(sum_vec, 2)
                        + vgetq_lane_f32(sum_vec, 3);

                    for k in simd_end..axis_size {
                        let e = (src[k] - max_val).exp();
                        dst[k] = e;
                        sum += e;
                    }

                    // SIMD normalization
                    let inv_sum = 1.0 / sum;
                    let inv_sum_vec = vdupq_n_f32(inv_sum);

                    j = 0;
                    while j < simd_end {
                        let v = vld1q_f32(dst.as_ptr().add(j));
                        let normalized = vmulq_f32(v, inv_sum_vec);
                        vst1q_f32(dst.as_mut_ptr().add(j), normalized);
                        j += 4;
                    }

                    for k in simd_end..axis_size {
                        dst[k] *= inv_sum;
                    }
                }
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            for i in 0..outer_size {
                let start = i * axis_size;
                let end = start + axis_size;
                let src = &data[start..end];
                let dst = &mut out_slice[start..end];
                unsafe {
                    crate::kernels::avx::norm::softmax(src, dst);
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            for i in 0..outer_size {
                let start = i * axis_size;
                let end = start + axis_size;
                let src = &data[start..end];
                let dst = &mut out_slice[start..end];
                unsafe {
                    crate::kernels::wasm::norm::softmax(src, dst);
                }
            }
        }
        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "wasm32",
            target_arch = "aarch64"
        )))]
        {
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
        }
    } else {
        unimplemented!("Softmax only supported on last dimension for now");
    }
    TensorView {
        data: Cow::Borrowed(out_slice),
        shape: std::borrow::Cow::Owned(input.shape.to_vec()),
    }
}

pub fn layer_norm<'b, 'a>(
    input: &TensorView<'b>,
    scale: &TensorView<'b>,
    bias: &TensorView<'b>,
    axis: i32,
    epsilon: f32,
    out_buf: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let ndim = input.shape.len();
    let axis = if axis < 0 { ndim as i32 + axis } else { axis } as usize;
    let outer_size: usize = input.shape[..axis].iter().product();
    let norm_size: usize = input.shape[axis..].iter().product();
    utils::ensure_capacity(out_buf, input.data.len());
    let out_slice =
        unsafe { std::slice::from_raw_parts_mut(out_buf.as_mut_ptr(), input.data.len()) };
    #[cfg(target_arch = "x86_64")]
    unsafe {
        crate::kernels::avx::norm::layer_norm_x86(
            input.data.as_ptr(),
            scale.data.as_ptr(),
            bias.data.as_ptr(),
            out_buf.as_mut_ptr(),
            norm_size,
            outer_size,
            epsilon,
        );
    }

    #[cfg(target_arch = "wasm32")]
    unsafe {
        crate::kernels::wasm::norm::layer_norm(
            input.data.as_ptr(),
            scale.data.as_ptr(),
            bias.data.as_ptr(),
            out_buf.as_mut_ptr(),
            norm_size,
            outer_size,
            epsilon,
        );
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "wasm32")))]
    {
        let src = &input.data;
        let gamma = &scale.data;
        let beta = &bias.data;

        #[cfg(target_arch = "aarch64")]
        unsafe {
            crate::kernels::neon::normalization::layer_norm_neon(
                src.as_ptr(),
                gamma.as_ptr(),
                beta.as_ptr(),
                out_slice.as_mut_ptr(),
                norm_size,
                outer_size,
                epsilon,
            );
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            let inv_n = 1.0 / norm_size as f32;
            for i in 0..outer_size {
                let offset = i * norm_size;
                let chunk = &src[offset..offset + norm_size];
                let out_chunk = &mut out_slice[offset..offset + norm_size];
                let mut sum = 0.0f32;
                let mut sumsq = 0.0f32;
                for &x in chunk.iter() {
                    sum += x;
                    sumsq += x * x;
                }
                let mean = sum * inv_n;
                let var = sumsq * inv_n - mean * mean;
                let inv_std = 1.0 / (var + epsilon).sqrt();
                for j in 0..norm_size {
                    out_chunk[j] = (chunk[j] - mean) * inv_std * gamma[j] + beta[j];
                }
            }
        }
    }
    TensorView {
        data: Cow::Borrowed(out_slice),
        shape: std::borrow::Cow::Owned(input.shape.to_vec()),
    }
}
pub fn batch_norm<'b, 'a>(
    input: &TensorView<'b>,
    scale: &TensorView<'b>,
    bias: &TensorView<'b>,
    mean: &TensorView<'b>,
    var: &TensorView<'b>,
    epsilon: f32,
    out_buf: &'a mut Vec<f32>,
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
        let inner_size: usize = if shape.len() > 2 {
            shape[2..].iter().product()
        } else {
            1
        };
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
        shape: Cow::Owned(shape.to_vec()),
    }
}

pub fn rms_norm<'b, 'a>(
    input: &TensorView<'b>,
    weight: &TensorView<'b>,
    axis: i32,
    epsilon: f32,
    out_buf: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let ndim = input.shape.len();
    let axis = if axis < 0 { ndim as i32 + axis } else { axis } as usize;
    let outer_size: usize = input.shape[..axis].iter().product();
    let norm_size: usize = input.shape[axis..].iter().product();
    utils::ensure_capacity(out_buf, input.data.len());
    let out_slice =
        unsafe { std::slice::from_raw_parts_mut(out_buf.as_mut_ptr(), input.data.len()) };

    #[cfg(target_arch = "aarch64")]
    unsafe {
        crate::kernels::neon::normalization::rms_norm_neon(
            input.data.as_ptr(),
            weight.data.as_ptr(),
            out_slice.as_mut_ptr(),
            norm_size,
            outer_size,
            epsilon,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let src = &input.data;
        let w = &weight.data;
        let inv_n = 1.0 / norm_size as f32;

        for i in 0..outer_size {
            let offset = i * norm_size;
            let chunk = &src[offset..offset + norm_size];
            let out_chunk = &mut out_slice[offset..offset + norm_size];

            // Compute sum of squares
            let mut sumsq = 0.0f32;
            for &x in chunk.iter() {
                sumsq += x * x;
            }

            // Compute RMS and normalize
            let mean_sq = sumsq * inv_n;
            let rms_inv = 1.0 / (mean_sq + epsilon).sqrt();

            for j in 0..norm_size {
                out_chunk[j] = chunk[j] * rms_inv * w[j];
            }
        }
    }

    TensorView {
        data: Cow::Borrowed(out_slice),
        shape: std::borrow::Cow::Owned(input.shape.to_vec()),
    }
}
