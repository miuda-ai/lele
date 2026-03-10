use crate::kernels::utils;
use crate::tensor::TensorView;
use std::borrow::Cow;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn softmax_avx2(src: &[f32], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    unsafe {
        let len = src.len();
        let simd_end = (len / 8) * 8;
        let src_ptr = src.as_ptr();
        let dst_ptr = dst.as_mut_ptr();

        // 1. Find max
        let mut max_vec = _mm256_set1_ps(f32::MIN);
        let mut j = 0;
        while j < simd_end {
            let v = _mm256_loadu_ps(src_ptr.add(j));
            max_vec = _mm256_max_ps(max_vec, v);
            j += 8;
        }
        let mut max_val = crate::kernels::avx::math::hsum_ps(_mm256_max_ps(
            max_vec,
            _mm256_permute2f128_ps(max_vec, max_vec, 1),
        ));
        // Actually use proper horizontal max
        max_val = {
            let hi = _mm256_extractf128_ps(max_vec, 1);
            let lo = _mm256_castps256_ps128(max_vec);
            let m128 = _mm_max_ps(lo, hi);
            let m64 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
            let m32 = _mm_max_ss(m64, _mm_shuffle_ps(m64, m64, 1));
            _mm_cvtss_f32(m32)
        };
        for k in simd_end..len {
            max_val = max_val.max(*src_ptr.add(k));
        }

        // 2. exp(x - max) and sum
        let max_broadcast = _mm256_set1_ps(max_val);
        let mut sum_vec = _mm256_setzero_ps();
        j = 0;
        while j < simd_end {
            let v = _mm256_loadu_ps(src_ptr.add(j));
            let shifted = _mm256_sub_ps(v, max_broadcast);
            let exp_val = crate::kernels::avx::math::avx2_exp_ps(shifted);
            _mm256_storeu_ps(dst_ptr.add(j), exp_val);
            sum_vec = _mm256_add_ps(sum_vec, exp_val);
            j += 8;
        }
        let mut sum = crate::kernels::avx::math::hsum_ps(sum_vec);
        for k in simd_end..len {
            let e = (*src_ptr.add(k) - max_val).exp();
            *dst_ptr.add(k) = e;
            sum += e;
        }

        // 3. Normalize
        let inv_sum = 1.0 / sum;
        let inv_sum_vec = _mm256_set1_ps(inv_sum);
        j = 0;
        while j < simd_end {
            let v = _mm256_loadu_ps(dst_ptr.add(j));
            let normalized = _mm256_mul_ps(v, inv_sum_vec);
            _mm256_storeu_ps(dst_ptr.add(j), normalized);
            j += 8;
        }
        for k in simd_end..len {
            *dst_ptr.add(k) *= inv_sum;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fast_exp_f32x4(x: float32x4_t) -> float32x4_t {
    // Fast exp approximation using exp(x) ≈ 2^(x/ln2)
    // Accurate to ~0.5% for typical softmax range [-10, 0]

    const LN2_RECIP: f32 = 1.442695041; // 1/ln(2)
    const C1: f32 = 1.0;
    const C2: f32 = 0.693147181;
    const C3: f32 = 0.240226507;
    const C4: f32 = 0.055504109;
    const C5: f32 = 0.009618129;

    let ln2_recip_vec = vdupq_n_f32(LN2_RECIP);
    let c1_vec = vdupq_n_f32(C1);
    let c2_vec = vdupq_n_f32(C2);
    let c3_vec = vdupq_n_f32(C3);
    let c4_vec = vdupq_n_f32(C4);
    let c5_vec = vdupq_n_f32(C5);

    // Convert to base-2: x' = x / ln(2)
    let x_scaled = vmulq_f32(x, ln2_recip_vec);

    // Split into integer and fractional parts
    let x_floor = vcvtq_s32_f32(x_scaled);
    let x_int = vcvtq_f32_s32(x_floor);
    let x_frac = vsubq_f32(x_scaled, x_int);

    // Polynomial approximation: 2^f ≈ 1 + f*C2 + f²*C3 + f³*C4 + f⁴*C5
    let f2 = vmulq_f32(x_frac, x_frac);
    let f3 = vmulq_f32(f2, x_frac);
    let f4 = vmulq_f32(f3, x_frac);

    let poly = vaddq_f32(
        c1_vec,
        vaddq_f32(
            vmulq_f32(x_frac, c2_vec),
            vaddq_f32(
                vmulq_f32(f2, c3_vec),
                vaddq_f32(vmulq_f32(f3, c4_vec), vmulq_f32(f4, c5_vec)),
            ),
        ),
    );

    // Scale by 2^(integer part) using bit manipulation
    let bias = vdupq_n_s32(127);
    let exponent = vaddq_s32(x_floor, bias);
    let result_bits = vshlq_n_s32(exponent, 23);
    let scale = vreinterpretq_f32_s32(result_bits);

    vmulq_f32(poly, scale)
}

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
            use std::arch::aarch64::*;
            for i in 0..outer_size {
                let start = i * axis_size;
                let end = start + axis_size;
                let src = &data[start..end];
                let dst = &mut out_slice[start..end];

                // SIMD max finding
                unsafe {
                    let mut max_vec = vdupq_n_f32(f32::MIN);
                    let mut j = 0;
                    let simd_end = (axis_size / 4) * 4;

                    while j < simd_end {
                        let v = vld1q_f32(src.as_ptr().add(j));
                        max_vec = vmaxq_f32(max_vec, v);
                        j += 4;
                    }

                    // Horizontal max
                    let max_val = vgetq_lane_f32(max_vec, 0)
                        .max(vgetq_lane_f32(max_vec, 1))
                        .max(vgetq_lane_f32(max_vec, 2))
                        .max(vgetq_lane_f32(max_vec, 3))
                        .max(src[simd_end..].iter().fold(f32::MIN, |a, &b| a.max(b)));

                    // SIMD exp and sum
                    let max_broadcast = vdupq_n_f32(max_val);
                    let mut sum_vec = vdupq_n_f32(0.0);

                    j = 0;
                    while j < simd_end {
                        let v = vld1q_f32(src.as_ptr().add(j));
                        let shifted = vsubq_f32(v, max_broadcast);

                        // Fast exp approximation (accurate enough for softmax)
                        let exp_val = fast_exp_f32x4(shifted);
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

        #[cfg(not(target_arch = "aarch64"))]
        {
            #[cfg(target_arch = "x86_64")]
            {
                for i in 0..outer_size {
                    let start = i * axis_size;
                    let end = start + axis_size;
                    let src = &data[start..end];
                    let dst = &mut out_slice[start..end];
                    unsafe {
                        softmax_avx2(src, dst);
                    }
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
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

    let src = &input.data;
    let gamma = &scale.data;
    let beta = &bias.data;

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        for i in 0..outer_size {
            let offset = i * norm_size;
            let chunk = &src[offset..offset + norm_size];
            let out_chunk = &mut out_slice[offset..offset + norm_size];

            unsafe {
                // Single-pass mean + variance with 4-way loop unrolling
                let mut sum_v0 = vdupq_n_f32(0.0);
                let mut sum_v1 = vdupq_n_f32(0.0);
                let mut sum_v2 = vdupq_n_f32(0.0);
                let mut sum_v3 = vdupq_n_f32(0.0);
                let mut sum_sq_v0 = vdupq_n_f32(0.0);
                let mut sum_sq_v1 = vdupq_n_f32(0.0);
                let mut sum_sq_v2 = vdupq_n_f32(0.0);
                let mut sum_sq_v3 = vdupq_n_f32(0.0);

                let mut j = 0;
                let unroll_end = (norm_size / 16) * 16;

                // 4-way unrolled loop: process 16 elements per iteration
                while j < unroll_end {
                    let v0 = vld1q_f32(chunk.as_ptr().add(j));
                    let v1 = vld1q_f32(chunk.as_ptr().add(j + 4));
                    let v2 = vld1q_f32(chunk.as_ptr().add(j + 8));
                    let v3 = vld1q_f32(chunk.as_ptr().add(j + 12));

                    sum_v0 = vaddq_f32(sum_v0, v0);
                    sum_v1 = vaddq_f32(sum_v1, v1);
                    sum_v2 = vaddq_f32(sum_v2, v2);
                    sum_v3 = vaddq_f32(sum_v3, v3);

                    sum_sq_v0 = vfmaq_f32(sum_sq_v0, v0, v0);
                    sum_sq_v1 = vfmaq_f32(sum_sq_v1, v1, v1);
                    sum_sq_v2 = vfmaq_f32(sum_sq_v2, v2, v2);
                    sum_sq_v3 = vfmaq_f32(sum_sq_v3, v3, v3);

                    j += 16;
                }

                // Combine accumulators
                sum_v0 = vaddq_f32(vaddq_f32(sum_v0, sum_v1), vaddq_f32(sum_v2, sum_v3));
                sum_sq_v0 = vaddq_f32(
                    vaddq_f32(sum_sq_v0, sum_sq_v1),
                    vaddq_f32(sum_sq_v2, sum_sq_v3),
                );

                // Cleanup: remaining 0-15 elements
                let simd_end = (norm_size / 4) * 4;
                while j < simd_end {
                    let v = vld1q_f32(chunk.as_ptr().add(j));
                    sum_v0 = vaddq_f32(sum_v0, v);
                    sum_sq_v0 = vfmaq_f32(sum_sq_v0, v, v);
                    j += 4;
                }

                // Horizontal reduction
                let mut sum = vgetq_lane_f32(sum_v0, 0)
                    + vgetq_lane_f32(sum_v0, 1)
                    + vgetq_lane_f32(sum_v0, 2)
                    + vgetq_lane_f32(sum_v0, 3);
                let mut sum_sq = vgetq_lane_f32(sum_sq_v0, 0)
                    + vgetq_lane_f32(sum_sq_v0, 1)
                    + vgetq_lane_f32(sum_sq_v0, 2)
                    + vgetq_lane_f32(sum_sq_v0, 3);

                // Remaining scalar elements
                while j < norm_size {
                    let val = chunk[j];
                    sum += val;
                    sum_sq += val * val;
                    j += 1;
                }

                let mean = sum / norm_size as f32;
                let var = (sum_sq / norm_size as f32) - (mean * mean);
                let inv_std = 1.0 / (var + epsilon).sqrt();

                let mean_v = vdupq_n_f32(mean);
                let inv_std_v = vdupq_n_f32(inv_std);

                // 4-way unrolled normalize
                j = 0;
                while j < unroll_end {
                    let v0 = vld1q_f32(chunk.as_ptr().add(j));
                    let v1 = vld1q_f32(chunk.as_ptr().add(j + 4));
                    let v2 = vld1q_f32(chunk.as_ptr().add(j + 8));
                    let v3 = vld1q_f32(chunk.as_ptr().add(j + 12));

                    let g0 = vld1q_f32(gamma.as_ptr().add(j));
                    let g1 = vld1q_f32(gamma.as_ptr().add(j + 4));
                    let g2 = vld1q_f32(gamma.as_ptr().add(j + 8));
                    let g3 = vld1q_f32(gamma.as_ptr().add(j + 12));

                    let b0 = vld1q_f32(beta.as_ptr().add(j));
                    let b1 = vld1q_f32(beta.as_ptr().add(j + 4));
                    let b2 = vld1q_f32(beta.as_ptr().add(j + 8));
                    let b3 = vld1q_f32(beta.as_ptr().add(j + 12));

                    let c0 = vsubq_f32(v0, mean_v);
                    let c1 = vsubq_f32(v1, mean_v);
                    let c2 = vsubq_f32(v2, mean_v);
                    let c3 = vsubq_f32(v3, mean_v);

                    let n0 = vmulq_f32(c0, inv_std_v);
                    let n1 = vmulq_f32(c1, inv_std_v);
                    let n2 = vmulq_f32(c2, inv_std_v);
                    let n3 = vmulq_f32(c3, inv_std_v);

                    let r0 = vfmaq_f32(b0, n0, g0);
                    let r1 = vfmaq_f32(b1, n1, g1);
                    let r2 = vfmaq_f32(b2, n2, g2);
                    let r3 = vfmaq_f32(b3, n3, g3);

                    vst1q_f32(out_chunk.as_mut_ptr().add(j), r0);
                    vst1q_f32(out_chunk.as_mut_ptr().add(j + 4), r1);
                    vst1q_f32(out_chunk.as_mut_ptr().add(j + 8), r2);
                    vst1q_f32(out_chunk.as_mut_ptr().add(j + 12), r3);

                    j += 16;
                }

                // Cleanup remaining elements
                while j < simd_end {
                    let v = vld1q_f32(chunk.as_ptr().add(j));
                    let g = vld1q_f32(gamma.as_ptr().add(j));
                    let b = vld1q_f32(beta.as_ptr().add(j));

                    let centered = vsubq_f32(v, mean_v);
                    let normalized = vmulq_f32(centered, inv_std_v);
                    let result = vfmaq_f32(b, normalized, g);
                    vst1q_f32(out_chunk.as_mut_ptr().add(j), result);
                    j += 4;
                }

                // Remaining scalar elements
                while j < norm_size {
                    out_chunk[j] = (chunk[j] - mean) * inv_std * gamma[j] + beta[j];
                    j += 1;
                }
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..outer_size {
            let offset = i * norm_size;
            let chunk = &src[offset..offset + norm_size];
            let out_chunk = &mut out_slice[offset..offset + norm_size];
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
