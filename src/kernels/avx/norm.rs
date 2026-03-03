#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2 single-pass LayerNorm: computes mean and variance in one pass using
/// sum + sum-of-squares, then normalize in a second pass.
/// Uses 4-way unrolling (32 elements per iteration) to maximize throughput.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn layer_norm_x86(
    input: *const f32,
    scale: *const f32,
    bias: *const f32,
    output: *mut f32,
    norm_size: usize,
    outer_size: usize,
    epsilon: f32,
) {
    unsafe {
        let inv_n = 1.0 / norm_size as f32;

        for i in 0..outer_size {
            let offset = i * norm_size;
            let in_ptr = input.add(offset);
            let out_ptr = output.add(offset);

            // Single pass: compute sum and sum-of-squares with 4-way unrolling
            let mut sum0 = _mm256_setzero_ps();
            let mut sum1 = _mm256_setzero_ps();
            let mut sum2 = _mm256_setzero_ps();
            let mut sum3 = _mm256_setzero_ps();
            let mut sq0 = _mm256_setzero_ps();
            let mut sq1 = _mm256_setzero_ps();
            let mut sq2 = _mm256_setzero_ps();
            let mut sq3 = _mm256_setzero_ps();
            let mut j = 0;
            while j + 32 <= norm_size {
                let v0 = _mm256_loadu_ps(in_ptr.add(j));
                let v1 = _mm256_loadu_ps(in_ptr.add(j + 8));
                let v2 = _mm256_loadu_ps(in_ptr.add(j + 16));
                let v3 = _mm256_loadu_ps(in_ptr.add(j + 24));
                sum0 = _mm256_add_ps(sum0, v0);
                sum1 = _mm256_add_ps(sum1, v1);
                sum2 = _mm256_add_ps(sum2, v2);
                sum3 = _mm256_add_ps(sum3, v3);
                sq0 = _mm256_fmadd_ps(v0, v0, sq0);
                sq1 = _mm256_fmadd_ps(v1, v1, sq1);
                sq2 = _mm256_fmadd_ps(v2, v2, sq2);
                sq3 = _mm256_fmadd_ps(v3, v3, sq3);
                j += 32;
            }
            // Merge accumulators
            let mut sum_v = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
            let mut sumsq_v = _mm256_add_ps(_mm256_add_ps(sq0, sq1), _mm256_add_ps(sq2, sq3));
            while j + 8 <= norm_size {
                let v = _mm256_loadu_ps(in_ptr.add(j));
                sum_v = _mm256_add_ps(sum_v, v);
                sumsq_v = _mm256_fmadd_ps(v, v, sumsq_v);
                j += 8;
            }
            // Horizontal reduce sum
            let temp_s = _mm_add_ps(
                _mm256_castps256_ps128(sum_v),
                _mm256_extractf128_ps(sum_v, 1),
            );
            let temp_s = _mm_add_ps(temp_s, _mm_movehl_ps(temp_s, temp_s));
            let temp_s = _mm_add_ss(temp_s, _mm_shuffle_ps(temp_s, temp_s, 1));
            let mut sum = _mm_cvtss_f32(temp_s);
            // Horizontal reduce sumsq
            let temp_sq = _mm_add_ps(
                _mm256_castps256_ps128(sumsq_v),
                _mm256_extractf128_ps(sumsq_v, 1),
            );
            let temp_sq = _mm_add_ps(temp_sq, _mm_movehl_ps(temp_sq, temp_sq));
            let temp_sq = _mm_add_ss(temp_sq, _mm_shuffle_ps(temp_sq, temp_sq, 1));
            let mut sumsq = _mm_cvtss_f32(temp_sq);
            // Scalar tail
            while j < norm_size {
                let v = *in_ptr.add(j);
                sum += v;
                sumsq += v * v;
                j += 1;
            }

            let mean = sum * inv_n;
            let var = sumsq * inv_n - mean * mean;
            let inv_std = 1.0 / (var + epsilon).sqrt();
            let mean_v = _mm256_set1_ps(mean);
            let inv_std_v = _mm256_set1_ps(inv_std);

            // Normalize, Scale & Shift with 4-way unrolling
            j = 0;
            while j + 32 <= norm_size {
                let v0 = _mm256_loadu_ps(in_ptr.add(j));
                let v1 = _mm256_loadu_ps(in_ptr.add(j + 8));
                let v2 = _mm256_loadu_ps(in_ptr.add(j + 16));
                let v3 = _mm256_loadu_ps(in_ptr.add(j + 24));
                let g0 = _mm256_loadu_ps(scale.add(j));
                let g1 = _mm256_loadu_ps(scale.add(j + 8));
                let g2 = _mm256_loadu_ps(scale.add(j + 16));
                let g3 = _mm256_loadu_ps(scale.add(j + 24));
                let b0 = _mm256_loadu_ps(bias.add(j));
                let b1 = _mm256_loadu_ps(bias.add(j + 8));
                let b2 = _mm256_loadu_ps(bias.add(j + 16));
                let b3 = _mm256_loadu_ps(bias.add(j + 24));
                let r0 = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_sub_ps(v0, mean_v), inv_std_v), g0, b0);
                let r1 = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_sub_ps(v1, mean_v), inv_std_v), g1, b1);
                let r2 = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_sub_ps(v2, mean_v), inv_std_v), g2, b2);
                let r3 = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_sub_ps(v3, mean_v), inv_std_v), g3, b3);
                _mm256_storeu_ps(out_ptr.add(j), r0);
                _mm256_storeu_ps(out_ptr.add(j + 8), r1);
                _mm256_storeu_ps(out_ptr.add(j + 16), r2);
                _mm256_storeu_ps(out_ptr.add(j + 24), r3);
                j += 32;
            }
            while j + 8 <= norm_size {
                let v_in = _mm256_loadu_ps(in_ptr.add(j));
                let v_gamma = _mm256_loadu_ps(scale.add(j));
                let v_beta = _mm256_loadu_ps(bias.add(j));
                let v_norm = _mm256_sub_ps(v_in, mean_v);
                let v_scaled = _mm256_mul_ps(v_norm, inv_std_v);
                let v_res = _mm256_fmadd_ps(v_scaled, v_gamma, v_beta);
                _mm256_storeu_ps(out_ptr.add(j), v_res);
                j += 8;
            }
            while j < norm_size {
                let val = *in_ptr.add(j);
                *out_ptr.add(j) = (val - mean) * inv_std * *scale.add(j) + *bias.add(j);
                j += 1;
            }
        }
    }
}

/// AVX2 softmax over a contiguous slice (inner_size == 1 case).
/// Uses 4-way unrolling (32 elements per iteration) for max, exp+sum, normalize passes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn softmax(src: &[f32], dst: &mut [f32]) {
    let len = src.len();
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    // 1. Find max with 4-way unrolling
    let mut max0 = _mm256_set1_ps(f32::MIN);
    let mut max1 = _mm256_set1_ps(f32::MIN);
    let mut max2 = _mm256_set1_ps(f32::MIN);
    let mut max3 = _mm256_set1_ps(f32::MIN);
    let mut j = 0;
    while j + 32 <= len {
        max0 = _mm256_max_ps(max0, _mm256_loadu_ps(src_ptr.add(j)));
        max1 = _mm256_max_ps(max1, _mm256_loadu_ps(src_ptr.add(j + 8)));
        max2 = _mm256_max_ps(max2, _mm256_loadu_ps(src_ptr.add(j + 16)));
        max3 = _mm256_max_ps(max3, _mm256_loadu_ps(src_ptr.add(j + 24)));
        j += 32;
    }
    let mut max_vec = _mm256_max_ps(_mm256_max_ps(max0, max1), _mm256_max_ps(max2, max3));
    while j + 8 <= len {
        max_vec = _mm256_max_ps(max_vec, _mm256_loadu_ps(src_ptr.add(j)));
        j += 8;
    }
    let mut max_val = {
        let hi = _mm256_extractf128_ps(max_vec, 1);
        let lo = _mm256_castps256_ps128(max_vec);
        let m128 = _mm_max_ps(lo, hi);
        let m64 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
        let m32 = _mm_max_ss(m64, _mm_shuffle_ps(m64, m64, 1));
        _mm_cvtss_f32(m32)
    };
    for k in j..len {
        max_val = max_val.max(*src_ptr.add(k));
    }

    // 2. exp(x - max) and sum with 4-way unrolling
    let max_broadcast = _mm256_set1_ps(max_val);
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();
    j = 0;
    while j + 32 <= len {
        let e0 = crate::kernels::avx::math::avx2_exp_ps(_mm256_sub_ps(_mm256_loadu_ps(src_ptr.add(j)), max_broadcast));
        let e1 = crate::kernels::avx::math::avx2_exp_ps(_mm256_sub_ps(_mm256_loadu_ps(src_ptr.add(j + 8)), max_broadcast));
        let e2 = crate::kernels::avx::math::avx2_exp_ps(_mm256_sub_ps(_mm256_loadu_ps(src_ptr.add(j + 16)), max_broadcast));
        let e3 = crate::kernels::avx::math::avx2_exp_ps(_mm256_sub_ps(_mm256_loadu_ps(src_ptr.add(j + 24)), max_broadcast));
        _mm256_storeu_ps(dst_ptr.add(j), e0);
        _mm256_storeu_ps(dst_ptr.add(j + 8), e1);
        _mm256_storeu_ps(dst_ptr.add(j + 16), e2);
        _mm256_storeu_ps(dst_ptr.add(j + 24), e3);
        sum0 = _mm256_add_ps(sum0, e0);
        sum1 = _mm256_add_ps(sum1, e1);
        sum2 = _mm256_add_ps(sum2, e2);
        sum3 = _mm256_add_ps(sum3, e3);
        j += 32;
    }
    let mut sum_vec = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    while j + 8 <= len {
        let e = crate::kernels::avx::math::avx2_exp_ps(_mm256_sub_ps(_mm256_loadu_ps(src_ptr.add(j)), max_broadcast));
        _mm256_storeu_ps(dst_ptr.add(j), e);
        sum_vec = _mm256_add_ps(sum_vec, e);
        j += 8;
    }
    let mut sum = crate::kernels::avx::math::hsum_ps(sum_vec);
    for k in j..len {
        let e = (*src_ptr.add(k) - max_val).exp();
        *dst_ptr.add(k) = e;
        sum += e;
    }

    // 3. Normalize with 4-way unrolling
    let inv_sum = 1.0 / sum;
    let inv_sum_vec = _mm256_set1_ps(inv_sum);
    j = 0;
    while j + 32 <= len {
        _mm256_storeu_ps(dst_ptr.add(j), _mm256_mul_ps(_mm256_loadu_ps(dst_ptr.add(j)), inv_sum_vec));
        _mm256_storeu_ps(dst_ptr.add(j + 8), _mm256_mul_ps(_mm256_loadu_ps(dst_ptr.add(j + 8)), inv_sum_vec));
        _mm256_storeu_ps(dst_ptr.add(j + 16), _mm256_mul_ps(_mm256_loadu_ps(dst_ptr.add(j + 16)), inv_sum_vec));
        _mm256_storeu_ps(dst_ptr.add(j + 24), _mm256_mul_ps(_mm256_loadu_ps(dst_ptr.add(j + 24)), inv_sum_vec));
        j += 32;
    }
    while j + 8 <= len {
        let v = _mm256_loadu_ps(dst_ptr.add(j));
        _mm256_storeu_ps(dst_ptr.add(j), _mm256_mul_ps(v, inv_sum_vec));
        j += 8;
    }
    for k in j..len {
        *dst_ptr.add(k) *= inv_sum;
    }
}

/// AVX2 RMSNorm: out[i] = x[i] * weight[i] / rms(x)
/// rms(x) = sqrt(mean(x^2) + epsilon)
/// Two-pass: first accumulate sum-of-squares, then normalize.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn rms_norm_x86(
    input: *const f32,
    weight: *const f32,
    output: *mut f32,
    norm_size: usize,
    outer_size: usize,
    epsilon: f32,
) {
    unsafe {
        let inv_n = 1.0 / norm_size as f32;

        for i in 0..outer_size {
            let offset = i * norm_size;
            let in_ptr = input.add(offset);
            let out_ptr = output.add(offset);

            // Pass 1: compute sum of squares with 4-way unrolling
            let mut sq0 = _mm256_setzero_ps();
            let mut sq1 = _mm256_setzero_ps();
            let mut sq2 = _mm256_setzero_ps();
            let mut sq3 = _mm256_setzero_ps();
            let mut j = 0;
            while j + 32 <= norm_size {
                let v0 = _mm256_loadu_ps(in_ptr.add(j));
                let v1 = _mm256_loadu_ps(in_ptr.add(j + 8));
                let v2 = _mm256_loadu_ps(in_ptr.add(j + 16));
                let v3 = _mm256_loadu_ps(in_ptr.add(j + 24));
                sq0 = _mm256_fmadd_ps(v0, v0, sq0);
                sq1 = _mm256_fmadd_ps(v1, v1, sq1);
                sq2 = _mm256_fmadd_ps(v2, v2, sq2);
                sq3 = _mm256_fmadd_ps(v3, v3, sq3);
                j += 32;
            }
            // Merge accumulators
            sq0 = _mm256_add_ps(_mm256_add_ps(sq0, sq1), _mm256_add_ps(sq2, sq3));
            while j + 8 <= norm_size {
                let v = _mm256_loadu_ps(in_ptr.add(j));
                sq0 = _mm256_fmadd_ps(v, v, sq0);
                j += 8;
            }
            // Horizontal sum
            let hi = _mm256_extractf128_ps(sq0, 1);
            let lo = _mm256_castps256_ps128(sq0);
            let s128 = _mm_add_ps(lo, hi);
            let s64 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
            let s32 = _mm_add_ss(s64, _mm_shuffle_ps(s64, s64, 1));
            let mut sumsq = _mm_cvtss_f32(s32);
            // Scalar tail
            while j < norm_size {
                let v = *in_ptr.add(j);
                sumsq += v * v;
                j += 1;
            }

            let mean_sq = sumsq * inv_n;
            let rms_inv = 1.0 / (mean_sq + epsilon).sqrt();
            let rms_inv_v = _mm256_set1_ps(rms_inv);

            // Pass 2: normalize and scale with 4-way unrolling.
            // Fuse rms_inv into weight: eff_w = weight * rms_inv, then out = x * eff_w.
            // This removes one multiply from the inner loop when weight is used multiple times
            // (only beneficial for outer_size > 1; for single pass still saves 1 dep chain level).
            j = 0;
            while j + 32 <= norm_size {
                let w0 = _mm256_mul_ps(_mm256_loadu_ps(weight.add(j)), rms_inv_v);
                let w1 = _mm256_mul_ps(_mm256_loadu_ps(weight.add(j + 8)), rms_inv_v);
                let w2 = _mm256_mul_ps(_mm256_loadu_ps(weight.add(j + 16)), rms_inv_v);
                let w3 = _mm256_mul_ps(_mm256_loadu_ps(weight.add(j + 24)), rms_inv_v);
                _mm256_storeu_ps(out_ptr.add(j), _mm256_mul_ps(_mm256_loadu_ps(in_ptr.add(j)), w0));
                _mm256_storeu_ps(out_ptr.add(j + 8), _mm256_mul_ps(_mm256_loadu_ps(in_ptr.add(j + 8)), w1));
                _mm256_storeu_ps(out_ptr.add(j + 16), _mm256_mul_ps(_mm256_loadu_ps(in_ptr.add(j + 16)), w2));
                _mm256_storeu_ps(out_ptr.add(j + 24), _mm256_mul_ps(_mm256_loadu_ps(in_ptr.add(j + 24)), w3));
                j += 32;
            }
            while j + 8 <= norm_size {
                let w = _mm256_mul_ps(_mm256_loadu_ps(weight.add(j)), rms_inv_v);
                _mm256_storeu_ps(out_ptr.add(j), _mm256_mul_ps(_mm256_loadu_ps(in_ptr.add(j)), w));
                j += 8;
            }
            while j < norm_size {
                *out_ptr.add(j) = *in_ptr.add(j) * rms_inv * *weight.add(j);
                j += 1;
            }
        }
    }
}

/// AVX2 batch_norm spatial kernel: out[i] = src[i] * scale_val + bias_val
/// Operates on a contiguous [spatial_size] slice with precomputed scale_val/bias_val per channel.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn batch_norm_spatial_x86(
    src: *const f32,
    out: *mut f32,
    scale_val: f32,
    bias_val: f32,
    spatial_size: usize,
) {
    unsafe {
        let sv = _mm256_set1_ps(scale_val);
        let bv = _mm256_set1_ps(bias_val);
        let mut i = 0;
        while i + 32 <= spatial_size {
            let v0 = _mm256_loadu_ps(src.add(i));
            let v1 = _mm256_loadu_ps(src.add(i + 8));
            let v2 = _mm256_loadu_ps(src.add(i + 16));
            let v3 = _mm256_loadu_ps(src.add(i + 24));
            _mm256_storeu_ps(out.add(i), _mm256_fmadd_ps(v0, sv, bv));
            _mm256_storeu_ps(out.add(i + 8), _mm256_fmadd_ps(v1, sv, bv));
            _mm256_storeu_ps(out.add(i + 16), _mm256_fmadd_ps(v2, sv, bv));
            _mm256_storeu_ps(out.add(i + 24), _mm256_fmadd_ps(v3, sv, bv));
            i += 32;
        }
        while i + 8 <= spatial_size {
            let v = _mm256_loadu_ps(src.add(i));
            _mm256_storeu_ps(out.add(i), _mm256_fmadd_ps(v, sv, bv));
            i += 8;
        }
        while i < spatial_size {
            *out.add(i) = *src.add(i) * scale_val + bias_val;
            i += 1;
        }
    }
}
