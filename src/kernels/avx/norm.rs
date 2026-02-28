#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2 single-pass LayerNorm: computes mean and variance in one pass using
/// sum + sum-of-squares, then normalize in a second pass.
/// Previous version used 3 passes (mean, variance, normalize).
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
        let _inv_n_v = _mm256_set1_ps(inv_n);

        for i in 0..outer_size {
            let offset = i * norm_size;
            let in_ptr = input.add(offset);
            let out_ptr = output.add(offset);

            // Single pass: compute sum and sum-of-squares simultaneously
            let mut sum_v = _mm256_setzero_ps();
            let mut sumsq_v = _mm256_setzero_ps();
            let mut j = 0;
            while j + 8 <= norm_size {
                let v = _mm256_loadu_ps(in_ptr.add(j));
                sum_v = _mm256_add_ps(sum_v, v);
                sumsq_v = _mm256_fmadd_ps(v, v, sumsq_v);
                j += 8;
            }
            // Horizontal reduce
            let temp_s = _mm_add_ps(
                _mm256_castps256_ps128(sum_v),
                _mm256_extractf128_ps(sum_v, 1),
            );
            let temp_s = _mm_add_ps(temp_s, _mm_movehl_ps(temp_s, temp_s));
            let temp_s = _mm_add_ss(temp_s, _mm_shuffle_ps(temp_s, temp_s, 1));
            let mut sum = _mm_cvtss_f32(temp_s);

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

            // Compute mean and variance from sum + sum_sq:
            // mean = sum / N
            // var = sum_sq / N - mean^2
            let mean = sum * inv_n;
            let var = sumsq * inv_n - mean * mean;
            let inv_std = 1.0 / (var + epsilon).sqrt();
            let mean_v = _mm256_set1_ps(mean);
            let inv_std_v = _mm256_set1_ps(inv_std);

            // Normalize, Scale & Shift
            j = 0;
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
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn softmax(src: &[f32], dst: &mut [f32]) {
    use std::arch::x86_64::*;
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
    let mut max_val = {
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
