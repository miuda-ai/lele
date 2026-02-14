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
