#![allow(unsafe_op_in_unsafe_fn)]
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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
