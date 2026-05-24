#![allow(unsafe_op_in_unsafe_fn)]
pub fn rfft_forward_f32(input: &[f32], output_re: &mut [f32], output_im: &mut [f32]) {
    let n = input.len();
    debug_assert!(n > 0 && (n & (n - 1)) == 0, "FFT length must be power of 2");
    let log2n = 32 - (n as u32).leading_zeros() as usize - 1;
    let half = n / 2 + 1;

    let mut re = vec![0.0f32; n];
    let mut im = vec![0.0f32; n];

    for i in 0..n {
        let j = bit_reverse(i, log2n);
        re[j] = input[i];
    }

    let mut size = 2usize;
    while size <= n {
        let half_size = size / 2;
        let step = n / size;
        let num_batches = n / size;
        for batch in 0..num_batches {
            let batch_start = batch * size;
            for k in 0..half_size {
                let even_idx = batch_start + k;
                let odd_idx = batch_start + half_size + k;
                let angle = -2.0 * core::f32::consts::PI * (k * step) as f32 / n as f32;
                let (wr, wi) = (angle.cos(), angle.sin());
                let tr = wr * re[odd_idx] - wi * im[odd_idx];
                let ti = wr * im[odd_idx] + wi * re[odd_idx];
                re[odd_idx] = re[even_idx] - tr;
                im[odd_idx] = im[even_idx] - ti;
                re[even_idx] += tr;
                im[even_idx] += ti;
            }
        }
        size *= 2;
    }

    output_re[0] = re[0];
    output_im[0] = 0.0;
    if half > 1 {
        output_re[half - 1] = re[n / 2];
        output_im[half - 1] = 0.0;
    }
    for k in 1..(half - 1) {
        output_re[k] = re[k];
        output_im[k] = im[k];
    }
}

pub fn rfft_forward_f32_precomputed(
    input: &[f32],
    twiddles_re: &[f32],
    twiddles_im: &[f32],
    bit_rev_table: &[usize],
    re_buf: &mut [f32],
    im_buf: &mut [f32],
    output_re: &mut [f32],
    output_im: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe {
                rfft_forward_f32_precomputed_avx2(
                    input, twiddles_re, twiddles_im, bit_rev_table,
                    re_buf, im_buf, output_re, output_im,
                );
            }
            return;
        }
    }
    rfft_forward_f32_precomputed_scalar(
        input, twiddles_re, twiddles_im, bit_rev_table,
        re_buf, im_buf, output_re, output_im,
    )
}

fn rfft_forward_f32_precomputed_scalar(
    input: &[f32],
    twiddles_re: &[f32],
    twiddles_im: &[f32],
    bit_rev_table: &[usize],
    re_buf: &mut [f32],
    im_buf: &mut [f32],
    output_re: &mut [f32],
    output_im: &mut [f32],
) {
    let n = input.len();
    let half = n / 2 + 1;

    for i in 0..n {
        let j = bit_rev_table[i];
        re_buf[j] = input[i];
    }
    im_buf.fill(0.0f32);

    let mut tw_offset = 0usize;
    let mut size = 2usize;
    while size <= n {
        let half_size = size / 2;
        let num_batches = n / size;
        for batch in 0..num_batches {
            let base = batch * size;
            for k in 0..half_size {
                let even_idx = base + k;
                let odd_idx = base + half_size + k;
                let wr = twiddles_re[tw_offset + k];
                let wi = twiddles_im[tw_offset + k];
                let odd_re = re_buf[odd_idx];
                let odd_im = im_buf[odd_idx];
                let tr = wr * odd_re - wi * odd_im;
                let ti = wr * odd_im + wi * odd_re;
                re_buf[odd_idx] = re_buf[even_idx] - tr;
                im_buf[odd_idx] = im_buf[even_idx] - ti;
                re_buf[even_idx] += tr;
                im_buf[even_idx] += ti;
            }
        }
        tw_offset += half_size;
        size *= 2;
    }

    output_re[0] = re_buf[0];
    output_im[0] = 0.0;
    if half > 1 {
        output_re[half - 1] = re_buf[n / 2];
        output_im[half - 1] = 0.0;
    }
    for k in 1..(half - 1) {
        output_re[k] = re_buf[k];
        output_im[k] = im_buf[k];
    }
}

pub fn precompute_twiddles(n: usize) -> (Vec<f32>, Vec<f32>, Vec<usize>) {
    let log2n = 32 - (n as u32).leading_zeros() as usize - 1;
    let mut bit_rev = vec![0usize; n];
    for i in 0..n {
        bit_rev[i] = bit_reverse(i, log2n);
    }

    let mut tw_re = Vec::new();
    let mut tw_im = Vec::new();
    let mut size = 2usize;
    while size <= n {
        let half_size = size / 2;
        let step = n / size;
        for k in 0..half_size {
            let angle = -2.0 * core::f32::consts::PI * (k * step) as f32 / n as f32;
            tw_re.push(angle.cos());
            tw_im.push(angle.sin());
        }
        size *= 2;
    }
    (tw_re, tw_im, bit_rev)
}

#[inline(always)]
fn bit_reverse(n: usize, log2n: usize) -> usize {
    let mut r = 0usize;
    let mut x = n;
    for _ in 0..log2n {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    r
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn rfft_forward_f32_precomputed_avx2(
    input: &[f32],
    twiddles_re: &[f32],
    twiddles_im: &[f32],
    bit_rev_table: &[usize],
    re_buf: &mut [f32],
    im_buf: &mut [f32],
    output_re: &mut [f32],
    output_im: &mut [f32],
) {
    use std::arch::x86_64::*;
    let n = input.len();
    let half = n / 2 + 1;

    for i in 0..n {
        let j = bit_rev_table[i];
        *re_buf.get_unchecked_mut(j) = *input.get_unchecked(i);
    }
    im_buf.fill(0.0f32);

    let mut tw_offset = 0usize;
    let mut size = 2usize;
    while size <= n {
        let half_size = size / 2;
        let num_batches = n / size;
        for batch in 0..num_batches {
            let base = batch * size;
            let mut k = 0usize;
            while k + 8 <= half_size {
                let ev_re = _mm256_loadu_ps(re_buf.as_ptr().add(base + k));
                let ev_im = _mm256_loadu_ps(im_buf.as_ptr().add(base + k));
                let od_re = _mm256_loadu_ps(re_buf.as_ptr().add(base + half_size + k));
                let od_im = _mm256_loadu_ps(im_buf.as_ptr().add(base + half_size + k));
                let wr = _mm256_loadu_ps(twiddles_re.as_ptr().add(tw_offset + k));
                let wi = _mm256_loadu_ps(twiddles_im.as_ptr().add(tw_offset + k));

                let t_re = _mm256_fmsub_ps(wr, od_re, _mm256_mul_ps(wi, od_im));
                let t_im = _mm256_fmadd_ps(wr, od_im, _mm256_mul_ps(wi, od_re));

                _mm256_storeu_ps(re_buf.as_mut_ptr().add(base + half_size + k), _mm256_sub_ps(ev_re, t_re));
                _mm256_storeu_ps(im_buf.as_mut_ptr().add(base + half_size + k), _mm256_sub_ps(ev_im, t_im));
                _mm256_storeu_ps(re_buf.as_mut_ptr().add(base + k), _mm256_add_ps(ev_re, t_re));
                _mm256_storeu_ps(im_buf.as_mut_ptr().add(base + k), _mm256_add_ps(ev_im, t_im));

                k += 8;
            }
            while k + 4 <= half_size {
                let ev_re = _mm_loadu_ps(re_buf.as_ptr().add(base + k));
                let ev_im = _mm_loadu_ps(im_buf.as_ptr().add(base + k));
                let od_re = _mm_loadu_ps(re_buf.as_ptr().add(base + half_size + k));
                let od_im = _mm_loadu_ps(im_buf.as_ptr().add(base + half_size + k));
                let wr = _mm_loadu_ps(twiddles_re.as_ptr().add(tw_offset + k));
                let wi = _mm_loadu_ps(twiddles_im.as_ptr().add(tw_offset + k));

                let t_re = _mm_fmsub_ps(wr, od_re, _mm_mul_ps(wi, od_im));
                let t_im = _mm_fmadd_ps(wr, od_im, _mm_mul_ps(wi, od_re));

                _mm_storeu_ps(re_buf.as_mut_ptr().add(base + half_size + k), _mm_sub_ps(ev_re, t_re));
                _mm_storeu_ps(im_buf.as_mut_ptr().add(base + half_size + k), _mm_sub_ps(ev_im, t_im));
                _mm_storeu_ps(re_buf.as_mut_ptr().add(base + k), _mm_add_ps(ev_re, t_re));
                _mm_storeu_ps(im_buf.as_mut_ptr().add(base + k), _mm_add_ps(ev_im, t_im));

                k += 4;
            }
            while k < half_size {
                let even_idx = base + k;
                let odd_idx = base + half_size + k;
                let wr = *twiddles_re.get_unchecked(tw_offset + k);
                let wi = *twiddles_im.get_unchecked(tw_offset + k);
                let odd_re = *re_buf.get_unchecked(odd_idx);
                let odd_im = *im_buf.get_unchecked(odd_idx);
                let tr = wr * odd_re - wi * odd_im;
                let ti = wr * odd_im + wi * odd_re;
                *re_buf.get_unchecked_mut(odd_idx) = *re_buf.get_unchecked(even_idx) - tr;
                *im_buf.get_unchecked_mut(odd_idx) = *im_buf.get_unchecked(even_idx) - ti;
                *re_buf.get_unchecked_mut(even_idx) += tr;
                *im_buf.get_unchecked_mut(even_idx) += ti;
                k += 1;
            }
        }
        tw_offset += half_size;
        size *= 2;
    }

    *output_re.get_unchecked_mut(0) = *re_buf.get_unchecked(0);
    *output_im.get_unchecked_mut(0) = 0.0;
    if half > 1 {
        *output_re.get_unchecked_mut(half - 1) = *re_buf.get_unchecked(n / 2);
        *output_im.get_unchecked_mut(half - 1) = 0.0;
    }
    for k in 1..(half - 1) {
        *output_re.get_unchecked_mut(k) = *re_buf.get_unchecked(k);
        *output_im.get_unchecked_mut(k) = *im_buf.get_unchecked(k);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rfft_simple() {
        let input: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
        let n = input.len();
        let half = n / 2 + 1;
        let mut re = vec![0.0f32; half];
        let mut im = vec![0.0f32; half];
        rfft_forward_f32(&input, &mut re, &mut im);

        let expected_re = [36.0, -4.0, -4.0, -4.0, -4.0];
        let expected_im = [0.0, 9.656854, 4.0, 1.656854, 0.0];
        for k in 0..half {
            assert!(
                (re[k] - expected_re[k]).abs() < 1e-3,
                "re[{}] = {} expected {}",
                k,
                re[k],
                expected_re[k]
            );
            assert!(
                (im[k] - expected_im[k]).abs() < 1e-3,
                "im[{}] = {} expected {}",
                k,
                im[k],
                expected_im[k]
            );
        }
    }

    #[test]
    fn test_rfft_avx2_matches_scalar() {
        let mut rng_state: u32 = 12345;
        let mut next_f32 = || -> f32 {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            (rng_state as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        for log_n in &[3, 4, 5, 6, 7, 8, 9, 10] {
            let n = 1usize << log_n;
            let half = n / 2 + 1;
            let input: Vec<f32> = (0..n).map(|_| next_f32()).collect();
            let (tw_re, tw_im, br) = precompute_twiddles(n);

            let mut re_buf_s = vec![0.0f32; n];
            let mut im_buf_s = vec![0.0f32; n];
            let mut out_re_s = vec![0.0f32; half];
            let mut out_im_s = vec![0.0f32; half];
            rfft_forward_f32_precomputed_scalar(
                &input, &tw_re, &tw_im, &br,
                &mut re_buf_s, &mut im_buf_s, &mut out_re_s, &mut out_im_s,
            );

            let mut re_buf_a = vec![0.0f32; n];
            let mut im_buf_a = vec![0.0f32; n];
            let mut out_re_a = vec![0.0f32; half];
            let mut out_im_a = vec![0.0f32; half];
            rfft_forward_f32_precomputed(
                &input, &tw_re, &tw_im, &br,
                &mut re_buf_a, &mut im_buf_a, &mut out_re_a, &mut out_im_a,
            );

            let mut max_diff_re = 0.0f32;
            let mut max_diff_im = 0.0f32;
            for k in 0..half {
                max_diff_re = max_diff_re.max((out_re_s[k] - out_re_a[k]).abs());
                max_diff_im = max_diff_im.max((out_im_s[k] - out_im_a[k]).abs());
            }
            assert!(
                max_diff_re < 1e-4,
                "n={}: max re diff = {}",
                n, max_diff_re
            );
            assert!(
                max_diff_im < 1e-4,
                "n={}: max im diff = {}",
                n, max_diff_im
            );
        }
    }
}
