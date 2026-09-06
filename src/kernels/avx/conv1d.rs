#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn conv1d_direct_k3_x86(
    batch_size: usize,
    in_channels: usize,
    input_len: usize,
    out_channels: usize,
    padding: usize,
    stride: usize,
    output_len: usize,
    relu: bool,
    bias: Option<*const f32>,
    input: *const f32,
    weights: *const f32,
    output: *mut f32,
) {
    let w_stride_oc = in_channels * 3;
    let in_stride_ch = input_len;
    let out_stride_ch = output_len;
    let zero_v = _mm256_setzero_ps();

    unsafe {
        for b in 0..batch_size {
            let in_base = input.add(b * in_channels * input_len);
            let out_base = output.add(b * out_channels * output_len);

            let mut oc = 0;
            // Process 4 Output Channels at a time
            while oc + 4 <= out_channels {
                let w_base0 = weights.add(oc * w_stride_oc);
                let w_base1 = weights.add((oc + 1) * w_stride_oc);
                let w_base2 = weights.add((oc + 2) * w_stride_oc);
                let w_base3 = weights.add((oc + 3) * w_stride_oc);

                let out_ptr0 = out_base.add(oc * out_stride_ch);
                let out_ptr1 = out_base.add((oc + 1) * out_stride_ch);
                let out_ptr2 = out_base.add((oc + 2) * out_stride_ch);
                let out_ptr3 = out_base.add((oc + 3) * out_stride_ch);

                let mut t = 0;

                // Only optimize stride=1 loop with AVX for now
                if stride == 1 {
                    while t + 8 <= output_len {
                        // Accumulators for 8 time steps
                        let mut acc0 = zero_v;
                        let mut acc1 = zero_v;
                        let mut acc2 = zero_v;
                        let mut acc3 = zero_v;

                        // Compute start index in input
                        let t_in_start = (t as isize) - (padding as isize);
                        let safe_start = t_in_start >= 0;

                        let limit = input_len as isize;
                        let safe_end = (t_in_start + 9) < limit;

                        if safe_start && safe_end {
                            // Fast path: direct loads
                            for ic in 0..in_channels {
                                let in_ptr_base = in_base.add(ic * in_stride_ch);

                                // Because unaligned loads are cheap on AVX2.
                                // Tap k covers input[t_in_start + k], k in 0..3.
                                let v_left = _mm256_loadu_ps(in_ptr_base.offset(t_in_start));
                                let v_center = _mm256_loadu_ps(in_ptr_base.offset(t_in_start + 1));
                                let v_right = _mm256_loadu_ps(in_ptr_base.offset(t_in_start + 2));

                                // Weights
                                let w0 = w_base0.add(ic * 3);
                                let w1 = w_base1.add(ic * 3);
                                let w2 = w_base2.add(ic * 3);
                                let w3 = w_base3.add(ic * 3);

                                let k0_0 = _mm256_broadcast_ss(&*w0);
                                let k0_1 = _mm256_broadcast_ss(&*w0.add(1));
                                let k0_2 = _mm256_broadcast_ss(&*w0.add(2));

                                acc0 = _mm256_fmadd_ps(v_left, k0_0, acc0);
                                acc0 = _mm256_fmadd_ps(v_center, k0_1, acc0);
                                acc0 = _mm256_fmadd_ps(v_right, k0_2, acc0);

                                let k1_0 = _mm256_broadcast_ss(&*w1);
                                let k1_1 = _mm256_broadcast_ss(&*w1.add(1));
                                let k1_2 = _mm256_broadcast_ss(&*w1.add(2));
                                acc1 = _mm256_fmadd_ps(v_left, k1_0, acc1);
                                acc1 = _mm256_fmadd_ps(v_center, k1_1, acc1);
                                acc1 = _mm256_fmadd_ps(v_right, k1_2, acc1);

                                let k2_0 = _mm256_broadcast_ss(&*w2);
                                let k2_1 = _mm256_broadcast_ss(&*w2.add(1));
                                let k2_2 = _mm256_broadcast_ss(&*w2.add(2));
                                acc2 = _mm256_fmadd_ps(v_left, k2_0, acc2);
                                acc2 = _mm256_fmadd_ps(v_center, k2_1, acc2);
                                acc2 = _mm256_fmadd_ps(v_right, k2_2, acc2);

                                let k3_0 = _mm256_broadcast_ss(&*w3);
                                let k3_1 = _mm256_broadcast_ss(&*w3.add(1));
                                let k3_2 = _mm256_broadcast_ss(&*w3.add(2));
                                acc3 = _mm256_fmadd_ps(v_left, k3_0, acc3);
                                acc3 = _mm256_fmadd_ps(v_center, k3_1, acc3);
                                acc3 = _mm256_fmadd_ps(v_right, k3_2, acc3);
                            }
                        } else {
                            // Slow path (boundary) handling
                            for ic in 0..in_channels {
                                let in_ptr_base = in_base.add(ic * in_stride_ch);
                                // Load using scalar loads into temporary array
                                let mut tmp = [0.0f32; 10]; // t_in_start .. t_in_start+9
                                for k in 0..10 {
                                    let idx = t_in_start + k as isize;
                                    if idx >= 0 && idx < limit {
                                        tmp[k] = *in_ptr_base.add(idx as usize);
                                    }
                                }

                                let v_left = _mm256_loadu_ps(tmp.as_ptr());
                                let v_center = _mm256_loadu_ps(tmp.as_ptr().add(1));
                                let v_right = _mm256_loadu_ps(tmp.as_ptr().add(2));

                                // Weights
                                let w0 = w_base0.add(ic * 3);
                                let w1 = w_base1.add(ic * 3);
                                let w2 = w_base2.add(ic * 3);
                                let w3 = w_base3.add(ic * 3);

                                let k0_0 = _mm256_broadcast_ss(&*w0);
                                let k0_1 = _mm256_broadcast_ss(&*w0.add(1));
                                let k0_2 = _mm256_broadcast_ss(&*w0.add(2));
                                acc0 = _mm256_fmadd_ps(v_left, k0_0, acc0);
                                acc0 = _mm256_fmadd_ps(v_center, k0_1, acc0);
                                acc0 = _mm256_fmadd_ps(v_right, k0_2, acc0);

                                let k1_0 = _mm256_broadcast_ss(&*w1);
                                let k1_1 = _mm256_broadcast_ss(&*w1.add(1));
                                let k1_2 = _mm256_broadcast_ss(&*w1.add(2));
                                acc1 = _mm256_fmadd_ps(v_left, k1_0, acc1);
                                acc1 = _mm256_fmadd_ps(v_center, k1_1, acc1);
                                acc1 = _mm256_fmadd_ps(v_right, k1_2, acc1);

                                let k2_0 = _mm256_broadcast_ss(&*w2);
                                let k2_1 = _mm256_broadcast_ss(&*w2.add(1));
                                let k2_2 = _mm256_broadcast_ss(&*w2.add(2));
                                acc2 = _mm256_fmadd_ps(v_left, k2_0, acc2);
                                acc2 = _mm256_fmadd_ps(v_center, k2_1, acc2);
                                acc2 = _mm256_fmadd_ps(v_right, k2_2, acc2);

                                let k3_0 = _mm256_broadcast_ss(&*w3);
                                let k3_1 = _mm256_broadcast_ss(&*w3.add(1));
                                let k3_2 = _mm256_broadcast_ss(&*w3.add(2));
                                acc3 = _mm256_fmadd_ps(v_left, k3_0, acc3);
                                acc3 = _mm256_fmadd_ps(v_center, k3_1, acc3);
                                acc3 = _mm256_fmadd_ps(v_right, k3_2, acc3);
                            }
                        }

                        // Store results
                        if bias.is_some() {
                            let b_ptr = bias.unwrap().add(oc);
                            let b0 = _mm256_broadcast_ss(&*b_ptr);
                            let b1 = _mm256_broadcast_ss(&*b_ptr.add(1));
                            let b2 = _mm256_broadcast_ss(&*b_ptr.add(2));
                            let b3 = _mm256_broadcast_ss(&*b_ptr.add(3));
                            acc0 = _mm256_add_ps(acc0, b0);
                            acc1 = _mm256_add_ps(acc1, b1);
                            acc2 = _mm256_add_ps(acc2, b2);
                            acc3 = _mm256_add_ps(acc3, b3);
                        }

                        if relu {
                            acc0 = _mm256_max_ps(acc0, zero_v);
                            acc1 = _mm256_max_ps(acc1, zero_v);
                            acc2 = _mm256_max_ps(acc2, zero_v);
                            acc3 = _mm256_max_ps(acc3, zero_v);
                        }

                        _mm256_storeu_ps(out_ptr0.add(t), acc0);
                        _mm256_storeu_ps(out_ptr1.add(t), acc1);
                        _mm256_storeu_ps(out_ptr2.add(t), acc2);
                        _mm256_storeu_ps(out_ptr3.add(t), acc3);

                        t += 8;
                    }
                } // end stride=1 check

                // Cleanup loop (handles stride!=1 OR remaining t)
                while t < output_len {
                    let t_in_start = (t * stride) as isize - (padding as isize);
                    let mut s0 = 0.0;
                    let mut s1 = 0.0;
                    let mut s2 = 0.0;
                    let mut s3 = 0.0;

                    for ic in 0..in_channels {
                        let in_ptr_base = in_base.add(ic * in_stride_ch);
                        let w0 = w_base0.add(ic * 3);
                        let w1 = w_base1.add(ic * 3);
                        let w2 = w_base2.add(ic * 3);
                        let w3 = w_base3.add(ic * 3);

                        // Convolve 3 elements
                        for k in 0..3 {
                            let idx = t_in_start + k;
                            if idx >= 0 && idx < (input_len as isize) {
                                let val = *in_ptr_base.add(idx as usize);
                                s0 += val * *w0.add(k as usize);
                                s1 += val * *w1.add(k as usize);
                                s2 += val * *w2.add(k as usize);
                                s3 += val * *w3.add(k as usize);
                            }
                        }
                    }

                    if let Some(b) = bias {
                        s0 += *b.add(oc);
                        s1 += *b.add(oc + 1);
                        s2 += *b.add(oc + 2);
                        s3 += *b.add(oc + 3);
                    }

                    if relu {
                        s0 = s0.max(0.0);
                        s1 = s1.max(0.0);
                        s2 = s2.max(0.0);
                        s3 = s3.max(0.0);
                    }

                    *out_ptr0.add(t) = s0;
                    *out_ptr1.add(t) = s1;
                    *out_ptr2.add(t) = s2;
                    *out_ptr3.add(t) = s3;

                    t += 1;
                }

                oc += 4;
            }

            // Handle remaining OCs (scalar loop for now, or just unroll generic)
            while oc < out_channels {
                let w_base = weights.add(oc * w_stride_oc);
                let out_ptr = out_base.add(oc * out_stride_ch);

                for t in 0..output_len {
                    let t_in_start = (t * stride) as isize - (padding as isize);
                    let mut s = 0.0;

                    for ic in 0..in_channels {
                        let in_ptr_base = in_base.add(ic * in_stride_ch);
                        let w = w_base.add(ic * 3);
                        for k in 0..3 {
                            let idx = t_in_start + k;
                            if idx >= 0 && idx < (input_len as isize) {
                                s += *in_ptr_base.add(idx as usize) * *w.add(k as usize);
                            }
                        }
                    }
                    if let Some(b) = bias {
                        s += *b.add(oc);
                    }
                    if relu {
                        s = s.max(0.0);
                    }
                    *out_ptr.add(t) = s;
                }
                oc += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn conv1d_dw_x86(
    batch_size: usize,
    channels: usize,
    input_len: usize,
    _out_channels: usize,
    padding: usize,
    stride: usize,
    output_len: usize,
    kernel_size: usize,
    relu: bool,
    bias: Option<*const f32>,
    input: *const f32,
    weights: *const f32,
    output: *mut f32,
) {
    unsafe {
        let zero_v = _mm256_setzero_ps();
        // println!("DEBUG: Calling AVX DW Kernel");
        for b in 0..batch_size {
            let in_base = input.add(b * channels * input_len);
            let out_base = output.add(b * channels * output_len);

            for c in 0..channels {
                let in_ptr = in_base.add(c * input_len);
                let out_ptr = out_base.add(c * output_len);
                let w_ptr = weights.add(c * kernel_size);
                let b_val = if let Some(bias_ptr) = bias {
                    *bias_ptr.add(c)
                } else {
                    0.0
                };
                let b_vec = _mm256_broadcast_ss(&b_val);

                let mut t = 0;
                if stride == 1 {
                    while t + 8 <= output_len {
                        let mut sum = b_vec;
                        let t_in_start = (t as isize) - (padding as isize);

                        for k in 0..kernel_size {
                            let w_val = *w_ptr.add(k);
                            let w_vec = _mm256_broadcast_ss(&w_val);

                            let idx = t_in_start + k as isize;
                            // Vector load input[idx..idx+8]
                            if idx >= 0 && (idx as usize + 7) < input_len {
                                let v_in = _mm256_loadu_ps(in_ptr.offset(idx));
                                sum = _mm256_fmadd_ps(v_in, w_vec, sum);
                            } else {
                                let mut tmp = [0.0f32; 8];
                                for iv in 0..8 {
                                    let i_idx = idx + iv as isize;
                                    if i_idx >= 0 && (i_idx as usize) < input_len {
                                        tmp[iv] = *in_ptr.add(i_idx as usize);
                                    }
                                }
                                let v_in = _mm256_loadu_ps(tmp.as_ptr());
                                sum = _mm256_fmadd_ps(v_in, w_vec, sum);
                            }
                        }

                        if relu {
                            sum = _mm256_max_ps(sum, zero_v);
                        }
                        _mm256_storeu_ps(out_ptr.add(t), sum);
                        t += 8;
                    }
                }

                while t < output_len {
                    let mut s = b_val;
                    let t_in = (t * stride) as isize - (padding as isize);
                    for k in 0..kernel_size {
                        let idx = t_in + k as isize;
                        if idx >= 0 && (idx as usize) < input_len {
                            s += *in_ptr.add(idx as usize) * *w_ptr.add(k);
                        }
                    }
                    if relu && s < 0.0 {
                        s = 0.0;
                    }
                    *out_ptr.add(t) = s;
                    t += 1;
                }
            }
        }
    }
}

/// AVX2-optimized Conv1d for single input channel with any kernel size/stride.
/// Avoids im2col + matmul overhead by computing dot products directly.
/// Processes 4 output channels at a time, using FMA for the kernel dot product.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn conv1d_single_channel_x86(
    batch_size: usize,
    input_len: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    output_len: usize,
    relu: bool,
    bias: Option<*const f32>,
    input: *const f32,
    weights: *const f32, // [out_channels, 1, kernel_size] row-major
    output: *mut f32,
) {
    unsafe {
        for b in 0..batch_size {
            let in_base = input.add(b * input_len);
            let out_base = output.add(b * out_channels * output_len);

            let mut oc = 0;
            // Process 4 output channels at a time
            while oc + 4 <= out_channels {
                let w0 = weights.add(oc * kernel_size);
                let w1 = weights.add((oc + 1) * kernel_size);
                let w2 = weights.add((oc + 2) * kernel_size);
                let w3 = weights.add((oc + 3) * kernel_size);

                for t in 0..output_len {
                    let in_ptr = in_base.add(t * stride);
                    let mut acc0 = _mm256_setzero_ps();
                    let mut acc1 = _mm256_setzero_ps();
                    let mut acc2 = _mm256_setzero_ps();
                    let mut acc3 = _mm256_setzero_ps();

                    let mut k = 0;
                    while k + 8 <= kernel_size {
                        let v_in = _mm256_loadu_ps(in_ptr.add(k));
                        acc0 = _mm256_fmadd_ps(v_in, _mm256_loadu_ps(w0.add(k)), acc0);
                        acc1 = _mm256_fmadd_ps(v_in, _mm256_loadu_ps(w1.add(k)), acc1);
                        acc2 = _mm256_fmadd_ps(v_in, _mm256_loadu_ps(w2.add(k)), acc2);
                        acc3 = _mm256_fmadd_ps(v_in, _mm256_loadu_ps(w3.add(k)), acc3);
                        k += 8;
                    }

                    // Horizontal sums
                    let mut s0 = super::math::hsum_ps(acc0);
                    let mut s1 = super::math::hsum_ps(acc1);
                    let mut s2 = super::math::hsum_ps(acc2);
                    let mut s3 = super::math::hsum_ps(acc3);

                    // Scalar tail for kernel_size not divisible by 8
                    while k < kernel_size {
                        let v = *in_ptr.add(k);
                        s0 += v * *w0.add(k);
                        s1 += v * *w1.add(k);
                        s2 += v * *w2.add(k);
                        s3 += v * *w3.add(k);
                        k += 1;
                    }

                    if let Some(b_ptr) = bias {
                        s0 += *b_ptr.add(oc);
                        s1 += *b_ptr.add(oc + 1);
                        s2 += *b_ptr.add(oc + 2);
                        s3 += *b_ptr.add(oc + 3);
                    }

                    if relu {
                        s0 = s0.max(0.0);
                        s1 = s1.max(0.0);
                        s2 = s2.max(0.0);
                        s3 = s3.max(0.0);
                    }

                    *out_base.add(oc * output_len + t) = s0;
                    *out_base.add((oc + 1) * output_len + t) = s1;
                    *out_base.add((oc + 2) * output_len + t) = s2;
                    *out_base.add((oc + 3) * output_len + t) = s3;
                }
                oc += 4;
            }

            // Handle remaining output channels (< 4)
            while oc < out_channels {
                let w_ptr = weights.add(oc * kernel_size);
                for t in 0..output_len {
                    let in_ptr = in_base.add(t * stride);
                    let mut acc = _mm256_setzero_ps();
                    let mut k = 0;
                    while k + 8 <= kernel_size {
                        let v_in = _mm256_loadu_ps(in_ptr.add(k));
                        acc = _mm256_fmadd_ps(v_in, _mm256_loadu_ps(w_ptr.add(k)), acc);
                        k += 8;
                    }
                    let mut s = super::math::hsum_ps(acc);
                    while k < kernel_size {
                        s += *in_ptr.add(k) * *w_ptr.add(k);
                        k += 1;
                    }
                    if let Some(b_ptr) = bias {
                        s += *b_ptr.add(oc);
                    }
                    if relu {
                        s = s.max(0.0);
                    }
                    *out_base.add(oc * output_len + t) = s;
                }
                oc += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn fuse_bias_relu_x86(
    output: *mut f32,
    bias: Option<*const f32>,
    relu: bool,
    batch_size: usize,
    out_channels: usize,
    output_len: usize,
) {
    unsafe {
        let zero = _mm256_setzero_ps();
        for b in 0..batch_size {
            for oc in 0..out_channels {
                let start = (b * out_channels + oc) * output_len;
                let out_ptr = output.add(start);

                let b_val = if let Some(b_ptr) = bias {
                    *b_ptr.add(oc)
                } else {
                    0.0
                };
                let b_vec = _mm256_set1_ps(b_val);

                let mut i = 0;
                while i + 8 <= output_len {
                    let v_out = _mm256_loadu_ps(out_ptr.add(i));
                    let mut v_res = if bias.is_some() {
                        _mm256_add_ps(v_out, b_vec)
                    } else {
                        v_out
                    };
                    if relu {
                        v_res = _mm256_max_ps(v_res, zero);
                    }
                    _mm256_storeu_ps(out_ptr.add(i), v_res);
                    i += 8;
                }

                while i < output_len {
                    let val = *out_ptr.add(i) + if bias.is_some() { b_val } else { 0.0 };
                    *out_ptr.add(i) = if relu && val < 0.0 { 0.0 } else { val };
                    i += 1;
                }
            }
        }
    }
}
