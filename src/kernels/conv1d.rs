use crate::kernels::utils;
use crate::tensor::TensorView;
use faer::{
    Accum,
    linalg::matmul::matmul,
    mat::{MatMut, MatRef},
};
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// Optimized Tiled Conv1d for K=3, Pad=1.
// Tiling Strategy: Unroll 4 Output Channels, Process 4 or 8 Time steps.
// This balances register pressure.
#[cfg(target_arch = "aarch64")]
unsafe fn conv1d_direct_k3_t4_oc4_neon(
    batch_size: usize,
    in_channels: usize,
    input_len: usize,
    out_channels: usize,
    padding: usize, // usually 1 for "same"
    stride: usize,
    output_len: usize,
    relu: bool,
    bias: Option<*const f32>,
    input: *const f32,
    weights: *const f32,
    output: *mut f32,
) {
    unsafe {
        let w_stride_oc = in_channels * 3;
        let in_stride_ch = input_len;
        let out_stride_ch = output_len;
        let zero_v = vdupq_n_f32(0.0);

        for b in 0..batch_size {
            let in_base = input.add(b * in_channels * input_len);
            let out_base = output.add(b * out_channels * output_len);

            let mut oc = 0;

            // Optimized dispatch for L=4 (Layer 1)
            if input_len == 4 {
                let zero = vdupq_n_f32(0.0);
                while oc + 4 <= out_channels {
                    let mut acc0 = zero;
                    let mut acc1 = zero;
                    let mut acc2 = zero;
                    let mut acc3 = zero;

                    if let Some(b_ptr) = bias {
                        acc0 = vdupq_n_f32(*b_ptr.add(oc));
                        acc1 = vdupq_n_f32(*b_ptr.add(oc + 1));
                        acc2 = vdupq_n_f32(*b_ptr.add(oc + 2));
                        acc3 = vdupq_n_f32(*b_ptr.add(oc + 3));
                    }

                    let w_base0 = weights.add(oc * w_stride_oc);
                    let w_base1 = weights.add((oc + 1) * w_stride_oc);
                    let w_base2 = weights.add((oc + 2) * w_stride_oc);
                    let w_base3 = weights.add((oc + 3) * w_stride_oc);

                    for ic in 0..in_channels {
                        let i_vec = vld1q_f32(in_base.add(ic * 4));

                        let i_m = i_vec;
                        let i_l = vextq_f32(zero, i_m, 3);
                        let i_r = vextq_f32(i_m, zero, 1);

                        let wb0 = w_base0.add(ic * 3);
                        let w0_0 = vld1q_dup_f32(wb0);
                        let w0_1 = vld1q_dup_f32(wb0.add(1));
                        let w0_2 = vld1q_dup_f32(wb0.add(2));
                        acc0 = vfmaq_f32(acc0, i_l, w0_0);
                        acc0 = vfmaq_f32(acc0, i_m, w0_1);
                        acc0 = vfmaq_f32(acc0, i_r, w0_2);

                        let wb1 = w_base1.add(ic * 3);
                        let w1_0 = vld1q_dup_f32(wb1);
                        let w1_1 = vld1q_dup_f32(wb1.add(1));
                        let w1_2 = vld1q_dup_f32(wb1.add(2));
                        acc1 = vfmaq_f32(acc1, i_l, w1_0);
                        acc1 = vfmaq_f32(acc1, i_m, w1_1);
                        acc1 = vfmaq_f32(acc1, i_r, w1_2);

                        let wb2 = w_base2.add(ic * 3);
                        let w2_0 = vld1q_dup_f32(wb2);
                        let w2_1 = vld1q_dup_f32(wb2.add(1));
                        let w2_2 = vld1q_dup_f32(wb2.add(2));
                        acc2 = vfmaq_f32(acc2, i_l, w2_0);
                        acc2 = vfmaq_f32(acc2, i_m, w2_1);
                        acc2 = vfmaq_f32(acc2, i_r, w2_2);

                        let wb3 = w_base3.add(ic * 3);
                        let w3_0 = vld1q_dup_f32(wb3);
                        let w3_1 = vld1q_dup_f32(wb3.add(1));
                        let w3_2 = vld1q_dup_f32(wb3.add(2));
                        acc3 = vfmaq_f32(acc3, i_l, w3_0);
                        acc3 = vfmaq_f32(acc3, i_m, w3_1);
                        acc3 = vfmaq_f32(acc3, i_r, w3_2);
                    }

                    if relu {
                        acc0 = vmaxq_f32(acc0, zero_v);
                        acc1 = vmaxq_f32(acc1, zero_v);
                        acc2 = vmaxq_f32(acc2, zero_v);
                        acc3 = vmaxq_f32(acc3, zero_v);
                    }

                    if stride == 2 {
                        // L=4, Stride=2 -> Output Len 2. Need T0, T2.
                        let r0 = vuzp1q_f32(acc0, acc0);
                        let r1 = vuzp1q_f32(acc1, acc1);
                        let r2 = vuzp1q_f32(acc2, acc2);
                        let r3 = vuzp1q_f32(acc3, acc3);
                        vst1_f32(out_base.add(oc * out_stride_ch), vget_low_f32(r0));
                        vst1_f32(out_base.add((oc + 1) * out_stride_ch), vget_low_f32(r1));
                        vst1_f32(out_base.add((oc + 2) * out_stride_ch), vget_low_f32(r2));
                        vst1_f32(out_base.add((oc + 3) * out_stride_ch), vget_low_f32(r3));
                    } else {
                        vst1q_f32(out_base.add(oc * 4), acc0);
                        vst1q_f32(out_base.add((oc + 1) * 4), acc1);
                        vst1q_f32(out_base.add((oc + 2) * 4), acc2);
                        vst1q_f32(out_base.add((oc + 3) * 4), acc3);
                    }
                    oc += 4;
                }
                continue;
            }

            // Optimized dispatch for L=1
            if input_len == 1 {
                while oc + 4 <= out_channels {
                    let w_base0 = weights.add(oc * w_stride_oc);
                    let w_base1 = weights.add((oc + 1) * w_stride_oc);
                    let w_base2 = weights.add((oc + 2) * w_stride_oc);
                    let w_base3 = weights.add((oc + 3) * w_stride_oc);

                    let mut v_sum0 = vdupq_n_f32(0.0);
                    let mut v_sum1 = vdupq_n_f32(0.0);
                    let mut v_sum2 = vdupq_n_f32(0.0);
                    let mut v_sum3 = vdupq_n_f32(0.0);

                    let mut ic = 0;
                    while ic + 4 <= in_channels {
                        let v_in = vld1q_f32(in_base.add(ic));
                        let w0 = vld3q_f32(w_base0.add(ic * 3)).1;
                        let w1 = vld3q_f32(w_base1.add(ic * 3)).1;
                        let w2 = vld3q_f32(w_base2.add(ic * 3)).1;
                        let w3 = vld3q_f32(w_base3.add(ic * 3)).1;
                        v_sum0 = vfmaq_f32(v_sum0, v_in, w0);
                        v_sum1 = vfmaq_f32(v_sum1, v_in, w1);
                        v_sum2 = vfmaq_f32(v_sum2, v_in, w2);
                        v_sum3 = vfmaq_f32(v_sum3, v_in, w3);
                        ic += 4;
                    }
                    let mut s0 = vaddvq_f32(v_sum0);
                    let mut s1 = vaddvq_f32(v_sum1);
                    let mut s2 = vaddvq_f32(v_sum2);
                    let mut s3 = vaddvq_f32(v_sum3);

                    for k_ic in ic..in_channels {
                        let v = *in_base.add(k_ic);
                        s0 += v * *w_base0.add(k_ic * 3 + 1);
                        s1 += v * *w_base1.add(k_ic * 3 + 1);
                        s2 += v * *w_base2.add(k_ic * 3 + 1);
                        s3 += v * *w_base3.add(k_ic * 3 + 1);
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

                    *out_base.add(oc * out_stride_ch) = s0;
                    *out_base.add((oc + 1) * out_stride_ch) = s1;
                    *out_base.add((oc + 2) * out_stride_ch) = s2;
                    *out_base.add((oc + 3) * out_stride_ch) = s3;
                    oc += 4;
                }
                continue;
            }

            // Optimized dispatch for L=3
            // Optimized dispatch for L=2
            if input_len == 2 {
                let zero = vdupq_n_f32(0.0);
                while oc + 4 <= out_channels {
                    let mut acc0 = zero;
                    let mut acc1 = zero;
                    let mut acc2 = zero;
                    let mut acc3 = zero;

                    if let Some(b_ptr) = bias {
                        acc0 = vdupq_n_f32(*b_ptr.add(oc));
                        acc1 = vdupq_n_f32(*b_ptr.add(oc + 1));
                        acc2 = vdupq_n_f32(*b_ptr.add(oc + 2));
                        acc3 = vdupq_n_f32(*b_ptr.add(oc + 3));
                    }

                    let w_base0 = weights.add(oc * w_stride_oc);
                    let w_base1 = weights.add((oc + 1) * w_stride_oc);
                    let w_base2 = weights.add((oc + 2) * w_stride_oc);
                    let w_base3 = weights.add((oc + 3) * w_stride_oc);

                    for ic in 0..in_channels {
                        let ptr = in_base.add(ic * 2);
                        // Safe load 2 floats
                        let low = vld1_f32(ptr);
                        // Combine into vec4 [i0, i1, 0, 0]
                        let i_vec = vcombine_f32(low, vget_low_f32(zero));

                        let i_m = i_vec;
                        let i_l = vextq_f32(zero, i_m, 3);
                        let i_r = vextq_f32(i_m, zero, 1);

                        let wb0 = w_base0.add(ic * 3);
                        let w0_0 = vld1q_dup_f32(wb0);
                        let w0_1 = vld1q_dup_f32(wb0.add(1));
                        let w0_2 = vld1q_dup_f32(wb0.add(2));
                        acc0 = vfmaq_f32(acc0, i_l, w0_0);
                        acc0 = vfmaq_f32(acc0, i_m, w0_1);
                        acc0 = vfmaq_f32(acc0, i_r, w0_2);

                        let wb1 = w_base1.add(ic * 3);
                        let w1_0 = vld1q_dup_f32(wb1);
                        let w1_1 = vld1q_dup_f32(wb1.add(1));
                        let w1_2 = vld1q_dup_f32(wb1.add(2));
                        acc1 = vfmaq_f32(acc1, i_l, w1_0);
                        acc1 = vfmaq_f32(acc1, i_m, w1_1);
                        acc1 = vfmaq_f32(acc1, i_r, w1_2);

                        let wb2 = w_base2.add(ic * 3);
                        let w2_0 = vld1q_dup_f32(wb2);
                        let w2_1 = vld1q_dup_f32(wb2.add(1));
                        let w2_2 = vld1q_dup_f32(wb2.add(2));
                        acc2 = vfmaq_f32(acc2, i_l, w2_0);
                        acc2 = vfmaq_f32(acc2, i_m, w2_1);
                        acc2 = vfmaq_f32(acc2, i_r, w2_2);

                        let wb3 = w_base3.add(ic * 3);
                        let w3_0 = vld1q_dup_f32(wb3);
                        let w3_1 = vld1q_dup_f32(wb3.add(1));
                        let w3_2 = vld1q_dup_f32(wb3.add(2));
                        acc3 = vfmaq_f32(acc3, i_l, w3_0);
                        acc3 = vfmaq_f32(acc3, i_m, w3_1);
                        acc3 = vfmaq_f32(acc3, i_r, w3_2);
                    }

                    if relu {
                        acc0 = vmaxq_f32(acc0, zero_v);
                        acc1 = vmaxq_f32(acc1, zero_v);
                        acc2 = vmaxq_f32(acc2, zero_v);
                        acc3 = vmaxq_f32(acc3, zero_v);
                    }

                    if stride == 2 {
                        // Output 1. Store SCALAR.
                        *out_base.add(oc * out_stride_ch) = vgetq_lane_f32(acc0, 0);
                        *out_base.add((oc + 1) * out_stride_ch) = vgetq_lane_f32(acc1, 0);
                        *out_base.add((oc + 2) * out_stride_ch) = vgetq_lane_f32(acc2, 0);
                        *out_base.add((oc + 3) * out_stride_ch) = vgetq_lane_f32(acc3, 0);
                    } else {
                        // L=2 S=1 -> Output 2. Store 2.
                        vst1_f32(out_base.add(oc * out_stride_ch), vget_low_f32(acc0));
                        vst1_f32(out_base.add((oc + 1) * out_stride_ch), vget_low_f32(acc1));
                        vst1_f32(out_base.add((oc + 2) * out_stride_ch), vget_low_f32(acc2));
                        vst1_f32(out_base.add((oc + 3) * out_stride_ch), vget_low_f32(acc3));
                    }
                    oc += 4;
                }
                continue;
            }

            // Optimized dispatch for L=3
            if input_len == 3 {
                while oc + 4 <= out_channels {
                    let w_base0 = weights.add(oc * w_stride_oc);
                    let w_base1 = weights.add((oc + 1) * w_stride_oc);
                    let w_base2 = weights.add((oc + 2) * w_stride_oc);
                    let w_base3 = weights.add((oc + 3) * w_stride_oc);

                    let mut s0_0 = vdupq_n_f32(0.0);
                    let mut s0_1 = vdupq_n_f32(0.0);
                    let mut s0_2 = vdupq_n_f32(0.0);
                    let mut s1_0 = vdupq_n_f32(0.0);
                    let mut s1_1 = vdupq_n_f32(0.0);
                    let mut s1_2 = vdupq_n_f32(0.0);
                    let mut s2_0 = vdupq_n_f32(0.0);
                    let mut s2_1 = vdupq_n_f32(0.0);
                    let mut s2_2 = vdupq_n_f32(0.0);
                    let mut s3_0 = vdupq_n_f32(0.0);
                    let mut s3_1 = vdupq_n_f32(0.0);
                    let mut s3_2 = vdupq_n_f32(0.0);

                    let mut ic = 0;
                    while ic + 4 <= in_channels {
                        let i_vecs = vld3q_f32(in_base.add(ic * 3));

                        let w = vld3q_f32(w_base0.add(ic * 3));
                        s0_0 = vfmaq_f32(s0_0, i_vecs.0, w.1);
                        s0_0 = vfmaq_f32(s0_0, i_vecs.1, w.2);
                        s0_1 = vfmaq_f32(s0_1, i_vecs.0, w.0);
                        s0_1 = vfmaq_f32(s0_1, i_vecs.1, w.1);
                        s0_1 = vfmaq_f32(s0_1, i_vecs.2, w.2);
                        s0_2 = vfmaq_f32(s0_2, i_vecs.1, w.0);
                        s0_2 = vfmaq_f32(s0_2, i_vecs.2, w.1);

                        let w = vld3q_f32(w_base1.add(ic * 3));
                        s1_0 = vfmaq_f32(s1_0, i_vecs.0, w.1);
                        s1_0 = vfmaq_f32(s1_0, i_vecs.1, w.2);
                        s1_1 = vfmaq_f32(s1_1, i_vecs.0, w.0);
                        s1_1 = vfmaq_f32(s1_1, i_vecs.1, w.1);
                        s1_1 = vfmaq_f32(s1_1, i_vecs.2, w.2);
                        s1_2 = vfmaq_f32(s1_2, i_vecs.1, w.0);
                        s1_2 = vfmaq_f32(s1_2, i_vecs.2, w.1);

                        let w = vld3q_f32(w_base2.add(ic * 3));
                        s2_0 = vfmaq_f32(s2_0, i_vecs.0, w.1);
                        s2_0 = vfmaq_f32(s2_0, i_vecs.1, w.2);
                        s2_1 = vfmaq_f32(s2_1, i_vecs.0, w.0);
                        s2_1 = vfmaq_f32(s2_1, i_vecs.1, w.1);
                        s2_1 = vfmaq_f32(s2_1, i_vecs.2, w.2);
                        s2_2 = vfmaq_f32(s2_2, i_vecs.1, w.0);
                        s2_2 = vfmaq_f32(s2_2, i_vecs.2, w.1);

                        let w = vld3q_f32(w_base3.add(ic * 3));
                        s3_0 = vfmaq_f32(s3_0, i_vecs.0, w.1);
                        s3_0 = vfmaq_f32(s3_0, i_vecs.1, w.2);
                        s3_1 = vfmaq_f32(s3_1, i_vecs.0, w.0);
                        s3_1 = vfmaq_f32(s3_1, i_vecs.1, w.1);
                        s3_1 = vfmaq_f32(s3_1, i_vecs.2, w.2);
                        s3_2 = vfmaq_f32(s3_2, i_vecs.1, w.0);
                        s3_2 = vfmaq_f32(s3_2, i_vecs.2, w.1);

                        ic += 4;
                    }

                    let mut v0_0 = vaddvq_f32(s0_0);
                    let mut v0_1 = vaddvq_f32(s0_1);
                    let mut v0_2 = vaddvq_f32(s0_2);
                    let mut v1_0 = vaddvq_f32(s1_0);
                    let mut v1_1 = vaddvq_f32(s1_1);
                    let mut v1_2 = vaddvq_f32(s1_2);
                    let mut v2_0 = vaddvq_f32(s2_0);
                    let mut v2_1 = vaddvq_f32(s2_1);
                    let mut v2_2 = vaddvq_f32(s2_2);
                    let mut v3_0 = vaddvq_f32(s3_0);
                    let mut v3_1 = vaddvq_f32(s3_1);
                    let mut v3_2 = vaddvq_f32(s3_2);

                    for k_ic in ic..in_channels {
                        let ptr = in_base.add(k_ic * 3);
                        let i0 = *ptr;
                        let i1 = *ptr.add(1);
                        let i2 = *ptr.add(2);

                        let wp = w_base0.add(k_ic * 3);
                        let w0 = *wp;
                        let w1 = *wp.add(1);
                        let w2 = *wp.add(2);
                        v0_0 += i0 * w1 + i1 * w2;
                        v0_1 += i0 * w0 + i1 * w1 + i2 * w2;
                        v0_2 += i1 * w0 + i2 * w1;

                        let wp = w_base1.add(k_ic * 3);
                        let w0 = *wp;
                        let w1 = *wp.add(1);
                        let w2 = *wp.add(2);
                        v1_0 += i0 * w1 + i1 * w2;
                        v1_1 += i0 * w0 + i1 * w1 + i2 * w2;
                        v1_2 += i1 * w0 + i2 * w1;

                        let wp = w_base2.add(k_ic * 3);
                        let w0 = *wp;
                        let w1 = *wp.add(1);
                        let w2 = *wp.add(2);
                        v2_0 += i0 * w1 + i1 * w2;
                        v2_1 += i0 * w0 + i1 * w1 + i2 * w2;
                        v2_2 += i1 * w0 + i2 * w1;

                        let wp = w_base3.add(k_ic * 3);
                        let w0 = *wp;
                        let w1 = *wp.add(1);
                        let w2 = *wp.add(2);
                        v3_0 += i0 * w1 + i1 * w2;
                        v3_1 += i0 * w0 + i1 * w1 + i2 * w2;
                        v3_2 += i1 * w0 + i2 * w1;
                    }

                    if let Some(b) = bias {
                        let b0 = *b.add(oc);
                        v0_0 += b0;
                        v0_1 += b0;
                        v0_2 += b0;
                        let b1 = *b.add(oc + 1);
                        v1_0 += b1;
                        v1_1 += b1;
                        v1_2 += b1;
                        let b2 = *b.add(oc + 2);
                        v2_0 += b2;
                        v2_1 += b2;
                        v2_2 += b2;
                        let b3 = *b.add(oc + 3);
                        v3_0 += b3;
                        v3_1 += b3;
                        v3_2 += b3;
                    }
                    if relu {
                        v0_0 = v0_0.max(0.0);
                        v0_1 = v0_1.max(0.0);
                        v0_2 = v0_2.max(0.0);
                        v1_0 = v1_0.max(0.0);
                        v1_1 = v1_1.max(0.0);
                        v1_2 = v1_2.max(0.0);
                        v2_0 = v2_0.max(0.0);
                        v2_1 = v2_1.max(0.0);
                        v2_2 = v2_2.max(0.0);
                        v3_0 = v3_0.max(0.0);
                        v3_1 = v3_1.max(0.0);
                        v3_2 = v3_2.max(0.0);
                    }

                    if stride == 2 {
                        *out_base.add(oc * out_stride_ch) = v0_0;
                        *out_base.add(oc * out_stride_ch + 1) = v0_2;

                        *out_base.add((oc + 1) * out_stride_ch) = v1_0;
                        *out_base.add((oc + 1) * out_stride_ch + 1) = v1_2;

                        *out_base.add((oc + 2) * out_stride_ch) = v2_0;
                        *out_base.add((oc + 2) * out_stride_ch + 1) = v2_2;

                        *out_base.add((oc + 3) * out_stride_ch) = v3_0;
                        *out_base.add((oc + 3) * out_stride_ch + 1) = v3_2;
                    } else {
                        *out_base.add(oc * out_stride_ch) = v0_0;
                        *out_base.add(oc * out_stride_ch + 1) = v0_1;
                        *out_base.add(oc * out_stride_ch + 2) = v0_2;

                        *out_base.add((oc + 1) * out_stride_ch) = v1_0;
                        *out_base.add((oc + 1) * out_stride_ch + 1) = v1_1;
                        *out_base.add((oc + 1) * out_stride_ch + 2) = v1_2;

                        *out_base.add((oc + 2) * out_stride_ch) = v2_0;
                        *out_base.add((oc + 2) * out_stride_ch + 1) = v2_1;
                        *out_base.add((oc + 2) * out_stride_ch + 2) = v2_2;

                        *out_base.add((oc + 3) * out_stride_ch) = v3_0;
                        *out_base.add((oc + 3) * out_stride_ch + 1) = v3_1;
                        *out_base.add((oc + 3) * out_stride_ch + 2) = v3_2;
                    }

                    oc += 4;
                }
                continue;
            }

            let mut oc = 0;
            // OC Loop unrolled by 4
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
                // Time Loop unrolled by 4
                while t + 4 <= input_len {
                    let mut acc0 = vdupq_n_f32(0.0);
                    let mut acc1 = vdupq_n_f32(0.0);
                    let mut acc2 = vdupq_n_f32(0.0);
                    let mut acc3 = vdupq_n_f32(0.0);

                    let start_offset = (t as isize) - (padding as isize);
                    let safe_start = start_offset >= 0;
                    let safe_end = (start_offset + 6) <= (input_len as isize);

                    for ic in 0..in_channels {
                        let in_ptr_row = in_base.add(ic * in_stride_ch);

                        // Input Vectors
                        // We need input[t-1..t+3], input[t..t+4], input[t+1..t+5]
                        // Total range: t-1 .. t+5 (6 elements)
                        let v_in_0: float32x4_t;
                        let v_in_1: float32x4_t;
                        let v_in_2: float32x4_t;

                        if safe_start && safe_end {
                            let ptr = in_ptr_row.offset(start_offset);
                            // Efficient loading: contiguous block of 6 floats?
                            // Can simplify to 3 vector loads if aligned, but unaligned:
                            v_in_0 = vld1q_f32(ptr);
                            v_in_1 = vld1q_f32(ptr.add(1));
                            v_in_2 = vld1q_f32(ptr.add(2));
                        } else {
                            // Boundary fall-back inside inner loop? SLOW.
                            // Optimization: Construct inputs array on stack
                            let mut tmp = [0.0f32; 6];
                            for k in 0..6 {
                                let idx = start_offset + k as isize;
                                if idx >= 0 && idx < input_len as isize {
                                    tmp[k] = *in_ptr_row.add(idx as usize);
                                }
                            }
                            v_in_0 = vld1q_f32(tmp.as_ptr());
                            v_in_1 = vld1q_f32(tmp.as_ptr().add(1));
                            v_in_2 = vld1q_f32(tmp.as_ptr().add(2));
                        }

                        // For each OC in the block of 4
                        // OC 0
                        let w_ptr = w_base0.add(ic * 3);
                        let w0 = vdupq_n_f32(*w_ptr);
                        let w1 = vdupq_n_f32(*w_ptr.add(1));
                        let w2 = vdupq_n_f32(*w_ptr.add(2));
                        acc0 = vfmaq_f32(acc0, v_in_0, w0);
                        acc0 = vfmaq_f32(acc0, v_in_1, w1);
                        acc0 = vfmaq_f32(acc0, v_in_2, w2);

                        // OC 1
                        let w_ptr = w_base1.add(ic * 3);
                        let w0 = vdupq_n_f32(*w_ptr);
                        let w1 = vdupq_n_f32(*w_ptr.add(1));
                        let w2 = vdupq_n_f32(*w_ptr.add(2));
                        acc1 = vfmaq_f32(acc1, v_in_0, w0);
                        acc1 = vfmaq_f32(acc1, v_in_1, w1);
                        acc1 = vfmaq_f32(acc1, v_in_2, w2);

                        // OC 2
                        let w_ptr = w_base2.add(ic * 3);
                        let w0 = vdupq_n_f32(*w_ptr);
                        let w1 = vdupq_n_f32(*w_ptr.add(1));
                        let w2 = vdupq_n_f32(*w_ptr.add(2));
                        acc2 = vfmaq_f32(acc2, v_in_0, w0);
                        acc2 = vfmaq_f32(acc2, v_in_1, w1);
                        acc2 = vfmaq_f32(acc2, v_in_2, w2);

                        // OC 3
                        let w_ptr = w_base3.add(ic * 3);
                        let w0 = vdupq_n_f32(*w_ptr);
                        let w1 = vdupq_n_f32(*w_ptr.add(1));
                        let w2 = vdupq_n_f32(*w_ptr.add(2));
                        acc3 = vfmaq_f32(acc3, v_in_0, w0);
                        acc3 = vfmaq_f32(acc3, v_in_1, w1);
                        acc3 = vfmaq_f32(acc3, v_in_2, w2);
                    }

                    vst1q_f32(out_ptr0.add(t), acc0);
                    vst1q_f32(out_ptr1.add(t), acc1);
                    vst1q_f32(out_ptr2.add(t), acc2);
                    vst1q_f32(out_ptr3.add(t), acc3);

                    t += 4;
                }

                // Remainder T Loop (Scalar with Specialized IC Vectorization for L=1,3)
                // Remainder T Loop (Scalar with Specialized IC Vectorization for L=1,3)
                while t < input_len {
                    if input_len == 1 {
                        // Optimized L=1 Fused for 4 OCs (Unrolls OC loop to reuse Input loads)
                        let w_base0 = weights.add(oc * w_stride_oc);
                        let w_base1 = weights.add((oc + 1) * w_stride_oc);
                        let w_base2 = weights.add((oc + 2) * w_stride_oc);
                        let w_base3 = weights.add((oc + 3) * w_stride_oc);

                        let mut v_sum0 = vdupq_n_f32(0.0);
                        let mut v_sum1 = vdupq_n_f32(0.0);
                        let mut v_sum2 = vdupq_n_f32(0.0);
                        let mut v_sum3 = vdupq_n_f32(0.0);

                        let mut ic = 0;
                        while ic + 4 <= in_channels {
                            // Load Input once
                            let v_in = vld1q_f32(in_base.add(ic));

                            // Load 4 Weight vectors
                            let w_mid0 = vld3q_f32(w_base0.add(ic * 3)).1;
                            let w_mid1 = vld3q_f32(w_base1.add(ic * 3)).1;
                            let w_mid2 = vld3q_f32(w_base2.add(ic * 3)).1;
                            let w_mid3 = vld3q_f32(w_base3.add(ic * 3)).1;

                            v_sum0 = vfmaq_f32(v_sum0, v_in, w_mid0);
                            v_sum1 = vfmaq_f32(v_sum1, v_in, w_mid1);
                            v_sum2 = vfmaq_f32(v_sum2, v_in, w_mid2);
                            v_sum3 = vfmaq_f32(v_sum3, v_in, w_mid3);
                            ic += 4;
                        }
                        let mut s0 = vaddvq_f32(v_sum0);
                        let mut s1 = vaddvq_f32(v_sum1);
                        let mut s2 = vaddvq_f32(v_sum2);
                        let mut s3 = vaddvq_f32(v_sum3);

                        for k_ic in ic..in_channels {
                            let i_val = *in_base.add(k_ic);
                            s0 += i_val * *w_base0.add(k_ic * 3 + 1);
                            s1 += i_val * *w_base1.add(k_ic * 3 + 1);
                            s2 += i_val * *w_base2.add(k_ic * 3 + 1);
                            s3 += i_val * *w_base3.add(k_ic * 3 + 1);
                        }
                        *out_base.add(oc * out_stride_ch + t) = s0;
                        *out_base.add((oc + 1) * out_stride_ch + t) = s1;
                        *out_base.add((oc + 2) * out_stride_ch + t) = s2;
                        *out_base.add((oc + 3) * out_stride_ch + t) = s3;
                    } else if input_len == 3 {
                        // Optimized L=3 Fused for 4 OCs
                        let w_base0 = weights.add(oc * w_stride_oc);
                        let w_base1 = weights.add((oc + 1) * w_stride_oc);
                        let w_base2 = weights.add((oc + 2) * w_stride_oc);
                        let w_base3 = weights.add((oc + 3) * w_stride_oc);

                        let mut v_sum0 = vdupq_n_f32(0.0);
                        let mut v_sum1 = vdupq_n_f32(0.0);
                        let mut v_sum2 = vdupq_n_f32(0.0);
                        let mut v_sum3 = vdupq_n_f32(0.0);

                        let mut ic = 0;
                        while ic + 4 <= in_channels {
                            let in_ptr = in_base.add(ic * 3);
                            let i_vecs = vld3q_f32(in_ptr);
                            let i0 = i_vecs.0;
                            let i1 = i_vecs.1;
                            let i2 = i_vecs.2;

                            // OC0
                            let w_vecs = vld3q_f32(w_base0.add(ic * 3));
                            if t == 0 {
                                v_sum0 = vfmaq_f32(v_sum0, i0, w_vecs.1);
                                v_sum0 = vfmaq_f32(v_sum0, i1, w_vecs.2);
                            } else if t == 1 {
                                v_sum0 = vfmaq_f32(v_sum0, i0, w_vecs.0);
                                v_sum0 = vfmaq_f32(v_sum0, i1, w_vecs.1);
                                v_sum0 = vfmaq_f32(v_sum0, i2, w_vecs.2);
                            } else {
                                v_sum0 = vfmaq_f32(v_sum0, i1, w_vecs.0);
                                v_sum0 = vfmaq_f32(v_sum0, i2, w_vecs.1);
                            }

                            // OC1
                            let w_vecs = vld3q_f32(w_base1.add(ic * 3));
                            if t == 0 {
                                v_sum1 = vfmaq_f32(v_sum1, i0, w_vecs.1);
                                v_sum1 = vfmaq_f32(v_sum1, i1, w_vecs.2);
                            } else if t == 1 {
                                v_sum1 = vfmaq_f32(v_sum1, i0, w_vecs.0);
                                v_sum1 = vfmaq_f32(v_sum1, i1, w_vecs.1);
                                v_sum1 = vfmaq_f32(v_sum1, i2, w_vecs.2);
                            } else {
                                v_sum1 = vfmaq_f32(v_sum1, i1, w_vecs.0);
                                v_sum1 = vfmaq_f32(v_sum1, i2, w_vecs.1);
                            }

                            // OC2
                            let w_vecs = vld3q_f32(w_base2.add(ic * 3));
                            if t == 0 {
                                v_sum2 = vfmaq_f32(v_sum2, i0, w_vecs.1);
                                v_sum2 = vfmaq_f32(v_sum2, i1, w_vecs.2);
                            } else if t == 1 {
                                v_sum2 = vfmaq_f32(v_sum2, i0, w_vecs.0);
                                v_sum2 = vfmaq_f32(v_sum2, i1, w_vecs.1);
                                v_sum2 = vfmaq_f32(v_sum2, i2, w_vecs.2);
                            } else {
                                v_sum2 = vfmaq_f32(v_sum2, i1, w_vecs.0);
                                v_sum2 = vfmaq_f32(v_sum2, i2, w_vecs.1);
                            }

                            // OC3
                            let w_vecs = vld3q_f32(w_base3.add(ic * 3));
                            if t == 0 {
                                v_sum3 = vfmaq_f32(v_sum3, i0, w_vecs.1);
                                v_sum3 = vfmaq_f32(v_sum3, i1, w_vecs.2);
                            } else if t == 1 {
                                v_sum3 = vfmaq_f32(v_sum3, i0, w_vecs.0);
                                v_sum3 = vfmaq_f32(v_sum3, i1, w_vecs.1);
                                v_sum3 = vfmaq_f32(v_sum3, i2, w_vecs.2);
                            } else {
                                v_sum3 = vfmaq_f32(v_sum3, i1, w_vecs.0);
                                v_sum3 = vfmaq_f32(v_sum3, i2, w_vecs.1);
                            }

                            ic += 4;
                        }
                        let mut s0 = vaddvq_f32(v_sum0);
                        let mut s1 = vaddvq_f32(v_sum1);
                        let mut s2 = vaddvq_f32(v_sum2);
                        let mut s3 = vaddvq_f32(v_sum3);

                        // Scalar cleanup for L=3
                        for k_ic in ic..in_channels {
                            let i_ptr = in_base.add(k_ic * 3);
                            let i0 = *i_ptr;
                            let i1 = *i_ptr.add(1);
                            let i2 = *i_ptr.add(2);

                            // OC0
                            let w_ptr = w_base0.add(k_ic * 3);
                            let w0 = *w_ptr;
                            let w1 = *w_ptr.add(1);
                            let w2 = *w_ptr.add(2);
                            if t == 0 {
                                s0 += i0 * w1 + i1 * w2;
                            } else if t == 1 {
                                s0 += i0 * w0 + i1 * w1 + i2 * w2;
                            } else {
                                s0 += i1 * w0 + i2 * w1;
                            }

                            // OC1
                            let w_ptr = w_base1.add(k_ic * 3);
                            let w0 = *w_ptr;
                            let w1 = *w_ptr.add(1);
                            let w2 = *w_ptr.add(2);
                            if t == 0 {
                                s1 += i0 * w1 + i1 * w2;
                            } else if t == 1 {
                                s1 += i0 * w0 + i1 * w1 + i2 * w2;
                            } else {
                                s1 += i1 * w0 + i2 * w1;
                            }

                            // OC2
                            let w_ptr = w_base2.add(k_ic * 3);
                            let w0 = *w_ptr;
                            let w1 = *w_ptr.add(1);
                            let w2 = *w_ptr.add(2);
                            if t == 0 {
                                s2 += i0 * w1 + i1 * w2;
                            } else if t == 1 {
                                s2 += i0 * w0 + i1 * w1 + i2 * w2;
                            } else {
                                s2 += i1 * w0 + i2 * w1;
                            }

                            // OC3
                            let w_ptr = w_base3.add(k_ic * 3);
                            let w0 = *w_ptr;
                            let w1 = *w_ptr.add(1);
                            let w2 = *w_ptr.add(2);
                            if t == 0 {
                                s3 += i0 * w1 + i1 * w2;
                            } else if t == 1 {
                                s3 += i0 * w0 + i1 * w1 + i2 * w2;
                            } else {
                                s3 += i1 * w0 + i2 * w1;
                            }
                        }
                        *out_base.add(oc * out_stride_ch + t) = s0;
                        *out_base.add((oc + 1) * out_stride_ch + t) = s1;
                        *out_base.add((oc + 2) * out_stride_ch + t) = s2;
                        *out_base.add((oc + 3) * out_stride_ch + t) = s3;
                    } else {
                        for sub_oc in 0..4 {
                            let real_oc = oc + sub_oc;
                            let w_base = weights.add(real_oc * w_stride_oc);
                            let out_ptr = out_base.add(real_oc * out_stride_ch);

                            let mut sum = 0.0;
                            for ic in 0..in_channels {
                                let w_ptr = w_base.add(ic * 3);
                                let in_ptr_row = in_base.add(ic * in_stride_ch);

                                let idx0 = (t as isize) - (padding as isize);
                                if idx0 >= 0 && idx0 < input_len as isize {
                                    sum += *in_ptr_row.add(idx0 as usize) * *w_ptr;
                                }
                                let idx1 = idx0 + 1;
                                if idx1 >= 0 && idx1 < input_len as i64 as isize {
                                    sum += *in_ptr_row.add(idx1 as usize) * *w_ptr.add(1);
                                }
                                let idx2 = idx0 + 2;
                                if idx2 >= 0 && idx2 < input_len as i64 as isize {
                                    sum += *in_ptr_row.add(idx2 as usize) * *w_ptr.add(2);
                                }
                            }
                            *out_ptr.add(t) = sum;
                        }
                    }
                    t += 1;
                }

                oc += 4;
            }

            // Remainder OC Loop... (If OutChannels not div by 4)
            while oc < out_channels {
                let w_base = weights.add(oc * w_stride_oc);
                let out_ptr = out_base.add(oc * out_stride_ch);

                for t in 0..output_len {
                    let mut sum = 0.0;
                    for ic in 0..in_channels {
                        let w_ptr = w_base.add(ic * 3);
                        let in_ptr_row = in_base.add(ic * in_stride_ch);

                        let idx0 = (t as isize) * (stride as isize) - (padding as isize);
                        if idx0 >= 0 && idx0 < input_len as isize {
                            sum += *in_ptr_row.add(idx0 as usize) * *w_ptr;
                        }
                        let idx1 = idx0 + 1;
                        if idx1 >= 0 && idx1 < input_len as isize {
                            sum += *in_ptr_row.add(idx1 as usize) * *w_ptr.add(1);
                        }
                        let idx2 = idx0 + 2;
                        if idx2 >= 0 && idx2 < input_len as isize {
                            sum += *in_ptr_row.add(idx2 as usize) * *w_ptr.add(2);
                        }
                    }
                    if let Some(b_ptr) = bias {
                        sum += *b_ptr.add(oc);
                    }
                    if relu {
                        sum = sum.max(0.0);
                    }
                    *out_ptr.add(t) = sum;
                }
                oc += 1;
            }
        }
    }
}

pub fn conv1d<'b, 'a>(
    input: &TensorView<'b>,
    weights: &TensorView<'b>,
    bias: Option<&TensorView<'b>>,
    dilations: &[i64],
    group: i64,
    pads: &[i64],
    strides: &[i64],
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    conv1d_fused(
        input, weights, bias, dilations, group, pads, strides, false, out,
    )
}

pub fn conv1d_fused<'b, 'a>(
    input: &TensorView<'b>,
    weights: &TensorView<'b>,
    bias: Option<&TensorView<'b>>,
    dilations: &[i64],
    group: i64,
    pads: &[i64],
    strides: &[i64],
    relu: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let in_shape = &input.shape;
    let w_shape = &weights.shape;
    let rank = in_shape.len();
    let (batch_size, in_channels, input_len) = if rank == 3 {
        (in_shape[0], in_shape[1], in_shape[2])
    } else if rank == 2 {
        (in_shape[0], 1, in_shape[1])
    } else {
        panic!("Conv1d: Unsupported input rank {}", rank);
    };
    let out_channels = w_shape[0];
    let kernel_size = w_shape[2];
    let dilation = if dilations.is_empty() {
        1
    } else {
        dilations[0] as usize
    };
    let stride = if strides.is_empty() {
        1
    } else {
        strides[0] as usize
    };
    let pad_left = if pads.is_empty() { 0 } else { pads[0] as usize };
    let pad_right = if pads.len() > 1 { pads[1] as usize } else { 0 };
    let output_len =
        (input_len + pad_left + pad_right - dilation * (kernel_size - 1) - 1) / stride + 1;
    let total_output_size = batch_size * out_channels * output_len;
    utils::ensure_capacity(out, total_output_size);
    unsafe {
        out.set_len(total_output_size);
    }
    let in_channels_per_group = in_channels / group as usize;
    let out_channels_per_group = out_channels / group as usize;
    let unfolded_rows = in_channels_per_group * kernel_size;

    // Optimization for Single-Channel Convolutions (e.g. STFT: 1->1, K=256)
    // Avoids im2col for large kernels by doing direct dot products.
    #[cfg(target_arch = "aarch64")]
    if in_channels == 1
        && out_channels == 1
        && group == 1
        && pad_left == 0
        && pad_right == 0
        && dilation == 1
    {
        unsafe {
            let in_ptr_base = input.data.as_ptr();
            let w_ptr = weights.data.as_ptr();
            let out_ptr_base = out.as_mut_ptr();

            for b in 0..batch_size {
                let in_ptr = in_ptr_base.add(b * input_len);
                let out_ptr = out_ptr_base.add(b * output_len);

                // For each output time step
                for t in 0..output_len {
                    let t_in = t * stride;
                    // Dot product of inputs[t_in..t_in+K] and weights[0..K]
                    let mut sum = 0.0;
                    let mut k = 0;
                    let mut v_sum = vdupq_n_f32(0.0);

                    // Vectorized Dot Product
                    while k + 4 <= kernel_size {
                        let v_i = vld1q_f32(in_ptr.add(t_in + k));
                        let v_w = vld1q_f32(w_ptr.add(k));
                        v_sum = vfmaq_f32(v_sum, v_i, v_w);
                        k += 4;
                    }
                    sum += vaddvq_f32(v_sum);

                    // Tail
                    while k < kernel_size {
                        sum += *in_ptr.add(t_in + k) * *w_ptr.add(k);
                        k += 1;
                    }

                    // Bias
                    if let Some(b_vec) = bias {
                        sum += b_vec.data[0];
                    }
                    // No ReLU support here yet (STFT doesn't use it)
                    *out_ptr.add(t) = sum;
                }
            }
        }
        return TensorView::from_slice(out, vec![batch_size, out_channels, output_len]);
    }

    #[cfg(target_arch = "aarch64")]
    if group == 1
        && kernel_size == 3
        && dilation == 1
        && (stride == 1 || stride == 2)
        && pad_left == 1
        && pad_right == 1
    {
        let bias_ptr = bias.map(|b| b.data.as_ptr());
        unsafe {
            conv1d_direct_k3_t4_oc4_neon(
                batch_size,
                in_channels,
                input_len,
                out_channels,
                1,
                stride,
                output_len,
                relu,
                bias_ptr,
                input.data.as_ptr(),
                weights.data.as_ptr(),
                out.as_mut_ptr(),
            );
        }

        return TensorView::from_slice(out, vec![batch_size, out_channels, output_len]);
    }

    #[cfg(target_arch = "x86_64")]
    if group as usize == in_channels
        && group as usize == out_channels
        && (stride == 1 || stride == 2)
    {
        // Depthwise Convolution
        let bias_ptr = bias.map(|b| b.data.as_ptr());
        unsafe {
            crate::kernels::avx::conv1d::conv1d_dw_x86(
                batch_size,
                in_channels,
                input_len,
                out_channels,
                pad_left, // Assuming symmetric padding or handling verify?
                stride,
                output_len,
                kernel_size,
                relu,
                bias_ptr,
                input.data.as_ptr(),
                weights.data.as_ptr(),
                out.as_mut_ptr(),
            );
        }
        return TensorView::from_slice(out, vec![batch_size, out_channels, output_len]);
    }

    #[cfg(target_arch = "x86_64")]
    if group == 1
        && kernel_size == 3
        && dilation == 1
        && (stride == 1 || stride == 2)
        && pad_left == 1
        && pad_right == 1
    {
        let bias_ptr = bias.map(|b| b.data.as_ptr());
        unsafe {
            crate::kernels::avx::conv1d::conv1d_direct_k3_x86(
                batch_size,
                in_channels,
                input_len,
                out_channels,
                1,
                stride,
                output_len,
                relu,
                bias_ptr,
                input.data.as_ptr(),
                weights.data.as_ptr(),
                out.as_mut_ptr(),
            );
        }

        return TensorView::from_slice(out, vec![batch_size, out_channels, output_len]);
    }

    let unfolded_size = unfolded_rows * output_len;

    thread_local! {
        static SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    SCRATCH.with(|scratch_cell| {
        let mut scratch = scratch_cell.borrow_mut();
        if scratch.len() < unfolded_size {
            scratch.resize(unfolded_size, 0.0);
        }
        let unfolded = &mut scratch[..unfolded_size];

        let is_fast_path =
            stride == 1 && dilation == 1 && pad_left == 0 && pad_right == 0 && kernel_size == 1;

        for b in 0..batch_size {
            for g in 0..group as usize {
                let in_group_offset = (b * in_channels + g * in_channels_per_group) * input_len;

                if !is_fast_path {
                    // Standard im2col path with optimizations
                    // Only zero out what we need
                    if pad_left > 0 || pad_right > 0 || dilation > 1 {
                        unfolded.fill(0.0);
                    }

                    for ic in 0..in_channels_per_group {
                        let in_row_offset = in_group_offset + ic * input_len;
                        let in_data = &input.data[in_row_offset..in_row_offset + input_len];

                        for k in 0..kernel_size {
                            let k_offset = k * dilation;
                            let unfolded_row_idx = ic * kernel_size + k;
                            let unfolded_row_offset = unfolded_row_idx * output_len;

                            // Optimize: calculate valid range to avoid per-element bounds checking
                            let first_valid_out = if pad_left > k_offset {
                                (pad_left - k_offset).div_ceil(stride).max(0)
                            } else {
                                0
                            };
                            let last_valid_out = (input_len + pad_left - k_offset)
                                .div_ceil(stride)
                                .min(output_len);

                            if first_valid_out < last_valid_out {
                                let unf_ptr = unfolded.as_mut_ptr();
                                let in_ptr = in_data.as_ptr();

                                if stride == 1 {
                                    // Optimized stride=1 with padding handling
                                    let t_in_start = -(pad_left as isize) + k_offset as isize;
                                    let dst_start = unfolded_row_offset;
                                    
                                    // prefix zeros
                                    let prefix_len = if t_in_start < 0 {
                                        (-t_in_start as usize).min(output_len)
                                    } else {
                                        0
                                    };
                                    
                                    // data copy
                                    let src_start = if t_in_start > 0 { t_in_start as usize } else { 0 };
                                    let data_dst_start = dst_start + prefix_len;
                                    let available_data = if input_len > src_start { input_len - src_start } else { 0 };
                                    let copy_len = available_data.min(output_len - prefix_len);
                                    
                                    if copy_len > 0 {
                                        unsafe {
                                            std::ptr::copy_nonoverlapping(
                                                in_ptr.add(src_start),
                                                unf_ptr.add(data_dst_start),
                                                copy_len,
                                            );
                                        }
                                    }
                                } else {
                                    // Strided copy
                                    for t_out in first_valid_out..last_valid_out {
                                        let t_in = (t_out * stride) as isize - pad_left as isize
                                            + k_offset as isize;
                                        if t_in >= 0 && (t_in as usize) < input_len {
                                            unsafe {
                                                *unf_ptr.add(unfolded_row_offset + t_out) =
                                                    *in_ptr.add(t_in as usize);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                let unf_ptr = if is_fast_path {
                    unsafe { input.data.as_ptr().add(in_group_offset) }
                } else {
                    unfolded.as_ptr()
                };

                let weight_group_offset =
                    (g * out_channels_per_group) * in_channels_per_group * kernel_size;
                let out_group_offset = (b * out_channels + g * out_channels_per_group) * output_len;
                unsafe {
                    let a = MatRef::<f32>::from_raw_parts(
                        weights.data.as_ptr().add(weight_group_offset),
                        out_channels_per_group,
                        unfolded_rows,
                        unfolded_rows as isize,
                        1,
                    );
                    let b = MatRef::<f32>::from_raw_parts(
                        unf_ptr,
                        unfolded_rows,
                        output_len,
                        output_len as isize,
                        1,
                    );
                    let c = MatMut::<f32>::from_raw_parts_mut(
                        out.as_mut_ptr().add(out_group_offset),
                        out_channels_per_group,
                        output_len,
                        output_len as isize,
                        1,
                    );
                    matmul(c, Accum::Replace, a, b, 1.0f32, utils::get_parallelism(out_channels_per_group, output_len, unfolded_rows));
                }
            }
        }

        // Post-processing: Bias + ReLU
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let out_ptr = out.as_mut_ptr();
            let total_len = batch_size * out_channels * output_len;

            if let Some(b_vec) = bias {
                let b_data = b_vec.data.as_ptr();
                if relu {
                    let zero = vdupq_n_f32(0.0);
                    for b in 0..batch_size {
                        for oc in 0..out_channels {
                            let start = (b * out_channels + oc) * output_len;
                            let b_val = *b_data.add(oc);
                            let v_bias = vdupq_n_f32(b_val);
                            let mut i = 0;
                            while i + 4 <= output_len {
                                let v_out = vld1q_f32(out_ptr.add(start + i));
                                let v_res = vmaxq_f32(vaddq_f32(v_out, v_bias), zero);
                                vst1q_f32(out_ptr.add(start + i), v_res);
                                i += 4;
                            }
                            // Tail
                            while i < output_len {
                                let val = *out_ptr.add(start + i) + b_val;
                                *out_ptr.add(start + i) = if val > 0.0 { val } else { 0.0 };
                                i += 1;
                            }
                        }
                    }
                } else {
                    for b in 0..batch_size {
                        for oc in 0..out_channels {
                            let start = (b * out_channels + oc) * output_len;
                            let b_val = *b_data.add(oc);
                            let v_bias = vdupq_n_f32(b_val);
                            let mut i = 0;
                            while i + 4 <= output_len {
                                let v_out = vld1q_f32(out_ptr.add(start + i));
                                let v_res = vaddq_f32(v_out, v_bias);
                                vst1q_f32(out_ptr.add(start + i), v_res);
                                i += 4;
                            }
                            while i < output_len {
                                *out_ptr.add(start + i) += b_val;
                                i += 1;
                            }
                        }
                    }
                }
            } else if relu {
                let zero = vdupq_n_f32(0.0);
                let mut i = 0;
                while i + 4 <= total_len {
                    let v_out = vld1q_f32(out_ptr.add(i));
                    let v_res = vmaxq_f32(v_out, zero);
                    vst1q_f32(out_ptr.add(i), v_res);
                    i += 4;
                }
                while i < total_len {
                    let ptr = out_ptr.add(i);
                    if *ptr < 0.0 {
                        *ptr = 0.0;
                    }
                    i += 1;
                }
            }
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            crate::kernels::avx::conv1d::fuse_bias_relu_x86(
                out.as_mut_ptr(),
                bias.map(|b| b.data.as_ptr()),
                relu,
                batch_size,
                out_channels,
                output_len,
            );
        }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        unsafe {
            let out_ptr = out.as_mut_ptr();
            if let Some(b_vec) = bias {
                if relu {
                    // Bias + ReLU
                    for b in 0..batch_size {
                        for oc in 0..out_channels {
                            let start = (b * out_channels + oc) * output_len;
                            let b_val = b_vec.data[oc];
                            for i in 0..output_len {
                                let ptr = out_ptr.add(start + i);
                                let val = *ptr + b_val;
                                *ptr = if val > 0.0 { val } else { 0.0 };
                            }
                        }
                    }
                } else {
                    // Bias only
                    for b in 0..batch_size {
                        for oc in 0..out_channels {
                            let start = (b * out_channels + oc) * output_len;
                            let b_val = b_vec.data[oc];
                            for i in 0..output_len {
                                *out_ptr.add(start + i) += b_val;
                            }
                        }
                    }
                }
            } else if relu {
                // ReLU only
                for i in 0..(batch_size * out_channels * output_len) {
                    let ptr = out_ptr.add(i);
                    let val = *ptr;
                    if val < 0.0 {
                        *ptr = 0.0;
                    }
                }
            }
        }

        TensorView::from_slice(out, vec![batch_size, out_channels, output_len])
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorView;
    #[test]
    fn test_conv1d_grouped() {
        let input_data = vec![1.0; 6];
        let input = TensorView::from_slice(&input_data, vec![1, 2, 3]);
        let weight_data = vec![1.0; 2];
        let weights = TensorView::from_slice(&weight_data, vec![2, 1, 1]);
        let mut out = Vec::new();
        let res = conv1d(&input, &weights, None, &[1], 2, &[0, 0], &[1], &mut out);
        assert_eq!(res.shape, vec![1, 2, 3]);
        assert_eq!(res.data, vec![1.0; 6]);
    }
    #[test]
    fn test_conv1d_simple() {
        let input_data = vec![1.0, 2.0, 3.0];
        let input = TensorView::from_slice(&input_data, vec![1, 1, 3]);
        let weight_data = vec![1.0, 1.0];
        let weights = TensorView::from_slice(&weight_data, vec![1, 1, 2]);
        let mut out = Vec::new();
        let res = conv1d(&input, &weights, None, &[1], 1, &[0, 0], &[1], &mut out);
        assert_eq!(res.shape, vec![1, 1, 2]);
        assert_eq!(res.data, vec![3.0, 5.0]);
    }

    #[test]
    fn test_conv1d_k3_opt() {
        // Input: 1 batch, 1 input channel, L=10
        let input_len = 10;
        let input_data: Vec<f32> = (0..input_len).map(|x| x as f32).collect();
        let input = TensorView::from_slice(&input_data, vec![1, 1, input_len]);

        // Weights: 1 output channel, 1 input channel, K=3
        // Filter [1, 1, 1] acts as sum of 3 window.
        let weight_data = vec![1.0, 1.0, 1.0];
        let weights = TensorView::from_slice(&weight_data, vec![1, 1, 3]);

        let mut out = Vec::new();
        // Pad=1, Stride=1
        // Dilation defaults to 1 passed as array? No, call needs `dilations` slice.
        // `conv1d` arg signature: ..., dilation: &[i64], group: i64, padding: &[i64], stride: &[i64], ...

        let res = conv1d(&input, &weights, None, &[1], 1, &[1, 1], &[1], &mut out);

        assert_eq!(res.shape, vec![1, 1, 10]);
        // T=0:  0(pad), 0, 1 -> 1
        // T=1:  0, 1, 2      -> 3
        // T=i: (i-1)+i+(i+1) = 3i
        // T=9: 8, 9, 0(pad)  -> 17

        let out_data = res.data;
        assert_eq!(out_data[0], 1.0);
        assert_eq!(out_data[1], 3.0);
        assert_eq!(out_data[5], 15.0); // 3*5
        assert_eq!(out_data[9], 17.0);
    }
}
