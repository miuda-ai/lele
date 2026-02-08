#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use rayon::prelude::*;
use crate::kernels::utils::{SendPtr, SendConstPtr};

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

    let s_input = SendConstPtr::new(input);
    let s_weights = SendConstPtr::new(weights);
    let s_output = SendPtr::new(output);
    let s_bias = bias.map(|b| SendConstPtr::new(b));

    let num_oc_blocks = (out_channels + 3) / 4;
    
    (0..batch_size * num_oc_blocks).into_par_iter().for_each(move |idx| {
        let b = idx / num_oc_blocks;
        let block_idx = idx % num_oc_blocks;
        let oc_start = block_idx * 4;
        let oc_end = std::cmp::min(oc_start + 4, out_channels);

        unsafe {
            let zero_v = _mm256_setzero_ps();
            let in_base = s_input.as_ptr().add(b * in_channels * input_len);
            let out_base = s_output.as_ptr().add(b * out_channels * output_len);

            let mut oc = oc_start;
            if oc + 4 <= oc_end {
                let w_base0 = s_weights.as_ptr().add(oc * w_stride_oc);
                let w_base1 = s_weights.as_ptr().add((oc + 1) * w_stride_oc);
                let w_base2 = s_weights.as_ptr().add((oc + 2) * w_stride_oc);
                let w_base3 = s_weights.as_ptr().add((oc + 3) * w_stride_oc);

                let out_ptr0 = out_base.add(oc * out_stride_ch);
                let out_ptr1 = out_base.add((oc + 1) * out_stride_ch);
                let out_ptr2 = out_base.add((oc + 2) * out_stride_ch);
                let out_ptr3 = out_base.add((oc + 3) * out_stride_ch);

                let mut t = 0;
                if stride == 1 {
                    while t + 8 <= output_len {
                        let mut acc0 = zero_v;
                        let mut acc1 = zero_v;
                        let mut acc2 = zero_v;
                        let mut acc3 = zero_v;

                        let t_in_start = (t as isize) - (padding as isize);
                        let safe_start = t_in_start >= 0;
                        let safe_end = (t_in_start + 9) < (input_len as isize); 

                        if safe_start && safe_end {
                            for ic in 0..in_channels {
                                let in_ptr_base = in_base.add(ic * in_stride_ch);
                                let v_left = _mm256_loadu_ps(in_ptr_base.offset(t_in_start - 1));
                                let v_center = _mm256_loadu_ps(in_ptr_base.offset(t_in_start));
                                let v_right = _mm256_loadu_ps(in_ptr_base.offset(t_in_start + 1));
                                
                                let wb0 = w_base0.add(ic * 3);
                                let wb1 = w_base1.add(ic * 3);
                                let wb2 = w_base2.add(ic * 3);
                                let wb3 = w_base3.add(ic * 3);
                                
                                acc0 = _mm256_fmadd_ps(v_left, _mm256_broadcast_ss(&*wb0), acc0);
                                acc0 = _mm256_fmadd_ps(v_center, _mm256_broadcast_ss(&*wb0.add(1)), acc0);
                                acc0 = _mm256_fmadd_ps(v_right, _mm256_broadcast_ss(&*wb0.add(2)), acc0);

                                acc1 = _mm256_fmadd_ps(v_left, _mm256_broadcast_ss(&*wb1), acc1);
                                acc1 = _mm256_fmadd_ps(v_center, _mm256_broadcast_ss(&*wb1.add(1)), acc1);
                                acc1 = _mm256_fmadd_ps(v_right, _mm256_broadcast_ss(&*wb1.add(2)), acc1);

                                acc2 = _mm256_fmadd_ps(v_left, _mm256_broadcast_ss(&*wb2), acc2);
                                acc2 = _mm256_fmadd_ps(v_center, _mm256_broadcast_ss(&*wb2.add(1)), acc2);
                                acc2 = _mm256_fmadd_ps(v_right, _mm256_broadcast_ss(&*wb2.add(2)), acc2);

                                acc3 = _mm256_fmadd_ps(v_left, _mm256_broadcast_ss(&*wb3), acc3);
                                acc3 = _mm256_fmadd_ps(v_center, _mm256_broadcast_ss(&*wb3.add(1)), acc3);
                                acc3 = _mm256_fmadd_ps(v_right, _mm256_broadcast_ss(&*wb3.add(2)), acc3);
                            }
                        } else {
                             for ic in 0..in_channels {
                                let in_ptr_base = in_base.add(ic * in_stride_ch);
                                let mut tmp = [0.0f32; 10];
                                for k in 0..10 {
                                    let idx = t_in_start - 1 + k as isize;
                                    if idx >= 0 && idx < input_len as isize {
                                        tmp[k] = *in_ptr_base.add(idx as usize);
                                    }
                                }
                                let v_left = _mm256_loadu_ps(tmp.as_ptr());
                                let v_center = _mm256_loadu_ps(tmp.as_ptr().add(1));
                                let v_right = _mm256_loadu_ps(tmp.as_ptr().add(2));
                                
                                let wb0 = w_base0.add(ic * 3);
                                let wb1 = w_base1.add(ic * 3);
                                let wb2 = w_base2.add(ic * 3);
                                let wb3 = w_base3.add(ic * 3);
                                
                                acc0 = _mm256_fmadd_ps(v_left, _mm256_broadcast_ss(&*wb0), acc0);
                                acc0 = _mm256_fmadd_ps(v_center, _mm256_broadcast_ss(&*wb0.add(1)), acc0);
                                acc0 = _mm256_fmadd_ps(v_right, _mm256_broadcast_ss(&*wb0.add(2)), acc0);

                                acc1 = _mm256_fmadd_ps(v_left, _mm256_broadcast_ss(&*wb1), acc1);
                                acc1 = _mm256_fmadd_ps(v_center, _mm256_broadcast_ss(&*wb1.add(1)), acc1);
                                acc1 = _mm256_fmadd_ps(v_right, _mm256_broadcast_ss(&*wb1.add(2)), acc1);

                                acc2 = _mm256_fmadd_ps(v_left, _mm256_broadcast_ss(&*wb2), acc2);
                                acc2 = _mm256_fmadd_ps(v_center, _mm256_broadcast_ss(&*wb2.add(1)), acc2);
                                acc2 = _mm256_fmadd_ps(v_right, _mm256_broadcast_ss(&*wb2.add(2)), acc2);

                                acc3 = _mm256_fmadd_ps(v_left, _mm256_broadcast_ss(&*wb3), acc3);
                                acc3 = _mm256_fmadd_ps(v_center, _mm256_broadcast_ss(&*wb3.add(1)), acc3);
                                acc3 = _mm256_fmadd_ps(v_right, _mm256_broadcast_ss(&*wb3.add(2)), acc3);
                            }
                        }

                        if let Some(b_ptr) = &s_bias {
                            let bp = b_ptr.as_ptr().add(oc);
                            acc0 = _mm256_add_ps(acc0, _mm256_broadcast_ss(&*bp));
                            acc1 = _mm256_add_ps(acc1, _mm256_broadcast_ss(&*bp.add(1)));
                            acc2 = _mm256_add_ps(acc2, _mm256_broadcast_ss(&*bp.add(2)));
                            acc3 = _mm256_add_ps(acc3, _mm256_broadcast_ss(&*bp.add(3)));
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
                }
                
                while t < output_len {
                     let t_in_start = (t * stride) as isize - (padding as isize);
                     let mut s0 = 0.0;
                     let mut s1 = 0.0;
                     let mut s2 = 0.0;
                     let mut s3 = 0.0;
                     
                     for ic in 0..in_channels {
                         let in_ptr_base = in_base.add(ic * in_stride_ch);
                         let wb0 = w_base0.add(ic * 3);
                         let wb1 = w_base1.add(ic * 3);
                         let wb2 = w_base2.add(ic * 3);
                         let wb3 = w_base3.add(ic * 3);
                         
                         for k in 0..3 {
                             let idx = t_in_start + k;
                             if idx >= 0 && idx < (input_len as isize) {
                                 let val = *in_ptr_base.add(idx as usize);
                                 s0 += val * *wb0.add(k as usize);
                                 s1 += val * *wb1.add(k as usize);
                                 s2 += val * *wb2.add(k as usize);
                                 s3 += val * *wb3.add(k as usize);
                             }
                         }
                     }
                     
                     if let Some(b) = &s_bias {
                         let bp = b.as_ptr();
                         s0 += *bp.add(oc);
                         s1 += *bp.add(oc + 1);
                         s2 += *bp.add(oc + 2);
                         s3 += *bp.add(oc + 3);
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

            while oc < oc_end {
                let wb = s_weights.as_ptr().add(oc * w_stride_oc);
                let out_ptr = out_base.add(oc * out_stride_ch);
                
                 for t in 0..output_len {
                     let t_in_start = (t * stride) as isize - (padding as isize);
                     let mut s = 0.0;
                     
                     for ic in 0..in_channels {
                          let in_ptr_base = in_base.add(ic * in_stride_ch);
                          let wb_ic = wb.add(ic * 3);
                           for k in 0..3 {
                             let idx = t_in_start + k;
                             if idx >= 0 && idx < (input_len as isize) {
                                 s += *in_ptr_base.add(idx as usize) * *wb_ic.add(k as usize);
                             }
                           }
                     }
                     if let Some(b) = &s_bias {
                         s += *b.as_ptr().add(oc);
                     }
                     if relu {
                         s = s.max(0.0);
                     }
                     *out_ptr.add(t) = s;
                 }
                 oc += 1;
            }
        }
    });
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
    let s_input = SendConstPtr::new(input);
    let s_weights = SendConstPtr::new(weights);
    let s_output = SendPtr::new(output);
    let s_bias = bias.map(|b| SendConstPtr::new(b));
    
    (0..batch_size * channels).into_par_iter().for_each(move |idx| {
        let b = idx / channels;
        let c = idx % channels;
        
        unsafe {
            let zero_v = _mm256_setzero_ps();
            let in_base = s_input.as_ptr().add(b * channels * input_len);
            let out_base = s_output.as_ptr().add(b * channels * output_len);
            
            let in_ptr = in_base.add(c * input_len);
            let out_ptr = out_base.add(c * output_len);
            let w_ptr = s_weights.as_ptr().add(c * kernel_size);
            let b_val = if let Some(bias_ptr) = &s_bias {
                *bias_ptr.as_ptr().add(c)
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
                if relu && s < 0.0 { s = 0.0; }
                *out_ptr.add(t) = s;
                t += 1;
            }
        }
    });
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
    let s_output = SendPtr::new(output);
    let s_bias = bias.map(|b| SendConstPtr::new(b));
    
    (0..batch_size * out_channels).into_par_iter().for_each(move |idx| {
        let b = idx / out_channels;
        let oc = idx % out_channels;
        
        unsafe {
            let zero = _mm256_setzero_ps();
            let start = (b * out_channels + oc) * output_len;
            let out_ptr = s_output.as_ptr().add(start);
            
            let b_val = if let Some(b_ptr) = &s_bias {
                *b_ptr.as_ptr().add(oc)
            } else {
                0.0
            };
            let b_vec = _mm256_set1_ps(b_val);
            
            let mut i = 0;
            while i + 8 <= output_len {
                 let v_out = _mm256_loadu_ps(out_ptr.add(i));
                 let mut v_res = if s_bias.is_some() { _mm256_add_ps(v_out, b_vec) } else { v_out };
                 if relu {
                     v_res = _mm256_max_ps(v_res, zero);
                 }
                 _mm256_storeu_ps(out_ptr.add(i), v_res);
                 i += 8;
            }
            
            while i < output_len {
                let val = *out_ptr.add(i) + if s_bias.is_some() { b_val } else { 0.0 };
                *out_ptr.add(i) = if relu && val < 0.0 { 0.0 } else { val };
                i += 1;
            }
        }
    });
}