use crate::kernels::bias_act::bias_act_inplace;
use crate::kernels::conv1d_direct::{
    Depthwise, Kernel3, SingleChannel, depthwise, kernel3, single_channel,
};
use crate::kernels::conv2d::Activation;
use crate::kernels::utils;
use fearless_simd::Level;
#[cfg(target_arch = "wasm32")]
use crate::kernels::wasm_matmul::{Accum, MatMut, MatRef, Par, matmul};
use crate::tensor::TensorView;
#[cfg(not(target_arch = "wasm32"))]
use faer::{
    Accum, Par,
    linalg::matmul::matmul,
    mat::{MatMut, MatRef},
};

/// Adds `bias[oc]` to every `[oc, ..]` row of `out` (laid out `[batch, out_channels, len]`)
/// and applies ReLU if asked.
fn bias_relu_rows(
    out: &mut [f32],
    bias: Option<&TensorView>,
    relu: bool,
    out_channels: usize,
    len: usize,
) {
    let act = if relu { Activation::Relu } else { Activation::None };
    match bias {
        Some(bias) => {
            for (i, row) in out.chunks_exact_mut(len).enumerate() {
                bias_act_inplace(row, bias.data[i % out_channels], act);
            }
        }
        None if relu => bias_act_inplace(out, 0.0, act),
        None => {}
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
    let result = conv1d_fused(
        input, weights, bias, dilations, group, pads, strides, false, out,
    );
    result
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
    conv1d_fused_at(
        Level::new(),
        input,
        weights,
        bias,
        dilations,
        group,
        pads,
        strides,
        relu,
        out,
    )
}

fn conv1d_fused_at<'b, 'a>(
    level: Level,
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

    // Optimization for Single-Channel Input Convolutions (e.g. STFT: 1->258, K=256)
    // Avoids im2col for large kernels by doing direct dot products.
    if in_channels == 1 && group == 1 && pad_left == 0 && pad_right == 0 && dilation == 1 {
        let shape = SingleChannel {
            batch: batch_size,
            input_len,
            out_channels,
            kernel: kernel_size,
            stride,
            out_len: output_len,
            bias: bias.map(|b| &b.data[..out_channels]),
            relu,
        };
        single_channel(
            level,
            &shape,
            &input.data[..batch_size * input_len],
            &weights.data[..out_channels * kernel_size],
            &mut out[..total_output_size],
        );
        return TensorView::from_slice(out, vec![batch_size, out_channels, output_len]);
    }

    if group as usize == in_channels && group as usize == out_channels && dilation == 1 {
        let shape = Depthwise {
            batch: batch_size,
            channels: in_channels,
            len: input_len,
            kernel: kernel_size,
            stride,
            pad_left,
            out_len: output_len,
            bias: bias.map(|b| &b.data[..out_channels]),
            relu,
        };
        depthwise(
            level,
            &shape,
            &input.data[..batch_size * in_channels * input_len],
            &weights.data[..out_channels * kernel_size],
            &mut out[..total_output_size],
        );
        return TensorView::from_slice(out, vec![batch_size, out_channels, output_len]);
    }

    if group == 1
        && kernel_size == 3
        && dilation == 1
        && stride == 1
        && pad_left == 1
        && pad_right == 1
    {
        let shape = Kernel3 {
            batch: batch_size,
            in_channels,
            out_channels,
            len: input_len,
            bias: bias.map(|b| &b.data[..out_channels]),
            relu,
        };
        kernel3(
            level,
            &shape,
            &input.data[..batch_size * in_channels * input_len],
            &weights.data[..out_channels * in_channels * 3],
            &mut out[..total_output_size],
        );
        return TensorView::from_slice(out, vec![batch_size, out_channels, output_len]);
    }

    // Fast path for K=1 pointwise convolution: direct GEMM without im2col or SCRATCH.
    // Conv1d with kernel_size=1 is equivalent to matmul: weights[OC, IC] × input[IC, T] = output[OC, T]
    // This avoids thread_local SCRATCH allocation and closure overhead.
    if kernel_size == 1
        && group == 1
        && stride == 1
        && dilation == 1
        && pad_left == 0
        && pad_right == 0
    {
        let out_slice = &mut out[..total_output_size];
        for b in 0..batch_size {
            let a_offset = b * out_channels * output_len;
            let in_offset = b * in_channels * input_len;

            #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
            unsafe {
                // Use Apple Accelerate AMX for pointwise conv (much faster than faer NEON)
                crate::kernels::gemm::accelerate_init();
                crate::kernels::gemm::accelerate_sgemm(
                    out_channels as i32,
                    output_len as i32,
                    in_channels as i32,
                    1.0,
                    weights.data.as_ptr(),
                    in_channels as i32,
                    input.data.as_ptr().add(in_offset),
                    output_len as i32,
                    0.0,
                    out_slice.as_mut_ptr().add(a_offset),
                    output_len as i32,
                );
            }

            #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
            unsafe {
                let a = MatRef::<f32>::from_raw_parts(
                    weights.data.as_ptr(),
                    out_channels,
                    in_channels,
                    in_channels as isize,
                    1,
                );
                let b_mat = MatRef::<f32>::from_raw_parts(
                    input.data.as_ptr().add(in_offset),
                    in_channels,
                    output_len,
                    output_len as isize,
                    1,
                );
                let c = MatMut::<f32>::from_raw_parts_mut(
                    out_slice.as_mut_ptr().add(a_offset),
                    out_channels,
                    output_len,
                    output_len as isize,
                    1,
                );
                matmul(c, Accum::Replace, a, b_mat, 1.0f32, Par::Seq);
            }
        }

        bias_relu_rows(out, bias, relu, out_channels, output_len);

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

                                if stride == 1 && pad_left == 0 {
                                    // Contiguous copy for stride=1, no padding
                                    let src_start = k_offset;
                                    let dst_start = unfolded_row_offset;
                                    let copy_len = (input_len - k_offset).min(output_len);
                                    unsafe {
                                        std::ptr::copy_nonoverlapping(
                                            in_ptr.add(src_start),
                                            unf_ptr.add(dst_start),
                                            copy_len,
                                        );
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
                    matmul(c, Accum::Replace, a, b, 1.0f32, Par::Seq);
                }
            }
        }

        bias_relu_rows(out, bias, relu, out_channels, output_len);

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

    /// One convolution to check: the shape of the problem, not its data.
    #[derive(Clone, Copy, Debug)]
    struct Case {
        batch: usize,
        in_ch: usize,
        out_ch: usize,
        len: usize,
        kernel: usize,
        stride: usize,
        dilation: usize,
        group: usize,
        pads: [usize; 2],
    }

    const fn case(
        in_ch: usize,
        out_ch: usize,
        len: usize,
        kernel: usize,
        stride: usize,
        pads: [usize; 2],
    ) -> Case {
        Case {
            batch: 1,
            in_ch,
            out_ch,
            len,
            kernel,
            stride,
            dilation: 1,
            group: 1,
            pads,
        }
    }

    /// Cases for every path `conv1d_fused` takes: a single input channel,
    /// depthwise, kernel 3 with padding 1, pointwise, and the general
    /// im2col one (strided, dilated, grouped, asymmetric padding), each at
    /// lengths and channel counts on both sides of the vector widths.
    fn cases() -> Vec<Case> {
        let mut cases = Vec::new();
        for &out_ch in &[1, 3, 4, 5, 9] {
            for &(kernel, stride) in &[(1, 1), (3, 1), (8, 1), (9, 2), (16, 4), (33, 3)] {
                cases.push(case(1, out_ch, 80, kernel, stride, [0, 0]));
            }
        }
        for &len in &[1, 2, 7, 8, 9, 16, 17, 33, 100] {
            for &(kernel, stride, pad) in &[(3, 1, 1), (3, 2, 1), (5, 1, 2), (5, 2, 2), (7, 1, 3)] {
                if len + 2 * pad >= kernel {
                    let ch = 6;
                    cases.push(Case {
                        group: ch,
                        ..case(ch, ch, len, kernel, stride, [pad, pad])
                    });
                }
            }
            for &(in_ch, out_ch) in &[(1, 1), (3, 1), (3, 4), (5, 5), (4, 9), (16, 8)] {
                cases.push(case(in_ch, out_ch, len, 3, 1, [1, 1]));
            }
            for &(in_ch, out_ch) in &[(3, 5), (16, 17)] {
                cases.push(case(in_ch, out_ch, len, 1, 1, [0, 0]));
            }
        }
        cases.extend([
            Case {
                group: 4,
                ..case(4, 4, 70, 5, 1, [1, 3])
            },
            Case {
                group: 4,
                ..case(4, 4, 70, 4, 3, [2, 0])
            },
            Case {
                group: 3,
                ..case(3, 3, 5, 7, 1, [3, 3])
            },
            Case {
                batch: 2,
                group: 5,
                ..case(5, 5, 47, 3, 1, [1, 1])
            },
            case(3, 5, 40, 3, 2, [1, 1]),
            case(4, 6, 40, 5, 1, [2, 2]),
            case(4, 6, 40, 3, 1, [0, 2]),
            case(4, 6, 40, 3, 1, [2, 0]),
            case(3, 5, 41, 4, 3, [1, 2]),
            Case {
                dilation: 2,
                ..case(3, 5, 40, 3, 1, [2, 2])
            },
            Case {
                group: 2,
                ..case(4, 6, 40, 3, 1, [1, 1])
            },
            Case {
                group: 2,
                ..case(4, 6, 40, 1, 1, [0, 0])
            },
            Case {
                batch: 2,
                ..case(1, 5, 80, 16, 4, [0, 0])
            },
            Case {
                batch: 3,
                ..case(3, 5, 17, 3, 1, [1, 1])
            },
            Case {
                batch: 2,
                group: 6,
                ..case(6, 6, 33, 3, 2, [1, 1])
            },
        ]);
        cases
    }

    /// The textbook definition, in f64: padding contributes zeros.
    fn reference(c: &Case, x: &[f32], w: &[f32], bias: Option<&[f32]>, relu: bool) -> Vec<f64> {
        let out_len = (c.len + c.pads[0] + c.pads[1] - c.dilation * (c.kernel - 1) - 1) / c.stride
            + 1;
        let (icg, ocg) = (c.in_ch / c.group, c.out_ch / c.group);
        let mut out = Vec::new();
        for b in 0..c.batch {
            for oc in 0..c.out_ch {
                let g = oc / ocg;
                for t in 0..out_len {
                    let mut s = bias.map_or(0.0, |b| b[oc] as f64);
                    for ic in 0..icg {
                        for k in 0..c.kernel {
                            let i = (t * c.stride + k * c.dilation) as isize - c.pads[0] as isize;
                            if i >= 0 && (i as usize) < c.len {
                                let xv = x[(b * c.in_ch + g * icg + ic) * c.len + i as usize];
                                s += xv as f64 * w[(oc * icg + ic) * c.kernel + k] as f64;
                            }
                        }
                    }
                    out.push(if relu { s.max(0.0) } else { s });
                }
            }
        }
        out
    }

    #[test]
    fn test_conv1d_matches_reference() {
        let mut rng = crate::kernels::test_util::Rng::new(11);
        for (level, c) in crate::kernels::test_util::levels()
            .into_iter()
            .flat_map(|level| cases().into_iter().map(move |c| (level, c)))
        {
            let x = rng.vec(c.batch * c.in_ch * c.len, -2.0, 2.0);
            let w = rng.vec(c.out_ch * (c.in_ch / c.group) * c.kernel, -1.0, 1.0);
            let bias = rng.vec(c.out_ch, -1.0, 1.0);
            let input = TensorView::from_slice(&x, vec![c.batch, c.in_ch, c.len]);
            let weights = TensorView::from_slice(&w, vec![c.out_ch, c.in_ch / c.group, c.kernel]);
            let bias_view = TensorView::from_slice(&bias, vec![c.out_ch]);
            for has_bias in [false, true] {
                for relu in [false, true] {
                    let bias_arg = has_bias.then_some(&bias_view);
                    let mut out = Vec::new();
                    let got = conv1d_fused_at(
                        level,
                        &input,
                        &weights,
                        bias_arg,
                        &[c.dilation as i64],
                        c.group as i64,
                        &[c.pads[0] as i64, c.pads[1] as i64],
                        &[c.stride as i64],
                        relu,
                        &mut out,
                    );
                    let want = reference(&c, &x, &w, has_bias.then_some(&bias[..]), relu);
                    crate::kernels::test_util::assert_close(
                        &got.data,
                        &want,
                        1e-5,
                        &format!("{level:?} {c:?} bias {has_bias} relu {relu}"),
                    );
                }
            }
        }
    }
}
