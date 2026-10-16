use crate::kernels::bias_act::{Act, act_inplace_at, bias_act_inplace_at};
use crate::kernels::conv1d_direct::{
    Depthwise, Kernel3, SingleChannel, depthwise, kernel3, single_channel,
};
use crate::kernels::conv2d::{IM2COL_BYTES, conv2d_activation};
use crate::kernels::gemm::WeightElem;
use crate::kernels::matmul::{Accum, MatMut, MatRef, Par, matmul};
use crate::kernels::utils;
use crate::tensor::TensorView;
use aligned_vec::AVec;
use fearless_simd::Level;

/// Adds `bias[oc]` to every `[oc, ..]` row of `out` (laid out `[batch, out_channels, len]`)
/// and applies the activation if any.
///
/// The activation goes over the whole output at once, not row by row: rows are
/// often short (a few frames), and each would pay a call and a last vector that
/// overlaps the one before it.
fn bias_act_rows(
    level: Level,
    out: &mut [f32],
    bias: Option<&TensorView>,
    act: Option<Act>,
    out_channels: usize,
    len: usize,
) {
    if let Some(bias) = bias {
        for (i, row) in out.chunks_exact_mut(len).enumerate() {
            bias_act_inplace_at(level, row, bias.data[i % out_channels], None);
        }
    }
    if let Some(act) = act {
        act_inplace_at(level, act, out);
    }
}

/// A `Conv` of kernel 1, one group, stride and dilation 1 and no padding, of an input
/// `[batch, in_channels, len]` with more than one channel (the case [`conv1d`] computes
/// as a matrix product), whose weights `[out_channels, in_channels, 1]` may be f16: they
/// are widened as they are read, and the result is that of [`conv1d`] on the widened
/// weights, bit for bit. Weights stored packed ([`PackedF16`](crate::kernels::PackedF16))
/// are read in panels from start to end and sum in another order. Then the activation, if
/// any, as its own kernel computes it.
pub fn conv1d_pointwise<'a, W: WeightElem>(
    input: &TensorView<'_>,
    weights: &TensorView<'_, W>,
    bias: Option<&TensorView<'_>>,
    act: Option<Act>,
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let [batch, in_channels, len] = input.shape[..] else {
        panic!("conv1d_pointwise: input of shape {:?}", input.shape);
    };
    let [out_channels, w_in, 1] = weights.shape[..] else {
        panic!("conv1d_pointwise: weights of shape {:?}", weights.shape);
    };
    assert!(
        w_in == in_channels && in_channels > 1,
        "conv1d_pointwise: {in_channels} input channels against weights of {w_in}"
    );
    // SAFETY: the product replaces every element.
    unsafe { utils::ensure_capacity(out, batch * out_channels * len) };
    pointwise(
        Level::new(),
        input,
        &weights.data,
        bias,
        act,
        (batch, in_channels, out_channels, len),
        out,
    )
}

/// Per batch, `weights[out_channels, in_channels] x input[in_channels, len]`, then the
/// bias and the activation (at `level`). `out` is the output's length.
fn pointwise<'a, W: WeightElem>(
    level: Level,
    input: &TensorView<'_>,
    weights: &[W],
    bias: Option<&TensorView<'_>>,
    act: Option<Act>,
    (batch_size, in_channels, out_channels, len): (usize, usize, usize, usize),
    out: &'a mut [f32],
) -> TensorView<'a> {
    let total = batch_size * out_channels * len;
    assert!(
        weights.len() >= out_channels * in_channels
            && input.data.len() >= batch_size * in_channels * len
    );
    let out_slice = &mut out[..total];
    for b in 0..batch_size {
        // SAFETY: the operands hold the elements addressed (asserted above).
        unsafe {
            let b_mat = MatRef::<f32>::from_raw_parts(
                input.data.as_ptr().add(b * in_channels * len),
                in_channels,
                len,
                len as isize,
                1,
            );
            let c = MatMut::<f32>::from_raw_parts_mut(
                out_slice.as_mut_ptr().add(b * out_channels * len),
                out_channels,
                len,
                len as isize,
                1,
            );
            W::matmul_left_into(
                c,
                Accum::Replace,
                weights.as_ptr(),
                out_channels,
                in_channels,
                b_mat,
            );
        }
    }
    bias_act_rows(level, out, bias, act, out_channels, len);
    TensorView::from_slice(out, vec![batch_size, out_channels, len])
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
        input, weights, bias, dilations, group, pads, strides, None, out,
    )
}

/// [`conv1d`], then `act`, applied as the output is written: the activation's own kernel
/// on the output, bit for bit.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_fused<'b, 'a>(
    input: &TensorView<'b>,
    weights: &TensorView<'b>,
    bias: Option<&TensorView<'b>>,
    dilations: &[i64],
    group: i64,
    pads: &[i64],
    strides: &[i64],
    act: Option<Act>,
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
        act,
        IM2COL_BYTES,
        out,
    )
}

/// `unfold_bytes` bounds the input unfolded whole ([`IM2COL_BYTES`]; others in tests).
fn conv1d_fused_at<'b, 'a>(
    level: Level,
    input: &TensorView<'b>,
    weights: &TensorView<'b>,
    bias: Option<&TensorView<'b>>,
    dilations: &[i64],
    group: i64,
    pads: &[i64],
    strides: &[i64],
    act: Option<Act>,
    unfold_bytes: usize,
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
    assert!(w_shape.len() == 3, "Conv1d: weights of shape {w_shape:?}");
    let out_channels = w_shape[0];
    let kernel_size = w_shape[2];
    // Each group's rows are computed from its share of the input; a group count
    // that does not divide both would leave output rows unwritten.
    let g = group as usize;
    assert!(
        group > 0
            && in_channels.is_multiple_of(g)
            && out_channels.is_multiple_of(g)
            && w_shape[1] * g == in_channels,
        "Conv1d: {in_channels} input and {out_channels} output channels in {group} groups \
         against weights of shape {w_shape:?}"
    );
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
    // An input shorter than the kernel's reach has no outputs.
    let padded_len = input_len + pad_left + pad_right;
    let reach = dilation * (kernel_size - 1) + 1;
    if padded_len < reach {
        return TensorView::from_slice(&out[..0], vec![batch_size, out_channels, 0]);
    }
    let output_len = (padded_len - reach) / stride + 1;
    let total_output_size = batch_size * out_channels * output_len;
    // SAFETY: every element is written below before any is read (the group
    // count divides the channels, asserted above).
    unsafe { utils::ensure_capacity(out, total_output_size) };
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
            act,
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

    // Dilated taps are only handled with stride 1.
    if group as usize == in_channels
        && group as usize == out_channels
        && (dilation == 1 || stride == 1)
    {
        let shape = Depthwise {
            batch: batch_size,
            channels: in_channels,
            len: input_len,
            kernel: kernel_size,
            stride,
            dilation,
            pad_left,
            out_len: output_len,
            bias: bias.map(|b| &b.data[..out_channels]),
            act,
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

    // A channel multiplier (every channel convolved with several kernels, as in a
    // depthwise convolution that widens, or a strided one that reduces a sequence while
    // widening it): one depthwise convolution for each multiple. Per group, the product
    // is a few rows against a few columns, and its fixed costs are the whole time.
    // (Of one channel, or with many multiples, each of its own pass over the input, the
    // product is the better way.)
    if g == in_channels
        && in_channels > 1
        && in_channels_per_group == 1
        && (2..=MAX_MULTIPLIER).contains(&out_channels_per_group)
        && (dilation == 1 || stride == 1)
    {
        depthwise_multiplied(
            level,
            (input, batch_size, in_channels, input_len),
            weights,
            bias,
            (kernel_size, stride, dilation, pad_left),
            act,
            (out_channels_per_group, output_len),
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
            act,
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
        return pointwise(
            level,
            input,
            &weights.data,
            bias,
            act,
            (batch_size, in_channels, out_channels, input_len),
            out,
        );
    }

    let unfolded_size = unfolded_rows * output_len;
    if unfolded_size * 4 > unfold_bytes {
        // Larger than cache, unfolded whole the input would be written out to
        // memory and read back: as one row of a 2D convolution, it is unfolded a
        // block at a time as the product reads it. (Smaller, the 2D convolution's
        // bookkeeping per row of columns would cost more than unfolding.)
        return conv1d_as_2d(
            level,
            (&input.data, [batch_size, in_channels, input_len]),
            weights,
            bias,
            (dilation, group, pad_left, pad_right, stride),
            act,
            out,
        );
    }

    thread_local! {
        static SCRATCH: std::cell::RefCell<AVec<f32>> = std::cell::RefCell::new(AVec::new(0));
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

        bias_act_rows(level, out, bias, act, out_channels, output_len);

        TensorView::from_slice(out, vec![batch_size, out_channels, output_len])
    })
}

/// The most kernels per channel that [`depthwise_multiplied`] takes.
const MAX_MULTIPLIER: usize = 8;

/// A depthwise convolution with `multiplier` kernels for each channel: output channel
/// `c * multiplier + j` is channel `c` convolved with kernel `c * multiplier + j`. Each
/// `j` is a depthwise convolution of its own, whose rows go to their place in `out`.
#[allow(clippy::too_many_arguments)]
fn depthwise_multiplied(
    level: Level,
    (input, batch, channels, len): (&TensorView, usize, usize, usize),
    weights: &TensorView,
    bias: Option<&TensorView>,
    (kernel, stride, dilation, pad_left): (usize, usize, usize, usize),
    act: Option<Act>,
    (multiplier, out_len): (usize, usize),
    out: &mut [f32],
) {
    thread_local! {
        /// The taps and bias of one multiple, and its output.
        static MULTIPLES: std::cell::RefCell<[AVec<f32>; 3]> =
            std::cell::RefCell::new([AVec::new(0), AVec::new(0), AVec::new(0)]);
    }
    MULTIPLES.with_borrow_mut(|[taps, biases, part]| {
        part.clear();
        part.resize(batch * channels * out_len, 0.0);
        for j in 0..multiplier {
            taps.clear();
            biases.clear();
            for c in 0..channels {
                let oc = c * multiplier + j;
                taps.extend_from_slice(&weights.data[oc * kernel..][..kernel]);
                if let Some(bias) = bias {
                    biases.push(bias.data[oc]);
                }
            }
            let shape = Depthwise {
                batch,
                channels,
                len,
                kernel,
                stride,
                dilation,
                pad_left,
                out_len,
                bias: bias.map(|_| &biases[..]),
                act,
            };
            depthwise(
                level,
                &shape,
                &input.data[..batch * channels * len],
                taps,
                part,
            );
            for b in 0..batch {
                for c in 0..channels {
                    let dst = (b * channels * multiplier + c * multiplier + j) * out_len;
                    out[dst..][..out_len]
                        .copy_from_slice(&part[(b * channels + c) * out_len..][..out_len]);
                }
            }
        }
    });
}

/// [`conv1d_fused_at`] as the convolution of a `[batch, channels, 1, len]` input.
/// The input is `[batch, in_channels, len]` (`dims`), the weights
/// `[out_channels, in_channels / group, kernel]`.
fn conv1d_as_2d<'a>(
    level: Level,
    (input, dims): (&[f32], [usize; 3]),
    weights: &TensorView<'_>,
    bias: Option<&TensorView<'_>>,
    (dilation, group, pad_left, pad_right, stride): (usize, i64, usize, usize, usize),
    act: Option<Act>,
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let [batch_size, in_channels, input_len] = dims;
    let [out_channels, in_channels_per_group, kernel_size] = weights.shape[..] else {
        unreachable!("weights of shape {:?}", weights.shape)
    };
    let input = TensorView::from_slice(
        &input[..batch_size * in_channels * input_len],
        vec![batch_size, in_channels, 1, input_len],
    );
    let weights = TensorView::from_slice(
        &weights.data[..out_channels * in_channels_per_group * kernel_size],
        vec![out_channels, in_channels_per_group, 1, kernel_size],
    );
    let pads = [0, pad_left as i64, 0, pad_right as i64];
    let (dilations, strides) = ([1, dilation as i64], [1, stride as i64]);
    let output_len = conv2d_activation(
        level,
        &input,
        &weights,
        bias,
        &dilations,
        group,
        &pads,
        &strides,
        act,
        IM2COL_BYTES,
        out,
    )
    .shape[3];
    TensorView::from_slice(out, vec![batch_size, out_channels, output_len])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::PackedF16;
    use crate::kernels::matmul::{PACK_COLS, pack_f16};
    use crate::kernels::test_util::{Rng, assert_close, assert_same_bits};
    use crate::tensor::TensorView;

    /// `conv1d_fused` with an activation the kernels do not apply themselves is
    /// `conv1d` then that activation's kernel, bit for bit: T-one's depthwise
    /// convolution and a dense one, outputs shorter and longer than a vector,
    /// with and without bias.
    #[test]
    fn test_conv1d_fused_is_conv1d_then_the_activation() {
        let mut rng = Rng::new(11);
        // (channels in, channels out, groups, kernel, padding, length)
        for (cin, cout, group, k, pad, len) in [
            (384, 384, 384, 31, 0, 40),
            (8, 8, 8, 31, 15, 5),
            (6, 10, 1, 5, 2, 23),
        ] {
            let x = rng.vec(2 * cin * len, -3.0, 3.0);
            let w = rng.vec(cout * (cin / group) * k, -0.5, 0.5);
            let b = rng.vec(cout, -1.0, 1.0);
            let x = TensorView::from_slice(&x, vec![2, cin, len]);
            let w = TensorView::from_slice(&w, vec![cout, cin / group, k]);
            let b = TensorView::from_slice(&b, vec![cout]);
            for (bias, act) in [None, Some(&b)]
                .into_iter()
                .flat_map(|b| [Act::Silu, Act::Sigmoid, Act::GeluErf].map(|a| (b, a)))
            {
                let (mut c, mut got) = (Vec::new(), Vec::new());
                let (pads, groups) = ([pad as i64; 2], group as i64);
                let conv = conv1d(&x, &w, bias, &[1], groups, &pads, &[1], &mut c);
                let mut want = conv.data.to_vec();
                crate::kernels::act_inplace(act, &mut want);
                let fused =
                    conv1d_fused(&x, &w, bias, &[1], groups, &pads, &[1], Some(act), &mut got);
                assert_eq!(fused.shape, conv.shape);
                let what = format!(
                    "{cin}->{cout}/{group} k{k} len {len} bias {} {act:?}",
                    bias.is_some()
                );
                assert_same_bits(&fused.data, &want, &what);
            }
        }
    }

    /// Padding a depthwise conv `s * stride` less on the left and
    /// `d * stride` less on the right drops `s` outputs at the start and `d`
    /// at the end, the rest bit for bit, which is what folding slices into
    /// its padding relies on: FireRedVAD's lookback filter as it folds in
    /// two steps, stride 2 and dilation 2.
    #[test]
    fn test_depthwise_conv1d_less_padding_is_sliced_conv1d() {
        let mut rng = Rng::new(13);
        // (channels, kernel, stride, dilation, pads, slice)
        for (c, k, stride, dilation, pads, (s, d)) in [
            (128, 20, 1, 1, [19, 19], (0, 19)),
            (128, 20, 1, 1, [19, 0], (19, 0)),
            (8, 5, 2, 1, [4, 5], (1, 2)),
            (8, 3, 1, 2, [4, 4], (3, 1)),
        ] {
            for len in [20, 29, 40] {
                let x = rng.vec(c * len, -3.0, 3.0);
                let w = rng.vec(c * k, -0.5, 0.5);
                let x = TensorView::from_slice(&x, vec![1, c, len]);
                let w = TensorView::from_slice(&w, vec![c, 1, k]);
                let (mut a, mut b) = (Vec::new(), Vec::new());
                let (g, dil) = (c as i64, [dilation]);
                let full = conv1d(&x, &w, None, &dil, g, &pads, &[stride], &mut a);
                let less = [pads[0] - s * stride, pads[1] - d * stride];
                let got = conv1d(&x, &w, None, &dil, g, &less, &[stride], &mut b);
                let n = full.shape[2];
                let keep = s as usize..n - d as usize;
                assert_eq!(*got.shape, [1, c, keep.len()]);
                let want: Vec<f32> = full
                    .data
                    .chunks_exact(n)
                    .flat_map(|row| row[keep.clone()].to_vec())
                    .collect();
                let what = format!("k{k} stride {stride} dilation {dilation} len {len}");
                assert_same_bits(&got.data, &want, &what);
            }
        }
    }

    /// An input shorter than the kernel, padding included, has no outputs.
    #[test]
    fn test_conv1d_short_input_is_empty() {
        let x = vec![1.0; 4 * 19];
        let w = vec![1.0; 4 * 20];
        let x = TensorView::from_slice(&x, vec![1, 4, 19]);
        let w = TensorView::from_slice(&w, vec![4, 1, 20]);
        let mut out = Vec::new();
        let y = conv1d(&x, &w, None, &[1], 4, &[0, 0], &[1], &mut out);
        assert_eq!(*y.shape, [1, 4, 0]);
    }

    /// `conv1d_pointwise` is `conv1d` for f32 weights, and for f16 weights `conv1d` of
    /// the widened ones, bit for bit (packed f16 weights sum in another order, so only
    /// to rounding): lengths through the dot products and the packed kernels, with and
    /// without bias and ReLU.
    #[test]
    fn test_conv1d_pointwise_matches_conv1d() {
        let mut rng = Rng::new(9);
        for (batch, cin, cout, len) in [
            (1, 384, 768, 10),
            (2, 3, 5, 1),
            (1, 16, 7, 40),
            (2, 33, 20, 13),
            (2, 40, 48, 3),
            (1, 600, 32, 7),
            (1, 32, 32, 300),
        ] {
            let input = rng.vec(batch * cin * len, -1.0, 1.0);
            let w16: Vec<half::f16> = rng
                .vec(cout * cin, -1.0, 1.0)
                .iter()
                .map(|&x| half::f16::from_f32(x))
                .collect();
            let w32: Vec<f32> = w16.iter().map(|x| x.to_f32()).collect();
            let bias = rng.vec(cout, -0.5, 0.5);
            let x = TensorView::from_slice(&input, vec![batch, cin, len]);
            let (t16, t32) = (
                TensorView::from_slice(&w16, vec![cout, cin, 1]),
                TensorView::from_slice(&w32, vec![cout, cin, 1]),
            );
            let tb = TensorView::from_slice(&bias, vec![cout]);
            let transposed: Vec<half::f16> = (0..cin * cout)
                .map(|i| w16[i % cout * cin + i / cout])
                .collect();
            let p16: Option<Vec<PackedF16>> = (cout % PACK_COLS == 0).then(|| {
                let packed = pack_f16(&transposed, cin, cout);
                packed.into_iter().map(PackedF16).collect()
            });
            let packed = p16
                .as_ref()
                .map(|p| TensorView::from_slice(p, vec![cout, cin, 1]));
            for (b, act) in [
                (None, None),
                (Some(&tb), Some(Act::Relu)),
                (Some(&tb), Some(Act::GeluErf)),
            ] {
                let mut want = Vec::new();
                let want = conv1d_fused(&x, &t32, b, &[1], 1, &[0, 0], &[1], act, &mut want);
                let what = format!("[{batch},{cin},{len}] -> {cout} {act:?}");
                let mut got = Vec::new();
                let got = conv1d_pointwise(&x, &t32, b, act, &mut got);
                assert_eq!(got.shape, want.shape);
                assert_same_bits(&got.data, &want.data, &format!("f32 {what}"));
                let mut got = Vec::new();
                let got = conv1d_pointwise(&x, &t16, b, act, &mut got);
                assert_same_bits(&got.data, &want.data, &format!("f16 {what}"));
                if let Some(packed) = &packed {
                    let mut got = Vec::new();
                    let got = conv1d_pointwise(&x, packed, b, act, &mut got);
                    let want: Vec<f64> = want.data.iter().map(|&x| x as f64).collect();
                    assert_close(&got.data, &want, 1e-5, &format!("packed {what}"));
                }
            }
        }
    }

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
                group: 5,
                dilation: 2,
                ..case(5, 5, 60, 5, 1, [4, 4])
            },
            Case {
                group: 5,
                dilation: 4,
                ..case(5, 5, 33, 5, 1, [0, 0])
            },
            Case {
                group: 4,
                dilation: 8,
                ..case(4, 4, 90, 3, 1, [8, 3])
            },
            Case {
                batch: 2,
                group: 3,
                dilation: 3,
                ..case(3, 3, 20, 4, 1, [3, 3])
            },
            Case {
                batch: 2,
                group: 5,
                ..case(5, 5, 47, 3, 1, [1, 1])
            },
            // As Hojo-TTS's encoder (a DAC encoder) has them: dilated residual
            // convolutions, a strided downsampling one, more than four output
            // channels per group, and deeper than one block of the product.
            Case {
                dilation: 3,
                ..case(8, 8, 150, 7, 1, [9, 9])
            },
            Case {
                dilation: 9,
                ..case(8, 8, 150, 7, 1, [27, 27])
            },
            case(8, 16, 151, 4, 2, [1, 1]),
            Case {
                group: 2,
                dilation: 2,
                ..case(8, 14, 90, 5, 1, [4, 4])
            },
            case(74, 7, 40, 7, 1, [3, 3]),
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
            // A channel multiplier: T-one's strided one (384 channels, four kernels
            // each, over 40 samples), and ones with padding, a batch, dilation, a bias
            // and ReLU, a multiplier past the limit, one channel, and dilation with a
            // stride (the general path).
            Case {
                group: 6,
                ..case(6, 24, 40, 3, 2, [0, 0])
            },
            Case {
                batch: 2,
                group: 5,
                ..case(5, 15, 33, 5, 1, [2, 2])
            },
            Case {
                group: 4,
                dilation: 3,
                ..case(4, 8, 50, 3, 1, [3, 3])
            },
            Case {
                group: 3,
                ..case(3, 30, 20, 3, 1, [1, 1])
            },
            Case {
                group: 1,
                ..case(1, 4, 20, 3, 1, [1, 1])
            },
            Case {
                group: 4,
                dilation: 2,
                ..case(4, 8, 40, 3, 2, [2, 2])
            },
        ]);
        cases
    }

    /// The textbook definition, in f64: padding contributes zeros.
    fn reference(c: &Case, x: &[f32], w: &[f32], bias: Option<&[f32]>) -> Vec<f64> {
        let out_len =
            (c.len + c.pads[0] + c.pads[1] - c.dilation * (c.kernel - 1) - 1) / c.stride + 1;
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
                    out.push(s);
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
            // Unfolded whole, and (no input being small enough) as a 2D convolution.
            for (has_bias, unfold_bytes) in [false, true]
                .into_iter()
                .flat_map(|b| [(b, IM2COL_BYTES), (b, 0)])
            {
                let bias_arg = has_bias.then_some(&bias_view);
                let conv = |act: Option<Act>| {
                    let mut out = Vec::new();
                    conv1d_fused_at(
                        level,
                        &input,
                        &weights,
                        bias_arg,
                        &[c.dilation as i64],
                        c.group as i64,
                        &[c.pads[0] as i64, c.pads[1] as i64],
                        &[c.stride as i64],
                        act,
                        unfold_bytes,
                        &mut out,
                    );
                    out
                };
                let plain = conv(None);
                let want = reference(&c, &x, &w, has_bias.then_some(&bias[..]));
                let what = format!("{level:?} {c:?} bias {has_bias} unfold {unfold_bytes}");
                // f32 rounding grows with the length of the dot products.
                let terms = (c.in_ch / c.group * c.kernel).max(64);
                let n = want.len();
                crate::kernels::test_util::assert_close(
                    &plain[..n],
                    &want,
                    1e-5 * terms as f64 / 64.0,
                    &what,
                );
                // An activation is the plain output, then its own kernel at that level.
                for act in [Act::Relu, Act::Silu, Act::Sigmoid, Act::GeluErf] {
                    let mut want = plain[..n].to_vec();
                    crate::kernels::bias_act::act_inplace_at(level, act, &mut want);
                    assert_same_bits(&conv(Some(act))[..n], &want, &format!("{what} {act:?}"));
                }
            }
        }
    }
}
