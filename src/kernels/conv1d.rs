use crate::kernels::utils;
use crate::tensor::TensorView;
use matrixmultiply::sgemm;
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
    let unfolded_size = unfolded_rows * output_len;
    let mut unfolded = vec![0.0; unfolded_size];
    for b in 0..batch_size {
        for g in 0..group as usize {
            unfolded.fill(0.0);
            let in_group_offset = (b * in_channels + g * in_channels_per_group) * input_len;
            for ic in 0..in_channels_per_group {
                let in_row_offset = in_group_offset + ic * input_len;
                let in_data = &input.data[in_row_offset..in_row_offset + input_len];
                for k in 0..kernel_size {
                    let k_offset = k * dilation;
                    let unfolded_row_idx = ic * kernel_size + k;
                    let unfolded_row_offset = unfolded_row_idx * output_len;
                    for t_out in 0..output_len {
                        let t_in =
                            (t_out * stride) as isize - pad_left as isize + k_offset as isize;
                        if t_in >= 0 && (t_in as usize) < input_len {
                            unfolded[unfolded_row_offset + t_out] = in_data[t_in as usize];
                        }
                    }
                }
            }
            let weight_group_offset =
                (g * out_channels_per_group) * in_channels_per_group * kernel_size;
            let out_group_offset = (b * out_channels + g * out_channels_per_group) * output_len;
            unsafe {
                sgemm(
                    out_channels_per_group,
                    unfolded_rows,
                    output_len,
                    1.0,
                    weights.data.as_ptr().add(weight_group_offset),
                    unfolded_rows as isize,
                    1,
                    unfolded.as_ptr(),
                    output_len as isize,
                    1,
                    0.0,
                    out.as_mut_ptr().add(out_group_offset),
                    output_len as isize,
                    1,
                );
            }
        }
    }
    if let Some(b_vec) = bias {
        for b in 0..batch_size {
            for oc in 0..out_channels {
                let start = (b * out_channels + oc) * output_len;
                let b_val = b_vec.data[oc];
                for i in 0..output_len {
                    out[start + i] += b_val;
                }
            }
        }
    }
    TensorView::from_slice(out, vec![batch_size, out_channels, output_len])
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
}
