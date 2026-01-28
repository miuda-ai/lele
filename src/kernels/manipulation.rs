use crate::kernels::utils;
use crate::tensor::TensorView;
pub fn concat<'b, 'a>(
    inputs: &[&TensorView<'b>],
    axis: i64,
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    if inputs.is_empty() {
        return TensorView::empty();
    }
    let non_empty_inputs: Vec<&TensorView<'b>> = inputs
        .iter()
        .filter(|inp| !inp.data.is_empty())
        .copied()
        .collect();
    if non_empty_inputs.is_empty() {
        return TensorView::empty();
    }
    let ndim = non_empty_inputs[0].dim();
    let axis = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };
    let mut out_shape = non_empty_inputs[0].shape.to_vec();
    out_shape[axis] = 0;
    for inp in &non_empty_inputs {
        assert_eq!(inp.dim(), ndim, "Concat: ranks mismatch");
        for d in 0..ndim {
            if d != axis {
                assert_eq!(inp.shape[d], out_shape[d], "Concat: inner dim mismatch");
            }
        }
        out_shape[axis] += inp.shape[axis];
    }
    let out_numel = out_shape.iter().product::<usize>();
    utils::ensure_capacity(out, out_numel);
    unsafe {
        out.set_len(out_numel);
    }
    let outer_dim: usize = non_empty_inputs[0].shape.iter().take(axis).product();
    let inner_dim: usize = non_empty_inputs[0].shape.iter().skip(axis + 1).product();
    let out_ptr = out.as_mut_ptr();
    let mut current_out_offset = 0;
    let mut inp_offsets = vec![0usize; non_empty_inputs.len()];
    for _ in 0..outer_dim {
        for (k, inp) in non_empty_inputs.iter().enumerate() {
            let axis_len = inp.shape[axis];
            let copy_len = axis_len * inner_dim;
            let inp_ptr = inp.data.as_ptr();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    inp_ptr.add(inp_offsets[k]),
                    out_ptr.add(current_out_offset),
                    copy_len,
                );
            }
            inp_offsets[k] += copy_len;
            current_out_offset += copy_len;
        }
    }
    TensorView::from_slice(out, out_shape)
}
pub fn slice<'b, 'a>(
    input: &TensorView<'b>,
    starts: &[i64],
    ends: &[i64],
    axes: &[i64],
    steps: &[i64],
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let ndim = input.dim();
    let num_ops = starts.len();
    let mut actual_starts = vec![0isize; ndim];
    let mut actual_ends = vec![0isize; ndim];
    let mut actual_steps = vec![1isize; ndim];
    for i in 0..ndim {
        actual_starts[i] = 0;
        actual_ends[i] = input.shape[i] as isize;
        actual_steps[i] = 1;
    }
    for i in 0..num_ops {
        let axis = if axes.is_empty() {
            i
        } else {
            let ax = axes[i];
            if ax < 0 {
                (ndim as i64 + ax) as usize
            } else {
                ax as usize
            }
        };
        let dim_size = input.shape[axis] as isize;
        let step = if i < steps.len() {
            steps[i] as isize
        } else {
            1
        };
        let start = starts[i] as isize;
        let end = ends[i] as isize;
        let norm_start = if start < 0 { start + dim_size } else { start };
        let norm_end = if end > 2_000_000_000 {
            if step > 0 {
                dim_size
            } else {
                -1
            }
        } else if end < -2_000_000_000 {
            if step > 0 {
                0
            } else {
                -1
            }
        } else if end < 0 {
            end + dim_size
        } else {
            end
        };
        let (s, e) = if step > 0 {
            (
                norm_start.max(0).min(dim_size),
                norm_end.max(0).min(dim_size),
            )
        } else {
            (
                norm_start.max(0).min(dim_size - 1),
                norm_end.max(-1).min(dim_size - 1),
            )
        };
        actual_starts[axis] = s;
        actual_ends[axis] = e;
        actual_steps[axis] = step;
    }
    let mut out_shape = vec![0; ndim];
    for i in 0..ndim {
        let start = actual_starts[i];
        let end = actual_ends[i];
        let step = actual_steps[i];
        let count = if step > 0 {
            if start >= end {
                0
            } else {
                (end - start + step - 1) / step
            }
        } else {
            if start <= end {
                0
            } else {
                (start - end + (-step) - 1) / (-step)
            }
        };
        out_shape[i] = count as usize;
    }
    let out_numel = out_shape.iter().product::<usize>();
    utils::ensure_capacity(out, out_numel);
    unsafe {
        out.set_len(out_numel);
    }
    if out_numel == 0 {
        return TensorView::from_slice(out, out_shape);
    }
    let in_strides = utils::compute_strides(&input.shape);
    let mut coords = vec![0; ndim];
    let out_slice = out.as_mut_slice();
    for i in 0..out_numel {
        let mut in_off = 0isize;
        for d in 0..ndim {
            let in_idx = actual_starts[d] + (coords[d] as isize) * actual_steps[d];
            in_off += in_idx * (in_strides[d] as isize);
        }
        out_slice[i] = input.data[in_off as usize];
        for d in (0..ndim).rev() {
            coords[d] += 1;
            if coords[d] < out_shape[d] {
                break;
            }
            coords[d] = 0;
        }
    }
    TensorView::from_slice(out, out_shape)
}
pub fn pad<'b, 'a>(
    input: &TensorView<'b>,
    pads: &TensorView<'b>,
    constant_value: Option<&TensorView<'b>>,
    _mode: &str,
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let p = pads.data.iter().map(|&x| x as usize).collect::<Vec<_>>();
    let rank = input.shape.len();
    if p.len() < rank * 2 {
        panic!(
            "Pad: pads length {} is less than rank {} * 2",
            p.len(),
            rank
        );
    }
    let mut new_shape = input.shape.to_vec();
    for i in 0..rank {
        new_shape[i] += p[i] + p[i + rank];
    }
    let total = new_shape.iter().product::<usize>();
    utils::ensure_capacity(out, total);
    unsafe {
        out.set_len(total);
    }
    let fill_val = constant_value
        .and_then(|t| t.data.first())
        .copied()
        .unwrap_or(0.0);
    out.fill(fill_val);
    if rank == 1 {
        let start = p[0];
        out[start..start + input.shape[0]].copy_from_slice(&input.data);
    } else if rank == 2 {
        for i in 0..input.shape[0] {
            let dst_row = i + p[0];
            let dst_start = dst_row * new_shape[1] + p[1];
            let src_start = i * input.shape[1];
            out[dst_start..dst_start + input.shape[1]]
                .copy_from_slice(&input.data[src_start..src_start + input.shape[1]]);
        }
    } else if rank == 3 {
        for i in 0..input.shape[0] {
            for j in 0..input.shape[1] {
                let dst_row = i + p[0];
                let dst_col = j + p[1];
                let dst_start = (dst_row * new_shape[1] + dst_col) * new_shape[2] + p[2];
                let src_start = (i * input.shape[1] + j) * input.shape[2];
                let copy_len = input.shape[2];
                out[dst_start..dst_start + copy_len]
                    .copy_from_slice(&input.data[src_start..src_start + copy_len]);
            }
        }
    } else if rank == 4 {
        for n in 0..input.shape[0] {
            for c in 0..input.shape[1] {
                for h in 0..input.shape[2] {
                    let dst_n = n + p[0];
                    let dst_c = c + p[1];
                    let dst_h = h + p[2];
                    let dst_start = (((dst_n * new_shape[1] + dst_c) * new_shape[2] + dst_h)
                        * new_shape[3])
                        + p[3];
                    let src_start =
                        ((n * input.shape[1] + c) * input.shape[2] + h) * input.shape[3];
                    let copy_len = input.shape[3];
                    out[dst_start..dst_start + copy_len]
                        .copy_from_slice(&input.data[src_start..src_start + copy_len]);
                }
            }
        }
    } else {
        panic!("Pad: Rank {} not fully implemented", rank);
    }
    TensorView::from_slice(out, new_shape)
}
pub fn gather<'b, 'a>(
    data: &TensorView<'b>,
    indices: &TensorView<'b>,
    axis: i64,
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let axis = if axis < 0 {
        (data.dim() as i64 + axis) as usize
    } else {
        axis as usize
    };
    let mut out_shape = Vec::new();
    for i in 0..axis {
        out_shape.push(data.shape[i]);
    }
    out_shape.extend_from_slice(&indices.shape);
    for i in (axis + 1)..data.dim() {
        out_shape.push(data.shape[i]);
    }
    let out_numel = out_shape.iter().product::<usize>();
    utils::ensure_capacity(out, out_numel);
    unsafe {
        out.set_len(out_numel);
    }
    let outer_dim: usize = data.shape[..axis].iter().product();
    let axis_dim = data.shape[axis];
    let inner_dim: usize = data.shape[axis + 1..].iter().product();
    let indices_len = indices.data.len();
    let mut out_idx = 0;
    let out_ptr = out.as_mut_ptr();
    let data_ptr = data.data.as_ptr();
    let indices_ptr = indices.data.as_ptr();
    for o in 0..outer_dim {
        for idx_i in 0..indices_len {
            unsafe {
                let idx_val = *indices_ptr.add(idx_i) as usize;
                let src_offset = o * axis_dim * inner_dim + idx_val * inner_dim;
                std::ptr::copy_nonoverlapping(
                    data_ptr.add(src_offset),
                    out_ptr.add(out_idx),
                    inner_dim,
                );
                out_idx += inner_dim;
            }
        }
    }
    TensorView::from_slice(out, out_shape)
}
pub fn cast<'a>(input: &TensorView<'a>, _to: i64) -> TensorView<'a> {
    input.clone()
}
pub fn transpose<'b, 'a>(
    input: &TensorView<'b>,
    perm: &[i64],
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let ndim = input.dim();
    let perm: Vec<usize> = if perm.is_empty() {
        (0..ndim).rev().collect()
    } else {
        perm.iter().map(|&x| x as usize).collect()
    };
    let mut out_shape = vec![0; ndim];
    for (i, &p) in perm.iter().enumerate() {
        out_shape[i] = input.shape[p];
    }
    let out_numel = input.data.len();
    utils::ensure_capacity(out, out_numel);
    unsafe {
        out.set_len(out_numel);
    }
    let in_strides = utils::compute_strides(&input.shape);
    let mut virtual_strides = vec![0; ndim];
    for i in 0..ndim {
        virtual_strides[i] = in_strides[perm[i]];
    }
    let mut coords = vec![0; ndim];
    let out_slice = out.as_mut_slice();
    let mut in_off = 0;
    for i in 0..out_numel {
        unsafe {
            *out_slice.get_unchecked_mut(i) = *input.data.get_unchecked(in_off);
        }
        for d in (0..ndim).rev() {
            coords[d] += 1;
            if coords[d] < out_shape[d] {
                in_off += virtual_strides[d];
                break;
            } else {
                in_off -= (coords[d] - 1) * virtual_strides[d];
                coords[d] = 0;
            }
        }
    }
    TensorView::from_slice(out, out_shape)
}
pub fn to_i64_vec(input: &TensorView) -> Vec<i64> {
    input.data.iter().map(|&x| x as i64).collect()
}
pub fn split<'a>(
    input: &TensorView,
    axis: i64,
    splits: &[i64],
    outputs: &'a mut [Vec<f32>],
) -> Vec<TensorView<'a>> {
    let ndim = input.dim();
    let axis = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };
    assert!(axis < ndim, "Split: axis out of bounds");
    let num_splits = splits.len();
    assert_eq!(
        outputs.len(),
        num_splits,
        "Split: output buffers count mismatch"
    );
    let total: i64 = splits.iter().sum();
    assert_eq!(
        total, input.shape[axis] as i64,
        "Split: splits sum mismatch"
    );
    let outer_dim: usize = input.shape[..axis].iter().product();
    let inner_dim: usize = input.shape[axis + 1..].iter().product();
    let mut results = Vec::with_capacity(num_splits);
    let mut axis_offset = 0;
    for (i, &split_size) in splits.iter().enumerate() {
        let split_size = split_size as usize;
        let mut out_shape = input.shape.to_vec();
        out_shape[axis] = split_size;
        let out_numel = out_shape.iter().product::<usize>();
        utils::ensure_capacity(&mut outputs[i], out_numel);
        for outer_idx in 0..outer_dim {
            let src_offset = outer_idx * input.shape[axis] * inner_dim + axis_offset * inner_dim;
            let dst_offset = outer_idx * split_size * inner_dim;
            let copy_len = split_size * inner_dim;
            let out_slice = &mut outputs[i][dst_offset..dst_offset + copy_len];
            out_slice.copy_from_slice(&input.data[src_offset..src_offset + copy_len]);
        }
        axis_offset += split_size;
    }
    for (i, &split_size) in splits.iter().enumerate() {
        let mut out_shape = input.shape.to_vec();
        out_shape[axis] = split_size as usize;
        results.push(TensorView::from_slice(&outputs[i], out_shape));
    }
    results
}
pub fn where_op<'b, 'a>(
    condition: &TensorView<'b>,
    x: &TensorView<'b>,
    y: &TensorView<'b>,
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let cond_data = &condition.data;
    let x_data = &x.data;
    let y_data = &y.data;
    if condition.shape == x.shape && x.shape == y.shape {
        let numel = cond_data.len();
        utils::ensure_capacity(out, numel);
        unsafe {
            out.set_len(0);
        }
        for i in 0..numel {
            out.push(if cond_data[i] != 0.0 {
                x_data[i]
            } else {
                y_data[i]
            });
        }
        return TensorView::from_slice(out, condition.shape.to_vec());
    }
    let out_shape = utils::broadcast_shapes(&condition.shape, &x.shape)
        .and_then(|s| utils::broadcast_shapes(&s, &y.shape))
        .unwrap_or_else(|| condition.shape.to_vec());
    let out_numel = out_shape.iter().product();
    utils::ensure_capacity(out, out_numel);
    unsafe {
        out.set_len(0);
    }
    let cond_strides = utils::compute_strides(&condition.shape);
    let x_strides = utils::compute_strides(&x.shape);
    let y_strides = utils::compute_strides(&y.shape);
    let mut coords = vec![0usize; out_shape.len()];
    for _ in 0..out_numel {
        let cond_idx = map_broadcast_index(&coords, &condition.shape, &cond_strides);
        let x_idx = map_broadcast_index(&coords, &x.shape, &x_strides);
        let y_idx = map_broadcast_index(&coords, &y.shape, &y_strides);
        out.push(if cond_data[cond_idx] != 0.0 {
            x_data[x_idx]
        } else {
            y_data[y_idx]
        });
        for d in (0..out_shape.len()).rev() {
            coords[d] += 1;
            if coords[d] < out_shape[d] {
                break;
            }
            coords[d] = 0;
        }
    }
    TensorView::from_slice(out, out_shape)
}
fn map_broadcast_index(out_coords: &[usize], shape: &[usize], strides: &[usize]) -> usize {
    let mut idx = 0;
    let rank_diff = out_coords.len().saturating_sub(shape.len());
    for (i, &coord) in out_coords.iter().skip(rank_diff).enumerate() {
        let dim_size = if i < shape.len() { shape[i] } else { 1 };
        let actual_coord = if dim_size == 1 { 0 } else { coord };
        if i < strides.len() {
            idx += actual_coord * strides[i];
        }
    }
    idx
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorView;
    #[test]
    fn test_concat() {
        let d1 = vec![1.0, 2.0, 3.0, 4.0];
        let t1 = TensorView::from_slice(&d1, vec![2, 2]);
        let d2 = vec![5.0, 6.0];
        let t2 = TensorView::from_slice(&d2, vec![2, 1]);
        let mut out = Vec::new();
        let res = concat(&[&t1, &t2], 1, &mut out);
        assert_eq!(res.shape, vec![2, 3]);
        assert_eq!(res.data, vec![1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
    }
}
