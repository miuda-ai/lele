use crate::kernels::utils;
use crate::tensor::TensorView;
pub fn concat<'b, 'a, T: Clone + Copy + std::fmt::Debug>(
    inputs: &[&TensorView<'b, T>],
    axis: i64,
    out: &'a mut Vec<T>,
) -> TensorView<'a, T> {
    if inputs.is_empty() {
        return TensorView::empty();
    }
    // Find first non-empty input to establish rank and base out_shape
    let mut first_non_empty = None;
    for (i, inp) in inputs.iter().enumerate() {
        if !inp.data.is_empty() {
            first_non_empty = Some((i, inp));
            break;
        }
    }

    let (_, first_inp) = match first_non_empty {
        Some(x) => x,
        None => return TensorView::empty(),
    };

    let ndim = first_inp.dim();
    let axis = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };

    // Calculate final shape and validate
    let mut out_shape = first_inp.shape.to_vec();
    out_shape[axis] = 0;

    for inp in inputs {
        if inp.data.is_empty() {
            continue;
        }
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
    if out_numel == 0 {
        return TensorView::from_slice(out, out_shape);
    }

    let outer_dim: usize = out_shape.iter().take(axis).product();
    let inner_dim: usize = out_shape.iter().skip(axis + 1).product();
    let out_ptr = out.as_mut_ptr();
    let mut current_out_offset = 0;

    // Direct copy without allocating offset vectors
    for outer_i in 0..outer_dim {
        for inp in inputs {
            if inp.data.is_empty() {
                continue;
            }
            let axis_len = inp.shape[axis];
            let copy_len = axis_len * inner_dim;
            let src_offset = outer_i * copy_len;
            let inp_ptr = inp.data.as_ptr();

            unsafe {
                std::ptr::copy_nonoverlapping(
                    inp_ptr.add(src_offset),
                    out_ptr.add(current_out_offset),
                    copy_len,
                );
            }
            current_out_offset += copy_len;
        }
    }
    TensorView::from_slice(out, out_shape)
}
pub fn slice<'b, 'a, T: Clone + Copy + std::fmt::Debug>(
    input: &TensorView<'b, T>,
    starts: &[i64],
    ends: &[i64],
    axes: &[i64],
    steps: &[i64],
    out: &'a mut Vec<T>,
) -> TensorView<'a, T> {
    // Fast path: simple contiguous slice along first dimension
    // This handles the most common case: slicing rows of a 2D tensor
    if !axes.is_empty() && axes.len() == 1 && steps.is_empty() {
        let axis = if axes[0] < 0 {
            (input.dim() as i64 + axes[0]) as usize
        } else {
            axes[0] as usize
        };

        if axis == 0 && starts.len() == 1 && ends.len() == 1 {
            let dim_size = input.shape[0] as i64;
            let start = if starts[0] < 0 {
                (starts[0] + dim_size).max(0) as usize
            } else {
                starts[0].min(dim_size).max(0) as usize
            };
            let end = if ends[0] < 0 {
                (ends[0] + dim_size).max(0) as usize
            } else if ends[0] > 2_000_000_000 {
                dim_size as usize
            } else {
                ends[0].min(dim_size).max(0) as usize
            };

            if end > start {
                let stride: usize = input.shape[1..].iter().product();
                let start_offset = start * stride;
                let end_offset = end * stride;

                utils::ensure_capacity(out, end_offset - start_offset);
                out.copy_from_slice(&input.data[start_offset..end_offset]);

                let mut out_shape = input.shape.to_vec();
                out_shape[0] = end - start;
                return TensorView::from_slice(out, out_shape);
            }
        }
    }

    // Standard path: complex multi-dimensional slicing
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
            if step > 0 { dim_size } else { -1 }
        } else if end < -2_000_000_000 {
            if step > 0 { 0 } else { -1 }
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
pub fn pad<'b, 'a, T: Clone + Copy + std::fmt::Debug>(
    input: &TensorView<'b, T>,
    pads: &[i64],
    constant_value: Option<&TensorView<'b, T>>,
    mode: &str,
    out: &'a mut Vec<T>,
) -> TensorView<'a, T> {
    let p = pads.iter().map(|&x| x as usize).collect::<Vec<_>>();
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
    // For constant mode: use provided constant_value or default to 0 (per ONNX spec)
    let fill_val: T = if let Some(cv) = constant_value {
        if !cv.data.is_empty() {
            cv.data[0]
        } else {
            // ONNX default constant value is 0
            unsafe { std::mem::zeroed() }
        }
    } else {
        // ONNX default constant value is 0
        unsafe { std::mem::zeroed() }
    };

    // First copy input data into correct position, then handle padding
    out.fill(fill_val);
    // Copy input data into the center
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
    // For edge mode, replicate edge values into padding regions
    if mode == "edge" {
        pad_edge_inplace(out, &input.shape, &new_shape, &p, rank);
    }
    TensorView::from_slice(out, new_shape)
}

/// Fill padding regions with edge (nearest) values for "edge" mode.
/// `out` already has the input data copied into the center region.
fn pad_edge_inplace<T: Clone + Copy>(
    out: &mut [T],
    input_shape: &[usize],
    new_shape: &[usize],
    p: &[usize],
    rank: usize,
) {
    // General n-dimensional edge padding via coordinate mapping
    let total: usize = new_shape.iter().product();
    let strides = utils::compute_strides(new_shape);

    // For each output element, clamp coordinates to the input range
    // to find the nearest edge value
    let mut coords = vec![0usize; rank];
    for idx in 0..total {
        // Check if this position is in the center (already filled)
        let mut in_center = true;
        for d in 0..rank {
            if coords[d] < p[d] || coords[d] >= p[d] + input_shape[d] {
                in_center = false;
                break;
            }
        }
        if !in_center {
            // Clamp coordinates to input region and read that value
            let mut src_idx = 0;
            for d in 0..rank {
                let clamped = if coords[d] < p[d] {
                    p[d]
                } else if coords[d] >= p[d] + input_shape[d] {
                    p[d] + input_shape[d] - 1
                } else {
                    coords[d]
                };
                src_idx += clamped * strides[d];
            }
            out[idx] = out[src_idx];
        }
        // Advance coordinates
        for d in (0..rank).rev() {
            coords[d] += 1;
            if coords[d] < new_shape[d] {
                break;
            }
            coords[d] = 0;
        }
    }
}
pub fn gather<'b, 'a, T, I>(
    data: &TensorView<'b, T>,
    indices: &TensorView<'b, I>,
    axis: i64,
    out: &'a mut Vec<T>,
) -> TensorView<'a, T>
where
    T: Copy + std::fmt::Debug,
    I: crate::kernels::utils::AsI64 + Copy + std::fmt::Debug,
{
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

    for o in 0..outer_dim {
        for idx_i in 0..indices_len {
            unsafe {
                let mut idx_val = indices.data[idx_i].as_i64();
                if idx_val < 0 {
                    idx_val += axis_dim as i64;
                }
                let idx_val = idx_val as usize;
                let src_offset = o * axis_dim * inner_dim + idx_val * inner_dim;
                for k in 0..inner_dim {
                    *out.as_mut_ptr().add(out_idx + k) = data.data[src_offset + k];
                }
                out_idx += inner_dim;
            }
        }
    }
    TensorView::from_slice(out, out_shape)
}
pub fn cast<'a>(input: &TensorView<'a>, _to: i64) -> TensorView<'a> {
    input.clone()
}
pub fn transpose<'b, 'a, T: Clone + Copy + std::fmt::Debug>(
    input: &TensorView<'b, T>,
    perm: &[i64],
    out: &'a mut Vec<T>,
) -> TensorView<'a, T> {
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
pub fn to_i64_vec<T: crate::kernels::utils::AsI64 + Copy + std::fmt::Debug>(
    input: &TensorView<T>,
) -> Vec<i64> {
    let mut out = Vec::with_capacity(input.data.len());
    for &val in input.data.iter() {
        out.push(val.as_i64());
    }
    out
}
pub fn split<'a, T: Clone + Copy + std::fmt::Debug>(
    input: &TensorView<'_, T>,
    axis: i64,
    splits: &[i64],
    outputs: &'a mut [Vec<T>],
) -> Vec<TensorView<'a, T>> {
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

    let actual_splits = if splits.iter().all(|&s| s == 0) && num_splits > 0 {
        let size = input.shape[axis];
        vec![ (size / num_splits) as i64; num_splits]
    } else {
        splits.to_vec()
    };

    let total: i64 = actual_splits.iter().sum();
    assert_eq!(
        total, input.shape[axis] as i64,
        "Split: splits sum mismatch"
    );
    let outer_dim: usize = input.shape[..axis].iter().product();
    let inner_dim: usize = input.shape[axis + 1..].iter().product();
    let mut results = Vec::with_capacity(num_splits);
    let mut axis_offset = 0;
    for (i, &split_size) in actual_splits.iter().enumerate() {
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
    for (i, &split_size) in actual_splits.iter().enumerate() {
        let mut out_shape = input.shape.to_vec();
        out_shape[axis] = split_size as usize;
        results.push(TensorView::from_slice(&outputs[i], out_shape));
    }
    results
}
pub fn where_op<'b, 'a, T, C>(
    condition: &TensorView<'b, C>,
    x: &TensorView<'b, T>,
    y: &TensorView<'b, T>,
    out: &'a mut Vec<T>,
) -> TensorView<'a, T>
where
    T: Clone + Copy + std::fmt::Debug,
    C: Clone + Copy + std::fmt::Debug + crate::kernels::utils::AsI64,
{
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
            out.push(if cond_data[i].as_i64() != 0 {
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
        out.push(if cond_data[cond_idx].as_i64() != 0 {
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
