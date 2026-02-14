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
        // Handle sentinel values (i64::MIN, i64::MAX) on the original i64
        // before casting to isize. On wasm32 (isize = 32-bit), casting
        // directly would truncate and break sentinel detection.
        let start_i64 = starts[i];
        let end_i64 = ends[i];

        // Detect sentinels BEFORE clamping. ONNX uses i64::MAX to mean
        // "to the end" (positive step) and i64::MIN to mean "to the
        // very beginning" (negative step).
        let end_is_max_sentinel = end_i64 > i64::MAX / 2;
        let end_is_min_sentinel = end_i64 < i64::MIN / 2;

        let start = if start_i64 > dim_size as i64 {
            dim_size
        } else if start_i64 < -(dim_size as i64) {
            -dim_size
        } else {
            start_i64 as isize
        };
        let end = if end_is_max_sentinel {
            dim_size
        } else if end_is_min_sentinel {
            -dim_size
        } else if end_i64 > dim_size as i64 {
            dim_size
        } else if end_i64 < -(dim_size as i64) {
            -dim_size
        } else {
            end_i64 as isize
        };
        let norm_start = if start < 0 { start + dim_size } else { start };
        let norm_end = if end_is_max_sentinel {
            if step > 0 { dim_size } else { -1 }
        } else if end_is_min_sentinel {
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
    // Safely convert i64 → usize, clamping negatives to 0
    let raw_p: Vec<usize> = pads
        .iter()
        .map(|&x| if x < 0 { 0usize } else { x as usize })
        .collect();
    let rank = input.shape.len();
    // If pads is shorter than rank*2, it covers fewer dims.
    // ONNX pads layout: [begin_0..begin_n, end_0..end_n].
    // Zero-pad for the leading (batch/channel) dimensions.
    let p = if raw_p.len() < rank * 2 {
        let half = raw_p.len() / 2;
        let missing = rank - half;
        let mut full = vec![0usize; rank * 2];
        // Copy begins into positions [missing..rank]
        for i in 0..half {
            full[missing + i] = raw_p[i];
        }
        // Copy ends into positions [rank+missing..rank*2]
        for i in 0..half {
            full[rank + missing + i] = raw_p[half + i];
        }
        full
    } else {
        raw_p
    };
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

    // Fast path: perm [0,2,1,3] for 4D tensors — copy contiguous blocks
    // [B, A, C, D] -> [B, C, A, D] — inner dim D is contiguous
    if ndim == 4 && perm == [0, 2, 1, 3] {
        let (b, a, c, d) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );
        let out_slice = out.as_mut_slice();
        for bi in 0..b {
            for ci in 0..c {
                for ai in 0..a {
                    let src_off = ((bi * a + ai) * c + ci) * d;
                    let dst_off = ((bi * c + ci) * a + ai) * d;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            input.data.as_ptr().add(src_off),
                            out_slice.as_mut_ptr().add(dst_off),
                            d,
                        );
                    }
                }
            }
        }
        return TensorView::from_slice(out, out_shape);
    }

    // Fast path: perm [0,2,3,1] for 4D tensors
    // [B, A, C, D] -> [B, C, D, A]
    if ndim == 4 && perm == [0, 2, 3, 1] {
        let (b, a, c, d) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );
        let out_slice = out.as_mut_slice();
        for bi in 0..b {
            for ai in 0..a {
                for ci in 0..c {
                    for di in 0..d {
                        let src_off = ((bi * a + ai) * c + ci) * d + di;
                        let dst_off = ((bi * c + ci) * d + di) * a + ai;
                        unsafe {
                            *out_slice.get_unchecked_mut(dst_off) =
                                *input.data.get_unchecked(src_off);
                        }
                    }
                }
            }
        }
        return TensorView::from_slice(out, out_shape);
    }

    // Fast path: perm [0,2,1] for 3D tensors — 2D transpose within each batch
    // [B, R, C] -> [B, C, R]
    if ndim == 3 && perm == [0, 2, 1] {
        let (b, r, c) = (input.shape[0], input.shape[1], input.shape[2]);
        let out_slice = out.as_mut_slice();
        for bi in 0..b {
            let base_in = bi * r * c;
            let base_out = bi * c * r;
            for ri in 0..r {
                for ci in 0..c {
                    unsafe {
                        *out_slice.get_unchecked_mut(base_out + ci * r + ri) =
                            *input.data.get_unchecked(base_in + ri * c + ci);
                    }
                }
            }
        }
        return TensorView::from_slice(out, out_shape);
    }

    // Generic fallback
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

    // Fast path: all same shape — direct elementwise, no coordinates
    if condition.shape == x.shape && x.shape == y.shape {
        let numel = cond_data.len();
        utils::ensure_capacity(out, numel);
        unsafe {
            out.set_len(numel);
        }
        let o = out.as_mut_slice();
        for i in 0..numel {
            unsafe {
                *o.get_unchecked_mut(i) = if cond_data.get_unchecked(i).as_i64() != 0 {
                    *x_data.get_unchecked(i)
                } else {
                    *y_data.get_unchecked(i)
                };
            }
        }
        return TensorView::from_slice(out, condition.shape.to_vec());
    }

    let out_shape = utils::broadcast_shapes(&condition.shape, &x.shape)
        .and_then(|s| utils::broadcast_shapes(&s, &y.shape))
        .unwrap_or_else(|| condition.shape.to_vec());
    let out_numel: usize = out_shape.iter().product();
    let dims = out_shape.len();
    utils::ensure_capacity(out, out_numel);
    unsafe {
        out.set_len(out_numel);
    }
    let o = out.as_mut_slice();

    if x.data.len() == out_numel && y.data.len() == out_numel {
        // Condition broadcasts, x and y are full — use stride-based cond index
        let cond_numel = cond_data.len();
        if cond_numel == 1 {
            // Scalar condition
            if cond_data[0].as_i64() != 0 {
                o.copy_from_slice(x_data);
            } else {
                o.copy_from_slice(y_data);
            }
        } else {
            // Condition is smaller, repeats over leading dims
            // E.g. cond=[T,T], out=[B,H,T,T]
            let repeat = out_numel / cond_numel;
            for r in 0..repeat {
                let base = r * cond_numel;
                for i in 0..cond_numel {
                    unsafe {
                        let idx = base + i;
                        *o.get_unchecked_mut(idx) = if cond_data.get_unchecked(i).as_i64() != 0 {
                            *x_data.get_unchecked(idx)
                        } else {
                            *y_data.get_unchecked(idx)
                        };
                    }
                }
            }
        }
        return TensorView::from_slice(out, out_shape);
    }

    // General broadcast: use coordinate-based indexing (slowest path)

    let mk_strides = |shape: &[usize]| -> Vec<usize> {
        let mut strides = vec![0; dims];
        let mut curr = 1;
        let offset = dims - shape.len();
        for i in (0..shape.len()).rev() {
            if shape[i] != 1 {
                strides[offset + i] = curr;
            }
            curr *= shape[i];
        }
        strides
    };
    let cs = mk_strides(&condition.shape);
    let xs = mk_strides(&x.shape);
    let ys = mk_strides(&y.shape);
    let mut coords = vec![0usize; dims];
    let mut off_c = 0usize;
    let mut off_x = 0usize;
    let mut off_y = 0usize;
    for j in 0..out_numel {
        unsafe {
            *o.get_unchecked_mut(j) = if cond_data.get_unchecked(off_c).as_i64() != 0 {
                *x_data.get_unchecked(off_x)
            } else {
                *y_data.get_unchecked(off_y)
            };
        }
        for d in (0..dims).rev() {
            coords[d] += 1;
            if coords[d] < out_shape[d] {
                off_c += cs[d];
                off_x += xs[d];
                off_y += ys[d];
                break;
            } else {
                off_c -= (coords[d] - 1) * cs[d];
                off_x -= (coords[d] - 1) * xs[d];
                off_y -= (coords[d] - 1) * ys[d];
                coords[d] = 0;
            }
        }
    }
    TensorView::from_slice(out, out_shape)
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
