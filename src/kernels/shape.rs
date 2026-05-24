use crate::tensor::TensorView;
pub fn reshape<'a, T: Clone + std::fmt::Debug>(
    input: &TensorView<'a, T>,
    target_shape_raw: &[i64],
) -> TensorView<'a, T> {
    let total_elements = input.data.len();

    // First pass: standard 0/-1 semantics
    if let Some(shape) = try_reshape_with_zeros(input, target_shape_raw, total_elements) {
        return TensorView {
            data: input.data.clone(),
            shape: std::borrow::Cow::Owned(shape),
        };
    }

    // Second pass: replace 0 with -1 (infer) and retry
    let fixed: Vec<i64> = target_shape_raw.iter().map(|&d| if d == 0 { -1 } else { d }).collect();
    if let Some(shape) = try_reshape_with_zeros(input, &fixed, total_elements) {
        return TensorView {
            data: input.data.clone(),
            shape: std::borrow::Cow::Owned(shape),
        };
    }

    // Third pass: collapse target to match source rank using -1
    // e.g. target [1, 96000, 64, 601] for source [1, 64, 601] → [1, -1, 601]
    let source_rank = input.dim();
    if target_shape_raw.len() > source_rank && source_rank > 0 {
        // Keep first dim, infer middle, keep last (source_rank - 1) dims
        let mut collapsed: Vec<i64> = Vec::new();
        // First dim from target
        let first = if target_shape_raw[0] > 0 { target_shape_raw[0] } else { -1 };
        collapsed.push(first);
        // -1 for inferred
        collapsed.push(-1);
        // Last (source_rank - 1) dims from target end
        for &dim in target_shape_raw.iter().rev().take(source_rank - 1).rev() {
            collapsed.push(if dim > 0 { dim } else { -1 });
        }
        if let Some(shape) = try_reshape_with_zeros(input, &collapsed, total_elements) {
            return TensorView {
                data: input.data.clone(),
                shape: std::borrow::Cow::Owned(shape),
            };
        }
    }

    panic!(
        "Reshape: element count mismatch (input={:?} target={:?})",
        input.shape, target_shape_raw
    );
}

fn try_reshape_with_zeros<T: Clone + std::fmt::Debug>(
    input: &TensorView<'_, T>,
    target: &[i64],
    total_elements: usize,
) -> Option<Vec<usize>> {
    let mut new_shape = Vec::new();
    let mut known_product = 1;
    let mut infer_idx = None;
    for (i, &dim) in target.iter().enumerate() {
        if dim == -1 {
            if infer_idx.is_some() {
                return None;
            }
            infer_idx = Some(i);
        } else if dim == 0 {
            if i < input.dim() {
                let d = input.shape[i];
                new_shape.push(d);
                known_product *= d;
            } else {
                return None;
            }
        } else {
            new_shape.push(dim as usize);
            known_product *= dim as usize;
        }
    }
    if let Some(idx) = infer_idx {
        if known_product == 0 || total_elements % known_product != 0 {
            return None;
        }
        let missing = total_elements / known_product;
        new_shape.insert(idx, missing);
    }
    if new_shape.iter().product::<usize>() == total_elements {
        Some(new_shape)
    } else {
        None
    }
}

pub fn size<T: Clone + std::fmt::Debug>(input: &TensorView<T>) -> TensorView<'static, i64> {
    let total = input.data.len() as i64;
    TensorView::from_owned(vec![total], vec![])
}

pub fn shape<T: Clone + std::fmt::Debug>(input: &TensorView<T>) -> TensorView<'static, i64> {
    let shape_data: Vec<i64> = input.shape.iter().map(|&x| x as i64).collect();
    let rank = input.dim();
    TensorView::from_owned(shape_data, vec![rank])
}
pub fn flatten<'a, T: Clone + std::fmt::Debug>(
    input: &TensorView<'a, T>,
    axis: i64,
) -> TensorView<'a, T> {
    let axis = if axis < 0 {
        input.dim() as i64 + axis
    } else {
        axis
    } as usize;
    let shape = &input.shape;
    let dim1: usize = shape.iter().take(axis).product();
    let dim2: usize = shape.iter().skip(axis).product();
    TensorView {
        data: input.data.clone(),
        shape: std::borrow::Cow::Owned(vec![dim1, dim2]),
    }
}
pub fn constant_of_shape<'a, 'b, T, V: Clone + std::fmt::Debug>(
    input: &TensorView<'a, T>,
    value: V,
    out: &'b mut Vec<V>,
) -> TensorView<'b, V>
where
    T: crate::kernels::utils::AsI64 + Copy + std::fmt::Debug,
{
    let shape: Vec<usize> = input.data.iter().map(|&x| x.as_i64() as usize).collect();
    let size: usize = shape.iter().product();
    out.clear();
    out.resize(size, value);
    TensorView::from_slice(out, shape)
}
pub fn unsqueeze<'a, T: Clone + std::fmt::Debug>(
    input: &TensorView<'a, T>,
    axes: &[i64],
) -> TensorView<'a, T> {
    let mut new_shape = input.shape.to_vec();
    let rank = input.dim() + axes.len();
    let mut sorted_axes = axes.to_vec();
    sorted_axes.sort();
    for &axis in &sorted_axes {
        let idx = if axis < 0 { rank as i64 + axis } else { axis } as usize;
        if idx <= new_shape.len() {
            new_shape.insert(idx, 1);
        } else {
            new_shape.push(1);
        }
    }
    TensorView {
        data: input.data.clone(),
        shape: std::borrow::Cow::Owned(new_shape),
    }
}
pub fn squeeze<'a, T: Clone + std::fmt::Debug>(
    input: &TensorView<'a, T>,
    axes: Option<&[i64]>,
) -> TensorView<'a, T> {
    let mut new_shape = Vec::new();
    if let Some(axes) = axes {
        let dims = input.dim();
        let mut axes_set = std::collections::HashSet::new();
        for &a in axes {
            axes_set.insert(if a < 0 { dims as i64 + a } else { a } as usize);
        }
        for (i, &d) in input.shape.iter().enumerate() {
            if d == 1 && axes_set.contains(&i) {
                continue;
            }
            new_shape.push(d);
        }
    } else {
        for &d in input.shape.iter() {
            if d != 1 {
                new_shape.push(d);
            }
        }
    }
    TensorView {
        data: input.data.clone(),
        shape: std::borrow::Cow::Owned(new_shape),
    }
}
pub fn identity<'a, T: Clone + std::fmt::Debug>(input: &TensorView<'a, T>) -> TensorView<'a, T> {
    input.clone()
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorView;
    #[test]
    fn test_reshape() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let t = TensorView::from_slice(&data, vec![2, 2]);
        let shape_data = [4];
        let res = reshape(&t, &shape_data);
        assert_eq!(res.shape, vec![4]);
        let shape_neg = [1, -1];
        let res2 = reshape(&t, &shape_neg);
        assert_eq!(res2.shape, vec![1, 4]);
    }
    #[test]
    fn test_flatten() {
        let data = vec![1.0; 24];
        let t = TensorView::from_slice(&data, vec![2, 3, 4]);
        let res = flatten(&t, 1);
        assert_eq!(res.shape, vec![2, 12]);
        let res2 = flatten(&t, 2);
        assert_eq!(res2.shape, vec![6, 4]);
    }
    #[test]
    fn test_squeeze_unsqueeze() {
        let data = vec![1.0];
        let t = TensorView::from_slice(&data, vec![1, 1]);
        let res = squeeze(&t, None);
        assert_eq!(res.shape, Vec::<usize>::new());
        let axes = vec![0];
        let res2 = unsqueeze(&res, &axes);
        assert_eq!(res2.shape, vec![1]);
    }
}
