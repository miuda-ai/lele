use crate::tensor::TensorView;
pub fn reshape<'a, T: Clone + std::fmt::Debug>(
    input: &TensorView<'a, T>,
    target_shape_raw: &[i64],
) -> TensorView<'a, T> {
    let total_elements = input.data.len();
    let mut new_shape = Vec::new();
    let mut known_product = 1;
    let mut infer_idx = None;
    for (i, &dim) in target_shape_raw.iter().enumerate() {
        if dim == -1 {
            if infer_idx.is_some() {
                panic!("Reshape: multiple -1 dimensions");
            }
            infer_idx = Some(i);
        } else if dim == 0 {
            if i < input.dim() {
                let d = input.shape[i];
                new_shape.push(d);
                known_product *= d;
            } else {
                panic!(
                    "Reshape: 0 dimension index out of bounds of input shape {} vs target rank {}",
                    input.dim(),
                    target_shape_raw.len()
                );
            }
        } else {
            new_shape.push(dim as usize);
            known_product *= dim as usize;
        }
    }
    if let Some(idx) = infer_idx {
        if known_product == 0 {
            panic!(
                "Reshape: known product is 0, cannot infer -1 dimension. Total elements: {}",
                total_elements
            );
        }
        let missing = total_elements / known_product;
        new_shape.insert(idx, missing);
    }
    assert_eq!(
        new_shape.iter().product::<usize>(),
        total_elements,
        "Reshape: element count mismatch (input={:?} target={:?})",
        input.shape,
        target_shape_raw
    );
    TensorView {
        data: input.data.clone(),
        shape: std::borrow::Cow::Owned(new_shape),
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
pub fn shape_slicing<T: Clone + std::fmt::Debug>(
    input: &TensorView<T>,
    start: i64,
    end: Option<i64>,
) -> TensorView<'static, i64> {
    let rank = input.dim() as i64;
    let start = if start < 0 { rank + start } else { start } as usize;
    let end = if let Some(e) = end {
        let e = if e < 0 { rank + e } else { e } as usize;
        e.min(rank as usize)
    } else {
        rank as usize
    };
    let start = start.min(rank as usize).min(end);
    let shape_slice = &input.shape[start..end];
    let shape_data: Vec<i64> = shape_slice.iter().map(|&x| x as i64).collect();
    let new_rank = shape_data.len();
    TensorView::from_owned(shape_data, vec![new_rank])
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
