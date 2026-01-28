pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut s = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = s;
        s *= shape[i];
    }
    strides
}
pub fn ensure_capacity(v: &mut Vec<f32>, len: usize) {
    if v.len() != len {
        v.clear();
        v.resize(len, 0.0);
    }
}
pub fn get_offset(indices: &[usize], strides: &[usize]) -> usize {
    indices
        .iter()
        .zip(strides)
        .map(|(&idx, &stride)| idx * stride)
        .sum()
}
pub fn offset_to_indices(mut offset: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
    for i in (0..shape.len()).rev() {
        indices[i] = offset % shape[i];
        offset /= shape[i];
    }
    indices
}
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let len = std::cmp::max(a.len(), b.len());
    let mut out_shape = vec![0; len];
    for i in 0..len {
        let a_dim = if i < len - a.len() {
            1
        } else {
            a[i - (len - a.len())]
        };
        let b_dim = if i < len - b.len() {
            1
        } else {
            b[i - (len - b.len())]
        };
        if a_dim == b_dim {
            out_shape[i] = a_dim;
        } else if a_dim == 1 {
            out_shape[i] = b_dim;
        } else if b_dim == 1 {
            out_shape[i] = a_dim;
        } else {
            return None;
        }
    }
    Some(out_shape)
}
