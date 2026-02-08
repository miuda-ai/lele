use crate::kernels::utils;
use crate::tensor::TensorView;
use std::borrow::Cow;

pub trait ElementOps: Copy + PartialOrd + PartialEq + std::fmt::Debug + Sized + 'static {
    fn constant_min() -> Self;
    fn constant_max() -> Self;
    fn pow(self, exp: Self) -> Self;
    fn clamp_val(self, min: Self, max: Self) -> Self;
    fn as_f32(self) -> f32;
    fn from_f32(v: f32) -> Self;
}

impl ElementOps for f32 {
    fn constant_min() -> Self {
        f32::NEG_INFINITY
    }
    fn constant_max() -> Self {
        f32::INFINITY
    }
    fn pow(self, exp: Self) -> Self {
        self.powf(exp)
    }
    fn clamp_val(self, min: Self, max: Self) -> Self {
        self.clamp(min, max)
    }
    fn as_f32(self) -> f32 {
        self
    }
    fn from_f32(v: f32) -> Self {
        v
    }
}

impl ElementOps for i64 {
    fn constant_min() -> Self {
        i64::MIN
    }
    fn constant_max() -> Self {
        i64::MAX
    }
    fn pow(self, exp: Self) -> Self {
        if exp < 0 { 0 } else { self.pow(exp as u32) }
    }
    fn clamp_val(self, min: Self, max: Self) -> Self {
        self.clamp(min, max)
    }
    fn as_f32(self) -> f32 {
        self as f32
    }
    fn from_f32(v: f32) -> Self {
        v as i64
    }
}

pub fn min_max<'b, 'a>(input: &TensorView<'b>, _output: &'a mut Vec<f32>) -> (f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &val in input.data.iter() {
        if val < min {
            min = val;
        }
        if val > max {
            max = val;
        }
    }
    (min, max)
}
fn broadcast_binary_op<'b, 'a, T, F>(
    a: &TensorView<'b, T>,
    b: &TensorView<'b, T>,
    output_buf: &'a mut Vec<T>,
    op: F,
) -> TensorView<'a, T>
where
    T: Clone + Copy + std::fmt::Debug,
    F: Fn(T, T) -> T,
{
    let out_shape = utils::broadcast_shapes(&a.shape, &b.shape).unwrap_or_else(|| {
        panic!("Shapes not broadcastable: {:?} and {:?}", a.shape, b.shape)
    });
    let numel = out_shape.iter().product::<usize>();
    utils::ensure_capacity(output_buf, numel);
    unsafe {
        output_buf.set_len(numel);
    }
    let dims = out_shape.len();
    if a.data.len() == 1 {
        let val_a = a.data[0];
        let b_slice = &b.data;
        let o_slice = output_buf.as_mut_slice();
        for i in 0..numel {
            o_slice[i] = op(val_a, b_slice[i]);
        }
        return TensorView {
            data: Cow::Borrowed(output_buf),
            shape: Cow::Owned(out_shape),
        };
    }
    if b.data.len() == 1 {
        let val_b = b.data[0];
        let a_slice = &a.data;
        let o_slice = output_buf.as_mut_slice();
        for i in 0..numel {
            o_slice[i] = op(a_slice[i], val_b);
        }
        return TensorView {
            data: Cow::Borrowed(output_buf),
            shape: Cow::Owned(out_shape),
        };
    }
    let mk_strides = |shape: &[usize], target_dims: usize| -> Vec<usize> {
        let mut strides = vec![0; target_dims];
        let mut curr = 1;
        let offset = target_dims - shape.len();
        for i in (0..shape.len()).rev() {
            if shape[i] != 1 {
                strides[offset + i] = curr;
            }
            curr *= shape[i];
        }
        strides
    };
    let a_strides = mk_strides(&a.shape, dims);
    let b_strides = mk_strides(&b.shape, dims);
    let mut coords = vec![0; dims];
    let mut off_a = 0;
    let mut off_b = 0;
    let o_slice = output_buf.as_mut_slice();
    for j in 0..numel {
        unsafe {
            *o_slice.get_unchecked_mut(j) =
                op(*a.data.get_unchecked(off_a), *b.data.get_unchecked(off_b));
        }
        for i in (0..dims).rev() {
            coords[i] += 1;
            if coords[i] < out_shape[i] {
                off_a += a_strides[i];
                off_b += b_strides[i];
                break;
            } else {
                off_a -= (coords[i] - 1) * a_strides[i];
                off_b -= (coords[i] - 1) * b_strides[i];
                coords[i] = 0;
            }
        }
    }
    TensorView {
        data: Cow::Borrowed(output_buf),
        shape: Cow::Owned(out_shape),
    }
}

fn broadcast_binary_op_to<'b, 'a, T, U, F>(
    a: &TensorView<'b, T>,
    b: &TensorView<'b, T>,
    output_buf: &'a mut Vec<U>,
    op: F,
) -> TensorView<'a, U>
where
    T: Clone + Copy + std::fmt::Debug,
    U: Clone + Copy + std::fmt::Debug,
    F: Fn(T, T) -> U,
{
    let out_shape = utils::broadcast_shapes(&a.shape, &b.shape).expect("Shapes not broadcastable");
    let numel = out_shape.iter().product::<usize>();
    utils::ensure_capacity(output_buf, numel);
    unsafe {
        output_buf.set_len(numel);
    }
    let dims = out_shape.len();
    if a.data.len() == 1 {
        let val_a = a.data[0];
        let b_slice = &b.data;
        let o_slice = output_buf.as_mut_slice();
        for i in 0..numel {
            o_slice[i] = op(val_a, b_slice[i]);
        }
        return TensorView {
            data: Cow::Borrowed(output_buf),
            shape: Cow::Owned(out_shape),
        };
    }
    if b.data.len() == 1 {
        let val_b = b.data[0];
        let a_slice = &a.data;
        let o_slice = output_buf.as_mut_slice();
        for i in 0..numel {
            o_slice[i] = op(a_slice[i], val_b);
        }
        return TensorView {
            data: Cow::Borrowed(output_buf),
            shape: Cow::Owned(out_shape),
        };
    }
    let mk_strides = |shape: &[usize], target_dims: usize| -> Vec<usize> {
        let mut strides = vec![0; target_dims];
        let mut curr = 1;
        let offset = target_dims - shape.len();
        for i in (0..shape.len()).rev() {
            if shape[i] != 1 {
                strides[offset + i] = curr;
            }
            curr *= shape[i];
        }
        strides
    };
    let a_strides = mk_strides(&a.shape, dims);
    let b_strides = mk_strides(&b.shape, dims);
    let mut coords = vec![0; dims];
    let mut off_a = 0;
    let mut off_b = 0;
    let o_slice = output_buf.as_mut_slice();
    for j in 0..numel {
        unsafe {
            *o_slice.get_unchecked_mut(j) =
                op(*a.data.get_unchecked(off_a), *b.data.get_unchecked(off_b));
        }
        for d in (0..dims).rev() {
            coords[d] += 1;
            if coords[d] < out_shape[d] {
                if a.shape.len() > d && a.shape[d] != 1 {
                    off_a += a_strides[d];
                }
                if b.shape.len() > d && b.shape[d] != 1 {
                    off_b += b_strides[d];
                }
                break;
            } else {
                coords[d] = 0;
                if a.shape.len() > d && a.shape[d] != 1 {
                    off_a -= a_strides[d] * (out_shape[d] - 1);
                }
                if b.shape.len() > d && b.shape[d] != 1 {
                    off_b -= b_strides[d] * (out_shape[d] - 1);
                }
            }
        }
    }
    TensorView {
        data: Cow::Borrowed(output_buf),
        shape: Cow::Owned(out_shape),
    }
}
pub fn add<'b, 'a, T: Clone + Copy + std::ops::Add<Output = T> + std::fmt::Debug>(
    a: &TensorView<'b, T>,
    b: &TensorView<'b, T>,
    out: &'a mut Vec<T>,
) -> TensorView<'a, T> {
    if a.data.len() == b.data.len() && a.shape == b.shape {
        let len = a.data.len();
        utils::ensure_capacity(out, len);
        let a_slice = &a.data;
        let b_slice = &b.data;
        let o_slice = out.as_mut_slice();
        for i in 0..len {
            unsafe {
                *o_slice.get_unchecked_mut(i) =
                    *a_slice.get_unchecked(i) + *b_slice.get_unchecked(i);
            }
        }
        return TensorView {
            data: Cow::Borrowed(out),
            shape: std::borrow::Cow::Owned(a.shape.to_vec()),
        };
    }
    broadcast_binary_op(a, b, out, |x, y| x + y)
}
pub fn mul<'b, 'a, T: Clone + Copy + std::ops::Mul<Output = T> + std::fmt::Debug>(
    a: &TensorView<'b, T>,
    b: &TensorView<'b, T>,
    out: &'a mut Vec<T>,
) -> TensorView<'a, T> {
    if a.data.len() == b.data.len() && a.shape == b.shape {
        let len = a.data.len();
        utils::ensure_capacity(out, len);
        let a_slice = &a.data;
        let b_slice = &b.data;
        let o_slice = out.as_mut_slice();
        for i in 0..len {
            unsafe {
                *o_slice.get_unchecked_mut(i) =
                    *a_slice.get_unchecked(i) * *b_slice.get_unchecked(i);
            }
        }
        return TensorView {
            data: Cow::Borrowed(out),
            shape: std::borrow::Cow::Owned(a.shape.to_vec()),
        };
    }
    broadcast_binary_op(a, b, out, |x, y| x * y)
}
pub fn sub<'b, 'a, T: Clone + Copy + std::ops::Sub<Output = T> + std::fmt::Debug>(
    a: &TensorView<'b, T>,
    b: &TensorView<'b, T>,
    out: &'a mut Vec<T>,
) -> TensorView<'a, T> {
    if a.data.len() == b.data.len() && a.shape == b.shape {
        let len = a.data.len();
        utils::ensure_capacity(out, len);
        let a_slice = &a.data;
        let b_slice = &b.data;
        let o_slice = out.as_mut_slice();
        for i in 0..len {
            unsafe {
                *o_slice.get_unchecked_mut(i) =
                    *a_slice.get_unchecked(i) - *b_slice.get_unchecked(i);
            }
        }
        return TensorView {
            data: Cow::Borrowed(out),
            shape: std::borrow::Cow::Owned(a.shape.to_vec()),
        };
    }
    broadcast_binary_op(a, b, out, |x, y| x - y)
}
pub fn reciprocal<'b, 'a>(input: &TensorView<'b>, out: &'a mut Vec<f32>) -> TensorView<'a> {
    let numel = input.data.len();
    utils::ensure_capacity(out, numel);
    for i in 0..numel {
        out[i] = 1.0 / input.data[i];
    }
    TensorView {
        data: Cow::Borrowed(out),
        shape: Cow::Owned(input.shape.to_vec()),
    }
}
pub fn erf<'b, 'a>(input: &TensorView<'b>, out: &'a mut Vec<f32>) -> TensorView<'a> {
    let numel = input.data.len();
    utils::ensure_capacity(out, numel);
    for i in 0..numel {
        out[i] = libm::erff(input.data[i]);
    }
    TensorView {
        data: Cow::Borrowed(out),
        shape: Cow::Owned(input.shape.to_vec()),
    }
}
pub fn softplus<'b, 'a>(input: &TensorView<'b>, out: &'a mut Vec<f32>) -> TensorView<'a> {
    let numel = input.data.len();
    utils::ensure_capacity(out, numel);
    for i in 0..numel {
        let x = input.data[i];
        out[i] = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
    }
    TensorView {
        data: Cow::Borrowed(out),
        shape: Cow::Owned(input.shape.to_vec()),
    }
}
pub fn tanh_kernel<'b, 'a>(input: &TensorView<'b>, out: &'a mut Vec<f32>) -> TensorView<'a> {
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    {
        crate::kernels::neon::math::tanh(input, out)
    }
    #[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
    {
        let numel = input.data.len();
        utils::ensure_capacity(out, numel);
        for i in 0..numel {
            out[i] = input.data[i].tanh();
        }
        TensorView {
            data: Cow::Borrowed(out),
            shape: Cow::Owned(input.shape.to_vec()),
        }
    }
}
pub fn div<'b, 'a, T: Clone + Copy + std::ops::Div<Output = T> + std::fmt::Debug>(
    a: &TensorView<'b, T>,
    b: &TensorView<'b, T>,
    out: &'a mut Vec<T>,
) -> TensorView<'a, T> {
    if a.data.len() == b.data.len() && a.shape == b.shape {
        let len = a.data.len();
        utils::ensure_capacity(out, len);
        let a_slice = &a.data;
        let b_slice = &b.data;
        let o_slice = out.as_mut_slice();
        for i in 0..len {
            unsafe {
                *o_slice.get_unchecked_mut(i) =
                    *a_slice.get_unchecked(i) / *b_slice.get_unchecked(i);
            }
        }
        return TensorView {
            data: Cow::Borrowed(out),
            shape: std::borrow::Cow::Owned(a.shape.to_vec()),
        };
    }
    broadcast_binary_op(a, b, out, |x, y| x / y)
}
pub fn equal<'b, 'a>(
    a: &TensorView<'b>,
    b: &TensorView<'b>,
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    broadcast_binary_op(a, b, out, |x, y| if x == y { 1.0 } else { 0.0 })
}

pub fn equal_i64<'b, 'a, T: ElementOps>(
    a: &TensorView<'b, T>,
    b: &TensorView<'b, T>,
    out: &'a mut Vec<i64>,
) -> TensorView<'a, i64> {
    broadcast_binary_op_to(a, b, out, |x, y| if x == y { 1 } else { 0 })
}
pub fn sigmoid<'b, 'a>(input: &TensorView<'b>, out: &'a mut Vec<f32>) -> TensorView<'a> {
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    {
        crate::kernels::neon::math::sigmoid(input, out)
    }
    #[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
    {
        let len = input.data.len();
        utils::ensure_capacity(out, len);
        let i_slice = &input.data;
        let o_slice = out.as_mut_slice();
        for i in 0..len {
            unsafe {
                use super::activations;

                *o_slice.get_unchecked_mut(i) = activations::sigmoid(*i_slice.get_unchecked(i));
            }
        }
        TensorView {
            data: Cow::Borrowed(out),
            shape: std::borrow::Cow::Owned(input.shape.to_vec()),
        }
    }
}
pub fn relu<'a, 'b>(input: &TensorView<'b>, out: &'a mut Vec<f32>) -> TensorView<'a> {
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    {
        crate::kernels::neon::math::relu(input, out)
    }
    #[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
    {
        let len = input.data.len();
        utils::ensure_capacity(out, len);
        let i_slice = &input.data;
        let o_slice = out.as_mut_slice();
        for i in 0..len {
            unsafe {
                *o_slice.get_unchecked_mut(i) = i_slice.get_unchecked(i).max(0.0);
            }
        }
        TensorView {
            data: Cow::Borrowed(out),
            shape: Cow::Owned(input.shape.to_vec()),
        }
    }
}
pub fn sqrt<'b, 'a>(input: &TensorView<'b>, out: &'a mut Vec<f32>) -> TensorView<'a> {
    let len = input.data.len();
    utils::ensure_capacity(out, len);
    let in_slice = &input.data;
    let out_slice = out.as_mut_slice();
    for i in 0..len {
        unsafe {
            *out_slice.get_unchecked_mut(i) = in_slice.get_unchecked(i).sqrt();
        }
    }
    TensorView {
        data: Cow::Borrowed(out),
        shape: std::borrow::Cow::Owned(input.shape.to_vec()),
    }
}
pub fn pow<'b, 'a, T: ElementOps>(
    a: &TensorView<'b, T>,
    b: &TensorView<'b, T>,
    out: &'a mut Vec<T>,
) -> TensorView<'a, T> {
    if a.data.len() == b.data.len() && a.shape == b.shape {
        let len = a.data.len();
        utils::ensure_capacity(out, len);
        let a_slice = &a.data;
        let b_slice = &b.data;
        unsafe {
            out.set_len(len);
        }
        let o_slice = out.as_mut_slice();
        for i in 0..len {
            unsafe {
                *o_slice.get_unchecked_mut(i) =
                    a_slice.get_unchecked(i).pow(*b_slice.get_unchecked(i));
            }
        }
        return TensorView {
            data: Cow::Borrowed(out),
            shape: std::borrow::Cow::Owned(a.shape.to_vec()),
        };
    }
    broadcast_binary_op(a, b, out, |x, y| x.pow(y))
}
pub fn not<'b, 'a, T: ElementOps>(
    input: &TensorView<'b, T>,
    out: &'a mut Vec<T>,
) -> TensorView<'a, T> {
    let len = input.data.len();
    utils::ensure_capacity(out, len);
    let i_slice = &input.data;
    let o_slice = out.as_mut_slice();
    let zero = T::from_f32(0.0);
    let one = T::from_f32(1.0);
    for i in 0..len {
        let x = unsafe { *i_slice.get_unchecked(i) };
        unsafe { *o_slice.get_unchecked_mut(i) = if x == zero { one } else { zero } };
    }
    TensorView {
        data: Cow::Borrowed(out),
        shape: std::borrow::Cow::Owned(input.shape.to_vec()),
    }
}
pub fn reduce_mean<'b, 'a>(
    input: &TensorView<'b>,
    axes: &[i64],
    keepdims: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let dims = input.dim();
    let mut resolved_axes: Vec<usize> = axes
        .iter()
        .map(|&x| {
            if x < 0 {
                (dims as i64 + x) as usize
            } else {
                x as usize
            }
        })
        .collect();
    resolved_axes.sort();
    resolved_axes.dedup();
    let mut out_shape = Vec::new();
    let mut reduce_mask = vec![false; dims];
    for &ax in &resolved_axes {
        reduce_mask[ax] = true;
    }
    for i in 0..dims {
        if !reduce_mask[i] {
            out_shape.push(input.shape[i]);
        } else if keepdims {
            out_shape.push(1);
        }
    }
    let out_numel = out_shape.iter().product::<usize>();
    if out.len() != out_numel {
        out.resize(out_numel, 0.0);
    }
    out.fill(0.0);
    let real_out_strides = utils::compute_strides(&out_shape);
    let mut input_to_out_strides = vec![0; dims];
    let mut out_dim_idx = 0;
    for i in 0..dims {
        if reduce_mask[i] {
            input_to_out_strides[i] = 0;
            if keepdims {
                out_dim_idx += 1;
            }
        } else {
            input_to_out_strides[i] = real_out_strides[out_dim_idx];
            out_dim_idx += 1;
        }
    }
    let mut coords = vec![0; dims];
    let total_elems = input.data.len();
    let i_slice = &input.data;
    let o_slice = out.as_mut_slice();
    for i in 0..total_elems {
        let val = unsafe { *i_slice.get_unchecked(i) };
        let mut out_off = 0;
        for d in 0..dims {
            out_off += coords[d] * input_to_out_strides[d];
        }
        unsafe {
            *o_slice.get_unchecked_mut(out_off) += val;
        }
        for d in (0..dims).rev() {
            coords[d] += 1;
            if coords[d] < input.shape[d] {
                break;
            }
            coords[d] = 0;
        }
    }
    let mut count = 1;
    for &ax in &resolved_axes {
        count *= input.shape[ax];
    }
        let scale = 1.0 / count as f32;
        for x in out.iter_mut() {
            *x *= scale;
        }
    
        TensorView {
            data: Cow::Borrowed(out),
            shape: Cow::Owned(out_shape),
        }
    }
    
    pub fn reduce_sum<'b, 'a>(
    input: &TensorView<'b>,
    axes: &[i64],
    keepdims: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let dims = input.dim();
    let mut resolved_axes: Vec<usize> = axes
        .iter()
        .map(|&x| {
            if x < 0 {
                (dims as i64 + x) as usize
            } else {
                x as usize
            }
        })
        .collect();
    resolved_axes.sort();
    resolved_axes.dedup();
    let mut out_shape = Vec::new();
    let mut reduce_mask = vec![false; dims];
    for &ax in &resolved_axes {
        reduce_mask[ax] = true;
    }
    for i in 0..dims {
        if !reduce_mask[i] {
            out_shape.push(input.shape[i]);
        } else if keepdims {
            out_shape.push(1);
        }
    }
    let out_numel = out_shape.iter().product::<usize>();
    if out.len() != out_numel {
        out.resize(out_numel, 0.0);
    }
    out.fill(0.0);
    let real_out_strides = utils::compute_strides(&out_shape);
    let mut input_to_out_strides = vec![0; dims];
    let mut out_dim_idx = 0;
    for i in 0..dims {
        if reduce_mask[i] {
            input_to_out_strides[i] = 0;
            if keepdims {
                out_dim_idx += 1;
            }
        } else {
            input_to_out_strides[i] = real_out_strides[out_dim_idx];
            out_dim_idx += 1;
        }
    }
    let mut coords = vec![0; dims];
    let total_elems = input.data.len();
    let i_slice = &input.data;
    let o_slice = out.as_mut_slice();
    for i in 0..total_elems {
        let val = unsafe { *i_slice.get_unchecked(i) };
        let mut out_off = 0;
        for d in 0..dims {
            out_off += coords[d] * input_to_out_strides[d];
        }
        unsafe {
            *o_slice.get_unchecked_mut(out_off) += val;
        }
        for d in (0..dims).rev() {
            coords[d] += 1;
            if coords[d] < input.shape[d] {
                break;
            }
            coords[d] = 0;
        }
    }
    TensorView {
        data: Cow::Borrowed(out),
        shape: Cow::Owned(out_shape),
    }
}

pub fn reduce_max<'b, 'a, T>(
    input: &TensorView<'b, T>,
    axes: &[i64],
    keepdims: bool,
    out: &'a mut Vec<T>,
) -> TensorView<'a, T>
where
    T: ElementOps,
{
    let dims = input.dim();
    let mut resolved_axes: Vec<usize> = axes
        .iter()
        .map(|&x| {
            if x < 0 {
                (dims as i64 + x) as usize
            } else {
                x as usize
            }
        })
        .collect();
    resolved_axes.sort();
    resolved_axes.dedup();
    let mut out_shape = Vec::new();
    let mut reduce_mask = vec![false; dims];
    for &ax in &resolved_axes {
        reduce_mask[ax] = true;
    }
    for i in 0..dims {
        if !reduce_mask[i] {
            out_shape.push(input.shape[i]);
        } else if keepdims {
            out_shape.push(1);
        }
    }
    let out_numel = out_shape.iter().product::<usize>();
    if out.len() != out_numel {
        out.resize(out_numel, T::constant_min());
    }
    out.fill(T::constant_min());
    let real_out_strides = utils::compute_strides(&out_shape);
    let mut input_to_out_strides = vec![0; dims];
    let mut out_dim_idx = 0;
    for i in 0..dims {
        if reduce_mask[i] {
            input_to_out_strides[i] = 0;
            if keepdims {
                out_dim_idx += 1;
            }
        } else {
            input_to_out_strides[i] = real_out_strides[out_dim_idx];
            out_dim_idx += 1;
        }
    }
    let mut coords = vec![0; dims];
    let total_elems = input.data.len();
    let i_slice = &input.data;
    let o_slice = out.as_mut_slice();
    for i in 0..total_elems {
        let val = unsafe { *i_slice.get_unchecked(i) };
        let mut out_off = 0;
        for d in 0..dims {
            out_off += coords[d] * input_to_out_strides[d];
        }
        unsafe {
            let p = o_slice.get_unchecked_mut(out_off);
            if val > *p {
                *p = val;
            }
        }
        for d in (0..dims).rev() {
            coords[d] += 1;
            if coords[d] < input.shape[d] {
                break;
            }
            coords[d] = 0;
        }
    }
    TensorView {
        data: Cow::Borrowed(out),
        shape: Cow::Owned(out_shape),
    }
}

pub fn clip<'b, 'a, T: ElementOps, U: ElementOps, V: ElementOps>(
    input: &TensorView<'b, T>,
    min: Option<&TensorView<U>>,
    max: Option<&TensorView<U>>,
    out: &'a mut Vec<V>,
) -> TensorView<'a, V> {
    let min_val = min
        .and_then(|t| t.data.first().cloned())
        .map(|v| T::from_f32(v.as_f32()))
        .unwrap_or(T::constant_min());
    let max_val = max
        .and_then(|t| t.data.first().cloned())
        .map(|v| T::from_f32(v.as_f32()))
        .unwrap_or(T::constant_max());
    let numel = input.data.len();
    utils::ensure_capacity(out, numel);
    unsafe {
        out.set_len(numel);
    }
    for i in 0..numel {
        let val = input.data[i].clamp_val(min_val, max_val);
        out[i] = V::from_f32(val.as_f32());
    }
    TensorView {
        data: Cow::Borrowed(out),
        shape: Cow::Owned(input.shape.to_vec()),
    }
}
pub fn prelu<'b, 'a>(
    input: &TensorView<'b>,
    slope: &TensorView<'b>,
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    if slope.data.len() == 1 {
        let s = slope.data[0];
        let numel = input.data.len();
        utils::ensure_capacity(out, numel);
        for i in 0..numel {
            let x = input.data[i];
            out[i] = if x < 0.0 { x * s } else { x };
        }
        return TensorView {
            data: Cow::Borrowed(out),
            shape: Cow::Owned(input.shape.to_vec()),
        };
    }
    broadcast_binary_op(input, slope, out, |x, s| if x < 0.0 { x * s } else { x })
}
pub fn range<'a>(
    start: &TensorView,
    limit: &TensorView,
    delta: &TensorView,
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let start_val = start.data.first().copied().unwrap_or(0.0);
    let limit_val = limit.data.first().copied().unwrap_or(0.0);
    let delta_val = delta.data.first().copied().unwrap_or(1.0);
    let n = if delta_val != 0.0 {
        ((limit_val - start_val) / delta_val).ceil().max(0.0) as usize
    } else {
        0
    };
    utils::ensure_capacity(out, n);
    unsafe {
        out.set_len(n);
    }
    let out_slice = out.as_mut_slice();
    for i in 0..n {
        out_slice[i] = start_val + (i as f32) * delta_val;
    }
    TensorView::from_slice(out, vec![n])
}
pub fn range_i64<'a>(
    start: &TensorView<'a, i64>,
    limit: &TensorView<'a, i64>,
    delta: &TensorView<'a, i64>,
    out: &'a mut Vec<i64>,
) -> TensorView<'a, i64> {
    let start_val = start.data.first().copied().unwrap_or(0);
    let limit_val = limit.data.first().copied().unwrap_or(0);
    let delta_val = delta.data.first().copied().unwrap_or(1);
    let n = if delta_val != 0 {
        let diff = (limit_val - start_val) as f64;
        let step = delta_val as f64;
        let n = (diff / step).ceil();
        if n > 0.0 { n as usize } else { 0 }
    } else {
        0
    };
    utils::ensure_capacity(out, n);
    unsafe {
        out.set_len(n);
    }
    let out_slice = out.as_mut_slice();
    for i in 0..n {
        out_slice[i] = start_val + (i as i64) * delta_val;
    }
    TensorView::from_slice(out, vec![n])
}
pub fn sin<'b, 'a>(input: &TensorView<'b>, out: &'a mut Vec<f32>) -> TensorView<'a> {
    let numel = input.data.len();
    utils::ensure_capacity(out, numel);
    unsafe {
        out.set_len(numel);
    }
    let out_slice = out.as_mut_slice();
    for i in 0..numel {
        out_slice[i] = input.data[i].sin();
    }
    TensorView::from_slice(out, input.shape.to_vec())
}
pub fn cos<'b, 'a>(input: &TensorView<'b>, out: &'a mut Vec<f32>) -> TensorView<'a> {
    let numel = input.data.len();
    utils::ensure_capacity(out, numel);
    unsafe {
        out.set_len(numel);
    }
    let out_slice = out.as_mut_slice();
    for i in 0..numel {
        out_slice[i] = input.data[i].cos();
    }
    TensorView::from_slice(out, input.shape.to_vec())
}
pub fn exp<'b, 'a>(input: &TensorView<'b>, out: &'a mut Vec<f32>) -> TensorView<'a> {
    let numel = input.data.len();
    utils::ensure_capacity(out, numel);
    unsafe {
        out.set_len(numel);
    }
    let out_slice = out.as_mut_slice();
    for i in 0..numel {
        out_slice[i] = input.data[i].exp();
    }
    TensorView::from_slice(out, input.shape.to_vec())
}
pub fn neg<'b, 'a>(input: &TensorView<'b>, out: &'a mut Vec<f32>) -> TensorView<'a> {
    let numel = input.data.len();
    utils::ensure_capacity(out, numel);
    unsafe {
        out.set_len(numel);
    }
    let out_slice = out.as_mut_slice();
    for i in 0..numel {
        out_slice[i] = -input.data[i];
    }
    TensorView::from_slice(out, input.shape.to_vec())
}
pub fn less<'b, 'a>(
    a: &TensorView<'b>,
    b: &TensorView<'b>,
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    broadcast_binary_op(a, b, out, |x, y| if x < y { 1.0 } else { 0.0 })
}
pub fn less_i64<'b, 'a, T: ElementOps>(
    a: &TensorView<'b, T>,
    b: &TensorView<'b, T>,
    out: &'a mut Vec<i64>,
) -> TensorView<'a, i64> {
    broadcast_binary_op_to(a, b, out, |x, y| if x < y { 1 } else { 0 })
}
pub fn expand<'b, 'a, T: Clone + Copy + std::fmt::Debug>(
    input: &TensorView<'b, T>,
    shape: &[i64],
    out: &'a mut Vec<T>,
) -> TensorView<'a, T> {
    let target_shape_vec: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
    let input_shape = &input.shape;
    let ndim_in = input_shape.len();
    let ndim_target = target_shape_vec.len();
    let ndim_out = std::cmp::max(ndim_in, ndim_target);
    let mut out_shape = vec![0; ndim_out];
    for i in 0..ndim_out {
        let offset_in = ndim_out - ndim_in;
        let dim_in = if i >= offset_in {
            input_shape[i - offset_in]
        } else {
            1
        };
        let offset_target = ndim_out - ndim_target;
        let dim_target = if i >= offset_target {
            target_shape_vec[i - offset_target]
        } else {
            1
        };
        if dim_in == dim_target {
            out_shape[i] = dim_in;
        } else if dim_in == 1 {
            out_shape[i] = dim_target;
        } else if dim_target == 1 {
            out_shape[i] = dim_in;
        } else {
            panic!(
                "Expand: incompatible dimensions at dim index {} (from left): in={} target={}. Full shapes: in={:?} target={:?}",
                i, dim_in, dim_target, input_shape, target_shape_vec
            );
        }
    }
    let numel: usize = out_shape.iter().product();
    utils::ensure_capacity(out, numel);
    unsafe {
        out.set_len(numel);
    }
    let input_strides = utils::compute_strides(input_shape);
    let mut virtual_strides = vec![0; ndim_out];
    for i in 0..ndim_out {
        let offset_in = ndim_out - ndim_in;
        if i < offset_in {
            virtual_strides[i] = 0;
        } else {
            let idx_in = i - offset_in;
            if input_shape[idx_in] == 1 {
                virtual_strides[i] = 0;
            } else {
                virtual_strides[i] = input_strides[idx_in];
            }
        }
    }
    let out_slice = out.as_mut_slice();
    let mut coords = vec![0; ndim_out];
    for i in 0..numel {
        let mut in_idx = 0;
        for d in 0..ndim_out {
            in_idx += coords[d] * virtual_strides[d];
        }
        out_slice[i] = input.data[in_idx];
        for d in (0..ndim_out).rev() {
            coords[d] += 1;
            if coords[d] < out_shape[d] {
                break;
            }
            coords[d] = 0;
        }
    }
    TensorView::from_slice(out, out_shape)
}
pub fn tile<'b, 'a, T: Clone + Copy + std::fmt::Debug>(
    input: &TensorView<'b, T>,
    repeats: &[i64],
    out: &'a mut Vec<T>,
) -> TensorView<'a, T> {
    let repeats_vec: Vec<usize> = repeats.iter().map(|&x| x as usize).collect();
    let ndim = input.shape.len();
    assert_eq!(
        repeats_vec.len(),
        ndim,
        "Tile: repeats length must match input rank"
    );
    let out_shape: Vec<usize> = input
        .shape
        .iter()
        .zip(&repeats_vec)
        .map(|(&dim, &rep)| dim * rep)
        .collect();
    let numel = out_shape.iter().product();
    utils::ensure_capacity(out, numel);
    unsafe {
        out.set_len(numel);
    }
    let out_slice = out.as_mut_slice();
    let mut coords = vec![0; ndim];
    for i in 0..numel {
        let mut in_idx = 0;
        let mut in_coords = vec![0; ndim];
        for d in 0..ndim {
            in_coords[d] = coords[d] % input.shape[d];
        }
        let input_strides = utils::compute_strides(&input.shape);
        for d in 0..ndim {
            in_idx += in_coords[d] * input_strides[d];
        }
        out_slice[i] = input.data[in_idx];
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
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorView;
    #[test]
    fn test_expand() {
        let input_data = vec![1.0, 2.0, 3.0];
        let shape_data = [1, 2, 3];
        let input = TensorView::from_slice(&input_data, vec![1, 1, 3]);
        let mut out = Vec::new();
        let res = expand(&input, &shape_data, &mut out);
        assert_eq!(res.shape, vec![1, 2, 3]);
        assert_eq!(res.data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }
    #[test]
    fn test_add_broadcast() {
        let a_data = vec![10.0, 20.0, 30.0];
        let a = TensorView::from_slice(&a_data, vec![1, 3]);
        let b_data = vec![1.0, 2.0];
        let b = TensorView::from_slice(&b_data, vec![2, 1]);
        let mut out = Vec::new();
        let res = add(&a, &b, &mut out);
        assert_eq!(res.shape, vec![2, 3]);
        assert_eq!(res.data, vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0]);
    }
    #[test]
    fn test_min_max() {
        let data = vec![1.0, -5.0, 10.0, 3.0];
        let t = TensorView::from_slice(&data, vec![4]);
        let mut buf = Vec::new();
        let (min_val, max_val) = min_max(&t, &mut buf);
        assert_eq!(min_val, -5.0);
        assert_eq!(max_val, 10.0);
    }
}
