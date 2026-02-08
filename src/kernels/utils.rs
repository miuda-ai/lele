pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut s = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = s;
        s *= shape[i];
    }
    strides
}
pub fn ensure_capacity<T>(v: &mut Vec<T>, len: usize) {
    if v.capacity() < len {
        let extra = len - v.len();
        v.try_reserve(extra).ok();
    }
    unsafe {
        v.set_len(len);
    }
}
pub fn get_offset(indices: &[usize], strides: &[usize]) -> usize {
    indices
        .iter()
        .zip(strides)
        .map(|(&idx, &stride)| idx * stride)
        .sum()
}

pub trait AsI64 {
    fn as_i64(self) -> i64;
}
impl AsI64 for f32 {
    #[inline]
    fn as_i64(self) -> i64 {
        self as i64
    }
}
impl AsI64 for i64 {
    #[inline]
    fn as_i64(self) -> i64 {
        self
    }
}
impl AsI64 for i32 {
    #[inline]
    fn as_i64(self) -> i64 {
        self as i64
    }
}

pub trait AsF32 {
    fn as_f32(self) -> f32;
}
impl AsF32 for f32 {
    #[inline]
    fn as_f32(self) -> f32 {
        self
    }
}
impl AsF32 for i64 {
    #[inline]
    fn as_f32(self) -> f32 {
        self as f32
    }
}
impl AsF32 for i32 {
    #[inline]
    fn as_f32(self) -> f32 {
        self as f32
    }
}

pub fn cast_to_f32<'a, 'b, T>(
    input: &crate::tensor::TensorView<'a, T>,
    out: &'b mut Vec<f32>,
) -> crate::tensor::TensorView<'b, f32>
where
    T: Copy + AsF32 + std::fmt::Debug,
{
    ensure_capacity(out, input.data.len());
    for i in 0..input.data.len() {
        out[i] = input.data[i].as_f32();
    }
    crate::tensor::TensorView::from_slice(out, input.shape.to_vec())
}

pub fn cast_to_i64<'a, 'b, T>(
    input: &crate::tensor::TensorView<'a, T>,
    out: &'b mut Vec<i64>,
) -> crate::tensor::TensorView<'b, i64>
where
    T: Copy + AsI64 + std::fmt::Debug,
{
    ensure_capacity(out, input.data.len());
    for i in 0..input.data.len() {
        out[i] = input.data[i].as_i64();
    }
    crate::tensor::TensorView::from_slice(out, input.shape.to_vec())
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

pub fn get_parallelism(m: usize, n: usize, k: usize) -> faer::Par {
    if m * n * k > 100_000_000 {
        if let Some(n) = std::num::NonZeroUsize::new(rayon::current_num_threads()) {
            faer::Par::Rayon(n)
        } else {
            faer::Par::Seq
        }
    } else {
        faer::Par::Seq
    }
}

#[derive(Copy, Clone)]
pub struct SendPtr<T>(pub usize, std::marker::PhantomData<T>);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    pub fn new(ptr: *mut T) -> Self {
        Self(ptr as usize, std::marker::PhantomData)
    }
    pub fn as_ptr(self) -> *mut T {
        self.0 as *mut T
    }
}

#[derive(Copy, Clone)]
pub struct SendConstPtr<T>(pub usize, std::marker::PhantomData<T>);
unsafe impl<T> Send for SendConstPtr<T> {}
unsafe impl<T> Sync for SendConstPtr<T> {}

impl<T> SendConstPtr<T> {
    pub fn new(ptr: *const T) -> Self {
        Self(ptr as usize, std::marker::PhantomData)
    }
    pub fn as_ptr(self) -> *const T {
        self.0 as *const T
    }
}
