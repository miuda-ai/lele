pub use half::{bf16, f16};
use std::borrow::Cow;

#[derive(Debug, Clone)]
pub struct TensorView<'a, T = f32>
where
    T: Clone + std::fmt::Debug,
    [T]: ToOwned<Owned = Vec<T>>,
{
    pub data: Cow<'a, [T]>,
    pub shape: Cow<'a, [usize]>,
}

pub type TensorViewF32<'a> = TensorView<'a, f32>;
pub type TensorViewI8<'a> = TensorView<'a, i8>;
pub type TensorViewU8<'a> = TensorView<'a, u8>;
pub type TensorViewI32<'a> = TensorView<'a, i32>;
pub type TensorViewI64<'a> = TensorView<'a, i64>;
pub type TensorViewF16<'a> = TensorView<'a, f16>;
pub type TensorViewBF16<'a> = TensorView<'a, bf16>;

impl<'a, T> TensorView<'a, T>
where
    T: Clone + std::fmt::Debug,
    [T]: ToOwned<Owned = Vec<T>>,
{
    pub fn new(data: &'a [T], shape: &'a [usize]) -> Self {
        let len: usize = shape.iter().product();
        assert_eq!(data.len(), len, "Data length mismatch");
        Self {
            data: Cow::Borrowed(data),
            shape: Cow::Borrowed(shape),
        }
    }

    pub fn from_owned(data: Vec<T>, shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        assert_eq!(data.len(), len, "Data length mismatch");
        Self {
            data: Cow::Owned(data),
            shape: Cow::Owned(shape),
        }
    }

    pub fn to_owned(&self) -> TensorView<'static, T> {
        TensorView::from_owned(self.data.to_vec(), self.shape.to_vec())
    }

    pub fn empty() -> Self {
        Self {
            data: Cow::Borrowed(&[]),
            shape: Cow::Borrowed(&[]),
        }
    }

    pub fn dim(&self) -> usize {
        self.shape.len()
    }

    pub fn size(&self, dim: usize) -> usize {
        self.shape[dim]
    }

    pub fn from_slice(data: &'a [T], shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        assert_eq!(data.len(), len, "Data length mismatch");
        Self {
            data: Cow::Borrowed(data),
            shape: Cow::Owned(shape),
        }
    }

    /// # Safety
    /// This function is unsafe because it bypasses the lifetime system to create a new `TensorView`
    /// that might outlive the data it points to.
    pub unsafe fn detach<'b>(&self) -> TensorView<'b, T> {
        unsafe {
            let slice = std::slice::from_raw_parts(self.data.as_ptr(), self.data.len());
            let shape_slice = std::slice::from_raw_parts(self.shape.as_ptr(), self.shape.len());
            TensorView {
                data: Cow::Borrowed(slice),
                shape: Cow::Borrowed(shape_slice),
            }
        }
    }
}

impl<'a> TensorView<'a, f32> {
    /// # Safety
    /// Reinterprets this TensorView<f32> as TensorView<u8>. This is only safe when the
    /// f32 values represent u8 data (e.g., from quantization operations).
    pub unsafe fn reinterpret_as_u8(&self) -> TensorView<'a, u8> {
        // The f32 vector actually contains u8 values stored as f32
        // We need to convert them back
        let u8_vec: Vec<u8> = self.data.iter().map(|&x| x as u8).collect();
        TensorView::from_owned(u8_vec, self.shape.to_vec())
    }
}

pub trait IntoLogits<'a, T>
where
    T: Clone + std::fmt::Debug,
    [T]: ToOwned<Owned = Vec<T>>,
{
    fn into_logits(self) -> TensorView<'a, T>;
}

impl<'a, T> IntoLogits<'a, T> for TensorView<'a, T>
where
    T: Clone + std::fmt::Debug,
    [T]: ToOwned<Owned = Vec<T>>,
{
    fn into_logits(self) -> TensorView<'a, T> {
        self
    }
}

impl<'a, T, U> IntoLogits<'a, T> for (TensorView<'a, T>, U)
where
    T: Clone + std::fmt::Debug,
    [T]: ToOwned<Owned = Vec<T>>,
{
    fn into_logits(self) -> TensorView<'a, T> {
        self.0
    }
}

/// Helper functions for creating TensorView from byte slices
impl<'a> TensorView<'a, f32> {
    /// Create a TensorView<f32> from a byte slice, handling alignment
    pub fn from_bytes_f32(bytes: &'a [u8], shape: &'a [usize]) -> Self {
        // Ensure 4-byte alignment for f32
        if bytes.as_ptr() as usize % 4 == 0 {
            let f32_slice = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4)
            };
            TensorView::new(f32_slice, shape)
        } else {
            // Fallback for unaligned data
            let mut f32_vec = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                let bytes_arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                f32_vec.push(f32::from_le_bytes(bytes_arr));
            }
            unsafe { std::mem::transmute(TensorView::from_owned(f32_vec, shape.to_vec())) }
        }
    }

    /// Create a TensorView<f32> from u8 byte slice (cast each byte to f32)
    pub fn from_bytes_u8(bytes: &[u8], shape: Vec<usize>) -> TensorView<'static, f32> {
        use std::cell::RefCell;
        use std::collections::HashMap;
        thread_local! {
            static CACHE: RefCell<HashMap<usize, Vec<f32>>> = RefCell::new(HashMap::new());
        }
        CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let key = bytes.as_ptr() as usize;
            let vec = cache
                .entry(key)
                .or_insert_with(|| bytes.iter().map(|&x| x as f32).collect());
            let data: &'static [f32] =
                unsafe { std::slice::from_raw_parts(vec.as_ptr(), vec.len()) };
            TensorView::from_slice(data, shape)
        })
    }

    /// Create a TensorView<f32> from i8 byte slice (cast each byte to i8 then f32)
    pub fn from_bytes_i8(bytes: &[u8], shape: Vec<usize>) -> TensorView<'static, f32> {
        use std::cell::RefCell;
        use std::collections::HashMap;
        thread_local! {
            static CACHE: RefCell<HashMap<usize, Vec<f32>>> = RefCell::new(HashMap::new());
        }
        CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let key = bytes.as_ptr() as usize;
            let vec = cache
                .entry(key)
                .or_insert_with(|| bytes.iter().map(|&x| x as i8 as f32).collect());
            let data: &'static [f32] =
                unsafe { std::slice::from_raw_parts(vec.as_ptr(), vec.len()) };
            TensorView::from_slice(data, shape)
        })
    }

    /// Create a TensorView<f32> from f16 byte slice
    pub fn from_bytes_f16(bytes: &[u8], shape: Vec<usize>) -> TensorView<'static, f32> {
        let mut f16_vec = Vec::with_capacity(bytes.len() / 2);
        for chunk in bytes.chunks_exact(2) {
            let bytes_arr: [u8; 2] = [chunk[0], chunk[1]];
            f16_vec.push(f16::from_bits(u16::from_le_bytes(bytes_arr)));
        }
        let f32_vec: Vec<f32> = f16_vec.iter().map(|&x| x.to_f32()).collect();
        TensorView::from_owned(f32_vec, shape)
    }

    /// Create a TensorView<f32> from i64 byte slice (cast to f32)
    pub fn from_bytes_i64_as_f32(bytes: &[u8], shape: Vec<usize>) -> TensorView<'static, f32> {
        let mut i64_vec = Vec::with_capacity(bytes.len() / 8);
        for chunk in bytes.chunks_exact(8) {
            let bytes_arr: [u8; 8] = [
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ];
            i64_vec.push(i64::from_le_bytes(bytes_arr));
        }
        let f32_vec: Vec<f32> = i64_vec.iter().map(|&x| x as f32).collect();
        TensorView::from_owned(f32_vec, shape)
    }

    /// Create a TensorView<f32> from i32 byte slice (cast to f32)
    pub fn from_bytes_i32_as_f32(bytes: &[u8], shape: Vec<usize>) -> TensorView<'static, f32> {
        let mut i32_vec = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            let bytes_arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
            i32_vec.push(i32::from_le_bytes(bytes_arr));
        }
        let f32_vec: Vec<f32> = i32_vec.iter().map(|&x| x as f32).collect();
        TensorView::from_owned(f32_vec, shape)
    }
}

impl<'a> TensorView<'a, i64> {
    /// Create a TensorView<i64> from byte slice
    pub fn from_bytes_i64(bytes: &[u8], shape: Vec<usize>) -> TensorView<'static, i64> {
        let mut i64_vec = Vec::with_capacity(bytes.len() / 8);
        for chunk in bytes.chunks_exact(8) {
            let bytes_arr: [u8; 8] = [
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ];
            i64_vec.push(i64::from_le_bytes(bytes_arr));
        }
        TensorView::from_owned(i64_vec, shape)
    }

    /// Create a TensorView<i64> from i32 byte slice (cast to i64)
    pub fn from_bytes_i32_as_i64(bytes: &[u8], shape: Vec<usize>) -> TensorView<'static, i64> {
        let mut i64_vec = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            let bytes_arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
            i64_vec.push(i32::from_le_bytes(bytes_arr) as i64);
        }
        TensorView::from_owned(i64_vec, shape)
    }
}

impl<'a> TensorView<'a, i32> {
    /// Create a TensorView<i32> from byte slice
    pub fn from_bytes_i32(bytes: &[u8], shape: Vec<usize>) -> TensorView<'static, i32> {
        let mut i32_vec = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            let bytes_arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
            i32_vec.push(i32::from_le_bytes(bytes_arr));
        }
        TensorView::from_owned(i32_vec, shape)
    }
}
