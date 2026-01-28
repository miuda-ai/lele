use std::borrow::Cow;
#[derive(Debug, Clone)]
pub struct TensorView<'a> {
    pub data: Cow<'a, [f32]>,
    pub shape: Cow<'a, [usize]>,
}
impl<'a> TensorView<'a> {
    pub fn new(data: &'a [f32], shape: &'a [usize]) -> Self {
        let len: usize = shape.iter().product();
        assert_eq!(data.len(), len, "Data length mismatch");
        Self {
            data: Cow::Borrowed(data),
            shape: Cow::Borrowed(shape),
        }
    }
    pub fn from_owned(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        assert_eq!(data.len(), len, "Data length mismatch");
        Self {
            data: Cow::Owned(data),
            shape: Cow::Owned(shape),
        }
    }
    pub fn to_owned(&self) -> TensorView<'static> {
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
    pub fn from_slice(data: &'a [f32], shape: Vec<usize>) -> Self {
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
    pub unsafe fn detach<'b>(&self) -> TensorView<'b> {
        let slice = std::slice::from_raw_parts(self.data.as_ptr(), self.data.len());
        let shape_slice = std::slice::from_raw_parts(self.shape.as_ptr(), self.shape.len());
        TensorView {
            data: Cow::Borrowed(slice),
            shape: Cow::Borrowed(shape_slice),
        }
    }
}
