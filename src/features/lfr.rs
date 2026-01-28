use crate::tensor::TensorView;
pub struct LfrConfig {
    pub m: usize,
    pub n: usize,
}
impl Default for LfrConfig {
    fn default() -> Self {
        Self { m: 7, n: 6 }
    }
}
pub struct Lfr {
    config: LfrConfig,
}
impl Lfr {
    pub fn new(config: LfrConfig) -> Self {
        Self { config }
    }
    pub fn compute(&self, input: &TensorView) -> TensorView<'static> {
        let (t, d) = if input.dim() == 2 {
            (input.size(0), input.size(1))
        } else if input.dim() == 3 && input.size(0) == 1 {
            (input.size(1), input.size(2))
        } else {
            panic!(
                "LFR expects [T, D] or [1, T, D] input, got {:?}",
                input.shape
            );
        };
        let m = self.config.m;
        let n = self.config.n;
        if t == 0 {
            return TensorView::from_owned(vec![], vec![0, d * m]);
        }
        let t_lfr = t.div_ceil(n);
        let d_out = d * m;
        let mut output_data = vec![0.0; t_lfr * d_out];
        let pad = (m - 1) / 2;
        let input_slice = &input.data;
        for i in 0..t_lfr {
            let current_out_row_start = i * d_out;
            let start_frame = i * n;
            for block in 0..m {
                let raw_idx = (start_frame + block) as isize - pad as isize;
                let clamped_idx = raw_idx.max(0).min((t - 1) as isize) as usize;
                let src_start = clamped_idx * d;
                let src_end = src_start + d;
                let src_row = &input_slice[src_start..src_end];
                let dst_start = current_out_row_start + block * d;
                let dst_end = dst_start + d;
                output_data[dst_start..dst_end].copy_from_slice(src_row);
            }
        }
        TensorView::from_owned(output_data, vec![t_lfr, d_out])
    }
}
