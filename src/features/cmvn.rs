use crate::tensor::TensorView;
pub struct Cmvn {
    eps: f32,
}
impl Default for Cmvn {
    fn default() -> Self {
        Self { eps: 1e-5 }
    }
}
impl Cmvn {
    pub fn new(eps: f32) -> Self {
        Self { eps }
    }
    pub fn compute(&self, input: &TensorView) -> TensorView<'static> {
        let shape = input.shape.as_ref();
        let (_batch_offset, t, d) = if shape.len() == 2 {
            (0, shape[0], shape[1])
        } else if shape.len() == 3 && shape[0] == 1 {
            (0, shape[1], shape[2])
        } else {
            panic!("CMVN expects [T, D] or [1, T, D] input, got {:?}", shape);
        };
        if t == 0 {
            return input.to_owned();
        }

        let input_data = &input.data;
        let mut sums = vec![0.0; d];
        let mut sq_sums = vec![0.0; d];

        // First pass: accumulate sums and squared sums
        for t_idx in 0..t {
            let offset = t_idx * d;
            let row = &input_data[offset..offset + d];
            for dim_idx in 0..d {
                let val = row[dim_idx];
                sums[dim_idx] += val;
                sq_sums[dim_idx] += val * val;
            }
        }

        let mut means = vec![0.0; d];
        let mut stds = vec![0.0; d];
        let t_f32 = t as f32;

        for dim_idx in 0..d {
            let mean = sums[dim_idx] / t_f32;
            let variance = (sq_sums[dim_idx] / t_f32 - mean * mean).max(0.0);
            let std = (variance + self.eps).sqrt();
            means[dim_idx] = mean;
            stds[dim_idx] = std;
        }

        // Second pass: normalize
        let mut output_data = vec![0.0; t * d];
        for t_idx in 0..t {
            let offset = t_idx * d;
            let row_in = &input_data[offset..offset + d];
            let row_out = &mut output_data[offset..offset + d];
            for dim_idx in 0..d {
                row_out[dim_idx] = (row_in[dim_idx] - means[dim_idx]) / stds[dim_idx];
            }
        }

        TensorView::from_owned(output_data, shape.to_vec())
    }
    pub fn apply_with_stats(
        &self,
        input: &TensorView,
        mean: &[f32],
        std: &[f32],
    ) -> TensorView<'static> {
        let shape = input.shape.as_ref();
        let (t, d) = if shape.len() == 2 {
            (shape[0], shape[1])
        } else if shape.len() == 3 && shape[0] == 1 {
            (shape[1], shape[2])
        } else {
            panic!("CMVN expects [T, D] or [1, T, D] input");
        };
        assert_eq!(mean.len(), d, "Mean dimension mismatch");
        assert_eq!(std.len(), d, "Std dimension mismatch");
        let input_data = &input.data;
        let mut output_data = vec![0.0; t * d];
        for t_idx in 0..t {
            for d_idx in 0..d {
                let idx = t_idx * d + d_idx;
                output_data[idx] = (input_data[idx] - mean[d_idx]) / (std[d_idx] + self.eps);
            }
        }
        TensorView::from_owned(output_data, shape.to_vec())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_cmvn_basic() {
        let data = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
        let input = TensorView::from_owned(data, vec![3, 2]);
        let cmvn = Cmvn::default();
        let output = cmvn.compute(&input);
        let out_data = output.data.as_ref();
        let col0_mean = (out_data[0] + out_data[2] + out_data[4]) / 3.0;
        assert!((col0_mean).abs() < 1e-5, "Mean should be near 0");
        assert!(out_data[0] < 0.0);
        assert!(out_data[2].abs() < 1e-5);
        assert!(out_data[4] > 0.0);
    }
}
