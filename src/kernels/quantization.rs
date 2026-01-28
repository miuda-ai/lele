use crate::kernels::matmul;
use crate::tensor::TensorView;
use std::borrow::Cow;

pub fn mat_mul_integer<'a, 'b, 'c>(
    a: &TensorView<'b>,
    b: &TensorView<'c>,
    a_zero_point: Option<&TensorView<'b>>,
    b_zero_point: Option<&TensorView<'c>>,
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let a_adjusted_data: Option<Vec<f32>> = if let Some(zp) = a_zero_point {
        let zp_val = zp.data[0];
        if zp_val != 0.0 {
            Some(a.data.iter().map(|&x| x - zp_val).collect())
        } else {
            None
        }
    } else {
        None
    };
    let b_adjusted_data: Option<Vec<f32>> = if let Some(zp) = b_zero_point {
        let zp_val = zp.data[0];
        if zp_val != 0.0 {
            Some(b.data.iter().map(|&x| x - zp_val).collect())
        } else {
            None
        }
    } else {
        None
    };
    let a_view = if let Some(d) = &a_adjusted_data {
        TensorView {
            data: Cow::Borrowed(d),
            shape: a.shape.clone(),
        }
    } else {
        TensorView {
            data: Cow::Borrowed(&a.data),
            shape: a.shape.clone(),
        }
    };
    let b_view = if let Some(d) = &b_adjusted_data {
        TensorView {
            data: Cow::Borrowed(d),
            shape: b.shape.clone(),
        }
    } else {
        TensorView {
            data: Cow::Borrowed(&b.data),
            shape: b.shape.clone(),
        }
    };
    matmul(&a_view, &b_view, out)
}

pub fn dynamic_quantize_linear<'a, 'b>(
    x: &TensorView<'b>,
    out_y: &'a mut Vec<f32>,
    out_scale: &'a mut Vec<f32>,
    out_zp: &'a mut Vec<f32>,
) -> (TensorView<'a>, TensorView<'a>, TensorView<'a>) {
    #[cfg(target_arch = "aarch64")]
    {
        crate::kernels::neon::quantization::dynamic_quantize_linear(x, out_y, out_scale, out_zp)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let len = x.data.len();
        if len == 0 {
            return (
                TensorView {
                    data: Cow::Borrowed(out_y),
                    shape: Cow::Owned(x.shape.to_vec()),
                },
                TensorView {
                    data: Cow::Borrowed(out_scale),
                    shape: Cow::Owned(vec![1]),
                },
                TensorView {
                    data: Cow::Borrowed(out_zp),
                    shape: Cow::Owned(vec![1]),
                },
            );
        }
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        for &v in x.data.iter() {
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }
        let adjusted_max = max_val.max(0.0);
        let adjusted_min = min_val.min(0.0);
        let range = (adjusted_max - adjusted_min).max(1e-5);
        let scale = range / 255.0;
        let zp = (-adjusted_min / scale).round().clamp(0.0, 255.0);
        utils::ensure_capacity(out_scale, 1);
        unsafe {
            out_scale.set_len(1);
        }
        out_scale[0] = scale;
        utils::ensure_capacity(out_zp, 1);
        unsafe {
            out_zp.set_len(1);
        }
        out_zp[0] = zp;
        utils::ensure_capacity(out_y, len);
        unsafe {
            out_y.set_len(len);
        }
        let inv_scale = 1.0 / scale;
        let x_data = &x.data;
        let y_data = out_y.as_mut_slice();
        for i in 0..len {
            y_data[i] = (x_data[i] * inv_scale + zp).round().clamp(0.0, 255.0);
        }
        (
            TensorView {
                data: Cow::Borrowed(out_y),
                shape: Cow::Owned(x.shape.to_vec()),
            },
            TensorView {
                data: Cow::Borrowed(out_scale),
                shape: Cow::Owned(vec![1]),
            },
            TensorView {
                data: Cow::Borrowed(out_zp),
                shape: Cow::Owned(vec![1]),
            },
        )
    }
}
