pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
pub fn tanh(x: f32) -> f32 {
    x.tanh()
}
pub fn hard_sigmoid(x: f32, alpha: f32, beta: f32) -> f32 {
    (alpha * x + beta).clamp(0.0, 1.0)
}
pub fn relu_scalar(x: f32) -> f32 {
    x.max(0.0)
}

pub fn gelu_tanh_kernel<'b, 'a>(
    input: &crate::tensor::TensorView<'b, f32>,
    out: &'a mut Vec<f32>,
) -> crate::tensor::TensorView<'a, f32> {
    use crate::tensor::TensorView;
    use std::borrow::Cow;
    let len = input.data.len();
    crate::kernels::utils::ensure_capacity(out, len);
    unsafe { out.set_len(len); }
    let c0 = 0.044715f32;
    let c1 = 0.7978845608028654f32;
    let x_slice = &input.data;
    let o_slice = out.as_mut_slice();

    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::*;
        let vc0 = unsafe { vdupq_n_f32(c0) };
        let vc1 = unsafe { vdupq_n_f32(c1) };
        let vhalf = unsafe { vdupq_n_f32(0.5) };
        let vone = unsafe { vdupq_n_f32(1.0) };
        let mut i = 0;
        unsafe {
            while i + 4 <= len {
                let xv = vld1q_f32(x_slice.as_ptr().add(i));
                let x2 = vmulq_f32(xv, xv);
                let x3 = vmulq_f32(x2, xv);
                let inner = vmulq_f32(vc1, vaddq_f32(xv, vmulq_f32(vc0, x3)));
                let e1 = crate::kernels::neon::math::neon_exp_f32x4(inner);
                let e2 = crate::kernels::neon::math::neon_exp_f32x4(vnegq_f32(inner));
                let tanh_v = vdivq_f32(vsubq_f32(e1, e2), vaddq_f32(e1, e2));
                let result = vmulq_f32(vhalf, vmulq_f32(xv, vaddq_f32(vone, tanh_v)));
                vst1q_f32(o_slice.as_mut_ptr().add(i), result);
                i += 4;
            }
        }
        for j in i..len {
            let xf = x_slice[j];
            let x3 = xf * xf * xf;
            let inner = c1 * (xf + c0 * x3);
            o_slice[j] = 0.5 * xf * (1.0 + inner.tanh());
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for j in 0..len {
            let xf = x_slice[j];
            let x3 = xf * xf * xf;
            let inner = c1 * (xf + c0 * x3);
            o_slice[j] = 0.5 * xf * (1.0 + inner.tanh());
        }
    }

    TensorView {
        data: Cow::Borrowed(out),
        shape: Cow::Owned(input.shape.to_vec()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_sigmoid() {
        let y = sigmoid(0.0);
        assert!((y - 0.5).abs() < 1e-6);
    }
    #[test]
    fn test_hard_sigmoid() {
        assert_eq!(hard_sigmoid(0.0, 0.2, 0.5), 0.5);
        assert_eq!(hard_sigmoid(3.0, 0.2, 0.5), 1.0);
        assert_eq!(hard_sigmoid(-3.0, 0.2, 0.5), 0.0);
    }
}
