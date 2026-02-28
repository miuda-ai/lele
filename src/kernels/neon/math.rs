use crate::kernels::utils;
use crate::tensor::TensorView;
use std::borrow::Cow;
use std::simd::StdFloat;
use std::simd::prelude::*;

/// Fast vectorized exp approximation using NEON intrinsics.
/// Based on Cephes/SSE2 approach: exp(x) = 2^(x * log2(e))
/// Accuracy: max relative error ~1e-6 over [-88, 88].
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn neon_exp_f32x4(
    x: core::arch::aarch64::float32x4_t,
) -> core::arch::aarch64::float32x4_t {
    use core::arch::aarch64::*;

    unsafe {
        let c_exp_hi = vdupq_n_f32(88.3762626647949f32);
        let c_exp_lo = vdupq_n_f32(-88.3762626647949f32);
        let c_log2ef = vdupq_n_f32(1.44269504088896341f32);
        let c_ln2_hi = vdupq_n_f32(0.693359375f32);
        let c_ln2_lo = vdupq_n_f32(-2.12194440e-4f32);

        let c_p0 = vdupq_n_f32(1.9875691500E-4);
        let c_p1 = vdupq_n_f32(1.3981999507E-3);
        let c_p2 = vdupq_n_f32(8.3334519073E-3);
        let c_p3 = vdupq_n_f32(4.1665795894E-2);
        let c_p4 = vdupq_n_f32(1.6666665459E-1);
        let c_p5 = vdupq_n_f32(5.0000001201E-1);
        let c_one = vdupq_n_f32(1.0);
        let c_half = vdupq_n_f32(0.5);
        let c_127 = vdupq_n_s32(127);

        // Clamp x
        let x = vminq_f32(vmaxq_f32(x, c_exp_lo), c_exp_hi);

        // fx = x * log2(e) + 0.5  (for rounding)
        let fx = vmlaq_f32(c_half, x, c_log2ef);

        // Convert to integer (floor)
        let fx_int = vcvtq_s32_f32(fx);
        let fx_floor = vcvtq_f32_s32(fx_int);

        // Adjust for negative: if fx_floor > fx, subtract 1
        let mask = vcgtq_f32(fx_floor, fx);
        let adj = vreinterpretq_f32_u32(vandq_u32(mask, vreinterpretq_u32_f32(c_one)));
        let fx_floor = vsubq_f32(fx_floor, adj);
        let n = vcvtq_s32_f32(fx_floor);

        // x = x - fx_floor * ln2
        let x = vmlsq_f32(x, fx_floor, c_ln2_hi);
        let x = vmlsq_f32(x, fx_floor, c_ln2_lo);

        // Polynomial approximation of exp(x) for x in [-ln2/2, ln2/2]
        // P(x) = p0*x^5 + p1*x^4 + p2*x^3 + p3*x^2 + p4*x + p5
        let mut y = vmlaq_f32(c_p1, c_p0, x);
        y = vmlaq_f32(c_p2, y, x);
        y = vmlaq_f32(c_p3, y, x);
        y = vmlaq_f32(c_p4, y, x);
        y = vmlaq_f32(c_p5, y, x);
        // exp(x) ≈ 1 + x + P(x) * x^2
        let xx = vmulq_f32(x, x);
        y = vmlaq_f32(x, y, xx); // y = x + P(x) * x^2
        y = vaddq_f32(y, c_one); // y = 1 + x + P(x) * x^2

        // Multiply by 2^n
        let pow2n = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(n, c_127), 23));
        vmulq_f32(y, pow2n)
    }
}

#[inline(always)]
pub(crate) fn simd_exp(x: f32x4) -> f32x4 {
    x.exp()
}

#[inline(always)]
pub(crate) fn simd_tanh(x: f32x4) -> f32x4 {
    let one = f32x4::splat(1.0);
    let zero = f32x4::splat(0.0);
    let two = f32x4::splat(2.0);

    let abs_x = x.abs();
    let neg_two_abs_x = zero - (two * abs_x);
    let e = simd_exp(neg_two_abs_x);

    let num = one - e;
    let den = one + e;
    let res_abs = num / den;

    // Restore sign: if x < 0, result is -res_abs
    let is_negative = x.simd_lt(zero);
    is_negative.select(zero - res_abs, res_abs)
}

#[inline(always)]
pub(crate) fn simd_sigmoid(x: f32x4) -> f32x4 {
    let one = f32x4::splat(1.0);
    let neg_x = f32x4::splat(0.0) - x;
    let e = simd_exp(neg_x);
    one / (one + e)
}

pub fn relu<'a>(input: &TensorView<'_>, output_buf: &'a mut Vec<f32>) -> TensorView<'a> {
    let len = input.data.len();
    utils::ensure_capacity(output_buf, len);
    unsafe {
        output_buf.set_len(len);
    }
    let (prefix, middle, _suffix) = input.data.as_simd::<4>();
    let out_slice = output_buf.as_mut_slice();
    let zero = f32x4::splat(0.0);
    for i in 0..prefix.len() {
        out_slice[i] = input.data[i].max(0.0);
    }
    let middle_out = &mut out_slice[prefix.len()..prefix.len() + middle.len() * 4];
    let (_, middle_out_simd, _) = middle_out.as_simd_mut::<4>();
    for i in 0..middle.len() {
        middle_out_simd[i] = middle[i].simd_max(zero);
    }
    let offset = prefix.len() + middle.len() * 4;
    for i in offset..len {
        out_slice[i] = input.data[i].max(0.0);
    }
    TensorView {
        data: Cow::Borrowed(output_buf),
        shape: Cow::Owned(input.shape.to_vec()),
    }
}

pub fn tanh<'a>(input: &TensorView<'_>, output_buf: &'a mut Vec<f32>) -> TensorView<'a> {
    let len = input.data.len();
    utils::ensure_capacity(output_buf, len);
    unsafe {
        output_buf.set_len(len);
    }
    let (prefix, middle, _suffix) = input.data.as_simd::<4>();
    let out_slice = output_buf.as_mut_slice();

    for i in 0..prefix.len() {
        out_slice[i] = input.data[i].tanh();
    }
    let offset_mid = prefix.len();
    for i in 0..middle.len() {
        let x = middle[i];
        let y = simd_tanh(x);
        y.copy_to_slice(&mut out_slice[offset_mid + i * 4..]);
    }
    let offset_suf = prefix.len() + middle.len() * 4;
    for i in offset_suf..len {
        out_slice[i] = input.data[i].tanh();
    }
    TensorView {
        data: Cow::Borrowed(output_buf),
        shape: Cow::Owned(input.shape.to_vec()),
    }
}

pub fn sigmoid<'a>(input: &TensorView<'_>, output_buf: &'a mut Vec<f32>) -> TensorView<'a> {
    let len = input.data.len();
    utils::ensure_capacity(output_buf, len);
    unsafe {
        output_buf.set_len(len);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let data = input.data.as_ptr();
        let out = output_buf.as_mut_ptr();
        let one = vdupq_n_f32(1.0);
        let zero = vdupq_n_f32(0.0);

        let mut i = 0;
        let simd_end = len & !3;
        while i < simd_end {
            let x = vld1q_f32(data.add(i));
            let neg_x = vsubq_f32(zero, x);
            let e = neon_exp_f32x4(neg_x);
            let sig = vdivq_f32(one, vaddq_f32(one, e));
            vst1q_f32(out.add(i), sig);
            i += 4;
        }
        while i < len {
            let x = *data.add(i);
            *out.add(i) = 1.0 / (1.0 + (-x).exp());
            i += 1;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let out_slice = output_buf.as_mut_slice();
        for i in 0..len {
            out_slice[i] = 1.0 / (1.0 + (-input.data[i]).exp());
        }
    }

    TensorView {
        data: Cow::Borrowed(output_buf),
        shape: Cow::Owned(input.shape.to_vec()),
    }
}

pub fn swish<'a>(input: &TensorView<'_>, output_buf: &'a mut Vec<f32>) -> TensorView<'a> {
    let len = input.data.len();
    utils::ensure_capacity(output_buf, len);
    unsafe {
        output_buf.set_len(len);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let data = input.data.as_ptr();
        let out = output_buf.as_mut_ptr();
        let one = vdupq_n_f32(1.0);
        let zero = vdupq_n_f32(0.0);

        let mut i = 0;
        let simd_end = len & !3;
        while i < simd_end {
            let x = vld1q_f32(data.add(i));
            let neg_x = vsubq_f32(zero, x);
            let e = neon_exp_f32x4(neg_x);
            let sig = vdivq_f32(one, vaddq_f32(one, e));
            let result = vmulq_f32(x, sig);
            vst1q_f32(out.add(i), result);
            i += 4;
        }
        while i < len {
            let x = *data.add(i);
            *out.add(i) = x / (1.0 + (-x).exp());
            i += 1;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let data = &input.data;
        let out = output_buf.as_mut_slice();
        for i in 0..len {
            let x = data[i];
            out[i] = x / (1.0 + (-x).exp());
        }
    }

    TensorView {
        data: Cow::Borrowed(output_buf),
        shape: Cow::Owned(input.shape.to_vec()),
    }
}

/// NEON-vectorized erf using Abramowitz & Stegun formula 7.1.26.
/// Max error ~1.5e-7, much faster than calling libm::erff per element.
pub fn erf<'b, 'a>(input: &TensorView<'b>, output_buf: &'a mut Vec<f32>) -> TensorView<'a> {
    let len = input.data.len();
    utils::ensure_capacity(output_buf, len);

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;

        let data = input.data.as_ptr();
        let out = output_buf.as_mut_ptr();

        // Abramowitz & Stegun constants for erf approximation (formula 7.1.26)
        let p = vdupq_n_f32(0.3275911f32);
        let a1 = vdupq_n_f32(0.254829592f32);
        let a2 = vdupq_n_f32(-0.284496736f32);
        let a3 = vdupq_n_f32(1.421413741f32);
        let a4 = vdupq_n_f32(-1.453152027f32);
        let a5 = vdupq_n_f32(1.061405429f32);
        let one = vdupq_n_f32(1.0f32);
        let neg_one = vdupq_n_f32(-1.0f32);

        let mut i = 0;
        let simd_end = len & !3;
        while i < simd_end {
            let x = vld1q_f32(data.add(i));
            // sign = x >= 0 ? 1 : -1
            let sign_mask = vcgeq_f32(x, vdupq_n_f32(0.0));
            let sign = vbslq_f32(sign_mask, one, neg_one);
            // x_abs = |x|
            let x_abs = vabsq_f32(x);
            // t = 1 / (1 + p * |x|)
            let t = vdivq_f32(one, vfmaq_f32(one, p, x_abs));
            // Horner's method: y = ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t
            let mut y = a5;
            y = vfmaq_f32(a4, y, t);
            y = vfmaq_f32(a3, y, t);
            y = vfmaq_f32(a2, y, t);
            y = vfmaq_f32(a1, y, t);
            y = vmulq_f32(y, t);
            // exp(-x*x)
            let neg_x2 = vnegq_f32(vmulq_f32(x_abs, x_abs));
            let exp_val = neon_exp_f32x4(neg_x2);
            // erf = sign * (1 - y * exp(-x²))
            let result = vmulq_f32(sign, vsubq_f32(one, vmulq_f32(y, exp_val)));
            vst1q_f32(out.add(i), result);
            i += 4;
        }
        // Scalar tail
        while i < len {
            *out.add(i) = libm::erff(*data.add(i));
            i += 1;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..len {
            output_buf[i] = libm::erff(input.data[i]);
        }
    }

    TensorView {
        data: Cow::Borrowed(output_buf),
        shape: Cow::Owned(input.shape.to_vec()),
    }
}

/// Standard GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
/// NEON-vectorized using our fast erf approximation.
pub fn gelu<'b, 'a>(input: &TensorView<'b>, output_buf: &'a mut Vec<f32>) -> TensorView<'a> {
    let len = input.data.len();
    utils::ensure_capacity(output_buf, len);
    unsafe {
        output_buf.set_len(len);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;

        let data = input.data.as_ptr();
        let out = output_buf.as_mut_ptr();

        // Constants for GELU
        let p = vdupq_n_f32(0.3275911f32);
        let a1 = vdupq_n_f32(0.254829592f32);
        let a2 = vdupq_n_f32(-0.284496736f32);
        let a3 = vdupq_n_f32(1.421413741f32);
        let a4 = vdupq_n_f32(-1.453152027f32);
        let a5 = vdupq_n_f32(1.061405429f32);
        let one = vdupq_n_f32(1.0f32);
        let neg_one = vdupq_n_f32(-1.0f32);
        let half = vdupq_n_f32(0.5f32);
        let inv_sqrt2 = vdupq_n_f32(0.7071067811865475f32); // 1/sqrt(2)

        let mut i = 0;
        let simd_end = len & !3;
        while i < simd_end {
            let x = vld1q_f32(data.add(i));

            // Compute erf(x / sqrt(2))
            let x_scaled = vmulq_f32(x, inv_sqrt2);
            let sign_mask = vcgeq_f32(x_scaled, vdupq_n_f32(0.0));
            let sign = vbslq_f32(sign_mask, one, neg_one);
            let x_abs = vabsq_f32(x_scaled);
            let t = vdivq_f32(one, vfmaq_f32(one, p, x_abs));
            let mut y = a5;
            y = vfmaq_f32(a4, y, t);
            y = vfmaq_f32(a3, y, t);
            y = vfmaq_f32(a2, y, t);
            y = vfmaq_f32(a1, y, t);
            y = vmulq_f32(y, t);
            let neg_x2 = vnegq_f32(vmulq_f32(x_abs, x_abs));
            let exp_val = neon_exp_f32x4(neg_x2);
            let erf_val = vmulq_f32(sign, vsubq_f32(one, vmulq_f32(y, exp_val)));

            // GELU = x * 0.5 * (1 + erf)
            let result = vmulq_f32(x, vmulq_f32(half, vaddq_f32(one, erf_val)));
            vst1q_f32(out.add(i), result);
            i += 4;
        }
        // Scalar tail
        while i < len {
            let x = *data.add(i);
            let erf_val = libm::erff(x * 0.7071067811865475);
            *out.add(i) = x * 0.5 * (1.0 + erf_val);
            i += 1;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..len {
            let x = input.data[i];
            let erf_val = libm::erff(x * 0.7071067811865475);
            output_buf[i] = x * 0.5 * (1.0 + erf_val);
        }
    }

    TensorView {
        data: Cow::Borrowed(output_buf),
        shape: Cow::Owned(input.shape.to_vec()),
    }
}

/// Fast GELU approximation (used in GPT-2, BERT):
/// fast_gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
/// This is faster than standard GELU and commonly used in transformers.
pub fn fast_gelu<'b, 'a>(input: &TensorView<'b>, output_buf: &'a mut Vec<f32>) -> TensorView<'a> {
    let len = input.data.len();
    utils::ensure_capacity(output_buf, len);
    unsafe {
        output_buf.set_len(len);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;

        let data = input.data.as_ptr();
        let out = output_buf.as_mut_ptr();

        // Constants for FastGELU
        let half = vdupq_n_f32(0.5f32);
        let one = vdupq_n_f32(1.0f32);
        let sqrt_2_over_pi = vdupq_n_f32(0.7978845608028654f32); // sqrt(2/pi)
        let coeff = vdupq_n_f32(0.044715f32);

        let mut i = 0;
        let simd_end = len & !3;
        while i < simd_end {
            let x = vld1q_f32(data.add(i));

            // x^3
            let x2 = vmulq_f32(x, x);
            let x3 = vmulq_f32(x2, x);

            // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
            let inner = vfmaq_f32(x, coeff, x3);
            let inner = vmulq_f32(sqrt_2_over_pi, inner);

            // tanh(inner) using exp
            // tanh(y) = (e^y - e^-y) / (e^y + e^-y) = (e^(2y) - 1) / (e^(2y) + 1)
            let two_inner = vmulq_f32(vdupq_n_f32(2.0), inner);
            let e2y = neon_exp_f32x4(two_inner);
            let tanh_val = vdivq_f32(vsubq_f32(e2y, one), vaddq_f32(e2y, one));

            // result = 0.5 * x * (1 + tanh(inner))
            let result = vmulq_f32(x, vmulq_f32(half, vaddq_f32(one, tanh_val)));
            vst1q_f32(out.add(i), result);
            i += 4;
        }
        // Scalar tail
        while i < len {
            let x = *data.add(i);
            let inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
            let tanh_val = inner.tanh();
            *out.add(i) = 0.5 * x * (1.0 + tanh_val);
            i += 1;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..len {
            let x = input.data[i];
            let inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
            let tanh_val = inner.tanh();
            output_buf[i] = 0.5 * x * (1.0 + tanh_val);
        }
    }

    TensorView {
        data: Cow::Borrowed(output_buf),
        shape: Cow::Owned(input.shape.to_vec()),
    }
}
// ─── NEON element-wise binary helpers ────────────────────────────────────────

/// NEON f32 add: out[i] = a[i] + b[i], 4x unrolled
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn add_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    use core::arch::aarch64::*;
    let mut i = 0;
    while i + 16 <= len {
        let va0 = vld1q_f32(a.add(i));
        let vb0 = vld1q_f32(b.add(i));
        let va1 = vld1q_f32(a.add(i + 4));
        let vb1 = vld1q_f32(b.add(i + 4));
        let va2 = vld1q_f32(a.add(i + 8));
        let vb2 = vld1q_f32(b.add(i + 8));
        let va3 = vld1q_f32(a.add(i + 12));
        let vb3 = vld1q_f32(b.add(i + 12));
        vst1q_f32(out.add(i), vaddq_f32(va0, vb0));
        vst1q_f32(out.add(i + 4), vaddq_f32(va1, vb1));
        vst1q_f32(out.add(i + 8), vaddq_f32(va2, vb2));
        vst1q_f32(out.add(i + 12), vaddq_f32(va3, vb3));
        i += 16;
    }
    while i + 4 <= len {
        let va = vld1q_f32(a.add(i));
        let vb = vld1q_f32(b.add(i));
        vst1q_f32(out.add(i), vaddq_f32(va, vb));
        i += 4;
    }
    while i < len {
        *out.add(i) = *a.add(i) + *b.add(i);
        i += 1;
    }
}

/// NEON f32 mul: out[i] = a[i] * b[i], 4x unrolled
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn mul_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    use core::arch::aarch64::*;
    let mut i = 0;
    while i + 16 <= len {
        let va0 = vld1q_f32(a.add(i));
        let vb0 = vld1q_f32(b.add(i));
        let va1 = vld1q_f32(a.add(i + 4));
        let vb1 = vld1q_f32(b.add(i + 4));
        let va2 = vld1q_f32(a.add(i + 8));
        let vb2 = vld1q_f32(b.add(i + 8));
        let va3 = vld1q_f32(a.add(i + 12));
        let vb3 = vld1q_f32(b.add(i + 12));
        vst1q_f32(out.add(i), vmulq_f32(va0, vb0));
        vst1q_f32(out.add(i + 4), vmulq_f32(va1, vb1));
        vst1q_f32(out.add(i + 8), vmulq_f32(va2, vb2));
        vst1q_f32(out.add(i + 12), vmulq_f32(va3, vb3));
        i += 16;
    }
    while i + 4 <= len {
        let va = vld1q_f32(a.add(i));
        let vb = vld1q_f32(b.add(i));
        vst1q_f32(out.add(i), vmulq_f32(va, vb));
        i += 4;
    }
    while i < len {
        *out.add(i) = *a.add(i) * *b.add(i);
        i += 1;
    }
}

/// NEON scalar-broadcast mul: out[i] = data[i] * scalar
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn mul_scalar_f32(data: *const f32, scalar: f32, out: *mut f32, len: usize) {
    use core::arch::aarch64::*;
    let mut i = 0;
    let vs = vdupq_n_f32(scalar);
    while i + 16 <= len {
        let v0 = vld1q_f32(data.add(i));
        let v1 = vld1q_f32(data.add(i + 4));
        let v2 = vld1q_f32(data.add(i + 8));
        let v3 = vld1q_f32(data.add(i + 12));
        vst1q_f32(out.add(i), vmulq_f32(v0, vs));
        vst1q_f32(out.add(i + 4), vmulq_f32(v1, vs));
        vst1q_f32(out.add(i + 8), vmulq_f32(v2, vs));
        vst1q_f32(out.add(i + 12), vmulq_f32(v3, vs));
        i += 16;
    }
    while i + 4 <= len {
        let v = vld1q_f32(data.add(i));
        vst1q_f32(out.add(i), vmulq_f32(v, vs));
        i += 4;
    }
    while i < len {
        *out.add(i) = *data.add(i) * scalar;
        i += 1;
    }
}
