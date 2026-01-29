use crate::kernels::utils;
use crate::tensor::TensorView;
use std::borrow::Cow;
use std::simd::prelude::*;
use std::simd::StdFloat;

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
    let (prefix, middle, _suffix) = input.data.as_simd::<4>();
    let out_slice = output_buf.as_mut_slice();

    for i in 0..prefix.len() {
        out_slice[i] = 1.0 / (1.0 + (-input.data[i]).exp());
    }
    let offset_mid = prefix.len();
    for i in 0..middle.len() {
        let x = middle[i];
        let y = simd_sigmoid(x);
        y.copy_to_slice(&mut out_slice[offset_mid + i * 4..]);
    }
    let offset_suf = prefix.len() + middle.len() * 4;
    for i in offset_suf..len {
        out_slice[i] = 1.0 / (1.0 + (-input.data[i]).exp());
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
    let (prefix, middle, _suffix) = input.data.as_simd::<4>();
    let out_slice = output_buf.as_mut_slice();

    for i in 0..prefix.len() {
        let x = input.data[i];
        out_slice[i] = x * (1.0 / (1.0 + (-x).exp()));
    }
    let offset_mid = prefix.len();
    for i in 0..middle.len() {
        let x = middle[i];
        let y = x * simd_sigmoid(x);
        y.copy_to_slice(&mut out_slice[offset_mid + i * 4..]);
    }
    let offset_suf = prefix.len() + middle.len() * 4;
    for i in offset_suf..len {
        let x = input.data[i];
        out_slice[i] = x * (1.0 / (1.0 + (-x).exp()));
    }
    TensorView {
        data: Cow::Borrowed(output_buf),
        shape: Cow::Owned(input.shape.to_vec()),
    }
}
