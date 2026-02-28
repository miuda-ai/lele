#![allow(unsafe_op_in_unsafe_fn)]
//! WASM SIMD128 math kernel implementations.
//! Only compiled when `target_arch = "wasm32"`.

use crate::kernels::utils;
use crate::tensor::TensorView;
use std::arch::wasm32::*;
use std::borrow::Cow;

// ─── Core exp approximation ─────────────────────────────────────────────────

/// Fast polynomial approximation of exp(x) for WASM SIMD128.
/// Uses the Schraudolph-style approach: clamp, scale, polynomial.
/// Accurate to ~1e-4 relative error in [-88, 88].
#[inline(always)]
pub unsafe fn exp_f32x4(x: v128) -> v128 {
    // Clamp to avoid overflow/underflow
    let x = f32x4_max(x, f32x4_splat(-88.0));
    let x = f32x4_min(x, f32x4_splat(88.0));

    // exp(x) = 2^(x / ln2) = 2^(n + f) where n = floor(x/ln2), f = frac
    let log2e = f32x4_splat(1.4426950408889634);
    let ln2 = f32x4_splat(0.6931471805599453);

    let t = f32x4_mul(x, log2e);
    let n = f32x4_floor(t);
    let f = f32x4_sub(x, f32x4_mul(n, ln2));

    // Polynomial approximation of exp(f) using Horner form:
    // 1 + f*(1 + f*(0.5 + f*(1/6 + f/24)))
    let c4 = f32x4_splat(1.0 / 24.0);
    let c3 = f32x4_splat(1.0 / 6.0);
    let c2 = f32x4_splat(0.5);
    let c1 = f32x4_splat(1.0);
    let c0 = f32x4_splat(1.0);

    let p = f32x4_add(f32x4_mul(c4, f), c3);
    let p = f32x4_add(f32x4_mul(p, f), c2);
    let p = f32x4_add(f32x4_mul(p, f), c1);
    let p = f32x4_add(f32x4_mul(p, f), c0);

    // Multiply by 2^n using integer arithmetic on the IEEE 754 exponent
    let n_i32 = i32x4_trunc_sat_f32x4(n);
    let bias = i32x4_splat(127);
    let shift = i32x4_shl(i32x4_add(n_i32, bias), 23);
    let pow2n: v128 = shift;

    f32x4_mul(p, pow2n)
}

// ─── erf approximation ──────────────────────────────────────────────────────

/// WASM SIMD128 erf approximation using Abramowitz & Stegun formula (7.1.28).
#[inline(always)]
pub unsafe fn erf_f32x4(x: v128) -> v128 {
    let sign_mask = f32x4_splat(-0.0);
    let sign = v128_and(x, sign_mask);
    let abs_x = f32x4_abs(x);

    let p = f32x4_splat(0.3275911);
    let a1 = f32x4_splat(0.254829592);
    let a2 = f32x4_splat(-0.284496736);
    let a3 = f32x4_splat(1.421413741);
    let a4 = f32x4_splat(-1.453152027);
    let a5 = f32x4_splat(1.061405429);
    let one = f32x4_splat(1.0);

    let t = f32x4_div(one, f32x4_add(one, f32x4_mul(p, abs_x)));

    let poly = f32x4_add(f32x4_mul(a5, t), a4);
    let poly = f32x4_add(f32x4_mul(poly, t), a3);
    let poly = f32x4_add(f32x4_mul(poly, t), a2);
    let poly = f32x4_add(f32x4_mul(poly, t), a1);
    let poly = f32x4_mul(poly, t);

    let neg_x2 = f32x4_neg(f32x4_mul(abs_x, abs_x));
    let exp_neg_x2 = exp_f32x4(neg_x2);

    let result = f32x4_sub(one, f32x4_mul(poly, exp_neg_x2));
    v128_xor(result, sign)
}

// ─── Unary activation kernels ────────────────────────────────────────────────

/// WASM SIMD128 tanh: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
pub unsafe fn tanh(input: *const f32, output: *mut f32, len: usize) {
    let mut i = 0;
    let one = f32x4_splat(1.0);
    let two = f32x4_splat(2.0);
    let neg_one = f32x4_splat(-1.0);

    while i + 4 <= len {
        let x = v128_load(input.add(i) as *const v128);
        let two_x = f32x4_mul(two, x);
        let exp_2x = exp_f32x4(two_x);
        let num = f32x4_sub(exp_2x, one);
        let den = f32x4_add(exp_2x, one);
        let result = f32x4_div(num, den);
        // Clamp to [-1, 1] for numerical safety
        let result = f32x4_max(result, neg_one);
        let result = f32x4_min(result, one);
        v128_store(output.add(i) as *mut v128, result);
        i += 4;
    }
    while i < len {
        *output.add(i) = (*input.add(i)).tanh();
        i += 1;
    }
}

/// WASM SIMD128 sigmoid: 1 / (1 + exp(-x))
pub unsafe fn sigmoid(input: *const f32, output: *mut f32, len: usize) {
    let mut i = 0;
    let one = f32x4_splat(1.0);

    while i + 4 <= len {
        let x = v128_load(input.add(i) as *const v128);
        let neg_x = f32x4_neg(x);
        let exp_neg_x = exp_f32x4(neg_x);
        let den = f32x4_add(one, exp_neg_x);
        let result = f32x4_div(one, den);
        v128_store(output.add(i) as *mut v128, result);
        i += 4;
    }
    while i < len {
        *output.add(i) = crate::kernels::activations::sigmoid(*input.add(i));
        i += 1;
    }
}

/// WASM SIMD128 SiLU: x * sigmoid(x) = x / (1 + exp(-x))
pub unsafe fn silu(input: *const f32, output: *mut f32, len: usize) {
    let mut i = 0;
    let one = f32x4_splat(1.0);

    while i + 4 <= len {
        let x = v128_load(input.add(i) as *const v128);
        let neg_x = f32x4_neg(x);
        let exp_neg_x = exp_f32x4(neg_x);
        let den = f32x4_add(one, exp_neg_x);
        let result = f32x4_div(x, den);
        v128_store(output.add(i) as *mut v128, result);
        i += 4;
    }
    while i < len {
        let x = *input.add(i);
        *output.add(i) = x / (1.0 + (-x).exp());
        i += 1;
    }
}

/// WASM SIMD128 ReLU: max(x, 0)
pub unsafe fn relu(input: *const f32, output: *mut f32, len: usize) {
    let zero = f32x4_splat(0.0);
    let mut i = 0;
    while i + 4 <= len {
        let v = v128_load(input.add(i) as *const v128);
        let r = f32x4_max(v, zero);
        v128_store(output.add(i) as *mut v128, r);
        i += 4;
    }
    while i < len {
        let v = *input.add(i);
        *output.add(i) = if v > 0.0 { v } else { 0.0 };
        i += 1;
    }
}

/// WASM SIMD128 erf for a buffer
pub unsafe fn erf(input: *const f32, output: *mut f32, len: usize) {
    let mut i = 0;
    while i + 4 <= len {
        let x = v128_load(input.add(i) as *const v128);
        let result = erf_f32x4(x);
        v128_store(output.add(i) as *mut v128, result);
        i += 4;
    }
    while i < len {
        *output.add(i) = libm::erff(*input.add(i));
        i += 1;
    }
}

/// WASM SIMD128 GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
pub unsafe fn gelu(input: *const f32, output: *mut f32, len: usize) {
    let mut i = 0;
    let half = f32x4_splat(0.5);
    let one = f32x4_splat(1.0);
    let inv_sqrt2 = f32x4_splat(0.7071067811865475);

    while i + 4 <= len {
        let x = v128_load(input.add(i) as *const v128);
        let erf_val = erf_f32x4(f32x4_mul(x, inv_sqrt2));
        let result = f32x4_mul(f32x4_mul(x, half), f32x4_add(one, erf_val));
        v128_store(output.add(i) as *mut v128, result);
        i += 4;
    }
    while i < len {
        let x = *input.add(i);
        let erf_val = libm::erff(x * 0.7071067811865475);
        *output.add(i) = x * 0.5 * (1.0 + erf_val);
        i += 1;
    }
}

/// WASM SIMD128 fast GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))
pub unsafe fn fast_gelu(input: *const f32, output: *mut f32, len: usize) {
    let mut i = 0;
    let half = f32x4_splat(0.5);
    let one = f32x4_splat(1.0);
    let two = f32x4_splat(2.0);
    let neg_one = f32x4_splat(-1.0);
    let sqrt_2_over_pi = f32x4_splat(0.7978845608028654);
    let coeff = f32x4_splat(0.044715);

    while i + 4 <= len {
        let x = v128_load(input.add(i) as *const v128);
        let x3 = f32x4_mul(f32x4_mul(x, x), x);
        let inner = f32x4_mul(sqrt_2_over_pi, f32x4_add(x, f32x4_mul(coeff, x3)));
        let two_inner = f32x4_mul(two, inner);
        let exp_2inner = exp_f32x4(two_inner);
        let tanh_val = f32x4_div(f32x4_sub(exp_2inner, one), f32x4_add(exp_2inner, one));
        let tanh_val = f32x4_max(f32x4_min(tanh_val, one), neg_one);
        let result = f32x4_mul(f32x4_mul(half, x), f32x4_add(one, tanh_val));
        v128_store(output.add(i) as *mut v128, result);
        i += 4;
    }
    while i < len {
        let x = *input.add(i);
        let inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
        *output.add(i) = 0.5 * x * (1.0 + inner.tanh());
        i += 1;
    }
}

// ─── Binary elementwise ops ──────────────────────────────────────────────────

/// WASM SIMD128 f32 add: out[i] = a[i] + b[i]
pub unsafe fn add_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let mut i = 0;
    while i + 16 <= len {
        let va0 = v128_load(a.add(i) as *const v128);
        let vb0 = v128_load(b.add(i) as *const v128);
        let va1 = v128_load(a.add(i + 4) as *const v128);
        let vb1 = v128_load(b.add(i + 4) as *const v128);
        let va2 = v128_load(a.add(i + 8) as *const v128);
        let vb2 = v128_load(b.add(i + 8) as *const v128);
        let va3 = v128_load(a.add(i + 12) as *const v128);
        let vb3 = v128_load(b.add(i + 12) as *const v128);
        v128_store(out.add(i) as *mut v128, f32x4_add(va0, vb0));
        v128_store(out.add(i + 4) as *mut v128, f32x4_add(va1, vb1));
        v128_store(out.add(i + 8) as *mut v128, f32x4_add(va2, vb2));
        v128_store(out.add(i + 12) as *mut v128, f32x4_add(va3, vb3));
        i += 16;
    }
    while i + 4 <= len {
        let va = v128_load(a.add(i) as *const v128);
        let vb = v128_load(b.add(i) as *const v128);
        v128_store(out.add(i) as *mut v128, f32x4_add(va, vb));
        i += 4;
    }
    while i < len {
        *out.add(i) = *a.add(i) + *b.add(i);
        i += 1;
    }
}

/// WASM SIMD128 f32 mul: out[i] = a[i] * b[i]
pub unsafe fn mul_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let mut i = 0;
    while i + 16 <= len {
        let va0 = v128_load(a.add(i) as *const v128);
        let vb0 = v128_load(b.add(i) as *const v128);
        let va1 = v128_load(a.add(i + 4) as *const v128);
        let vb1 = v128_load(b.add(i + 4) as *const v128);
        let va2 = v128_load(a.add(i + 8) as *const v128);
        let vb2 = v128_load(b.add(i + 8) as *const v128);
        let va3 = v128_load(a.add(i + 12) as *const v128);
        let vb3 = v128_load(b.add(i + 12) as *const v128);
        v128_store(out.add(i) as *mut v128, f32x4_mul(va0, vb0));
        v128_store(out.add(i + 4) as *mut v128, f32x4_mul(va1, vb1));
        v128_store(out.add(i + 8) as *mut v128, f32x4_mul(va2, vb2));
        v128_store(out.add(i + 12) as *mut v128, f32x4_mul(va3, vb3));
        i += 16;
    }
    while i + 4 <= len {
        let va = v128_load(a.add(i) as *const v128);
        let vb = v128_load(b.add(i) as *const v128);
        v128_store(out.add(i) as *mut v128, f32x4_mul(va, vb));
        i += 4;
    }
    while i < len {
        *out.add(i) = *a.add(i) * *b.add(i);
        i += 1;
    }
}

/// WASM SIMD128 f32 scalar broadcast add: out[i] = src[i] + scalar
/// Handles both src+scalar and scalar+src (same result for addition).
pub unsafe fn add_scalar_f32(src: *const f32, scalar: f32, out: *mut f32, n: usize) {
    let vs = f32x4_splat(scalar);
    let mut i = 0;
    while i + 16 <= n {
        let a0 = v128_load(src.add(i) as *const v128);
        let a1 = v128_load(src.add(i + 4) as *const v128);
        let a2 = v128_load(src.add(i + 8) as *const v128);
        let a3 = v128_load(src.add(i + 12) as *const v128);
        v128_store(out.add(i) as *mut v128, f32x4_add(a0, vs));
        v128_store(out.add(i + 4) as *mut v128, f32x4_add(a1, vs));
        v128_store(out.add(i + 8) as *mut v128, f32x4_add(a2, vs));
        v128_store(out.add(i + 12) as *mut v128, f32x4_add(a3, vs));
        i += 16;
    }
    while i + 4 <= n {
        let a = v128_load(src.add(i) as *const v128);
        v128_store(out.add(i) as *mut v128, f32x4_add(a, vs));
        i += 4;
    }
    while i < n {
        *out.add(i) = *src.add(i) + scalar;
        i += 1;
    }
}

/// WASM SIMD128 f32 scalar broadcast mul: out[i] = src[i] * scalar
pub unsafe fn mul_scalar_f32(src: *const f32, scalar: f32, out: *mut f32, n: usize) {
    let vs = f32x4_splat(scalar);
    let mut i = 0;
    while i + 16 <= n {
        let a0 = v128_load(src.add(i) as *const v128);
        let a1 = v128_load(src.add(i + 4) as *const v128);
        let a2 = v128_load(src.add(i + 8) as *const v128);
        let a3 = v128_load(src.add(i + 12) as *const v128);
        v128_store(out.add(i) as *mut v128, f32x4_mul(a0, vs));
        v128_store(out.add(i + 4) as *mut v128, f32x4_mul(a1, vs));
        v128_store(out.add(i + 8) as *mut v128, f32x4_mul(a2, vs));
        v128_store(out.add(i + 12) as *mut v128, f32x4_mul(a3, vs));
        i += 16;
    }
    while i + 4 <= n {
        let a = v128_load(src.add(i) as *const v128);
        v128_store(out.add(i) as *mut v128, f32x4_mul(a, vs));
        i += 4;
    }
    while i < n {
        *out.add(i) = *src.add(i) * scalar;
        i += 1;
    }
}

/// WASM SIMD128 f32 channel-wise add broadcast for NCHW: [1,C,H,W] + [1,C,1,1] (or [C]).
/// `src` is [1,C,H,W] dense buffer (c*hw elements), `scalars` is [C] (c elements).
/// `out` receives src[ch*hw..ch*hw+hw] + scalars[ch] for each channel ch.
pub unsafe fn add_channel_broadcast_f32(
    src: *const f32,
    scalars: *const f32,
    out: *mut f32,
    c: usize,
    hw: usize,
) {
    for ch in 0..c {
        let scalar = *scalars.add(ch);
        let vs = f32x4_splat(scalar);
        let base = ch * hw;
        let src_ch = src.add(base);
        let out_ch = out.add(base);
        let mut i = 0;
        while i + 16 <= hw {
            let a0 = v128_load(src_ch.add(i) as *const v128);
            let a1 = v128_load(src_ch.add(i + 4) as *const v128);
            let a2 = v128_load(src_ch.add(i + 8) as *const v128);
            let a3 = v128_load(src_ch.add(i + 12) as *const v128);
            v128_store(out_ch.add(i) as *mut v128, f32x4_add(a0, vs));
            v128_store(out_ch.add(i + 4) as *mut v128, f32x4_add(a1, vs));
            v128_store(out_ch.add(i + 8) as *mut v128, f32x4_add(a2, vs));
            v128_store(out_ch.add(i + 12) as *mut v128, f32x4_add(a3, vs));
            i += 16;
        }
        while i + 4 <= hw {
            let a = v128_load(src_ch.add(i) as *const v128);
            v128_store(out_ch.add(i) as *mut v128, f32x4_add(a, vs));
            i += 4;
        }
        while i < hw {
            *out_ch.add(i) = *src_ch.add(i) + scalar;
            i += 1;
        }
    }
}

/// WASM SIMD128 f32 channel-wise mul broadcast for NCHW: [1,C,H,W] * [1,C,1,1] (or [C]).
pub unsafe fn mul_channel_broadcast_f32(
    src: *const f32,
    scalars: *const f32,
    out: *mut f32,
    c: usize,
    hw: usize,
) {
    for ch in 0..c {
        let scalar = *scalars.add(ch);
        let vs = f32x4_splat(scalar);
        let base = ch * hw;
        let src_ch = src.add(base);
        let out_ch = out.add(base);
        let mut i = 0;
        while i + 16 <= hw {
            let a0 = v128_load(src_ch.add(i) as *const v128);
            let a1 = v128_load(src_ch.add(i + 4) as *const v128);
            let a2 = v128_load(src_ch.add(i + 8) as *const v128);
            let a3 = v128_load(src_ch.add(i + 12) as *const v128);
            v128_store(out_ch.add(i) as *mut v128, f32x4_mul(a0, vs));
            v128_store(out_ch.add(i + 4) as *mut v128, f32x4_mul(a1, vs));
            v128_store(out_ch.add(i + 8) as *mut v128, f32x4_mul(a2, vs));
            v128_store(out_ch.add(i + 12) as *mut v128, f32x4_mul(a3, vs));
            i += 16;
        }
        while i + 4 <= hw {
            let a = v128_load(src_ch.add(i) as *const v128);
            v128_store(out_ch.add(i) as *mut v128, f32x4_mul(a, vs));
            i += 4;
        }
        while i < hw {
            *out_ch.add(i) = *src_ch.add(i) * scalar;
            i += 1;
        }
    }
}

/// WASM SIMD128 f32 sub: out[i] = a[i] - b[i]
pub unsafe fn sub_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let mut i = 0;
    while i + 16 <= len {
        let va0 = v128_load(a.add(i) as *const v128);
        let vb0 = v128_load(b.add(i) as *const v128);
        let va1 = v128_load(a.add(i + 4) as *const v128);
        let vb1 = v128_load(b.add(i + 4) as *const v128);
        let va2 = v128_load(a.add(i + 8) as *const v128);
        let vb2 = v128_load(b.add(i + 8) as *const v128);
        let va3 = v128_load(a.add(i + 12) as *const v128);
        let vb3 = v128_load(b.add(i + 12) as *const v128);
        v128_store(out.add(i) as *mut v128, f32x4_sub(va0, vb0));
        v128_store(out.add(i + 4) as *mut v128, f32x4_sub(va1, vb1));
        v128_store(out.add(i + 8) as *mut v128, f32x4_sub(va2, vb2));
        v128_store(out.add(i + 12) as *mut v128, f32x4_sub(va3, vb3));
        i += 16;
    }
    while i + 4 <= len {
        let va = v128_load(a.add(i) as *const v128);
        let vb = v128_load(b.add(i) as *const v128);
        v128_store(out.add(i) as *mut v128, f32x4_sub(va, vb));
        i += 4;
    }
    while i < len {
        *out.add(i) = *a.add(i) - *b.add(i);
        i += 1;
    }
}

/// WASM SIMD128 f32 div: out[i] = a[i] / b[i]
pub unsafe fn div_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let mut i = 0;
    while i + 16 <= len {
        let va0 = v128_load(a.add(i) as *const v128);
        let vb0 = v128_load(b.add(i) as *const v128);
        let va1 = v128_load(a.add(i + 4) as *const v128);
        let vb1 = v128_load(b.add(i + 4) as *const v128);
        let va2 = v128_load(a.add(i + 8) as *const v128);
        let vb2 = v128_load(b.add(i + 8) as *const v128);
        let va3 = v128_load(a.add(i + 12) as *const v128);
        let vb3 = v128_load(b.add(i + 12) as *const v128);
        v128_store(out.add(i) as *mut v128, f32x4_div(va0, vb0));
        v128_store(out.add(i + 4) as *mut v128, f32x4_div(va1, vb1));
        v128_store(out.add(i + 8) as *mut v128, f32x4_div(va2, vb2));
        v128_store(out.add(i + 12) as *mut v128, f32x4_div(va3, vb3));
        i += 16;
    }
    while i + 4 <= len {
        let va = v128_load(a.add(i) as *const v128);
        let vb = v128_load(b.add(i) as *const v128);
        v128_store(out.add(i) as *mut v128, f32x4_div(va, vb));
        i += 4;
    }
    while i < len {
        *out.add(i) = *a.add(i) / *b.add(i);
        i += 1;
    }
}

// ─── Public TensorView-level functions (mirror the neon/math.rs style) ──────

pub fn relu_kernel<'b, 'a>(input: &TensorView<'b>, out: &'a mut Vec<f32>) -> TensorView<'a> {
    let len = input.data.len();
    utils::ensure_capacity(out, len);
    unsafe {
        relu(input.data.as_ptr(), out.as_mut_ptr(), len);
    }
    TensorView {
        data: Cow::Borrowed(out),
        shape: Cow::Owned(input.shape.to_vec()),
    }
}

pub fn tanh_kernel<'b, 'a>(input: &TensorView<'b>, out: &'a mut Vec<f32>) -> TensorView<'a> {
    let len = input.data.len();
    utils::ensure_capacity(out, len);
    unsafe {
        tanh(input.data.as_ptr(), out.as_mut_ptr(), len);
    }
    TensorView {
        data: Cow::Borrowed(out),
        shape: Cow::Owned(input.shape.to_vec()),
    }
}

pub fn sigmoid_kernel<'b, 'a>(input: &TensorView<'b>, out: &'a mut Vec<f32>) -> TensorView<'a> {
    let len = input.data.len();
    utils::ensure_capacity(out, len);
    unsafe {
        sigmoid(input.data.as_ptr(), out.as_mut_ptr(), len);
    }
    TensorView {
        data: Cow::Borrowed(out),
        shape: Cow::Owned(input.shape.to_vec()),
    }
}

pub fn silu_kernel<'b, 'a>(input: &TensorView<'b>, out: &'a mut Vec<f32>) -> TensorView<'a> {
    let len = input.data.len();
    utils::ensure_capacity(out, len);
    unsafe {
        silu(input.data.as_ptr(), out.as_mut_ptr(), len);
    }
    TensorView {
        data: Cow::Borrowed(out),
        shape: Cow::Owned(input.shape.to_vec()),
    }
}

pub fn erf_kernel<'b, 'a>(input: &TensorView<'b>, out: &'a mut Vec<f32>) -> TensorView<'a> {
    let len = input.data.len();
    utils::ensure_capacity(out, len);
    unsafe {
        erf(input.data.as_ptr(), out.as_mut_ptr(), len);
    }
    TensorView {
        data: Cow::Borrowed(out),
        shape: Cow::Owned(input.shape.to_vec()),
    }
}

pub fn gelu_kernel<'b, 'a>(input: &TensorView<'b>, out: &'a mut Vec<f32>) -> TensorView<'a> {
    let len = input.data.len();
    utils::ensure_capacity(out, len);
    unsafe {
        gelu(input.data.as_ptr(), out.as_mut_ptr(), len);
    }
    TensorView {
        data: Cow::Borrowed(out),
        shape: Cow::Owned(input.shape.to_vec()),
    }
}

pub fn fast_gelu_kernel<'b, 'a>(input: &TensorView<'b>, out: &'a mut Vec<f32>) -> TensorView<'a> {
    let len = input.data.len();
    utils::ensure_capacity(out, len);
    unsafe {
        fast_gelu(input.data.as_ptr(), out.as_mut_ptr(), len);
    }
    TensorView {
        data: Cow::Borrowed(out),
        shape: Cow::Owned(input.shape.to_vec()),
    }
}
