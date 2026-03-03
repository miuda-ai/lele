#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Fast exp approximation using AVX2 + FMA.
/// exp(x) ≈ 2^(x / ln2) via polynomial minimax on fractional part.
/// Accurate to ~1e-6 relative error in [-87, 88].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn avx2_exp_ps(x: __m256) -> __m256 {
    const LN2_HI: f32 = 0.693359375;
    const LN2_LO: f32 = -2.12194440e-4;
    const LOG2EF: f32 = 1.44269504088896341;
    const C1: f32 = 0.5;
    const C2: f32 = 0.166666671633720398;
    const C3: f32 = 0.0416657844442129135;
    const C4: f32 = 0.00833345670066840443;
    const C5: f32 = 0.00139712726883569741;
    const C6: f32 = 0.000198712018891638893;

    let log2ef = _mm256_set1_ps(LOG2EF);
    let ln2_hi = _mm256_set1_ps(LN2_HI);
    let ln2_lo = _mm256_set1_ps(LN2_LO);
    let c1 = _mm256_set1_ps(C1);
    let c2 = _mm256_set1_ps(C2);
    let c3 = _mm256_set1_ps(C3);
    let c4 = _mm256_set1_ps(C4);
    let c5 = _mm256_set1_ps(C5);
    let c6 = _mm256_set1_ps(C6);
    let one = _mm256_set1_ps(1.0);

    // Clamp input to avoid overflow/underflow
    let x = _mm256_max_ps(x, _mm256_set1_ps(-87.33654));
    let x = _mm256_min_ps(x, _mm256_set1_ps(88.72284));

    // fx = round(x * log2e)
    let fx = _mm256_round_ps(
        _mm256_mul_ps(x, log2ef),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );

    // x = x - fx * ln2  (high + low for precision)
    let x = _mm256_fnmadd_ps(fx, ln2_hi, x);
    let x = _mm256_fnmadd_ps(fx, ln2_lo, x);

    // Polynomial: 1 + x*(1 + x*(C1 + x*(C2 + x*(C3 + x*(C4 + x*(C5 + x*C6))))))
    let mut y = _mm256_fmadd_ps(c6, x, c5);
    y = _mm256_fmadd_ps(y, x, c4);
    y = _mm256_fmadd_ps(y, x, c3);
    y = _mm256_fmadd_ps(y, x, c2);
    y = _mm256_fmadd_ps(y, x, c1);
    y = _mm256_fmadd_ps(y, x, one);
    y = _mm256_fmadd_ps(y, x, one);

    // Scale by 2^n: construct float from integer exponent
    let emm0 = _mm256_cvtps_epi32(fx);
    let emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(127));
    let emm0 = _mm256_slli_epi32(emm0, 23);
    let pow2n = _mm256_castsi256_ps(emm0);

    _mm256_mul_ps(y, pow2n)
}

/// AVX2 SIMD sigmoid: 1 / (1 + exp(-x))
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn avx2_sigmoid_ps(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let neg_x = _mm256_xor_ps(x, _mm256_set1_ps(-0.0)); // negate
    let exp_neg = unsafe { avx2_exp_ps(neg_x) };
    let denom = _mm256_add_ps(one, exp_neg);
    _mm256_div_ps(one, denom)
}

/// AVX2 SIMD tanh: (1 - exp(-2x)) / (1 + exp(-2x))
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn avx2_tanh_ps(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let two = _mm256_set1_ps(2.0);
    let neg_two_x = _mm256_mul_ps(_mm256_xor_ps(x, _mm256_set1_ps(-0.0)), two);
    let e = unsafe { avx2_exp_ps(neg_two_x) };
    let num = _mm256_sub_ps(one, e);
    let den = _mm256_add_ps(one, e);
    let abs_result = _mm256_div_ps(num, den);

    // Restore sign
    let sign_mask = _mm256_set1_ps(-0.0);
    let sign_bit = _mm256_and_ps(x, sign_mask);
    let abs_x_result = _mm256_andnot_ps(sign_mask, abs_result);
    _mm256_or_ps(abs_x_result, sign_bit)
}

/// AVX2 SIMD SiLU: x * sigmoid(x) = x / (1 + exp(-x))
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn avx2_silu_ps(x: __m256) -> __m256 {
    let sig = unsafe { avx2_sigmoid_ps(x) };
    _mm256_mul_ps(x, sig)
}

/// AVX2 SIMD erf approximation using Abramowitz & Stegun formula 7.1.26
/// Maximum error ~1.5e-7 over the entire range.
/// erf(x) = 1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5) * exp(-x^2)
/// where t = 1 / (1 + 0.3275911 * |x|)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn avx2_erf_ps(x: __m256) -> __m256 {
    let sign_mask = _mm256_set1_ps(-0.0);
    let one = _mm256_set1_ps(1.0);

    // Constants from Abramowitz & Stegun
    let p = _mm256_set1_ps(0.3275911);
    let a1 = _mm256_set1_ps(0.254829592);
    let a2 = _mm256_set1_ps(-0.284496736);
    let a3 = _mm256_set1_ps(1.421413741);
    let a4 = _mm256_set1_ps(-1.453152027);
    let a5 = _mm256_set1_ps(1.061405429);

    // Save sign and work with |x|
    let sign_bit = _mm256_and_ps(x, sign_mask);
    let abs_x = _mm256_andnot_ps(sign_mask, x);

    // t = 1 / (1 + p * |x|)
    let t = _mm256_div_ps(one, _mm256_fmadd_ps(p, abs_x, one));

    // Horner's method: poly = a1 + t*(a2 + t*(a3 + t*(a4 + t*a5)))
    let mut poly = _mm256_fmadd_ps(a5, t, a4);
    poly = _mm256_fmadd_ps(poly, t, a3);
    poly = _mm256_fmadd_ps(poly, t, a2);
    poly = _mm256_fmadd_ps(poly, t, a1);

    // result = 1 - poly * t * exp(-x^2)
    let neg_x_sq = _mm256_xor_ps(_mm256_mul_ps(abs_x, abs_x), sign_mask);
    let exp_val = unsafe { avx2_exp_ps(neg_x_sq) };
    let result = _mm256_fnmadd_ps(_mm256_mul_ps(poly, t), exp_val, one);

    // Restore sign: erf(-x) = -erf(x)
    _mm256_or_ps(result, sign_bit)
}

/// Horizontal sum of __m256 into a single f32
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn hsum_ps(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_shuffle_ps(sum64, sum64, 1);
    let sum32 = _mm_add_ss(sum64, hi32);
    _mm_cvtss_f32(sum32)
}
// ─── Buffer-level element-wise kernels ───────────────────────────────────────

/// AVX2 f32 add: out[i] = a[i] + b[i]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn add_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 32 <= len {
        let va0 = _mm256_loadu_ps(a.add(i));
        let vb0 = _mm256_loadu_ps(b.add(i));
        let va1 = _mm256_loadu_ps(a.add(i + 8));
        let vb1 = _mm256_loadu_ps(b.add(i + 8));
        let va2 = _mm256_loadu_ps(a.add(i + 16));
        let vb2 = _mm256_loadu_ps(b.add(i + 16));
        let va3 = _mm256_loadu_ps(a.add(i + 24));
        let vb3 = _mm256_loadu_ps(b.add(i + 24));
        _mm256_storeu_ps(out.add(i), _mm256_add_ps(va0, vb0));
        _mm256_storeu_ps(out.add(i + 8), _mm256_add_ps(va1, vb1));
        _mm256_storeu_ps(out.add(i + 16), _mm256_add_ps(va2, vb2));
        _mm256_storeu_ps(out.add(i + 24), _mm256_add_ps(va3, vb3));
        i += 32;
    }
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.add(i));
        let vb = _mm256_loadu_ps(b.add(i));
        _mm256_storeu_ps(out.add(i), _mm256_add_ps(va, vb));
        i += 8;
    }
    while i < len {
        *out.add(i) = *a.add(i) + *b.add(i);
        i += 1;
    }
}

/// AVX2 f32 mul: out[i] = a[i] * b[i]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn mul_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 32 <= len {
        let va0 = _mm256_loadu_ps(a.add(i));
        let vb0 = _mm256_loadu_ps(b.add(i));
        let va1 = _mm256_loadu_ps(a.add(i + 8));
        let vb1 = _mm256_loadu_ps(b.add(i + 8));
        let va2 = _mm256_loadu_ps(a.add(i + 16));
        let vb2 = _mm256_loadu_ps(b.add(i + 16));
        let va3 = _mm256_loadu_ps(a.add(i + 24));
        let vb3 = _mm256_loadu_ps(b.add(i + 24));
        _mm256_storeu_ps(out.add(i), _mm256_mul_ps(va0, vb0));
        _mm256_storeu_ps(out.add(i + 8), _mm256_mul_ps(va1, vb1));
        _mm256_storeu_ps(out.add(i + 16), _mm256_mul_ps(va2, vb2));
        _mm256_storeu_ps(out.add(i + 24), _mm256_mul_ps(va3, vb3));
        i += 32;
    }
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.add(i));
        let vb = _mm256_loadu_ps(b.add(i));
        _mm256_storeu_ps(out.add(i), _mm256_mul_ps(va, vb));
        i += 8;
    }
    while i < len {
        *out.add(i) = *a.add(i) * *b.add(i);
        i += 1;
    }
}

/// AVX2 erf buffer kernel
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn erf_kernel(input: *const f32, output: *mut f32, len: usize) {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= len {
        let v = _mm256_loadu_ps(input.add(i));
        let r = avx2_erf_ps(v);
        _mm256_storeu_ps(output.add(i), r);
        i += 8;
    }
    while i < len {
        *output.add(i) = libm::erff(*input.add(i));
        i += 1;
    }
}

/// AVX2 tanh buffer kernel
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn tanh_kernel(input: *const f32, output: *mut f32, len: usize) {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= len {
        let v = _mm256_loadu_ps(input.add(i));
        let r = avx2_tanh_ps(v);
        _mm256_storeu_ps(output.add(i), r);
        i += 8;
    }
    while i < len {
        *output.add(i) = (*input.add(i)).tanh();
        i += 1;
    }
}

/// AVX2 SiLU buffer kernel
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn silu_kernel(input: *const f32, output: *mut f32, len: usize) {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= len {
        let v = _mm256_loadu_ps(input.add(i));
        let r = avx2_silu_ps(v);
        _mm256_storeu_ps(output.add(i), r);
        i += 8;
    }
    while i < len {
        let x = *input.add(i);
        *output.add(i) = x / (1.0 + (-x).exp());
        i += 1;
    }
}

/// AVX2 sigmoid buffer kernel
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn sigmoid_kernel(input: *const f32, output: *mut f32, len: usize) {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= len {
        let v = _mm256_loadu_ps(input.add(i));
        let r = avx2_sigmoid_ps(v);
        _mm256_storeu_ps(output.add(i), r);
        i += 8;
    }
    while i < len {
        *output.add(i) = crate::kernels::activations::sigmoid(*input.add(i));
        i += 1;
    }
}

/// AVX2 ReLU buffer kernel
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn relu_kernel(input: *const f32, output: *mut f32, len: usize) {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= len {
        let v = _mm256_loadu_ps(input.add(i));
        let r = _mm256_max_ps(v, zero);
        _mm256_storeu_ps(output.add(i), r);
        i += 8;
    }
    while i < len {
        let v = *input.add(i);
        *output.add(i) = if v > 0.0 { v } else { 0.0 };
        i += 1;
    }
}

/// AVX2 sqrt buffer kernel
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn sqrt_kernel(input: *const f32, output: *mut f32, len: usize) {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= len {
        let v = _mm256_loadu_ps(input.add(i));
        let r = _mm256_sqrt_ps(v);
        _mm256_storeu_ps(output.add(i), r);
        i += 8;
    }
    while i < len {
        *output.add(i) = (*input.add(i)).sqrt();
        i += 1;
    }
}

/// AVX2 fused bias + SiLU: out[i] = silu(out[i] + bias)
/// Operates in-place on `data` buffer. `bias` is a scalar broadcast.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn bias_silu_inplace(data: *mut f32, len: usize, bias: f32) {
    use std::arch::x86_64::*;
    let bias_v = _mm256_set1_ps(bias);
    let mut i = 0;
    while i + 16 <= len {
        let v0 = _mm256_add_ps(_mm256_loadu_ps(data.add(i)), bias_v);
        let v1 = _mm256_add_ps(_mm256_loadu_ps(data.add(i + 8)), bias_v);
        _mm256_storeu_ps(data.add(i), avx2_silu_ps(v0));
        _mm256_storeu_ps(data.add(i + 8), avx2_silu_ps(v1));
        i += 16;
    }
    while i + 8 <= len {
        let v = _mm256_add_ps(_mm256_loadu_ps(data.add(i)), bias_v);
        _mm256_storeu_ps(data.add(i), avx2_silu_ps(v));
        i += 8;
    }
    while i < len {
        let x = *data.add(i) + bias;
        *data.add(i) = x / (1.0 + (-x).exp());
        i += 1;
    }
}

/// AVX2 fused bias + ReLU: out[i] = max(0, out[i] + bias)
/// Operates in-place on `data` buffer. `bias` is a scalar broadcast.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn bias_relu_inplace(data: *mut f32, len: usize, bias: f32) {
    use std::arch::x86_64::*;
    let bias_v = _mm256_set1_ps(bias);
    let zero = _mm256_setzero_ps();
    let mut i = 0;
    while i + 16 <= len {
        let v0 = _mm256_add_ps(_mm256_loadu_ps(data.add(i)), bias_v);
        let v1 = _mm256_add_ps(_mm256_loadu_ps(data.add(i + 8)), bias_v);
        _mm256_storeu_ps(data.add(i), _mm256_max_ps(v0, zero));
        _mm256_storeu_ps(data.add(i + 8), _mm256_max_ps(v1, zero));
        i += 16;
    }
    while i + 8 <= len {
        let v = _mm256_add_ps(_mm256_loadu_ps(data.add(i)), bias_v);
        _mm256_storeu_ps(data.add(i), _mm256_max_ps(v, zero));
        i += 8;
    }
    while i < len {
        let x = *data.add(i) + bias;
        *data.add(i) = if x > 0.0 { x } else { 0.0 };
        i += 1;
    }
}

/// AVX2 bias-only addition: out[i] = out[i] + bias (in-place)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn bias_add_inplace(data: *mut f32, len: usize, bias: f32) {
    use std::arch::x86_64::*;
    let bias_v = _mm256_set1_ps(bias);
    let mut i = 0;
    while i + 16 <= len {
        let v0 = _mm256_add_ps(_mm256_loadu_ps(data.add(i)), bias_v);
        let v1 = _mm256_add_ps(_mm256_loadu_ps(data.add(i + 8)), bias_v);
        _mm256_storeu_ps(data.add(i), v0);
        _mm256_storeu_ps(data.add(i + 8), v1);
        i += 16;
    }
    while i + 8 <= len {
        let v = _mm256_add_ps(_mm256_loadu_ps(data.add(i)), bias_v);
        _mm256_storeu_ps(data.add(i), v);
        i += 8;
    }
    while i < len {
        *data.add(i) += bias;
        i += 1;
    }
}

/// AVX2 SiLU activation only (in-place, no bias): out[i] = silu(out[i])
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn silu_inplace(data: *mut f32, len: usize) {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 16 <= len {
        let v0 = _mm256_loadu_ps(data.add(i));
        let v1 = _mm256_loadu_ps(data.add(i + 8));
        _mm256_storeu_ps(data.add(i), avx2_silu_ps(v0));
        _mm256_storeu_ps(data.add(i + 8), avx2_silu_ps(v1));
        i += 16;
    }
    while i + 8 <= len {
        let v = _mm256_loadu_ps(data.add(i));
        _mm256_storeu_ps(data.add(i), avx2_silu_ps(v));
        i += 8;
    }
    while i < len {
        let x = *data.add(i);
        *data.add(i) = x / (1.0 + (-x).exp());
        i += 1;
    }
}

/// AVX2 ReLU activation only (in-place, no bias): out[i] = max(0, out[i])
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn relu_inplace(data: *mut f32, len: usize) {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_ps();
    let mut i = 0;
    while i + 16 <= len {
        let v0 = _mm256_loadu_ps(data.add(i));
        let v1 = _mm256_loadu_ps(data.add(i + 8));
        _mm256_storeu_ps(data.add(i), _mm256_max_ps(v0, zero));
        _mm256_storeu_ps(data.add(i + 8), _mm256_max_ps(v1, zero));
        i += 16;
    }
    while i + 8 <= len {
        let v = _mm256_loadu_ps(data.add(i));
        _mm256_storeu_ps(data.add(i), _mm256_max_ps(v, zero));
        i += 8;
    }
    while i < len {
        let x = *data.add(i);
        *data.add(i) = if x > 0.0 { x } else { 0.0 };
        i += 1;
    }
}

/// Scalar broadcast multiply: out[i] = a[i] * scalar
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn mul_scalar_f32(a: *const f32, scalar: f32, out: *mut f32, len: usize) {
    let vs = _mm256_set1_ps(scalar);
    let mut i = 0;
    while i + 32 <= len {
        let v0 = _mm256_loadu_ps(a.add(i));
        let v1 = _mm256_loadu_ps(a.add(i + 8));
        let v2 = _mm256_loadu_ps(a.add(i + 16));
        let v3 = _mm256_loadu_ps(a.add(i + 24));
        _mm256_storeu_ps(out.add(i), _mm256_mul_ps(v0, vs));
        _mm256_storeu_ps(out.add(i + 8), _mm256_mul_ps(v1, vs));
        _mm256_storeu_ps(out.add(i + 16), _mm256_mul_ps(v2, vs));
        _mm256_storeu_ps(out.add(i + 24), _mm256_mul_ps(v3, vs));
        i += 32;
    }
    while i + 8 <= len {
        _mm256_storeu_ps(out.add(i), _mm256_mul_ps(_mm256_loadu_ps(a.add(i)), vs));
        i += 8;
    }
    while i < len {
        *out.add(i) = *a.add(i) * scalar;
        i += 1;
    }
}

/// AVX2 exp buffer kernel: out[i] = exp(input[i])
/// Uses the fast avx2_exp_ps approximation (max error ~1e-6).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn exp_kernel(input: *const f32, output: *mut f32, len: usize) {
    let mut i = 0;
    while i + 32 <= len {
        let v0 = _mm256_loadu_ps(input.add(i));
        let v1 = _mm256_loadu_ps(input.add(i + 8));
        let v2 = _mm256_loadu_ps(input.add(i + 16));
        let v3 = _mm256_loadu_ps(input.add(i + 24));
        _mm256_storeu_ps(output.add(i), avx2_exp_ps(v0));
        _mm256_storeu_ps(output.add(i + 8), avx2_exp_ps(v1));
        _mm256_storeu_ps(output.add(i + 16), avx2_exp_ps(v2));
        _mm256_storeu_ps(output.add(i + 24), avx2_exp_ps(v3));
        i += 32;
    }
    while i + 8 <= len {
        _mm256_storeu_ps(output.add(i), avx2_exp_ps(_mm256_loadu_ps(input.add(i))));
        i += 8;
    }
    while i < len {
        *output.add(i) = (*input.add(i)).exp();
        i += 1;
    }
}

/// AVX2 GELU buffer kernel: out[i] = x * 0.5 * (1 + erf(x / sqrt(2)))
/// Uses avx2_erf_ps for accurate computation (max error ~1.5e-7).
/// 4-way unrolled to match throughput of add_f32/mul_f32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gelu_kernel(input: *const f32, output: *mut f32, len: usize) {
    let inv_sqrt2 = _mm256_set1_ps(0.7071067811865475f32);
    let half = _mm256_set1_ps(0.5f32);
    let one = _mm256_set1_ps(1.0f32);
    let mut i = 0;
    while i + 32 <= len {
        let x0 = _mm256_loadu_ps(input.add(i));
        let x1 = _mm256_loadu_ps(input.add(i + 8));
        let x2 = _mm256_loadu_ps(input.add(i + 16));
        let x3 = _mm256_loadu_ps(input.add(i + 24));
        let r0 = _mm256_mul_ps(_mm256_mul_ps(x0, half), _mm256_add_ps(one, avx2_erf_ps(_mm256_mul_ps(x0, inv_sqrt2))));
        let r1 = _mm256_mul_ps(_mm256_mul_ps(x1, half), _mm256_add_ps(one, avx2_erf_ps(_mm256_mul_ps(x1, inv_sqrt2))));
        let r2 = _mm256_mul_ps(_mm256_mul_ps(x2, half), _mm256_add_ps(one, avx2_erf_ps(_mm256_mul_ps(x2, inv_sqrt2))));
        let r3 = _mm256_mul_ps(_mm256_mul_ps(x3, half), _mm256_add_ps(one, avx2_erf_ps(_mm256_mul_ps(x3, inv_sqrt2))));
        _mm256_storeu_ps(output.add(i), r0);
        _mm256_storeu_ps(output.add(i + 8), r1);
        _mm256_storeu_ps(output.add(i + 16), r2);
        _mm256_storeu_ps(output.add(i + 24), r3);
        i += 32;
    }
    while i + 8 <= len {
        let x = _mm256_loadu_ps(input.add(i));
        let r = _mm256_mul_ps(_mm256_mul_ps(x, half), _mm256_add_ps(one, avx2_erf_ps(_mm256_mul_ps(x, inv_sqrt2))));
        _mm256_storeu_ps(output.add(i), r);
        i += 8;
    }
    while i < len {
        let x = *input.add(i);
        *output.add(i) = x * 0.5 * (1.0 + libm::erff(x * 0.7071067811865475));
        i += 1;
    }
}

/// AVX2 fast GELU buffer kernel: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
/// Uses avx2_tanh_ps. 4-way unrolled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn fast_gelu_kernel(input: *const f32, output: *mut f32, len: usize) {
    let sqrt_2_over_pi = _mm256_set1_ps(0.7978845608028654f32);
    let coeff = _mm256_set1_ps(0.044715f32);
    let half = _mm256_set1_ps(0.5f32);
    let one = _mm256_set1_ps(1.0f32);
    let mut i = 0;
    // Inline helper: fast_gelu for one __m256
    macro_rules! fgelu {
        ($x:expr) => {{
            let x3 = _mm256_mul_ps(_mm256_mul_ps($x, $x), $x);
            let inner = _mm256_mul_ps(sqrt_2_over_pi, _mm256_fmadd_ps(coeff, x3, $x));
            _mm256_mul_ps(_mm256_mul_ps($x, half), _mm256_add_ps(one, avx2_tanh_ps(inner)))
        }};
    }
    while i + 32 <= len {
        let x0 = _mm256_loadu_ps(input.add(i));
        let x1 = _mm256_loadu_ps(input.add(i + 8));
        let x2 = _mm256_loadu_ps(input.add(i + 16));
        let x3 = _mm256_loadu_ps(input.add(i + 24));
        _mm256_storeu_ps(output.add(i), fgelu!(x0));
        _mm256_storeu_ps(output.add(i + 8), fgelu!(x1));
        _mm256_storeu_ps(output.add(i + 16), fgelu!(x2));
        _mm256_storeu_ps(output.add(i + 24), fgelu!(x3));
        i += 32;
    }
    while i + 8 <= len {
        let x = _mm256_loadu_ps(input.add(i));
        _mm256_storeu_ps(output.add(i), fgelu!(x));
        i += 8;
    }
    while i < len {
        let x = *input.add(i);
        let inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
        *output.add(i) = 0.5 * x * (1.0 + inner.tanh());
        i += 1;
    }
}

/// AVX2 f32 div: out[i] = a[i] / b[i]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn div_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let mut i = 0;
    while i + 32 <= len {
        let va0 = _mm256_loadu_ps(a.add(i));
        let vb0 = _mm256_loadu_ps(b.add(i));
        let va1 = _mm256_loadu_ps(a.add(i + 8));
        let vb1 = _mm256_loadu_ps(b.add(i + 8));
        let va2 = _mm256_loadu_ps(a.add(i + 16));
        let vb2 = _mm256_loadu_ps(b.add(i + 16));
        let va3 = _mm256_loadu_ps(a.add(i + 24));
        let vb3 = _mm256_loadu_ps(b.add(i + 24));
        _mm256_storeu_ps(out.add(i), _mm256_div_ps(va0, vb0));
        _mm256_storeu_ps(out.add(i + 8), _mm256_div_ps(va1, vb1));
        _mm256_storeu_ps(out.add(i + 16), _mm256_div_ps(va2, vb2));
        _mm256_storeu_ps(out.add(i + 24), _mm256_div_ps(va3, vb3));
        i += 32;
    }
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.add(i));
        let vb = _mm256_loadu_ps(b.add(i));
        _mm256_storeu_ps(out.add(i), _mm256_div_ps(va, vb));
        i += 8;
    }
    while i < len {
        *out.add(i) = *a.add(i) / *b.add(i);
        i += 1;
    }
}

/// AVX2 sub: out[i] = a[i] - b[i]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn sub_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let mut i = 0;
    while i + 32 <= len {
        let va0 = _mm256_loadu_ps(a.add(i));
        let vb0 = _mm256_loadu_ps(b.add(i));
        let va1 = _mm256_loadu_ps(a.add(i + 8));
        let vb1 = _mm256_loadu_ps(b.add(i + 8));
        let va2 = _mm256_loadu_ps(a.add(i + 16));
        let vb2 = _mm256_loadu_ps(b.add(i + 16));
        let va3 = _mm256_loadu_ps(a.add(i + 24));
        let vb3 = _mm256_loadu_ps(b.add(i + 24));
        _mm256_storeu_ps(out.add(i), _mm256_sub_ps(va0, vb0));
        _mm256_storeu_ps(out.add(i + 8), _mm256_sub_ps(va1, vb1));
        _mm256_storeu_ps(out.add(i + 16), _mm256_sub_ps(va2, vb2));
        _mm256_storeu_ps(out.add(i + 24), _mm256_sub_ps(va3, vb3));
        i += 32;
    }
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.add(i));
        let vb = _mm256_loadu_ps(b.add(i));
        _mm256_storeu_ps(out.add(i), _mm256_sub_ps(va, vb));
        i += 8;
    }
    while i < len {
        *out.add(i) = *a.add(i) - *b.add(i);
        i += 1;
    }
}

/// Scalar broadcast add: out[i] = a[i] + scalar
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn add_scalar_f32(a: *const f32, scalar: f32, out: *mut f32, len: usize) {
    let vs = _mm256_set1_ps(scalar);
    let mut i = 0;
    while i + 32 <= len {
        let v0 = _mm256_loadu_ps(a.add(i));
        let v1 = _mm256_loadu_ps(a.add(i + 8));
        let v2 = _mm256_loadu_ps(a.add(i + 16));
        let v3 = _mm256_loadu_ps(a.add(i + 24));
        _mm256_storeu_ps(out.add(i), _mm256_add_ps(v0, vs));
        _mm256_storeu_ps(out.add(i + 8), _mm256_add_ps(v1, vs));
        _mm256_storeu_ps(out.add(i + 16), _mm256_add_ps(v2, vs));
        _mm256_storeu_ps(out.add(i + 24), _mm256_add_ps(v3, vs));
        i += 32;
    }
    while i + 8 <= len {
        _mm256_storeu_ps(out.add(i), _mm256_add_ps(_mm256_loadu_ps(a.add(i)), vs));
        i += 8;
    }
    while i < len {
        *out.add(i) = *a.add(i) + scalar;
        i += 1;
    }
}

/// AVX2 memset: fill memory with zeros
#[target_feature(enable = "avx2")]
pub unsafe fn memset_zero_f32(ptr: *mut f32, len: usize) {
    let zero = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= len {
        _mm256_storeu_ps(ptr.add(i), zero);
        i += 8;
    }
    while i < len {
        *ptr.add(i) = 0.0;
        i += 1;
    }
}

/// AVX2 memory copy (non-temporal hint for large copies)
#[target_feature(enable = "avx2")]
pub unsafe fn copy_f32(src: *const f32, dst: *mut f32, len: usize) {
    let mut i = 0;
    // Use non-temporal hint for large copies to avoid cache pollution
    while i + 32 <= len {
        let v0 = _mm256_loadu_ps(src.add(i));
        let v1 = _mm256_loadu_ps(src.add(i + 8));
        let v2 = _mm256_loadu_ps(src.add(i + 16));
        let v3 = _mm256_loadu_ps(src.add(i + 24));
        _mm256_storeu_ps(dst.add(i), v0);
        _mm256_storeu_ps(dst.add(i + 8), v1);
        _mm256_storeu_ps(dst.add(i + 16), v2);
        _mm256_storeu_ps(dst.add(i + 24), v3);
        i += 32;
    }
    while i + 8 <= len {
        _mm256_storeu_ps(dst.add(i), _mm256_loadu_ps(src.add(i)));
        i += 8;
    }
    while i < len {
        *dst.add(i) = *src.add(i);
        i += 1;
    }
}
