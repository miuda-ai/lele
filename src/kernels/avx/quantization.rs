//! AVX2 quantize/dequantize of activations. (The int8 matrix products are
//! [`crate::kernels::qgemm`].)

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::kernels::utils;
use crate::tensor::TensorView;

/// Fused `QuantizeLinear` → `DequantizeLinear` round trip for a per-tensor
/// scale and zero point.
///
/// Division by exact `vdivps` on purpose. A reciprocal-plus-FMA-refine
/// replacement was measured here and dropped: it is bit-identical on
/// well-separated inputs but sits within half an ulp of every rounding tie,
/// and the kernel is memory-bound anyway — removing the divide bought only
/// 3-7% of the kernel and ~0.4 ms of a 58 ms inference, inside run-to-run
/// noise. Keep the arithmetic identical to the scalar fallback and the other
/// layouts; if the divide ever shows in a profile, revisit with a
/// bit-identicalness proof in hand.
///
/// NaN inputs clamp rather than propagate, because `vmaxps`/`vminps` return
/// their second operand for NaN; quantized activations are finite by
/// construction.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn fake_quantize_per_tensor_avx2(
    src: *const f32,
    dst: *mut f32,
    len: usize,
    scale: f32,
    zero_point: f32,
    qmin: f32,
    qmax: f32,
) {
    unsafe {
        let sv = _mm256_set1_ps(scale);
        let zv = _mm256_set1_ps(zero_point);
        let lo = _mm256_set1_ps(qmin);
        let hi = _mm256_set1_ps(qmax);

        macro_rules! round_trip {
            ($x:expr) => {{
                let x = $x;
                let q = _mm256_div_ps(x, sv);
                let q = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(q);
                let c = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(q, zv), lo), hi);
                _mm256_mul_ps(_mm256_sub_ps(c, zv), sv)
            }};
        }

        // Ordinary stores, deliberately: the output is read back by the very
        // next operator, so a non-temporal store only forces a round trip to
        // memory. Measured on Ice Lake it doubles the cost of the large
        // tensors rather than saving the read-for-ownership traffic.
        let mut i = 0;
        while i + 32 <= len {
            let r0 = round_trip!(_mm256_loadu_ps(src.add(i)));
            let r1 = round_trip!(_mm256_loadu_ps(src.add(i + 8)));
            let r2 = round_trip!(_mm256_loadu_ps(src.add(i + 16)));
            let r3 = round_trip!(_mm256_loadu_ps(src.add(i + 24)));
            _mm256_storeu_ps(dst.add(i), r0);
            _mm256_storeu_ps(dst.add(i + 8), r1);
            _mm256_storeu_ps(dst.add(i + 16), r2);
            _mm256_storeu_ps(dst.add(i + 24), r3);
            i += 32;
        }
        while i + 8 <= len {
            _mm256_storeu_ps(dst.add(i), round_trip!(_mm256_loadu_ps(src.add(i))));
            i += 8;
        }
        // The tail uses the same exact division as the vector body.
        while i < len {
            let v = *src.add(i);
            let q = (v / scale).round_ties_even() + zero_point;
            *dst.add(i) = (q.clamp(qmin, qmax) - zero_point) * scale;
            i += 1;
        }
    }
}

/// AVX2-optimized dynamic quantize linear
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dynamic_quantize_linear_avx2<'a, 'b>(
    x: &TensorView<'b, f32>,
    out_y: &'a mut Vec<f32>,
    out_scale: &'a mut Vec<f32>,
    out_zp: &'a mut Vec<f32>,
) -> (
    TensorView<'a, f32>,
    TensorView<'a, f32>,
    TensorView<'a, f32>,
) {
    unsafe {
        let len = x.data.len();
        if len == 0 {
            return (
                TensorView::from_owned(vec![], x.shape.to_vec()),
                TensorView::from_owned(vec![1.0], vec![1]),
                TensorView::from_owned(vec![0.0], vec![1]),
            );
        }

        // SIMD min/max finding
        let mut min_vec = _mm256_set1_ps(f32::MAX);
        let mut max_vec = _mm256_set1_ps(f32::MIN);
        let mut i = 0;
        let simd_end = (len / 8) * 8;
        let ptr = x.data.as_ptr();

        while i < simd_end {
            let v = _mm256_loadu_ps(ptr.add(i));
            min_vec = _mm256_min_ps(min_vec, v);
            max_vec = _mm256_max_ps(max_vec, v);
            i += 8;
        }

        // Horizontal min/max
        let mut min_val = hmin_ps(min_vec);
        let mut max_val = hmax_ps(max_vec);

        for j in simd_end..len {
            let v = *ptr.add(j);
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
        let inv_scale = 1.0 / scale;

        out_scale.clear();
        out_scale.push(scale);
        out_zp.clear();
        out_zp.push(zp);

        utils::ensure_capacity(out_y, len);

        // SIMD quantization
        let inv_scale_vec = _mm256_set1_ps(inv_scale);
        let zp_vec = _mm256_set1_ps(zp);
        let zero_vec = _mm256_setzero_ps();
        let max_255 = _mm256_set1_ps(255.0);
        let out_ptr = out_y.as_mut_ptr();

        i = 0;
        while i + 8 <= len {
            let v = _mm256_loadu_ps(ptr.add(i));
            let scaled = _mm256_fmadd_ps(v, inv_scale_vec, zp_vec);
            let rounded = _mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            let clamped = _mm256_min_ps(_mm256_max_ps(rounded, zero_vec), max_255);
            _mm256_storeu_ps(out_ptr.add(i), clamped);
            i += 8;
        }

        for j in i..len {
            *out_ptr.add(j) = (*ptr.add(j) * inv_scale + zp).round().clamp(0.0, 255.0);
        }

        (
            TensorView::from_slice(out_y, x.shape.to_vec()),
            TensorView::from_slice(out_scale, vec![1]),
            TensorView::from_slice(out_zp, vec![1]),
        )
    }
}

/// Horizontal min of 8 f32s in __m256
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hmin_ps(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let m128 = _mm_min_ps(lo, hi);
    let m64 = _mm_min_ps(m128, _mm_movehl_ps(m128, m128));
    let m32 = _mm_min_ss(m64, _mm_shuffle_ps(m64, m64, 1));
    _mm_cvtss_f32(m32)
}

/// Horizontal max of 8 f32s in __m256
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hmax_ps(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let m128 = _mm_max_ps(lo, hi);
    let m64 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
    let m32 = _mm_max_ss(m64, _mm_shuffle_ps(m64, m64, 1));
    _mm_cvtss_f32(m32)
}
