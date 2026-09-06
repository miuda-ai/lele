//! Static-QDQ matmul against int8 weights.
//!
//! ONNX QDQ graphs express a quantized linear layer as a dequantized activation
//! multiplied by a dequantized weight. Folding the weight's `DequantizeLinear`
//! into an f32 constant, as constant folding otherwise would, throws away the
//! only thing that makes an integer kernel possible. The compiler instead keeps
//! the int8 weight and emits a call here.
//!
//! Which kernel actually runs is a runtime decision made once per weight, when
//! it is prepared:
//!
//! * With AVX-512 VNNI, `vpdpbusd` does 64 MAC per instruction on two ports —
//!   four times the f32 FMA rate — so the weight is packed for
//!   [`crate::kernels::avx512::qgemm`] and the multiply runs in integers.
//! * Everywhere else the weight is dequantized to f32 once and the ordinary
//!   [`matmul`] runs, which is bit-identical to not having taken this path at
//!   all. The AVX2 integer kernels are not used: without VNNI the exact
//!   `vpmaddwd` sequence is barely faster than f32 FMA and `vpmaddubsw`
//!   saturates for realistic weights.

use crate::tensor::TensorView;

#[cfg(target_arch = "x86_64")]
use crate::kernels::avx512::qgemm::{PackedI8Weights, has_vnni, pack_i8_weights, qgemm_u8s8_f32};

/// A weight tensor prepared for whichever quantized path this machine can run.
pub enum QuantizedWeights {
    /// Packed for `vpdpbusd`, with the per-output-channel scale alongside.
    #[cfg(target_arch = "x86_64")]
    Vnni {
        packed: PackedI8Weights,
        scale: Vec<f32>,
    },
    /// Dequantized to f32 `[k, n]`; the ordinary f32 GEMM handles it.
    Float { data: Vec<f32>, k: usize, n: usize },
}

impl QuantizedWeights {
    /// True when the integer kernel will run, rather than the f32 fallback.
    pub fn is_integer(&self) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            matches!(self, QuantizedWeights::Vnni { .. })
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }
}

/// Prepares a row-major `[k, n]` int8 weight with a per-output-channel scale.
///
/// `raw` is the weight's bytes straight out of the model blob, reinterpreted as
/// i8. `w_scale` holds either one scale for the whole tensor or one per output
/// channel. Symmetric quantization (zero point 0) is assumed; the compiler only
/// routes weights here after checking that.
pub fn prepare_quantized_weights(
    raw: &[u8],
    k: usize,
    n: usize,
    w_scale: &[f32],
) -> QuantizedWeights {
    debug_assert_eq!(raw.len(), k * n);
    debug_assert!(w_scale.len() == 1 || w_scale.len() == n);

    #[cfg(target_arch = "x86_64")]
    if has_vnni() {
        // SAFETY: i8 and u8 have the same size and alignment, and every bit
        // pattern is valid for both.
        let as_i8: &[i8] = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const i8, k * n) };
        let scale = if w_scale.len() == 1 {
            vec![w_scale[0]; n]
        } else {
            w_scale.to_vec()
        };
        return QuantizedWeights::Vnni {
            packed: pack_i8_weights(as_i8, k, n),
            scale,
        };
    }

    let mut data = vec![0f32; k * n];
    for kk in 0..k {
        for j in 0..n {
            let s = if w_scale.len() == 1 { w_scale[0] } else { w_scale[j] };
            data[kk * n + j] = (raw[kk * n + j] as i8) as f32 * s;
        }
    }
    QuantizedWeights::Float { data, k, n }
}

/// Recovers the integer codes from an activation that a `QuantizeLinear` ->
/// `DequantizeLinear` pair already snapped onto the quantization grid.
///
/// The dequantized value is `scale * (code - zero_point)` exactly, so dividing
/// it back out lands within a fraction of an ulp of an integer and rounding
/// recovers the code. Rows are written at stride `lda` so K can be padded up to
/// the multiple of four `vpdpbusd` consumes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn requantize_rows_avx512(
    src: *const f32,
    dst: *mut u8,
    rows: usize,
    k: usize,
    lda: usize,
    scale: f32,
    zero_point: i32,
) {
    use std::arch::x86_64::*;
    unsafe {
        let sv = _mm512_set1_ps(scale);
        let zv = _mm512_set1_epi32(zero_point);
        let zero = _mm512_setzero_si512();
        for r in 0..rows {
            let s = src.add(r * k);
            let d = dst.add(r * lda);
            let mut i = 0;
            while i + 16 <= k {
                let v = _mm512_loadu_ps(s.add(i));
                // Default rounding is round-to-nearest-even, matching the
                // round_ties_even the quantize kernel used.
                let q = _mm512_cvtps_epi32(_mm512_div_ps(v, sv));
                let q = _mm512_max_epi32(_mm512_add_epi32(q, zv), zero);
                _mm_storeu_si128(d.add(i) as *mut __m128i, _mm512_cvtusepi32_epi8(q));
                i += 16;
            }
            while i < k {
                let q = (*s.add(i) / scale).round_ties_even() as i32 + zero_point;
                *d.add(i) = q.clamp(0, 255) as u8;
                i += 1;
            }
            // Zero the padding so it cannot perturb the accumulator even if the
            // packed weight bytes were ever non-zero there.
            for i in k..lda {
                *d.add(i) = 0;
            }
        }
    }
}

/// `out = a @ dequantize(w)`, computed in integers where the hardware allows.
///
/// `a` is the already-dequantized activation, still on the quantization grid,
/// with per-tensor `a_scale` and `a_zero_point`.
pub fn qmatmul_i8<'a>(
    a: &TensorView<'_, f32>,
    a_scale: &TensorView<'_, f32>,
    a_zero_point: &TensorView<'_, f32>,
    qw: &QuantizedWeights,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    match qw {
        QuantizedWeights::Float { data, k, n } => {
            let w = TensorView::from_slice(data.as_slice(), vec![*k, *n]);
            crate::kernels::matmul(a, &w, out)
        }
        #[cfg(target_arch = "x86_64")]
        QuantizedWeights::Vnni { packed, scale } => {
            let dims = a.shape.len();
            let m = a.shape[dims - 2];
            let k = a.shape[dims - 1];
            let n = packed.n;
            let batch: usize = a.shape[..dims - 2].iter().product::<usize>().max(1);

            let output_len = batch * m * n;
            crate::kernels::utils::ensure_capacity(out, output_len);
            out.resize(output_len, 0.0);

            let sa = a_scale.data.first().copied().unwrap_or(1.0);
            let za = a_zero_point.data.first().copied().unwrap_or(0.0) as i32;
            let lda = packed.k4;

            // The requantized activation is scratch, and this runs dozens of
            // times per forward pass, so keep the buffer rather than allocating
            // one each call.
            thread_local! {
                static A_U8: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
            }
            A_U8.with(|cell| {
                let mut a_u8 = cell.borrow_mut();
                a_u8.clear();
                a_u8.resize(m * lda, 0);
                let a_ptr = a.data.as_ptr();
                for b in 0..batch {
                    unsafe {
                        requantize_rows_avx512(
                            a_ptr.add(b * m * k),
                            a_u8.as_mut_ptr(),
                            m,
                            k,
                            lda,
                            sa,
                            za,
                        );
                        qgemm_u8s8_f32(
                            a_u8.as_ptr(),
                            m,
                            lda,
                            packed,
                            za,
                            sa,
                            scale.as_ptr(),
                            scale.len(),
                            None,
                            out.as_mut_ptr().add(b * m * n),
                            n,
                        );
                    }
                }
            });

            let mut shape = a.shape[..dims - 2].to_vec();
            shape.push(m);
            shape.push(n);
            TensorView::from_slice(out.as_slice(), shape)
        }
    }
}
