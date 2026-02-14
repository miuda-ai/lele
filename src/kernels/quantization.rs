use crate::tensor::TensorView;

// Re-export ARM prepared weights
#[cfg(target_arch = "aarch64")]
pub use crate::kernels::neon::quantization::{PreparedWeightsArm, prepare_weights_arm};

// MatMulInteger operation: accepts f32 tensors and converts internally to u8
pub fn mat_mul_integer<'a, 'b, 'c>(
    a: &TensorView<'b, f32>,
    b: &TensorView<'c, f32>,
    a_zero_point: Option<&TensorView<'b, f32>>,
    b_zero_point: Option<&TensorView<'c, f32>>,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    mat_mul_integer_with_scale_bias(a, b, a_zero_point, b_zero_point, None, None, out)
}

// MatMulInteger with optional bias fusion (backward compatibility)
pub fn mat_mul_integer_with_bias<'a, 'b, 'c>(
    a: &TensorView<'b, f32>,
    b: &TensorView<'c, f32>,
    a_zero_point: Option<&TensorView<'b, f32>>,
    b_zero_point: Option<&TensorView<'c, f32>>,
    bias: Option<&TensorView<'b, f32>>,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    mat_mul_integer_with_scale_bias(a, b, a_zero_point, b_zero_point, None, bias, out)
}

// MatMulInteger with optional scale and bias fusion (full fusion)
pub fn mat_mul_integer_with_scale_bias<'a, 'b, 'c>(
    a: &TensorView<'b, f32>,
    b: &TensorView<'c, f32>,
    a_zero_point: Option<&TensorView<'b, f32>>,
    b_zero_point: Option<&TensorView<'c, f32>>,
    scale: Option<&TensorView<'b, f32>>,
    bias: Option<&TensorView<'b, f32>>,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    mat_mul_integer_with_scale_bias_activation(
        a,
        b,
        a_zero_point,
        b_zero_point,
        scale,
        bias,
        false,
        out,
    )
}

// MatMulInteger with optional scale, bias, and ReLU fusion
pub fn mat_mul_integer_with_scale_bias_relu<'a, 'b, 'c>(
    a: &TensorView<'b, f32>,
    b: &TensorView<'c, f32>,
    a_zero_point: Option<&TensorView<'b, f32>>,
    b_zero_point: Option<&TensorView<'c, f32>>,
    scale: Option<&TensorView<'b, f32>>,
    bias: Option<&TensorView<'b, f32>>,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    mat_mul_integer_with_scale_bias_activation(
        a,
        b,
        a_zero_point,
        b_zero_point,
        scale,
        bias,
        true,
        out,
    )
}

/// Fully-fused quantized linear: DynamicQuantizeLinear + MatMulInteger + Scale + Bias [+ ReLU].
/// Eliminates intermediate allocations by using thread-local scratch buffers.
/// On x86_64, uses cached B weight transposes for near-zero overhead on repeated calls.
pub fn fused_quantized_linear<'a>(
    input: &TensorView<'_, f32>,
    weight_int8: &TensorView<'_, f32>,
    weight_scale: &TensorView<'_, f32>,
    weight_zero: &TensorView<'_, f32>,
    bias: &TensorView<'_, f32>,
    apply_relu: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    #[cfg(target_arch = "x86_64")]
    {
        let input_dims = input.shape.len();
        let b_dims = weight_int8.shape.len();
        let m = input.shape[input_dims - 2];
        let k = input.shape[input_dims - 1];
        let n = weight_int8.shape[b_dims - 1];

        let batch: usize = input.shape[..input_dims - 2].iter().product();
        let output_len = batch * m * n;
        crate::kernels::utils::ensure_capacity(out, output_len);
        out.resize(output_len, 0.0);

        let weight_zp = weight_zero.data.first().map(|&v| v as i32).unwrap_or(0);
        let stride_in = m * k;
        let stride_b = k * n;
        let stride_out = m * n;

        for b_i in 0..batch {
            let in_offset = b_i * stride_in;
            let b_offset = 0; // weights are not batched
            let out_offset = b_i * stride_out;

            unsafe {
                crate::kernels::avx::quantization::fused_dq_gemm_avx2(
                    input.data[in_offset..in_offset + stride_in].as_ptr(),
                    m,
                    k,
                    n,
                    weight_int8.data[b_offset..b_offset + stride_b].as_ptr(),
                    weight_scale.data.as_ptr(),
                    weight_scale.data.len(),
                    weight_zp,
                    if bias.data.is_empty() {
                        None
                    } else {
                        Some(bias.data.as_ptr())
                    },
                    apply_relu,
                    out[out_offset..out_offset + stride_out].as_mut_ptr(),
                );
            }
        }

        let mut output_shape = input.shape[..input_dims - 2].to_vec();
        output_shape.push(m);
        output_shape.push(n);
        TensorView::from_slice(out, output_shape)
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        use std::cell::RefCell;
        thread_local! {
            static BUF_Q: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
            static BUF_S: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
            static BUF_Z: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
            static BUF_SM: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
        }

        let mut buf_q = BUF_Q.with(|c| std::mem::take(&mut *c.borrow_mut()));
        let mut buf_s = BUF_S.with(|c| std::mem::take(&mut *c.borrow_mut()));
        let mut buf_z = BUF_Z.with(|c| std::mem::take(&mut *c.borrow_mut()));
        let mut buf_sm = BUF_SM.with(|c| std::mem::take(&mut *c.borrow_mut()));

        let (q, s, z) = dynamic_quantize_linear(input, &mut buf_q, &mut buf_s, &mut buf_z);
        let combined_scale = crate::kernels::mul(&s, weight_scale, &mut buf_sm);
        let result = mat_mul_integer_with_scale_bias_activation(
            &q,
            weight_int8,
            Some(&z),
            Some(weight_zero),
            Some(&combined_scale),
            Some(bias),
            apply_relu,
            out,
        );
        BUF_Q.with(|c| *c.borrow_mut() = buf_q);
        BUF_S.with(|c| *c.borrow_mut() = buf_s);
        BUF_Z.with(|c| *c.borrow_mut() = buf_z);
        BUF_SM.with(|c| *c.borrow_mut() = buf_sm);
        result
    }
}

/// Optimized version that takes raw u8 weight data directly, avoiding f32->u8 conversion.
/// `a` is quantized input (f32 values 0-255), `b_u8_data`/`b_shape` is the raw u8 weight.
pub fn mat_mul_integer_u8_weights<'a, 'b>(
    a: &TensorView<'b, f32>,
    b_u8_data: &[u8],
    b_shape: &[usize],
    a_zero_point: Option<f32>,
    b_zero_point: Option<u8>,
    scale: Option<&TensorView<'b, f32>>,
    bias: Option<&TensorView<'b, f32>>,
    apply_relu: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    // Convert only A from f32->u8 (small: M×K, typically 93×512)
    let a_u8: Vec<u8> = a.data.iter().map(|&x| x as u8).collect();
    let a_u8_view = TensorView::from_slice(&a_u8, a.shape.to_vec());
    let b_u8_view = TensorView::from_slice(b_u8_data, b_shape.to_vec());

    let a_zp_u8 = a_zero_point.map(|z| TensorView::from_owned(vec![z as u8], vec![1]));
    let b_zp_u8 = b_zero_point.map(|z| TensorView::from_owned(vec![z], vec![1]));

    mat_mul_integer_u8(
        &a_u8_view,
        &b_u8_view,
        a_zp_u8.as_ref(),
        b_zp_u8.as_ref(),
        scale,
        bias,
        apply_relu,
        out,
    )
}

/// Pre-processed weight data for fast quantized GEMM.
/// Stores the transposed+XOR'd weight matrix and column sums.
pub struct PreparedWeights {
    /// B transposed [N, K_padded] with XOR 0x80 applied (i8 reinterpretation)
    pub b_t: Vec<u8>,
    /// Column sums of original B u8 values (for zero-point correction)
    pub col_sums_b_u8: Vec<i32>,
    /// Original K dimension
    pub k: usize,
    /// K padded to multiple of 16
    pub k_padded: usize,
    /// N dimension (number of output columns)
    pub n: usize,
}

/// Pre-process weight matrix B [K, N] for fast quantized GEMM.
/// Transposes to [N, K_padded] and XORs with 0x80 for VPMADDWD compatibility.
pub fn prepare_weights(b_data: &[u8], k: usize, n: usize) -> PreparedWeights {
    let k_padded = (k + 15) & !15;

    #[cfg(target_arch = "x86_64")]
    {
        unsafe { prepare_weights_avx2(b_data, k, n, k_padded) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        prepare_weights_scalar(b_data, k, n, k_padded)
    }
}

#[allow(dead_code)]
fn prepare_weights_scalar(b_data: &[u8], k: usize, n: usize, k_padded: usize) -> PreparedWeights {
    let mut b_t = vec![0u8; n * k_padded];
    let mut col_sums_b_u8 = vec![0i32; n];

    for jj in 0..n {
        let mut csum: i32 = 0;
        for kk in 0..k {
            let b_val = b_data[kk * n + jj];
            b_t[jj * k_padded + kk] = b_val ^ 0x80;
            csum += b_val as i32;
        }
        col_sums_b_u8[jj] = csum;
    }

    PreparedWeights {
        b_t,
        col_sums_b_u8,
        k,
        k_padded,
        n,
    }
}

/// AVX2-optimized prepare_weights: 8×8 in-register transpose + XOR + vectorized column sums.
/// Two-pass approach:
///   Pass 1: Column sums via sequential row reads with AVX2 u8→i32 widening
///   Pass 2: Transpose + XOR using SSE2 8×8 byte transpose in registers
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn prepare_weights_avx2(
    b_data: &[u8],
    k: usize,
    n: usize,
    k_padded: usize,
) -> PreparedWeights {
    use std::arch::x86_64::*;

    // Allocate output. Only zero the padding region (k..k_padded) per row later.
    let total = n * k_padded;
    let mut b_t = Vec::<u8>::with_capacity(total);
    unsafe { b_t.set_len(total) };
    let mut col_sums_b_u8 = vec![0i32; n];

    // --- Pass 1: column sums (row-sequential reads, AVX2 u8→i32 accumulation) ---
    let sums = col_sums_b_u8.as_mut_ptr();
    for kk in 0..k {
        let row = unsafe { b_data.as_ptr().add(kk * n) };
        let mut jj = 0usize;
        while jj + 32 <= n {
            unsafe {
                let v = _mm256_loadu_si256(row.add(jj) as *const __m256i);
                let lo_128 = _mm256_castsi256_si128(v);
                let hi_128 = _mm256_extracti128_si256(v, 1);

                let s0 = _mm256_cvtepu8_epi32(lo_128);
                let s1 = _mm256_cvtepu8_epi32(_mm_srli_si128(lo_128, 8));
                let s2 = _mm256_cvtepu8_epi32(hi_128);
                let s3 = _mm256_cvtepu8_epi32(_mm_srli_si128(hi_128, 8));

                let a0 = _mm256_loadu_si256(sums.add(jj) as *const __m256i);
                let a1 = _mm256_loadu_si256(sums.add(jj + 8) as *const __m256i);
                let a2 = _mm256_loadu_si256(sums.add(jj + 16) as *const __m256i);
                let a3 = _mm256_loadu_si256(sums.add(jj + 24) as *const __m256i);

                _mm256_storeu_si256(sums.add(jj) as *mut __m256i, _mm256_add_epi32(a0, s0));
                _mm256_storeu_si256(sums.add(jj + 8) as *mut __m256i, _mm256_add_epi32(a1, s1));
                _mm256_storeu_si256(sums.add(jj + 16) as *mut __m256i, _mm256_add_epi32(a2, s2));
                _mm256_storeu_si256(sums.add(jj + 24) as *mut __m256i, _mm256_add_epi32(a3, s3));
            }
            jj += 32;
        }
        while jj < n {
            unsafe {
                *sums.add(jj) += *row.add(jj) as i32;
            }
            jj += 1;
        }
    }

    // --- Pass 2: Transpose + XOR in 8×8 blocks using SSE2 in-register transpose ---
    let b_t_ptr = b_t.as_mut_ptr();
    let xor_mask_128 = _mm_set1_epi8(0x80u8 as i8);

    // 8×8 byte transpose macro: loads 8 rows of 8 bytes, XORs with 0x80,
    // transposes in-register, stores 8 cols of 8 bytes.
    // Input: src_ptrs[0..8] point to the 8 bytes in each row
    // Output: dst_ptrs[0..8] point to where each transposed column goes
    macro_rules! transpose_8x8_xor {
        ($src:expr, $src_stride:expr, $dst:expr, $dst_stride:expr) => {
            unsafe {
                // Load 8 rows of 8 bytes each into the low 64 bits of __m128i
                let r0 = _mm_xor_si128(
                    _mm_loadl_epi64($src.add(0 * $src_stride) as *const __m128i),
                    xor_mask_128,
                );
                let r1 = _mm_xor_si128(
                    _mm_loadl_epi64($src.add(1 * $src_stride) as *const __m128i),
                    xor_mask_128,
                );
                let r2 = _mm_xor_si128(
                    _mm_loadl_epi64($src.add(2 * $src_stride) as *const __m128i),
                    xor_mask_128,
                );
                let r3 = _mm_xor_si128(
                    _mm_loadl_epi64($src.add(3 * $src_stride) as *const __m128i),
                    xor_mask_128,
                );
                let r4 = _mm_xor_si128(
                    _mm_loadl_epi64($src.add(4 * $src_stride) as *const __m128i),
                    xor_mask_128,
                );
                let r5 = _mm_xor_si128(
                    _mm_loadl_epi64($src.add(5 * $src_stride) as *const __m128i),
                    xor_mask_128,
                );
                let r6 = _mm_xor_si128(
                    _mm_loadl_epi64($src.add(6 * $src_stride) as *const __m128i),
                    xor_mask_128,
                );
                let r7 = _mm_xor_si128(
                    _mm_loadl_epi64($src.add(7 * $src_stride) as *const __m128i),
                    xor_mask_128,
                );

                // Step 1: interleave bytes (8-bit)
                let t0 = _mm_unpacklo_epi8(r0, r1); // [a00 a10 a01 a11 ... a07 a17]
                let t2 = _mm_unpacklo_epi8(r2, r3);
                let t4 = _mm_unpacklo_epi8(r4, r5);
                let t6 = _mm_unpacklo_epi8(r6, r7);

                // Step 2: interleave 16-bit words
                let u0 = _mm_unpacklo_epi16(t0, t2); // [a00 a10 a20 a30 a01 a11 a21 a31 | ...]
                let u1 = _mm_unpackhi_epi16(t0, t2);
                let u4 = _mm_unpacklo_epi16(t4, t6);
                let u5 = _mm_unpackhi_epi16(t4, t6);

                // Step 3: interleave 32-bit dwords → each result has 2 transposed columns
                let c01 = _mm_unpacklo_epi32(u0, u4); // col0 (low 64) | col1 (high 64)
                let c23 = _mm_unpackhi_epi32(u0, u4);
                let c45 = _mm_unpacklo_epi32(u1, u5);
                let c67 = _mm_unpackhi_epi32(u1, u5);

                // Store: each transposed column is 8 consecutive bytes in the output
                _mm_storel_epi64($dst.add(0 * $dst_stride) as *mut __m128i, c01);
                _mm_storel_epi64(
                    $dst.add(1 * $dst_stride) as *mut __m128i,
                    _mm_srli_si128(c01, 8),
                );
                _mm_storel_epi64($dst.add(2 * $dst_stride) as *mut __m128i, c23);
                _mm_storel_epi64(
                    $dst.add(3 * $dst_stride) as *mut __m128i,
                    _mm_srli_si128(c23, 8),
                );
                _mm_storel_epi64($dst.add(4 * $dst_stride) as *mut __m128i, c45);
                _mm_storel_epi64(
                    $dst.add(5 * $dst_stride) as *mut __m128i,
                    _mm_srli_si128(c45, 8),
                );
                _mm_storel_epi64($dst.add(6 * $dst_stride) as *mut __m128i, c67);
                _mm_storel_epi64(
                    $dst.add(7 * $dst_stride) as *mut __m128i,
                    _mm_srli_si128(c67, 8),
                );
            }
        }
    }

    // Process 8×8 blocks
    let k8 = k & !7;
    let n8 = n & !7;

    for kk_base in (0..k8).step_by(8) {
        for jj_base in (0..n8).step_by(8) {
            let src = unsafe { b_data.as_ptr().add(kk_base * n + jj_base) };
            let dst = unsafe { b_t_ptr.add(jj_base * k_padded + kk_base) };
            transpose_8x8_xor!(src, n, dst, k_padded);
        }
        // Remainder N columns (< 8)
        for jj in n8..n {
            for dk in 0..8usize {
                let kk = kk_base + dk;
                unsafe {
                    *b_t_ptr.add(jj * k_padded + kk) = *b_data.as_ptr().add(kk * n + jj) ^ 0x80;
                }
            }
        }
    }
    // Remainder K rows (< 8)
    for kk in k8..k {
        for jj in 0..n {
            unsafe {
                *b_t_ptr.add(jj * k_padded + kk) = *b_data.as_ptr().add(kk * n + jj) ^ 0x80;
            }
        }
    }

    // Zero padding bytes (k..k_padded) for each row
    if k_padded > k {
        let pad = k_padded - k;
        for jj in 0..n {
            unsafe {
                std::ptr::write_bytes(b_t_ptr.add(jj * k_padded + k), 0, pad);
            }
        }
    }

    PreparedWeights {
        b_t,
        col_sums_b_u8,
        k,
        k_padded,
        n,
    }
}

/// Fused DynamicQuantizeLinear + prepared-weight GEMM for x86.
/// Fully inlined: reuses ALL scratch buffers (DynQuant + a_u8 + combined_scale),
/// calls gemm_row_avx2 directly. Zero per-call heap allocations.
#[cfg(target_arch = "x86_64")]
pub fn fused_dq_gemm_prepared_x86<'a>(
    input: &TensorView<'_, f32>,
    pw: &PreparedWeights,
    b_zero_point: Option<u8>,
    weight_scale: &TensorView<'_, f32>,
    bias: Option<&TensorView<'_, f32>>,
    apply_relu: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    use std::cell::RefCell;
    thread_local! {
        static SCRATCH: RefCell<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<u8>)> =
            RefCell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()));
    }

    let a_dims = input.shape.len();
    let m = input.shape[a_dims - 2];
    let k = pw.k;
    let n = pw.n;
    let k_padded = pw.k_padded;
    let batch_shape = &input.shape[..a_dims.saturating_sub(2)];
    let batch: usize = batch_shape.iter().product();
    let total_batch = batch.max(1);

    let output_len = total_batch * m * n;
    crate::kernels::utils::ensure_capacity(out, output_len);
    out.resize(output_len, 0.0);

    SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        let (buf_q, buf_s, buf_z, buf_sm, buf_a_u8) = &mut *scratch;

        // Dynamic quantize input (reuses scratch buffers)
        let (q, s, z) = dynamic_quantize_linear(input, buf_q, buf_s, buf_z);

        // Compute combined scale = dyn_scale * weight_scale (AVX2 vectorized)
        let dyn_scale = s.data[0];
        let ws_data = &weight_scale.data;
        let ws_len = ws_data.len();
        crate::kernels::utils::ensure_capacity(buf_sm, ws_len);
        unsafe {
            buf_sm.set_len(ws_len);
            let src = ws_data.as_ptr();
            let dst = buf_sm.as_mut_ptr();
            use std::arch::x86_64::*;
            let scale_vec = _mm256_set1_ps(dyn_scale);
            let mut i = 0;
            while i + 8 <= ws_len {
                let v = _mm256_loadu_ps(src.add(i));
                _mm256_storeu_ps(dst.add(i), _mm256_mul_ps(v, scale_vec));
                i += 8;
            }
            while i < ws_len {
                *dst.add(i) = dyn_scale * *src.add(i);
                i += 1;
            }
        }

        // Convert quantized input f32 → u8 (reuse scratch buffer, zero alloc)
        let q_len = q.data.len();
        if buf_a_u8.capacity() < q_len {
            buf_a_u8.reserve(q_len - buf_a_u8.len());
        }
        unsafe {
            buf_a_u8.set_len(q_len);
            crate::kernels::avx::quantization::f32_to_u8_avx2(
                q.data.as_ptr(),
                buf_a_u8.as_mut_ptr(),
                q_len,
            );
        }

        // GEMM with prepared weights
        let zp_a = z.data.first().map(|&v| v as i32).unwrap_or(0);
        let zp_b = b_zero_point.unwrap_or(0) as i32;
        let k_zp_b = k as i32 * zp_b;
        let corr_128_minus_zpb = 128 - zp_b;
        let k_aligned = k == k_padded;

        let scale_data_ptr = Some(buf_sm.as_ptr());
        let scale_len = ws_len;
        let bias_data_ptr = bias.map(|b| b.data.as_ptr());

        // N-tiled GEMM for cache efficiency
        let l2_budget = 192 * 1024;
        let tile_n = if k_padded > 0 {
            (l2_budget / k_padded).max(8).min(n)
        } else {
            n
        };
        let tile_n = ((tile_n + 7) & !7).min(n);

        let stride_a = m * k;
        let stride_out = m * n;

        for b_i in 0..total_batch {
            let a_batch = &buf_a_u8[b_i * stride_a..(b_i + 1) * stride_a];
            let out_batch = &mut out[b_i * stride_out..(b_i + 1) * stride_out];

            // Precompute row sums for multi-row kernel
            let mut row_sums: Vec<i32> = Vec::with_capacity(m);
            unsafe {
                for i in 0..m {
                    row_sums.push(crate::kernels::avx::quantization::row_sum_u8_avx2(
                        &a_batch[i * k..i * k + k],
                    ));
                }
            }

            if tile_n >= n {
                // Process pairs of rows with 2-row kernel
                let mut i = 0;
                while i + 2 <= m {
                    unsafe {
                        let (out0, out1) = out_batch.split_at_mut((i + 1) * n);
                        crate::kernels::avx::quantization::gemm_2rows_avx2(
                            &a_batch[i * k..i * k + k],
                            &a_batch[(i + 1) * k..(i + 1) * k + k],
                            pw.b_t.as_ptr(),
                            k,
                            n,
                            k_padded,
                            k_aligned,
                            pw.col_sums_b_u8.as_ptr(),
                            zp_a,
                            zp_b,
                            k_zp_b,
                            corr_128_minus_zpb,
                            scale_data_ptr,
                            scale_len,
                            bias_data_ptr,
                            apply_relu,
                            row_sums[i],
                            row_sums[i + 1],
                            &mut out0[i * n..(i + 1) * n],
                            &mut out1[0..n],
                        );
                    }
                    i += 2;
                }
                // Handle odd last row with single-row kernel
                if i < m {
                    unsafe {
                        crate::kernels::avx::quantization::gemm_row_avx2(
                            &a_batch[i * k..i * k + k],
                            pw.b_t.as_ptr(),
                            k,
                            n,
                            k_padded,
                            k_aligned,
                            pw.col_sums_b_u8.as_ptr(),
                            zp_a,
                            zp_b,
                            k_zp_b,
                            corr_128_minus_zpb,
                            scale_data_ptr,
                            scale_len,
                            bias_data_ptr,
                            apply_relu,
                            &mut out_batch[i * n..(i + 1) * n],
                        );
                    }
                }
            } else {
                let mut n_start = 0;
                while n_start < n {
                    let n_end = (n_start + tile_n).min(n);
                    let cur_n = n_end - n_start;
                    // Process pairs of rows with 2-row kernel
                    let mut i = 0;
                    while i + 2 <= m {
                        unsafe {
                            let (out0, out1) = out_batch.split_at_mut((i + 1) * n);
                            crate::kernels::avx::quantization::gemm_2rows_avx2(
                                &a_batch[i * k..i * k + k],
                                &a_batch[(i + 1) * k..(i + 1) * k + k],
                                pw.b_t.as_ptr().add(n_start * k_padded),
                                k,
                                cur_n,
                                k_padded,
                                k_aligned,
                                pw.col_sums_b_u8.as_ptr().add(n_start),
                                zp_a,
                                zp_b,
                                k_zp_b,
                                corr_128_minus_zpb,
                                scale_data_ptr.map(|p| {
                                    if scale_len == 1 {
                                        p
                                    } else {
                                        p.add(n_start)
                                    }
                                }),
                                scale_len,
                                bias_data_ptr.map(|p| p.add(n_start)),
                                apply_relu,
                                row_sums[i],
                                row_sums[i + 1],
                                &mut out0[i * n + n_start..i * n + n_end],
                                &mut out1[n_start..n_end],
                            );
                        }
                        i += 2;
                    }
                    // Handle odd last row
                    if i < m {
                        unsafe {
                            crate::kernels::avx::quantization::gemm_row_avx2(
                                &a_batch[i * k..i * k + k],
                                pw.b_t.as_ptr().add(n_start * k_padded),
                                k,
                                cur_n,
                                k_padded,
                                k_aligned,
                                pw.col_sums_b_u8.as_ptr().add(n_start),
                                zp_a,
                                zp_b,
                                k_zp_b,
                                corr_128_minus_zpb,
                                scale_data_ptr.map(|p| {
                                    if scale_len == 1 {
                                        p
                                    } else {
                                        p.add(n_start)
                                    }
                                }),
                                scale_len,
                                bias_data_ptr.map(|p| p.add(n_start)),
                                apply_relu,
                                &mut out_batch[i * n + n_start..i * n + n_end],
                            );
                        }
                    }
                    n_start = n_end;
                }
            }
        }
    });

    // Create TensorView from `out` after closure
    let mut output_shape = batch_shape.to_vec();
    output_shape.push(m);
    output_shape.push(n);
    TensorView::from_slice(out, output_shape)
}

/// Fast GEMM with pre-processed weights. Avoids per-call transpose.
/// Uses multi-threaded row parallelism for large M dimensions.
/// `a` is f32 quantized input (values 0-255), `pw` is pre-processed weight data.
pub fn mat_mul_integer_prepared<'a, 'b>(
    a: &TensorView<'b, f32>,
    pw: &PreparedWeights,
    a_zero_point: Option<f32>,
    b_zero_point: Option<u8>,
    scale: Option<&TensorView<'b, f32>>,
    bias: Option<&TensorView<'b, f32>>,
    apply_relu: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    // Convert A from f32 to u8 (vectorized on x86)
    let len = a.data.len();
    let mut a_u8: Vec<u8> = Vec::with_capacity(len);
    unsafe {
        a_u8.set_len(len);
        #[cfg(target_arch = "x86_64")]
        {
            crate::kernels::avx::quantization::f32_to_u8_avx2(
                a.data.as_ptr(),
                a_u8.as_mut_ptr(),
                len,
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let src = a.data.as_ptr();
            let dst = a_u8.as_mut_ptr();
            for i in 0..len {
                *dst.add(i) = *src.add(i) as u8;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        let a_dims = a.shape.len();
        let m = a.shape[a_dims - 2];
        let batch: usize = a.shape[..a_dims - 2].iter().product();
        let batch_shape = &a.shape[..a_dims - 2];

        let total_batch = batch.max(1);
        let output_len = total_batch * m * pw.n;
        crate::kernels::utils::ensure_capacity(out, output_len);
        out.resize(output_len, 0.0);

        let zp_a = a_zero_point.unwrap_or(0.0) as i32;
        let zp_b = b_zero_point.unwrap_or(0) as i32;
        let k = pw.k;
        let n = pw.n;
        let k_padded = pw.k_padded;
        let k_zp_b = k as i32 * zp_b;
        let corr_128_minus_zpb = 128 - zp_b;
        let k_aligned = k == k_padded;
        let stride_a = m * k;
        let stride_out = m * n;

        // Extract scale/bias data for gemm_row_avx2
        let scale_data_ptr = scale.map(|s| s.data.as_ptr());
        let scale_len = scale.map(|s| s.data.len()).unwrap_or(0);
        let bias_data_ptr = bias.map(|b| b.data.as_ptr());

        // N-tiled GEMM for cache efficiency: keep B_T tile in L2
        let l2_budget = 192 * 1024; // 192KB for B_T tile
        let tile_n = if k_padded > 0 {
            (l2_budget / k_padded).max(8).min(n)
        } else {
            n
        };
        let tile_n = ((tile_n + 7) & !7).min(n);

        for b_i in 0..total_batch {
            let a_batch = &a_u8[b_i * stride_a..(b_i + 1) * stride_a];
            let out_batch = &mut out[b_i * stride_out..(b_i + 1) * stride_out];

            if tile_n >= n {
                // No tiling needed
                for i in 0..m {
                    unsafe {
                        crate::kernels::avx::quantization::gemm_row_avx2(
                            &a_batch[i * k..i * k + k],
                            pw.b_t.as_ptr(),
                            k,
                            n,
                            k_padded,
                            k_aligned,
                            pw.col_sums_b_u8.as_ptr(),
                            zp_a,
                            zp_b,
                            k_zp_b,
                            corr_128_minus_zpb,
                            scale_data_ptr,
                            scale_len,
                            bias_data_ptr,
                            apply_relu,
                            &mut out_batch[i * n..(i + 1) * n],
                        );
                    }
                }
            } else {
                // N-tiled: process tile_n columns at a time for all rows
                let mut n_start = 0;
                while n_start < n {
                    let n_end = (n_start + tile_n).min(n);
                    let cur_n = n_end - n_start;

                    for i in 0..m {
                        unsafe {
                            crate::kernels::avx::quantization::gemm_row_avx2(
                                &a_batch[i * k..i * k + k],
                                pw.b_t.as_ptr().add(n_start * k_padded),
                                k,
                                cur_n,
                                k_padded,
                                k_aligned,
                                pw.col_sums_b_u8.as_ptr().add(n_start),
                                zp_a,
                                zp_b,
                                k_zp_b,
                                corr_128_minus_zpb,
                                scale_data_ptr
                                    .map(|p| if scale_len == 1 { p } else { p.add(n_start) }),
                                scale_len,
                                bias_data_ptr.map(|p| p.add(n_start)),
                                apply_relu,
                                &mut out_batch[i * n + n_start..i * n + n_end],
                            );
                        }
                    }

                    n_start = n_end;
                }
            }
        }

        let mut output_shape = batch_shape.to_vec();
        output_shape.push(m);
        output_shape.push(n);
        TensorView::from_slice(out, output_shape)
    }
    #[cfg(target_arch = "aarch64")]
    {
        // Use the optimized NEON prepared kernel directly
        let a_dims = a.shape.len();
        let m = a.shape[a_dims - 2];
        let batch: usize = a.shape[..a_dims - 2].iter().product();
        let batch_shape = &a.shape[..a_dims - 2];

        let total_batch = batch.max(1);
        let output_len = total_batch * m * pw.n;
        crate::kernels::utils::ensure_capacity(out, output_len);
        out.resize(output_len, 0.0);

        let zp_a = a_zero_point.unwrap_or(0.0) as i32;
        let zp_b = b_zero_point.unwrap_or(0) as i32;
        let k = pw.k;
        let n = pw.n;
        let stride_a = m * k;
        let stride_out = m * n;

        // Reconstruct B u8 from the x86-format PreparedWeights (transpose + XOR)
        let mut b_u8 = vec![0u8; pw.k * pw.n];
        for jj in 0..pw.n {
            for kk in 0..pw.k {
                b_u8[kk * pw.n + jj] = pw.b_t[jj * pw.k_padded + kk] ^ 0x80;
            }
        }

        // Create ARM prepared weights from reconstructed B
        let pw_arm = crate::kernels::neon::quantization::prepare_weights_arm(&b_u8, k, n);

        for b_i in 0..total_batch {
            let a_batch = &a_u8[b_i * stride_a..(b_i + 1) * stride_a];

            if total_batch == 1 {
                crate::kernels::neon::quantization::mat_mul_integer_prepared_neon(
                    a_batch, m, k, &pw_arm, zp_a, zp_b, scale, bias, apply_relu, out,
                );
            } else {
                let mut batch_out = vec![0f32; stride_out];
                crate::kernels::neon::quantization::mat_mul_integer_prepared_neon(
                    a_batch,
                    m,
                    k,
                    &pw_arm,
                    zp_a,
                    zp_b,
                    scale,
                    bias,
                    apply_relu,
                    &mut batch_out,
                );
                let out_offset = b_i * stride_out;
                out[out_offset..out_offset + stride_out].copy_from_slice(&batch_out);
            }
        }

        let mut output_shape = batch_shape.to_vec();
        output_shape.push(m);
        output_shape.push(n);
        TensorView::from_slice(out, output_shape)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        // Generic fallback: reconstruct original B layout and use existing path
        let a_u8_view = TensorView::from_slice(&a_u8, a.shape.to_vec());
        let mut b_u8 = vec![0u8; pw.k * pw.n];
        for jj in 0..pw.n {
            for kk in 0..pw.k {
                b_u8[kk * pw.n + jj] = pw.b_t[jj * pw.k_padded + kk] ^ 0x80;
            }
        }
        let b_view = TensorView::from_slice(&b_u8, vec![pw.k, pw.n]);
        let a_zp = a_zero_point.map(|z| TensorView::from_owned(vec![z as u8], vec![1]));
        let b_zp = b_zero_point.map(|z| TensorView::from_owned(vec![z], vec![1]));
        mat_mul_integer_u8(
            &a_u8_view,
            &b_view,
            a_zp.as_ref(),
            b_zp.as_ref(),
            scale,
            bias,
            apply_relu,
            out,
        )
    }
}

// Internal function with activation parameter
fn mat_mul_integer_with_scale_bias_activation<'a, 'b, 'c>(
    a: &TensorView<'b, f32>,
    b: &TensorView<'c, f32>,
    a_zero_point: Option<&TensorView<'b, f32>>,
    b_zero_point: Option<&TensorView<'c, f32>>,
    scale: Option<&TensorView<'b, f32>>,
    bias: Option<&TensorView<'b, f32>>,
    apply_relu: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    #[cfg(target_arch = "x86_64")]
    {
        // Fused x86 path: AVX2-vectorized f32→u8 + tiled B transpose + N-tiled GEMM
        let a_dims = a.shape.len();
        let b_dims = b.shape.len();
        let m = a.shape[a_dims - 2];
        let k = a.shape[a_dims - 1];
        let n = b.shape[b_dims - 1];

        let batch_a: usize = a.shape[..a_dims - 2].iter().product();
        let batch_b: usize = b.shape[..b_dims - 2].iter().product();
        let final_batch = batch_a.max(batch_b);

        let output_len = final_batch * m * n;
        crate::kernels::utils::ensure_capacity(out, output_len);
        out.resize(output_len, 0.0);

        let zp_a = a_zero_point
            .and_then(|z| z.data.first())
            .map(|&v| v as i32)
            .unwrap_or(0);
        let zp_b = b_zero_point
            .and_then(|z| z.data.first())
            .map(|&v| v as i32)
            .unwrap_or(0);

        let stride_a = m * k;
        let stride_b = k * n;
        let stride_out = m * n;

        for b_i in 0..final_batch {
            let a_offset = if batch_a == 1 { 0 } else { b_i * stride_a };
            let b_offset = if batch_b == 1 { 0 } else { b_i * stride_b };
            let out_offset = b_i * stride_out;

            let a_slice = &a.data[a_offset..a_offset + stride_a];
            let b_slice = &b.data[b_offset..b_offset + stride_b];
            let out_slice = &mut out[out_offset..out_offset + stride_out];

            unsafe {
                crate::kernels::avx::quantization::mat_mul_integer_fused_f32_avx2(
                    a_slice, b_slice, m, k, n, zp_a, zp_b, scale, bias, apply_relu, out_slice,
                );
            }
        }

        let mut output_shape = if batch_a >= batch_b {
            a.shape[..a_dims - 2].to_vec()
        } else {
            b.shape[..b_dims - 2].to_vec()
        };
        output_shape.push(m);
        output_shape.push(n);

        TensorView::from_slice(out, output_shape)
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        // Generic fallback: convert f32 tensors to u8
        let a_u8: Vec<u8> = a.data.iter().map(|&x| x as u8).collect();
        let b_u8: Vec<u8> = b.data.iter().map(|&x| x as u8).collect();

        let a_u8_view = TensorView::from_slice(&a_u8, a.shape.to_vec());
        let b_u8_view = TensorView::from_slice(&b_u8, b.shape.to_vec());

        let a_zp_u8 = a_zero_point.map(|z| {
            let data: Vec<u8> = z.data.iter().map(|&x| x as u8).collect();
            TensorView::from_owned(data, z.shape.to_vec())
        });
        let b_zp_u8 = b_zero_point.map(|z| {
            let data: Vec<u8> = z.data.iter().map(|&x| x as u8).collect();
            TensorView::from_owned(data, z.shape.to_vec())
        });

        mat_mul_integer_u8(
            &a_u8_view,
            &b_u8_view,
            a_zp_u8.as_ref(),
            b_zp_u8.as_ref(),
            scale,
            bias,
            apply_relu,
            out,
        )
    }
}

// True quantization version (u8 x u8 -> f32 output)
fn mat_mul_integer_u8<'a, 'b, 'c>(
    a: &TensorView<'b, u8>,
    b: &TensorView<'c, u8>,
    a_zero_point: Option<&TensorView<'b, u8>>,
    b_zero_point: Option<&TensorView<'c, u8>>,
    scale: Option<&TensorView<'b, f32>>,
    bias: Option<&TensorView<'b, f32>>,
    apply_relu: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    #[cfg(target_arch = "aarch64")]
    {
        crate::kernels::neon::quantization::mat_mul_integer_u8(
            a,
            b,
            a_zero_point,
            b_zero_point,
            scale,
            bias,
            apply_relu,
            out,
        )
    }
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            crate::kernels::avx::quantization::mat_mul_integer_u8_avx2(
                a,
                b,
                a_zero_point,
                b_zero_point,
                scale,
                bias,
                apply_relu,
                out,
            )
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        use crate::kernels::utils;
        let zp_a_ref: &[u8] = a_zero_point.map(|z| z.data.as_ref()).unwrap_or(&[]);
        let zp_b_ref: &[u8] = b_zero_point.map(|z| z.data.as_ref()).unwrap_or(&[]);
        let zp_a_scalar = if zp_a_ref.len() == 1 {
            zp_a_ref[0] as f32
        } else {
            0.0
        };
        let zp_b_scalar = if zp_b_ref.len() == 1 {
            zp_b_ref[0] as f32
        } else {
            0.0
        };

        let a_dims = a.shape.len();
        let b_dims = b.shape.len();
        let m = a.shape[a_dims - 2];
        let k = a.shape[a_dims - 1];
        let n = b.shape[b_dims - 1];

        // Batch handling
        let batch_a: usize = a.shape[..a_dims - 2].iter().product();
        let batch_b: usize = b.shape[..b_dims - 2].iter().product();
        let final_batch = batch_a.max(batch_b);

        let output_len = final_batch * m * n;
        crate::kernels::utils::ensure_capacity(out, output_len);

        // Ensure exact size for safety
        out.resize(output_len, 0.0);

        let stride_a = m * k;
        let stride_b = k * n;
        let stride_out = m * n;

        // Naive loop with f32 accumulation (slow) - Fallback
        for b_i in 0..final_batch {
            let a_offset = if batch_a == 1 { 0 } else { b_i * stride_a };
            let b_offset = if batch_b == 1 { 0 } else { b_i * stride_b };
            let out_offset = b_i * stride_out;

            let a_data = &a.data[a_offset..a_offset + stride_a];
            let b_data = &b.data[b_offset..b_offset + stride_b];
            let out_data = &mut out[out_offset..out_offset + stride_out];

            for i in 0..m {
                let global_row = b_i * m + i;
                let zp_a = if zp_a_ref.len() > 1 {
                    zp_a_ref[global_row % zp_a_ref.len()] as f32
                } else {
                    zp_a_scalar
                };

                for j in 0..n {
                    let zp_b = if zp_b_ref.len() > 1 {
                        zp_b_ref[j] as f32
                    } else {
                        zp_b_scalar
                    };

                    let mut sum = 0.0;
                    for l in 0..k {
                        let val_a = a_data[i * k + l] as f32 - zp_a;
                        let val_b = b_data[l * n + j] as f32 - zp_b;
                        sum += val_a * val_b;
                    }

                    // Apply scale if provided
                    if let Some(scale_data) = scale {
                        if scale_data.data.len() == 1 {
                            sum *= scale_data.data[0];
                        } else {
                            sum *= scale_data.data[j];
                        }
                    }

                    // Apply bias if provided (per-column)
                    if let Some(bias_data) = bias {
                        sum += bias_data.data[j];
                    }

                    // Apply ReLU if requested
                    if apply_relu && sum < 0.0 {
                        sum = 0.0;
                    }

                    out_data[i * n + j] = sum;
                }
            }
        }

        let mut output_shape = if batch_a >= batch_b {
            a.shape[..a_dims - 2].to_vec()
        } else {
            b.shape[..b_dims - 2].to_vec()
        };
        output_shape.push(m);
        output_shape.push(n);

        TensorView::from_slice(out, output_shape)
    }
}

#[cfg(target_arch = "aarch64")]
pub fn fused_dq_gemm_prepared_arm<'a>(
    input: &TensorView<'_, f32>,
    pw_arm: &PreparedWeightsArm,
    b_zero_point: Option<u8>,
    weight_scale: &TensorView<'_, f32>,
    bias: Option<&TensorView<'_, f32>>,
    apply_relu: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    use std::cell::RefCell;
    thread_local! {
        static SCRATCH: RefCell<(Vec<u8>, Vec<f32>)> = RefCell::new((Vec::new(), Vec::new()));
    }

    let a_dims = input.shape.len();
    let m = input.shape[a_dims - 2];
    let k = input.shape[a_dims - 1];
    let batch_shape = &input.shape[..a_dims.saturating_sub(2)];
    let batch: usize = batch_shape.iter().product();
    let total_batch = batch.max(1);
    let zp_b = b_zero_point.unwrap_or(0) as i32;
    let n = pw_arm.n;

    if total_batch <= 1 {
        // Fast path: no batching
        SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let (a_u8, scale_buf) = &mut *scratch;

            crate::kernels::neon::quantization::fused_dq_gemm_neon(
                &input.data,
                m,
                k,
                pw_arm,
                zp_b,
                &weight_scale.data,
                bias.map(|b| &*b.data),
                apply_relu,
                a_u8,
                scale_buf,
                out,
            );
        });
        // Preserve batch dimensions in output shape
        if batch_shape.is_empty() {
            TensorView::from_slice(out, vec![m, n])
        } else {
            let mut output_shape = batch_shape.to_vec();
            output_shape.push(m);
            output_shape.push(n);
            TensorView::from_slice(out, output_shape)
        }
    } else {
        // Batch path
        let stride_in = m * k;
        let stride_out = m * n;
        let output_len = total_batch * stride_out;
        crate::kernels::utils::ensure_capacity(out, output_len);
        out.resize(output_len, 0.0);

        SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let (a_u8, scale_buf) = &mut *scratch;

            for b_i in 0..total_batch {
                let batch_start = b_i * stride_in;
                let batch_end = batch_start + stride_in;
                let batch_data = &input.data[batch_start..batch_end];

                let mut batch_out = vec![0f32; stride_out];
                // Create a temporary TensorView-like slice for this batch
                crate::kernels::neon::quantization::fused_dq_gemm_neon(
                    batch_data,
                    m,
                    k,
                    pw_arm,
                    zp_b,
                    &weight_scale.data,
                    bias.map(|b| &*b.data),
                    apply_relu,
                    a_u8,
                    scale_buf,
                    &mut batch_out,
                );
                let out_offset = b_i * stride_out;
                out[out_offset..out_offset + stride_out].copy_from_slice(&batch_out);
            }
        });

        let mut output_shape = batch_shape.to_vec();
        output_shape.push(m);
        output_shape.push(n);
        TensorView::from_slice(out, output_shape)
    }
}

#[cfg(target_arch = "aarch64")]
pub fn mat_mul_integer_prepared_arm<'a, 'b>(
    a: &TensorView<'b, f32>,
    pw_arm: &PreparedWeightsArm,
    a_zero_point: Option<f32>,
    b_zero_point: Option<u8>,
    scale: Option<&TensorView<'b, f32>>,
    bias: Option<&TensorView<'b, f32>>,
    apply_relu: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    // Convert A from f32 to u8
    let len = a.data.len();
    let mut a_u8: Vec<u8> = Vec::with_capacity(len);
    unsafe {
        a_u8.set_len(len);
        let src = a.data.as_ptr();
        let dst = a_u8.as_mut_ptr();
        // NEON vectorized f32→u8 conversion
        let mut i = 0;
        while i + 16 <= len {
            use core::arch::aarch64::*;
            let v0 = vld1q_f32(src.add(i));
            let v1 = vld1q_f32(src.add(i + 4));
            let v2 = vld1q_f32(src.add(i + 8));
            let v3 = vld1q_f32(src.add(i + 12));
            let u0 = vcvtq_u32_f32(v0);
            let u1 = vcvtq_u32_f32(v1);
            let u2 = vcvtq_u32_f32(v2);
            let u3 = vcvtq_u32_f32(v3);
            let n0 = vqmovn_u32(u0);
            let n1 = vqmovn_u32(u1);
            let n2 = vqmovn_u32(u2);
            let n3 = vqmovn_u32(u3);
            let nn0 = vcombine_u16(n0, n1);
            let nn1 = vcombine_u16(n2, n3);
            let b0 = vqmovn_u16(nn0);
            let b1 = vqmovn_u16(nn1);
            let res = vcombine_u8(b0, b1);
            vst1q_u8(dst.add(i), res);
            i += 16;
        }
        while i < len {
            *dst.add(i) = *src.add(i) as u8;
            i += 1;
        }
    }

    let a_dims = a.shape.len();
    let m = a.shape[a_dims - 2];
    let k = a.shape[a_dims - 1];
    let batch: usize = a.shape[..a_dims - 2].iter().product();
    let batch_shape = &a.shape[..a_dims - 2];

    let total_batch = batch.max(1);
    let zp_a = a_zero_point.unwrap_or(0.0) as i32;
    let zp_b = b_zero_point.unwrap_or(0) as i32;
    let n = pw_arm.n;
    let stride_a = m * k;
    let stride_out = m * n;
    let output_len = total_batch * stride_out;
    crate::kernels::utils::ensure_capacity(out, output_len);
    out.resize(output_len, 0.0);

    for b_i in 0..total_batch {
        let a_batch = &a_u8[b_i * stride_a..(b_i + 1) * stride_a];

        if total_batch == 1 {
            crate::kernels::neon::quantization::mat_mul_integer_prepared_neon(
                a_batch, m, k, pw_arm, zp_a, zp_b, scale, bias, apply_relu, out,
            );
        } else {
            let mut batch_out = vec![0f32; stride_out];
            crate::kernels::neon::quantization::mat_mul_integer_prepared_neon(
                a_batch,
                m,
                k,
                pw_arm,
                zp_a,
                zp_b,
                scale,
                bias,
                apply_relu,
                &mut batch_out,
            );
            let out_offset = b_i * stride_out;
            out[out_offset..out_offset + stride_out].copy_from_slice(&batch_out);
        }
    }

    let mut output_shape = batch_shape.to_vec();
    output_shape.push(m);
    output_shape.push(n);
    TensorView::from_slice(out, output_shape)
}

#[cfg(target_arch = "aarch64")]
pub fn prepare_weights_arm_from_i8(b_i8_bytes: &[u8], k: usize, n: usize) -> PreparedWeightsArm {
    prepare_weights_arm(b_i8_bytes, k, n)
}

pub fn dynamic_quantize_linear<'a, 'b>(
    x: &TensorView<'b, f32>,
    out_y_storage: &'a mut Vec<f32>,
    out_scale: &'a mut Vec<f32>,
    out_zp: &'a mut Vec<f32>,
) -> (
    TensorView<'a, f32>,
    TensorView<'a, f32>,
    TensorView<'a, f32>,
) {
    #[cfg(target_arch = "aarch64")]
    {
        crate::kernels::neon::quantization::dynamic_quantize_linear(
            x,
            out_y_storage,
            out_scale,
            out_zp,
        )
    }
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            crate::kernels::avx::quantization::dynamic_quantize_linear_avx2(
                x,
                out_y_storage,
                out_scale,
                out_zp,
            )
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let len = x.data.len();

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
        let inv_scale = 1.0 / scale;

        out_scale.clear();
        out_scale.push(scale);

        out_zp.clear();
        out_zp.push(zp);

        // Calculate and write directly to output
        out_y_storage.clear();
        out_y_storage.reserve(len);
        for i in 0..len {
            let q = (x.data[i] * inv_scale + zp).round().clamp(0.0, 255.0);
            out_y_storage.push(q);
        }

        (
            TensorView::from_slice(out_y_storage, x.shape.to_vec()),
            TensorView::from_slice(out_scale, vec![1]),
            TensorView::from_slice(out_zp, vec![1]),
        )
    }
}
