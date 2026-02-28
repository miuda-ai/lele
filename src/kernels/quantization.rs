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
        };
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
                                scale_data_ptr.map(
                                    |p| {
                                        if scale_len == 1 { p } else { p.add(n_start) }
                                    },
                                ),
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
                                scale_data_ptr.map(
                                    |p| {
                                        if scale_len == 1 { p } else { p.add(n_start) }
                                    },
                                ),
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

    #[cfg(target_arch = "wasm32")]
    {
        // WASM SIMD path: work directly in f32 space, avoid Vec<u8> allocation.
        // Both `a` (dynamic-quantized) and `b` (weight_int8) hold u8-range values
        // stored as f32 — no conversion needed. Use SIMD outer-product accumulator
        // with i,l,j loop order so inner j accesses B/C rows contiguously.
        let a_dims = a.shape.len();
        let b_dims = b.shape.len();
        let m = a.shape[a_dims - 2];
        let k = a.shape[a_dims - 1];
        let n = b.shape[b_dims - 1];

        let batch_a: usize = a.shape[..a_dims - 2].iter().product::<usize>().max(1);
        let batch_b: usize = b.shape[..b_dims - 2].iter().product::<usize>().max(1);
        let final_batch = batch_a.max(batch_b);

        let zp_a_val = a_zero_point
            .and_then(|z| z.data.first())
            .copied()
            .unwrap_or(0.0);
        let zp_b_val = b_zero_point
            .and_then(|z| z.data.first())
            .copied()
            .unwrap_or(0.0);

        let output_len = final_batch * m * n;
        crate::kernels::utils::ensure_capacity(out, output_len);
        out.resize(output_len, 0.0);

        let stride_a = m * k;
        let stride_b = k * n;
        let stride_out = m * n;

        for b_i in 0..final_batch {
            let a_offset = if batch_a == 1 { 0 } else { b_i * stride_a };
            let b_offset = if batch_b == 1 { 0 } else { b_i * stride_b };
            let out_offset = b_i * stride_out;
            unsafe {
                wasm_mat_mul_integer_simd(
                    a.data[a_offset..].as_ptr(),
                    b.data[b_offset..].as_ptr(),
                    out[out_offset..].as_mut_ptr(),
                    m,
                    k,
                    n,
                    zp_a_val,
                    zp_b_val,
                );
                wasm_apply_scale_bias_relu(
                    out[out_offset..].as_mut_ptr(),
                    scale.map(|s| s.data.as_ref()),
                    bias.map(|b| b.data.as_ref()),
                    m,
                    n,
                    apply_relu,
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

    #[cfg(not(any(target_arch = "x86_64", target_arch = "wasm32")))]
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
    // On macOS aarch64, use Apple Accelerate cblas_sgemm with dequantized fp32 weights
    // This leverages the AMX hardware which is much faster than NEON UDOT for these sizes
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        return fused_dq_gemm_accelerate(
            input,
            pw_arm,
            b_zero_point,
            weight_scale,
            bias,
            apply_relu,
            out,
        );
    }

    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    {
        fused_dq_gemm_prepared_arm_neon(
            input,
            pw_arm,
            b_zero_point,
            weight_scale,
            bias,
            apply_relu,
            out,
        )
    }
}

/// Apple Accelerate AMX-based implementation: dequantize weights lazily, then use cblas_sgemm
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn fused_dq_gemm_accelerate<'a>(
    input: &TensorView<'_, f32>,
    pw_arm: &PreparedWeightsArm,
    b_zero_point: Option<u8>,
    weight_scale: &TensorView<'_, f32>,
    bias: Option<&TensorView<'_, f32>>,
    apply_relu: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    crate::kernels::gemm::accelerate_init();

    let a_dims = input.shape.len();
    let m = input.shape[a_dims - 2];
    let k = input.shape[a_dims - 1];
    let batch_shape = &input.shape[..a_dims.saturating_sub(2)];
    let batch: usize = batch_shape.iter().product();
    let total_batch = batch.max(1);
    let n = pw_arm.n;

    // Get or compute the dequantized fp32 weight matrix [K, N]
    let weight_f32 = pw_arm.get_dequantized_weights(b_zero_point, weight_scale);

    let output_len = total_batch * m * n;
    crate::kernels::utils::ensure_capacity(out, output_len);
    unsafe {
        out.set_len(output_len);
    }

    let stride_in = m * k;
    let stride_out = m * n;
    let has_bias = bias.is_some();

    for b_i in 0..total_batch {
        let a_offset = b_i * stride_in;
        let out_offset = b_i * stride_out;

        // C = A * B  (beta=0, no pre-fill needed)
        unsafe {
            crate::kernels::gemm::accelerate_sgemm(
                m as i32,
                n as i32,
                k as i32,
                1.0f32,
                input.data.as_ptr().add(a_offset),
                k as i32,
                weight_f32.as_ptr(),
                n as i32,
                0.0f32,
                out.as_mut_ptr().add(out_offset),
                n as i32,
            );
        }

        // Fused bias add + ReLU in a single vectorized pass
        if has_bias || apply_relu {
            let bias_data = bias.map(|b| b.data.as_ptr());
            let out_ptr = out.as_mut_ptr();
            unsafe {
                use core::arch::aarch64::*;
                let zero = vdupq_n_f32(0.0);
                for row in 0..m {
                    let row_offset = out_offset + row * n;
                    let mut j = 0;
                    if has_bias && apply_relu {
                        let bp = bias_data.unwrap();
                        while j + 16 <= n {
                            let mut v0 = vld1q_f32(out_ptr.add(row_offset + j));
                            let mut v1 = vld1q_f32(out_ptr.add(row_offset + j + 4));
                            let mut v2 = vld1q_f32(out_ptr.add(row_offset + j + 8));
                            let mut v3 = vld1q_f32(out_ptr.add(row_offset + j + 12));
                            v0 = vaddq_f32(v0, vld1q_f32(bp.add(j)));
                            v1 = vaddq_f32(v1, vld1q_f32(bp.add(j + 4)));
                            v2 = vaddq_f32(v2, vld1q_f32(bp.add(j + 8)));
                            v3 = vaddq_f32(v3, vld1q_f32(bp.add(j + 12)));
                            vst1q_f32(out_ptr.add(row_offset + j), vmaxq_f32(v0, zero));
                            vst1q_f32(out_ptr.add(row_offset + j + 4), vmaxq_f32(v1, zero));
                            vst1q_f32(out_ptr.add(row_offset + j + 8), vmaxq_f32(v2, zero));
                            vst1q_f32(out_ptr.add(row_offset + j + 12), vmaxq_f32(v3, zero));
                            j += 16;
                        }
                        while j < n {
                            let val = *out_ptr.add(row_offset + j) + *bp.add(j);
                            *out_ptr.add(row_offset + j) = if val > 0.0 { val } else { 0.0 };
                            j += 1;
                        }
                    } else if has_bias {
                        let bp = bias_data.unwrap();
                        while j + 16 <= n {
                            let v0 = vaddq_f32(
                                vld1q_f32(out_ptr.add(row_offset + j)),
                                vld1q_f32(bp.add(j)),
                            );
                            let v1 = vaddq_f32(
                                vld1q_f32(out_ptr.add(row_offset + j + 4)),
                                vld1q_f32(bp.add(j + 4)),
                            );
                            let v2 = vaddq_f32(
                                vld1q_f32(out_ptr.add(row_offset + j + 8)),
                                vld1q_f32(bp.add(j + 8)),
                            );
                            let v3 = vaddq_f32(
                                vld1q_f32(out_ptr.add(row_offset + j + 12)),
                                vld1q_f32(bp.add(j + 12)),
                            );
                            vst1q_f32(out_ptr.add(row_offset + j), v0);
                            vst1q_f32(out_ptr.add(row_offset + j + 4), v1);
                            vst1q_f32(out_ptr.add(row_offset + j + 8), v2);
                            vst1q_f32(out_ptr.add(row_offset + j + 12), v3);
                            j += 16;
                        }
                        while j < n {
                            *out_ptr.add(row_offset + j) += *bp.add(j);
                            j += 1;
                        }
                    } else {
                        // relu only
                        while j + 16 <= n {
                            let v0 = vld1q_f32(out_ptr.add(row_offset + j));
                            let v1 = vld1q_f32(out_ptr.add(row_offset + j + 4));
                            let v2 = vld1q_f32(out_ptr.add(row_offset + j + 8));
                            let v3 = vld1q_f32(out_ptr.add(row_offset + j + 12));
                            vst1q_f32(out_ptr.add(row_offset + j), vmaxq_f32(v0, zero));
                            vst1q_f32(out_ptr.add(row_offset + j + 4), vmaxq_f32(v1, zero));
                            vst1q_f32(out_ptr.add(row_offset + j + 8), vmaxq_f32(v2, zero));
                            vst1q_f32(out_ptr.add(row_offset + j + 12), vmaxq_f32(v3, zero));
                            j += 16;
                        }
                        while j < n {
                            let val = *out_ptr.add(row_offset + j);
                            *out_ptr.add(row_offset + j) = if val > 0.0 { val } else { 0.0 };
                            j += 1;
                        }
                    }
                }
            }
        }
    }

    let output_shape = if batch_shape.is_empty() {
        vec![m, n]
    } else {
        let mut s = batch_shape.to_vec();
        s.push(m);
        s.push(n);
        s
    };
    TensorView::from_slice(out, output_shape)
}

/// NEON UDOT-based implementation (non-macOS or fallback)
#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
fn fused_dq_gemm_prepared_arm_neon<'a>(
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
    #[cfg(target_arch = "wasm32")]
    {
        use std::arch::wasm32::*;
        let len = x.data.len();
        let ptr = x.data.as_ptr();

        // Phase 1: SIMD min/max scan
        let mut min_v = f32x4_splat(f32::MAX);
        let mut max_v = f32x4_splat(f32::MIN);
        let mut i = 0usize;
        unsafe {
            while i + 16 <= len {
                let v0 = v128_load(ptr.add(i) as *const v128);
                let v1 = v128_load(ptr.add(i + 4) as *const v128);
                let v2 = v128_load(ptr.add(i + 8) as *const v128);
                let v3 = v128_load(ptr.add(i + 12) as *const v128);
                min_v = f32x4_min(min_v, f32x4_min(v0, f32x4_min(v1, f32x4_min(v2, v3))));
                max_v = f32x4_max(max_v, f32x4_max(v0, f32x4_max(v1, f32x4_max(v2, v3))));
                i += 16;
            }
            while i + 4 <= len {
                let v = v128_load(ptr.add(i) as *const v128);
                min_v = f32x4_min(min_v, v);
                max_v = f32x4_max(max_v, v);
                i += 4;
            }
        }
        // Horizontal reduce
        let mut min_val = f32x4_extract_lane::<0>(min_v)
            .min(f32x4_extract_lane::<1>(min_v))
            .min(f32x4_extract_lane::<2>(min_v))
            .min(f32x4_extract_lane::<3>(min_v));
        let mut max_val = f32x4_extract_lane::<0>(max_v)
            .max(f32x4_extract_lane::<1>(max_v))
            .max(f32x4_extract_lane::<2>(max_v))
            .max(f32x4_extract_lane::<3>(max_v));
        // Handle remainder
        for j in i..len {
            let v = x.data[j];
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

        // Phase 2: SIMD vectorized quantization
        out_y_storage.clear();
        out_y_storage.resize(len, 0.0);
        let dst = out_y_storage.as_mut_ptr();
        let inv_scale_v = f32x4_splat(inv_scale);
        let zp_v = f32x4_splat(zp);
        let zero_v = f32x4_splat(0.0);
        let max255_v = f32x4_splat(255.0);
        let half_v = f32x4_splat(0.5);
        let mut i = 0usize;
        unsafe {
            while i + 4 <= len {
                let v = v128_load(ptr.add(i) as *const v128);
                // round via floor(x + 0.5) to match .round() semantics
                let q = f32x4_floor(f32x4_add(
                    f32x4_add(f32x4_mul(v, inv_scale_v), zp_v),
                    half_v,
                ));
                let q = f32x4_max(f32x4_min(q, max255_v), zero_v);
                v128_store(dst.add(i) as *mut v128, q);
                i += 4;
            }
        }
        for j in i..len {
            out_y_storage[j] = (x.data[j] * inv_scale + zp).round().clamp(0.0, 255.0);
        }

        return (
            TensorView::from_slice(out_y_storage, x.shape.to_vec()),
            TensorView::from_slice(out_scale, vec![1]),
            TensorView::from_slice(out_zp, vec![1]),
        );
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        target_arch = "wasm32"
    )))]
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

// ─── WASM SIMD INT8 MatMul helpers ────────────────────────────────────────────

/// WASM SIMD f32 INT8-style matmul: C = (A - zp_a) × (B - zp_b)
///
/// Register-tiled micro-kernel: 4 rows × 16 cols = 16 v128 accumulators.
/// - A precompute: `a_adj = A - zp_a` computed once per row-group, reused
///   across all N/16 j-panels (eliminates ~4.5M redundant subtracts).
/// - zp_b correction: `C -= zp_b * row_sum(a_adj)` applied at store time.
/// - C never touches memory during K sweep — pure register accumulation.
///
/// Register budget: 16 accum + 4 B loads + 4 A splats = 24 v128.
/// Fits in ARM64 NEON (32 regs). On x86-64 (16 XMM) JIT may spill some.
///
/// A: [M, K] row-major f32. B: [K, N] row-major f32. C: [M, N] output.
#[cfg(target_arch = "wasm32")]
#[inline(never)]
unsafe fn wasm_mat_mul_integer_simd(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    zp_a: f32,
    zp_b: f32,
) {
    use std::arch::wasm32::*;

    let n16 = n & !15;
    let m4 = m - (m % 4); // round down to multiple of 4

    // Reusable buffer for pre-adjusted A values (avoids redundant zp_a subtracts)
    let mut a_adj_buf = vec![0.0f32; 4 * k];

    // ── Main loop: 4 rows at a time ──────────────────────────────────────
    let mut i = 0;
    while i < m4 {
        // Pre-compute a_adj[r][l] = A[i+r, l] - zp_a and row_sums
        let mut row_sums = [0.0f32; 4];
        for r in 0..4 {
            let a_row = a.add((i + r) * k);
            let base = r * k;
            let mut sum = 0.0f32;
            for l in 0..k {
                let v = *a_row.add(l) - zp_a;
                *a_adj_buf.as_mut_ptr().add(base + l) = v;
                sum += v;
            }
            row_sums[r] = sum;
        }

        let corr0 = f32x4_splat(zp_b * row_sums[0]);
        let corr1 = f32x4_splat(zp_b * row_sums[1]);
        let corr2 = f32x4_splat(zp_b * row_sums[2]);
        let corr3 = f32x4_splat(zp_b * row_sums[3]);

        let cr0 = c.add(i * n);
        let cr1 = c.add((i + 1) * n);
        let cr2 = c.add((i + 2) * n);
        let cr3 = c.add((i + 3) * n);

        let a0 = a_adj_buf.as_ptr();
        let a1 = a0.add(k);
        let a2 = a0.add(2 * k);
        let a3 = a0.add(3 * k);

        // Process 16 columns at a time — all accumulators in registers
        let mut j = 0;
        while j < n16 {
            let mut c00 = f32x4_splat(0.0);
            let mut c01 = f32x4_splat(0.0);
            let mut c02 = f32x4_splat(0.0);
            let mut c03 = f32x4_splat(0.0);
            let mut c10 = f32x4_splat(0.0);
            let mut c11 = f32x4_splat(0.0);
            let mut c12 = f32x4_splat(0.0);
            let mut c13 = f32x4_splat(0.0);
            let mut c20 = f32x4_splat(0.0);
            let mut c21 = f32x4_splat(0.0);
            let mut c22 = f32x4_splat(0.0);
            let mut c23 = f32x4_splat(0.0);
            let mut c30 = f32x4_splat(0.0);
            let mut c31 = f32x4_splat(0.0);
            let mut c32 = f32x4_splat(0.0);
            let mut c33 = f32x4_splat(0.0);

            for l in 0..k {
                let av0 = f32x4_splat(*a0.add(l));
                let av1 = f32x4_splat(*a1.add(l));
                let av2 = f32x4_splat(*a2.add(l));
                let av3 = f32x4_splat(*a3.add(l));
                let bp = b.add(l * n + j);
                let b0 = v128_load(bp as *const v128);
                let b1 = v128_load(bp.add(4) as *const v128);
                let b2 = v128_load(bp.add(8) as *const v128);
                let b3 = v128_load(bp.add(12) as *const v128);

                c00 = f32x4_add(c00, f32x4_mul(av0, b0));
                c01 = f32x4_add(c01, f32x4_mul(av0, b1));
                c02 = f32x4_add(c02, f32x4_mul(av0, b2));
                c03 = f32x4_add(c03, f32x4_mul(av0, b3));
                c10 = f32x4_add(c10, f32x4_mul(av1, b0));
                c11 = f32x4_add(c11, f32x4_mul(av1, b1));
                c12 = f32x4_add(c12, f32x4_mul(av1, b2));
                c13 = f32x4_add(c13, f32x4_mul(av1, b3));
                c20 = f32x4_add(c20, f32x4_mul(av2, b0));
                c21 = f32x4_add(c21, f32x4_mul(av2, b1));
                c22 = f32x4_add(c22, f32x4_mul(av2, b2));
                c23 = f32x4_add(c23, f32x4_mul(av2, b3));
                c30 = f32x4_add(c30, f32x4_mul(av3, b0));
                c31 = f32x4_add(c31, f32x4_mul(av3, b1));
                c32 = f32x4_add(c32, f32x4_mul(av3, b2));
                c33 = f32x4_add(c33, f32x4_mul(av3, b3));
            }

            // Apply correction and store
            v128_store(cr0.add(j) as *mut v128, f32x4_sub(c00, corr0));
            v128_store(cr0.add(j + 4) as *mut v128, f32x4_sub(c01, corr0));
            v128_store(cr0.add(j + 8) as *mut v128, f32x4_sub(c02, corr0));
            v128_store(cr0.add(j + 12) as *mut v128, f32x4_sub(c03, corr0));
            v128_store(cr1.add(j) as *mut v128, f32x4_sub(c10, corr1));
            v128_store(cr1.add(j + 4) as *mut v128, f32x4_sub(c11, corr1));
            v128_store(cr1.add(j + 8) as *mut v128, f32x4_sub(c12, corr1));
            v128_store(cr1.add(j + 12) as *mut v128, f32x4_sub(c13, corr1));
            v128_store(cr2.add(j) as *mut v128, f32x4_sub(c20, corr2));
            v128_store(cr2.add(j + 4) as *mut v128, f32x4_sub(c21, corr2));
            v128_store(cr2.add(j + 8) as *mut v128, f32x4_sub(c22, corr2));
            v128_store(cr2.add(j + 12) as *mut v128, f32x4_sub(c23, corr2));
            v128_store(cr3.add(j) as *mut v128, f32x4_sub(c30, corr3));
            v128_store(cr3.add(j + 4) as *mut v128, f32x4_sub(c31, corr3));
            v128_store(cr3.add(j + 8) as *mut v128, f32x4_sub(c32, corr3));
            v128_store(cr3.add(j + 12) as *mut v128, f32x4_sub(c33, corr3));

            j += 16;
        }

        // Remainder columns (< 16) — 4-row scalar
        while j < n {
            let mut cv0: f32 = 0.0;
            let mut cv1: f32 = 0.0;
            let mut cv2: f32 = 0.0;
            let mut cv3: f32 = 0.0;
            for l in 0..k {
                let bv = *b.add(l * n + j);
                cv0 += *a0.add(l) * bv;
                cv1 += *a1.add(l) * bv;
                cv2 += *a2.add(l) * bv;
                cv3 += *a3.add(l) * bv;
            }
            *cr0.add(j) = cv0 - zp_b * row_sums[0];
            *cr1.add(j) = cv1 - zp_b * row_sums[1];
            *cr2.add(j) = cv2 - zp_b * row_sums[2];
            *cr3.add(j) = cv3 - zp_b * row_sums[3];
            j += 1;
        }

        i += 4;
    }

    // ── Remainder rows (m % 4) — one row at a time with register tiling ──
    while i < m {
        let c_row = c.add(i * n);
        let a_row = a.add(i * k);

        // Pre-compute a_adj into first row of buffer
        let mut row_sum: f32 = 0.0;
        let abuf = a_adj_buf.as_mut_ptr();
        for l in 0..k {
            let v = *a_row.add(l) - zp_a;
            *abuf.add(l) = v;
            row_sum += v;
        }
        let corr = f32x4_splat(zp_b * row_sum);
        let ap = a_adj_buf.as_ptr();

        let mut j = 0;
        while j < n16 {
            let mut c0 = f32x4_splat(0.0);
            let mut c1 = f32x4_splat(0.0);
            let mut c2 = f32x4_splat(0.0);
            let mut c3 = f32x4_splat(0.0);

            for l in 0..k {
                let a_v = f32x4_splat(*ap.add(l));
                let bp = b.add(l * n + j);
                let b0 = v128_load(bp as *const v128);
                let b1 = v128_load(bp.add(4) as *const v128);
                let b2 = v128_load(bp.add(8) as *const v128);
                let b3 = v128_load(bp.add(12) as *const v128);
                c0 = f32x4_add(c0, f32x4_mul(a_v, b0));
                c1 = f32x4_add(c1, f32x4_mul(a_v, b1));
                c2 = f32x4_add(c2, f32x4_mul(a_v, b2));
                c3 = f32x4_add(c3, f32x4_mul(a_v, b3));
            }

            v128_store(c_row.add(j) as *mut v128, f32x4_sub(c0, corr));
            v128_store(c_row.add(j + 4) as *mut v128, f32x4_sub(c1, corr));
            v128_store(c_row.add(j + 8) as *mut v128, f32x4_sub(c2, corr));
            v128_store(c_row.add(j + 12) as *mut v128, f32x4_sub(c3, corr));
            j += 16;
        }
        while j < n {
            let mut cv: f32 = 0.0;
            for l in 0..k {
                cv += *ap.add(l) * *b.add(l * n + j);
            }
            *c_row.add(j) = cv - zp_b * row_sum;
            j += 1;
        }

        i += 1;
    }
}

/// Apply per-column scale and bias (and optional ReLU) to output matrix in-place.
/// scale: None = no scaling, Some([1]) = scalar, Some([N]) = per-column.
/// bias:  None = no bias,    Some([N]) = per-column.
#[cfg(target_arch = "wasm32")]
unsafe fn wasm_apply_scale_bias_relu(
    c: *mut f32,
    scale: Option<&[f32]>,
    bias: Option<&[f32]>,
    m: usize,
    n: usize,
    apply_relu: bool,
) {
    use std::arch::wasm32::*;
    if scale.is_none() && bias.is_none() && !apply_relu {
        return;
    }
    let zero_v = f32x4_splat(0.0);
    for i in 0..m {
        let c_row = c.add(i * n);
        let mut j = 0;
        while j + 4 <= n {
            let mut cv = v128_load(c_row.add(j) as *const v128);
            if let Some(s) = scale {
                let sv = if s.len() == 1 {
                    f32x4_splat(s[0])
                } else {
                    v128_load(s.as_ptr().add(j) as *const v128)
                };
                cv = f32x4_mul(cv, sv);
            }
            if let Some(b) = bias {
                let bv = v128_load(b.as_ptr().add(j) as *const v128);
                cv = f32x4_add(cv, bv);
            }
            if apply_relu {
                cv = f32x4_max(cv, zero_v);
            }
            v128_store(c_row.add(j) as *mut v128, cv);
            j += 4;
        }
        while j < n {
            let mut v = *c_row.add(j);
            if let Some(s) = scale {
                v *= if s.len() == 1 { s[0] } else { s[j] };
            }
            if let Some(b) = bias {
                v += b[j];
            }
            if apply_relu && v < 0.0 {
                v = 0.0;
            }
            *c_row.add(j) = v;
            j += 1;
        }
    }
}
