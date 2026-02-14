#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::cell::RefCell;
use crate::kernels::utils;
use crate::tensor::TensorView;

/// Cached pre-transposed B weight data for quantized GEMM.
/// Avoids re-transposing static weights on every call (~65ms saved per forward pass).
/// Uses content fingerprinting instead of pointer-based keys to handle the case where
/// weight_u8() creates temporary Vec<f32> whose addresses get recycled by the allocator.
struct BWeightCacheEntry {
    k: usize,
    n: usize,
    #[allow(dead_code)]
    k_padded: usize,
    // Content fingerprint: 4 elements sampled from the weight data.
    // Uniquely identifies a weight matrix regardless of pointer address.
    fp0: u32,
    fp1: u32,
    fp2: u32,
    fp3: u32,
    b_t: Vec<u8>,    // Transposed [N, K_padded] with XOR 0x80
    col_sums: Vec<i32>,
}

/// Compute a 4-element content fingerprint for cache lookup.
/// Samples first, 1/3, 2/3, and last elements — sufficient to uniquely identify
/// each weight matrix in a typical model (hundreds of weights with same shape).
#[inline]
fn b_cache_fingerprint(data: *const f32, len: usize) -> (u32, u32, u32, u32) {
    if len == 0 {
        return (0, 0, 0, 0);
    }
    unsafe {
        let fp0 = (*data).to_bits();
        let fp1 = (*data.add(len / 3)).to_bits();
        let fp2 = (*data.add(len * 2 / 3)).to_bits();
        let fp3 = (*data.add(len - 1)).to_bits();
        (fp0, fp1, fp2, fp3)
    }
}

/// Content-addressed B weight cache lookup or insert. Returns the index.
/// Keys by (k, n, fingerprint) — stable across allocator address reuse.
#[inline]
unsafe fn b_cache_lookup_or_insert(
    b_cache: &mut Vec<BWeightCacheEntry>,
    b_f32: *const f32,
    k: usize,
    n: usize,
    k_padded: usize,
) -> usize {
    let (fp0, fp1, fp2, fp3) = b_cache_fingerprint(b_f32, k * n);

    // Content-addressed lookup: match by shape + fingerprint (pointer-independent)
    if let Some(idx) = b_cache.iter().position(|e|
        e.k == k && e.n == n &&
        e.fp0 == fp0 && e.fp1 == fp1 && e.fp2 == fp2 && e.fp3 == fp3
    ) {
        return idx;
    }

    // Create fresh entry
    let mut b_t = vec![0u8; n * k_padded];
    let mut col_sums = vec![0i32; n];
    unsafe {
        transpose_b_from_f32_avx2(b_f32, k, n, b_t.as_mut_ptr(), k_padded, col_sums.as_mut_ptr());
    }
    b_cache.push(BWeightCacheEntry { k, n, k_padded, fp0, fp1, fp2, fp3, b_t, col_sums });
    b_cache.len() - 1
}

thread_local! {
    static B_WEIGHT_CACHE: RefCell<Vec<BWeightCacheEntry>> = const { RefCell::new(Vec::new()) };
    static SCRATCH_A: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static SCRATCH_RS: RefCell<Vec<i32>> = const { RefCell::new(Vec::new()) };
    static SCRATCH_CS: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Combined DynQuant + u8 conversion + row_sums in 2 passes (min/max then quantize+convert+sum).
/// Eliminates the intermediate f32 buffer, saving ~420KB of memory traffic per call.
/// Returns (scale, zero_point).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dq_to_u8_rowsums_avx2(
    input: *const f32,
    m: usize,
    k: usize,
    a_u8: *mut u8,
    row_sums: *mut i32,
) -> (f32, f32) {
    unsafe {
        let len = m * k;

        // Pass 1: find min/max
        let mut min_vec = _mm256_set1_ps(f32::MAX);
        let mut max_vec = _mm256_set1_ps(f32::MIN);
        let mut i = 0;
        let simd_end = (len / 8) * 8;
        while i < simd_end {
            let v = _mm256_loadu_ps(input.add(i));
            min_vec = _mm256_min_ps(min_vec, v);
            max_vec = _mm256_max_ps(max_vec, v);
            i += 8;
        }
        let mut min_val = hmin_ps(min_vec);
        let mut max_val = hmax_ps(max_vec);
        for j in simd_end..len {
            let v = *input.add(j);
            if v < min_val { min_val = v; }
            if v > max_val { max_val = v; }
        }

        let adjusted_max = max_val.max(0.0);
        let adjusted_min = min_val.min(0.0);
        let range = (adjusted_max - adjusted_min).max(1e-5);
        let scale = range / 255.0;
        let zp = (-adjusted_min / scale).round().clamp(0.0, 255.0);
        let inv_scale = 1.0 / scale;

        // Pass 2: quantize f32 → u8 directly + compute row_sums
        let inv_scale_vec = _mm256_set1_ps(inv_scale);
        let zp_vec = _mm256_set1_ps(zp);
        let zero_f = _mm256_setzero_ps();
        let max_255 = _mm256_set1_ps(255.0);

        for row in 0..m {
            let row_start = row * k;
            let mut sum_vec = _mm256_setzero_si256();
            let mut kk = 0;

            // Process 16 elements at a time (2 × 8 f32 → 16 u8)
            while kk + 16 <= k {
                let v0 = _mm256_loadu_ps(input.add(row_start + kk));
                let v1 = _mm256_loadu_ps(input.add(row_start + kk + 8));

                let s0 = _mm256_fmadd_ps(v0, inv_scale_vec, zp_vec);
                let s1 = _mm256_fmadd_ps(v1, inv_scale_vec, zp_vec);
                let r0 = _mm256_round_ps(s0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                let r1 = _mm256_round_ps(s1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                let c0 = _mm256_min_ps(_mm256_max_ps(r0, zero_f), max_255);
                let c1 = _mm256_min_ps(_mm256_max_ps(r1, zero_f), max_255);

                let i0 = _mm256_cvtps_epi32(c0);
                let i1 = _mm256_cvtps_epi32(c1);

                // Accumulate row sums from i32 values (exact, no precision loss)
                sum_vec = _mm256_add_epi32(sum_vec, _mm256_add_epi32(i0, i1));

                // Pack i32 → i16 → u8
                let s16 = _mm256_packs_epi32(i0, i1);
                let s16 = _mm256_permute4x64_epi64(s16, 0b11_01_10_00);
                let u8v = _mm256_packus_epi16(s16, _mm256_setzero_si256());
                let u8v = _mm256_permute4x64_epi64(u8v, 0b11_01_10_00);
                _mm_storeu_si128(
                    a_u8.add(row_start + kk) as *mut __m128i,
                    _mm256_castsi256_si128(u8v),
                );
                kk += 16;
            }

            // Process 8 elements
            if kk + 8 <= k {
                let v = _mm256_loadu_ps(input.add(row_start + kk));
                let s = _mm256_fmadd_ps(v, inv_scale_vec, zp_vec);
                let r = _mm256_round_ps(s, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                let c = _mm256_min_ps(_mm256_max_ps(r, zero_f), max_255);
                let iv = _mm256_cvtps_epi32(c);
                sum_vec = _mm256_add_epi32(sum_vec, iv);
                let s16 = _mm256_packs_epi32(iv, _mm256_setzero_si256());
                let s16 = _mm256_permute4x64_epi64(s16, 0b11_01_10_00);
                let u8v = _mm256_packus_epi16(s16, _mm256_setzero_si256());
                let u8v = _mm256_permute4x64_epi64(u8v, 0b11_01_10_00);
                let lo = _mm256_castsi256_si128(u8v);
                std::ptr::copy_nonoverlapping(
                    &lo as *const __m128i as *const u8,
                    a_u8.add(row_start + kk),
                    8,
                );
                kk += 8;
            }

            // Scalar remainder
            let mut scalar_sum: i32 = 0;
            while kk < k {
                let v = *input.add(row_start + kk);
                let q = (v * inv_scale + zp).round().clamp(0.0, 255.0) as u8;
                *a_u8.add(row_start + kk) = q;
                scalar_sum += q as i32;
                kk += 1;
            }

            *row_sums.add(row) = hsum_epi32(sum_vec) + scalar_sum;
        }

        (scale, zp)
    }
}

/// Fully-fused quantized linear on AVX2: raw f32 input → DynQuant → u8 + GEMM → f32 output.
/// Eliminates the intermediate f32 quantized buffer entirely.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn fused_dq_gemm_avx2(
    input: *const f32,
    m: usize,
    k: usize,
    n: usize,
    b_f32: *const f32,
    weight_scale: *const f32,
    weight_scale_len: usize,
    weight_zp: i32,
    bias: Option<*const f32>,
    apply_relu: bool,
    out: *mut f32,
) {
    unsafe {
        let k_padded = (k + 15) & !15;
        let k_aligned = k == k_padded;

        // Take scratch buffers from thread-local
        let mut a_u8 = SCRATCH_A.with(|c| std::mem::take(&mut *c.borrow_mut()));
        let mut row_sums = SCRATCH_RS.with(|c| std::mem::take(&mut *c.borrow_mut()));
        let mut combined_scale = SCRATCH_CS.with(|c| std::mem::take(&mut *c.borrow_mut()));

        a_u8.resize(m * k, 0);
        row_sums.resize(m, 0);

        // Fused DynQuant: min/max → quantize to u8 + row_sums (2 passes, no f32 intermediate)
        let (dyn_scale, zp_a_f32) = dq_to_u8_rowsums_avx2(
            input, m, k,
            a_u8.as_mut_ptr(), row_sums.as_mut_ptr(),
        );
        let zp_a = zp_a_f32 as i32;

        // Compute combined_scale = dyn_scale * weight_scale[j] for each column
        combined_scale.resize(weight_scale_len.max(1), 0.0);
        if weight_scale_len <= 1 {
            combined_scale[0] = dyn_scale * *weight_scale;
        } else {
            let dyn_scale_vec = _mm256_set1_ps(dyn_scale);
            let mut j = 0;
            while j + 8 <= weight_scale_len {
                let ws = _mm256_loadu_ps(weight_scale.add(j));
                _mm256_storeu_ps(combined_scale.as_mut_ptr().add(j), _mm256_mul_ps(dyn_scale_vec, ws));
                j += 8;
            }
            while j < weight_scale_len {
                combined_scale[j] = dyn_scale * *weight_scale.add(j);
                j += 1;
            }
        }

        // B cache lookup/transpose (content-addressed)
        let mut b_cache = B_WEIGHT_CACHE.with(|c| std::mem::take(&mut *c.borrow_mut()));
        let entry_idx = b_cache_lookup_or_insert(&mut b_cache, b_f32, k, n, k_padded);
        let bt_ptr = b_cache[entry_idx].b_t.as_ptr();
        let cs_ptr = b_cache[entry_idx].col_sums.as_ptr();

        let k_zp_b = k as i32 * weight_zp;
        let corr_128_minus_zpb = 128 - weight_zp;
        let scale_data_ptr = Some(combined_scale.as_ptr());
        let scale_len = combined_scale.len();
        let bias_data_ptr = bias;

        // N-tiling and 2-row GEMM dispatch (same logic as mat_mul_integer_fused_f32_avx2)
        let l2_budget = 384 * 1024;
        let tile_n = if k_padded > 0 {
            (l2_budget / k_padded).max(8).min(n)
        } else {
            n
        };
        let tile_n = ((tile_n + 7) & !7).min(n);
        let out_slice = std::slice::from_raw_parts_mut(out, m * n);

        if tile_n >= n {
            let mut i = 0;
            while i + 2 <= m {
                let out_row0 = std::slice::from_raw_parts_mut(out.add(i * n), n);
                let out_row1 = std::slice::from_raw_parts_mut(out.add((i + 1) * n), n);
                gemm_2rows_avx2(
                    &a_u8[i * k..i * k + k],
                    &a_u8[(i + 1) * k..(i + 1) * k + k],
                    bt_ptr, k, n, k_padded, k_aligned, cs_ptr,
                    zp_a, weight_zp, k_zp_b, corr_128_minus_zpb,
                    scale_data_ptr, scale_len, bias_data_ptr, apply_relu,
                    row_sums[i], row_sums[i + 1],
                    out_row0, out_row1,
                );
                i += 2;
            }
            if i < m {
                gemm_row_avx2(
                    &a_u8[i * k..i * k + k],
                    bt_ptr, k, n, k_padded, k_aligned, cs_ptr,
                    zp_a, weight_zp, k_zp_b, corr_128_minus_zpb,
                    scale_data_ptr, scale_len, bias_data_ptr, apply_relu,
                    &mut out_slice[i * n..(i + 1) * n],
                );
            }
        } else {
            let mut n_start = 0;
            while n_start < n {
                let n_end = (n_start + tile_n).min(n);
                let cur_n = n_end - n_start;
                let tile_scale = scale_data_ptr.map(|p| if scale_len == 1 { p } else { p.add(n_start) });
                let tile_bias = bias_data_ptr.map(|p| p.add(n_start));
                let tile_bt = bt_ptr.add(n_start * k_padded);
                let tile_cs = cs_ptr.add(n_start);

                let mut i = 0;
                while i + 2 <= m {
                    let out_row0 = std::slice::from_raw_parts_mut(out.add(i * n + n_start), cur_n);
                    let out_row1 = std::slice::from_raw_parts_mut(out.add((i + 1) * n + n_start), cur_n);
                    gemm_2rows_avx2(
                        &a_u8[i * k..i * k + k],
                        &a_u8[(i + 1) * k..(i + 1) * k + k],
                        tile_bt, k, cur_n, k_padded, k_aligned, tile_cs,
                        zp_a, weight_zp, k_zp_b, corr_128_minus_zpb,
                        tile_scale, scale_len, tile_bias, apply_relu,
                        row_sums[i], row_sums[i + 1],
                        out_row0, out_row1,
                    );
                    i += 2;
                }
                if i < m {
                    gemm_row_avx2(
                        &a_u8[i * k..i * k + k],
                        tile_bt, k, cur_n, k_padded, k_aligned, tile_cs,
                        zp_a, weight_zp, k_zp_b, corr_128_minus_zpb,
                        tile_scale, scale_len, tile_bias, apply_relu,
                        &mut out_slice[i * n + n_start..i * n + n_end],
                    );
                }
                n_start = n_end;
            }
        }

        // Return scratch buffers
        SCRATCH_A.with(|c| *c.borrow_mut() = a_u8);
        SCRATCH_RS.with(|c| *c.borrow_mut() = row_sums);
        SCRATCH_CS.with(|c| *c.borrow_mut() = combined_scale);
        B_WEIGHT_CACHE.with(|c| *c.borrow_mut() = b_cache);
    }
}

/// AVX2 vectorized f32→u8 conversion.
/// Each f32 value is expected to be in [0, 255]; values are truncated to u8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn f32_to_u8_avx2(src: *const f32, dst: *mut u8, len: usize) {
    unsafe {
        let mut i = 0;
        // Process 16 f32s → 16 u8s per iteration
        while i + 16 <= len {
            let v0 = _mm256_loadu_ps(src.add(i));
            let v1 = _mm256_loadu_ps(src.add(i + 8));
            // f32 → i32
            let i0 = _mm256_cvtps_epi32(v0);
            let i1 = _mm256_cvtps_epi32(v1);
            // i32 → i16 (saturate)
            let s16 = _mm256_packs_epi32(i0, i1);
            // Fix lane interleaving from packs: [a0..a3,b0..b3,a4..a7,b4..b7] → [a0..a7,b0..b7]
            let s16 = _mm256_permute4x64_epi64(s16, 0b11_01_10_00);
            // i16 → u8 (saturate)
            let u8x16 = _mm256_packus_epi16(s16, _mm256_setzero_si256());
            let u8x16 = _mm256_permute4x64_epi64(u8x16, 0b11_01_10_00);
            // Store lower 16 bytes
            _mm_storeu_si128(
                dst.add(i) as *mut __m128i,
                _mm256_castsi256_si128(u8x16),
            );
            i += 16;
        }
        // Process 8 f32s → 8 u8s
        if i + 8 <= len {
            let v = _mm256_loadu_ps(src.add(i));
            let iv = _mm256_cvtps_epi32(v);
            let s16 = _mm256_packs_epi32(iv, _mm256_setzero_si256());
            let s16 = _mm256_permute4x64_epi64(s16, 0b11_01_10_00);
            let u8v = _mm256_packus_epi16(s16, _mm256_setzero_si256());
            let u8v = _mm256_permute4x64_epi64(u8v, 0b11_01_10_00);
            // Store lower 8 bytes
            let lo = _mm256_castsi256_si128(u8v);
            std::ptr::copy_nonoverlapping(
                &lo as *const __m128i as *const u8,
                dst.add(i),
                8,
            );
            i += 8;
        }
        while i < len {
            *dst.add(i) = *src.add(i) as u8;
            i += 1;
        }
    }
}

/// Fused B transpose from f32: reads B[K,N] as f32 (values 0-255), writes B_T[N, K_padded]
/// as u8 XOR 0x80 (for VPMADDWD compatibility), and computes column sums.
/// Uses tiled access pattern for cache efficiency.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn transpose_b_from_f32_avx2(
    b_f32: *const f32,
    k: usize,
    n: usize,
    b_t: *mut u8,
    k_padded: usize,
    col_sums: *mut i32,
) {
    unsafe {
        // Zero col_sums
        std::ptr::write_bytes(col_sums, 0, n);
        // Zero-fill b_t (for padding region)
        std::ptr::write_bytes(b_t, 0, n * k_padded);

        // Tile dimensions for L1 cache efficiency
        // Each tile: TILE_K rows × TILE_N columns of B
        // Working set: TILE_K * TILE_N * 4 bytes (f32 input) ≈ 16KB for 64×64
        const TILE_K: usize = 64;
        const TILE_N: usize = 64;

        let mut k0 = 0;
        while k0 < k {
            let k_end = (k0 + TILE_K).min(k);
            let mut j0 = 0;
            while j0 < n {
                let j_end = (j0 + TILE_N).min(n);

                for kk in k0..k_end {
                    for jj in j0..j_end {
                        let val = *b_f32.add(kk * n + jj) as u8;
                        *b_t.add(jj * k_padded + kk) = val ^ 0x80;
                        *col_sums.add(jj) += val as i32;
                    }
                }

                j0 = j_end;
            }
            k0 = k_end;
        }
    }
}

/// Full fused mat_mul_integer for x86: takes f32 inputs (A quantized, B weight),
/// converts A to u8, transposes B from f32 directly, then runs GEMM.
/// This avoids the intermediate b_u8 allocation and separate transpose pass.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn mat_mul_integer_fused_f32_avx2<'a>(
    a_f32: &[f32],
    b_f32: &[f32],
    m: usize,
    k: usize,
    n: usize,
    zp_a: i32,
    zp_b: i32,
    scale: Option<&TensorView<'_, f32>>,
    bias: Option<&TensorView<'_, f32>>,
    apply_relu: bool,
    out: &'a mut [f32],
) {
    unsafe {
        let k_padded = (k + 15) & !15;
        let k_aligned = k == k_padded;

        // Take scratch buffers from thread-local (preserving capacity)
        let mut a_u8 = SCRATCH_A.with(|c| std::mem::take(&mut *c.borrow_mut()));
        let mut row_sums = SCRATCH_RS.with(|c| std::mem::take(&mut *c.borrow_mut()));

        a_u8.resize(m * k, 0);
        row_sums.resize(m, 0);

        // Convert A from f32 to u8 (vectorized)
        f32_to_u8_avx2(a_f32.as_ptr(), a_u8.as_mut_ptr(), m * k);

        // Cache B transpose: content-addressed lookup by (k, n, fingerprint)
        let mut b_cache = B_WEIGHT_CACHE.with(|c| std::mem::take(&mut *c.borrow_mut()));
        let entry_idx = b_cache_lookup_or_insert(&mut b_cache, b_f32.as_ptr(), k, n, k_padded);
        // Get stable pointers into heap-allocated Vecs inside the cache entry
        let bt_ptr = b_cache[entry_idx].b_t.as_ptr();
        let cs_ptr = b_cache[entry_idx].col_sums.as_ptr();
        let k_zp_b = k as i32 * zp_b;
        let corr_128_minus_zpb = 128 - zp_b;

        let scale_data_ptr = scale.map(|s| s.data.as_ptr());
        let scale_len = scale.map(|s| s.data.len()).unwrap_or(0);
        let bias_data_ptr = bias.map(|b| b.data.as_ptr());

        // Pre-compute all row sums (vectorized) for 2-row dispatch
        for i in 0..m {
            row_sums[i] = row_sum_u8_avx2(&a_u8[i * k..i * k + k]);
        }

        // N-tiled GEMM for cache efficiency
        // Choose tile size so that tile_n * k_padded fits well in L2 cache
        // Zen 3/4 has 512KB L2; Skylake+ has 256KB L2; use 75% of 512KB = 384KB
        let l2_budget = 384 * 1024; // 384KB for B_T tile in L2
        let tile_n = if k_padded > 0 {
            (l2_budget / k_padded).max(8).min(n)
        } else {
            n
        };
        // Round tile_n up to multiple of 8 for SIMD alignment
        let tile_n = (tile_n + 7) & !7;
        let tile_n = tile_n.min(n);

        let out_ptr = out.as_mut_ptr();

        if tile_n >= n {
            // No tiling needed, process all N columns at once
            // 2-row dispatch: share B loads between row pairs (~50% B bandwidth reduction)
            let mut i = 0;
            while i + 2 <= m {
                let out_row0 = std::slice::from_raw_parts_mut(out_ptr.add(i * n), n);
                let out_row1 = std::slice::from_raw_parts_mut(out_ptr.add((i + 1) * n), n);
                gemm_2rows_avx2(
                    &a_u8[i * k..i * k + k],
                    &a_u8[(i + 1) * k..(i + 1) * k + k],
                    bt_ptr,
                    k,
                    n,
                    k_padded,
                    k_aligned,
                    cs_ptr,
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
                    out_row0,
                    out_row1,
                );
                i += 2;
            }
            if i < m {
                gemm_row_avx2(
                    &a_u8[i * k..i * k + k],
                    bt_ptr,
                    k,
                    n,
                    k_padded,
                    k_aligned,
                    cs_ptr,
                    zp_a,
                    zp_b,
                    k_zp_b,
                    corr_128_minus_zpb,
                    scale_data_ptr,
                    scale_len,
                    bias_data_ptr,
                    apply_relu,
                    &mut out[i * n..(i + 1) * n],
                );
            }
        } else {
            // N-tiled: process tile_n columns at a time for all rows
            let mut n_start = 0;
            while n_start < n {
                let n_end = (n_start + tile_n).min(n);
                let cur_n = n_end - n_start;

                let tile_scale = scale_data_ptr.map(|p| if scale_len == 1 { p } else { p.add(n_start) });
                let tile_bias = bias_data_ptr.map(|p| p.add(n_start));
                let tile_bt = bt_ptr.add(n_start * k_padded);
                let tile_cs = cs_ptr.add(n_start);

                // 2-row dispatch for tiled path
                let mut i = 0;
                while i + 2 <= m {
                    let out_row0 = std::slice::from_raw_parts_mut(out_ptr.add(i * n + n_start), cur_n);
                    let out_row1 = std::slice::from_raw_parts_mut(out_ptr.add((i + 1) * n + n_start), cur_n);
                    gemm_2rows_avx2(
                        &a_u8[i * k..i * k + k],
                        &a_u8[(i + 1) * k..(i + 1) * k + k],
                        tile_bt,
                        k,
                        cur_n,
                        k_padded,
                        k_aligned,
                        tile_cs,
                        zp_a,
                        zp_b,
                        k_zp_b,
                        corr_128_minus_zpb,
                        tile_scale,
                        scale_len,
                        tile_bias,
                        apply_relu,
                        row_sums[i],
                        row_sums[i + 1],
                        out_row0,
                        out_row1,
                    );
                    i += 2;
                }
                if i < m {
                    gemm_row_avx2(
                        &a_u8[i * k..i * k + k],
                        tile_bt,
                        k,
                        cur_n,
                        k_padded,
                        k_aligned,
                        tile_cs,
                        zp_a,
                        zp_b,
                        k_zp_b,
                        corr_128_minus_zpb,
                        tile_scale,
                        scale_len,
                        tile_bias,
                        apply_relu,
                        &mut out[i * n + n_start..i * n + n_end],
                    );
                }

                n_start = n_end;
            }
        }

        // Return scratch buffers and cache to thread-local for reuse
        SCRATCH_A.with(|c| *c.borrow_mut() = a_u8);
        SCRATCH_RS.with(|c| *c.borrow_mut() = row_sums);
        B_WEIGHT_CACHE.with(|c| *c.borrow_mut() = b_cache);
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

/// AVX2-optimized u8 matrix multiply using VPMADDUBSW integer dot product.
/// Uses true int8 SIMD: processes 32 u8 multiplies per instruction (4x over f32 FMA path).
///
/// Math: out[i][j] = sum_k (a[i][k] - zp_a) * (b[k][j] - zp_b)
///
/// Rewrite for VPMADDUBSW (u8 × i8 → i16):
///   b_u8 as i8 = b_u8 - 128 (reinterpret cast, wraps correctly)
///   So: a_u8 * b_i8_reinterp = a_u8 * (b_u8 - 128)
///   Then: (a - zp_a)(b - zp_b) = a*b_i8 + a*(128 - zp_b) - zp_a*(b_col_sum - K*zp_b)
///   where b_col_sum = sum_k b[k][j] (original u8 values)
///
/// We pre-transpose B to [N, K_padded] and store as i8 (reinterpreted from u8 by XOR 0x80).
/// Per-row correction: row_sum_a * (128 - zp_b) - zp_a * col_sum_b_adj
/// where col_sum_b_adj = col_sum_b_u8 - K * 128  (sum of the reinterpreted i8 values)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn mat_mul_integer_u8_avx2<'a, 'b, 'c>(
    a: &TensorView<'b, u8>,
    b: &TensorView<'c, u8>,
    a_zero_point: Option<&TensorView<'b, u8>>,
    b_zero_point: Option<&TensorView<'c, u8>>,
    scale: Option<&TensorView<'b, f32>>,
    bias: Option<&TensorView<'b, f32>>,
    apply_relu: bool,
    out: &'a mut Vec<f32>,
) -> TensorView<'a, f32> {
    unsafe {
        let zp_a = a_zero_point.map(|z| z.data[0]).unwrap_or(0u8) as i32;
        let zp_b = b_zero_point.map(|z| z.data[0]).unwrap_or(0u8) as i32;

        let a_dims = a.shape.len();
        let b_dims = b.shape.len();
        let m = a.shape[a_dims - 2];
        let k = a.shape[a_dims - 1];
        let n = b.shape[b_dims - 1];

        let batch_a: usize = a.shape[..a_dims - 2].iter().product();
        let batch_b: usize = b.shape[..b_dims - 2].iter().product();
        let final_batch = batch_a.max(batch_b);
        let output_len = final_batch * m * n;

        utils::ensure_capacity(out, output_len);
        out.resize(output_len, 0.0);

        let stride_a = m * k;
        let stride_b = k * n;
        let stride_out = m * n;

        // Pad K to multiple of 16 for SIMD (VPMADDWD processes 16 pairs per instruction)
        let k_padded = (k + 15) & !15;

        // Pre-transpose B to [N, K_padded] with u8 XOR 0x80 → reinterpret as i8
        // Also compute col_sums of reinterpreted i8 values for zero-point correction
        let mut b_t = vec![0u8; n * k_padded]; // stored as u8, interpreted as i8 by sign-extend

        // Correction constant per column:
        // (a - zp_a)(b - zp_b) = a*(b^0x80 - 128 + 128 - zp_b) - zp_a*(b - zp_b)
        //                      = a*(b_reinterp) + a*(128 - zp_b) - zp_a*col_sum_b_orig + zp_a*zp_b*K
        // where b_reinterp = b^0x80 interpreted as i8 = b_u8 - 128
        // So VPMADDUBSW(a_u8, b_reinterp_i8) gives us: sum(a * (b - 128))
        // We need: sum((a - zp_a)(b - zp_b)) = sum(a*(b-128)) + sum(a)*(128 - zp_b) - zp_a*sum(b-zp_b)
        //        = dot_result + row_sum_a * (128 - zp_b) - zp_a * (col_sum_b_u8 - K * zp_b)

        // col_sum_b_u8[j] = sum over k of b_u8[k][j]
        let mut col_sums_b_u8 = vec![0i32; n];

        for b_i in 0..final_batch {
            let a_offset = if batch_a == 1 { 0 } else { b_i * stride_a };
            let b_offset = if batch_b == 1 { 0 } else { b_i * stride_b };
            let out_offset = b_i * stride_out;

            let a_slice = &a.data[a_offset..a_offset + stride_a];
            let b_slice = &b.data[b_offset..b_offset + stride_b];
            let out_slice = &mut out[out_offset..out_offset + stride_out];

            // Transpose B and XOR with 0x80: B[kk][jj] -> B_T[jj][kk] = B[kk][jj] ^ 0x80
            if b_i == 0 || batch_b > 1 {
                for jj in 0..n {
                    let mut csum: i32 = 0;
                    for kk in 0..k {
                        let b_val = b_slice[kk * n + jj];
                        b_t[jj * k_padded + kk] = b_val ^ 0x80; // reinterpret as i8
                        csum += b_val as i32;
                    }
                    // Zero-pad remainder (0x00 as i8 = 0, correct for zero-padding)
                    for kk in k..k_padded {
                        b_t[jj * k_padded + kk] = 0x80; // 0x80 ^ 0x80 = 0 as i8, but we XOR'd already
                        // Actually: padding should contribute 0 to the dot product.
                        // a_padded[kk] = 0 (we'll pad A with 0), so b_t value doesn't matter
                        // But A is padded with 0, so this is fine.
                        b_t[jj * k_padded + kk] = 0; // doesn't matter, A padding is 0
                    }
                    col_sums_b_u8[jj] = csum;
                }
            }

            // Pre-compute per-column correction: zp_a * (col_sum_b_u8[j] - K * zp_b)
            let k_zp_b = k as i32 * zp_b;
            let corr_128_minus_zpb = 128 - zp_b; // (128 - zp_b) factor

            // Prepare A with zero-padding to k_padded if needed
            // We'll copy A row by row with padding during the loop

            for i in 0..m {
                let a_row_start = i * k;
                let a_row = &a_slice[a_row_start..a_row_start + k];

                // Compute row_sum_a for zero-point correction
                let mut row_sum_a: i32 = 0;
                for &av in a_row {
                    row_sum_a += av as i32;
                }

                // Prepare padded A row (stack for small K, heap for large)
                let mut a_padded_heap;
                let mut a_padded_stack = [0u8; 2080]; // 2048 + 32
                let a_padded: &mut [u8] = if k_padded <= 2080 {
                    a_padded_stack[..k_padded].fill(0);
                    &mut a_padded_stack[..k_padded]
                } else {
                    a_padded_heap = vec![0u8; k_padded];
                    &mut a_padded_heap
                };
                a_padded[..k].copy_from_slice(a_row);

                let a_ptr = a_padded.as_ptr();

                // Per-row correction part: row_sum_a * (128 - zp_b)
                let row_corr = row_sum_a * corr_128_minus_zpb;

                let _zero = _mm256_setzero_si256();

                let mut j = 0;
                while j + 4 <= n {
                    let bt_ptr0 = b_t.as_ptr().add(j * k_padded);
                    let bt_ptr1 = b_t.as_ptr().add((j + 1) * k_padded);
                    let bt_ptr2 = b_t.as_ptr().add((j + 2) * k_padded);
                    let bt_ptr3 = b_t.as_ptr().add((j + 3) * k_padded);

                    let mut iacc0 = _mm256_setzero_si256();
                    let mut iacc1 = _mm256_setzero_si256();
                    let mut iacc2 = _mm256_setzero_si256();
                    let mut iacc3 = _mm256_setzero_si256();

                    let mut kk = 0;
                    while kk + 16 <= k_padded {
                        // Load 16 u8 values from A, zero-extend to 16 x i16
                        let a_128 = _mm_loadu_si128(a_ptr.add(kk) as *const __m128i);
                        let va_lo = _mm256_cvtepu8_epi16(a_128);

                        // Load 16 bytes from B_T, sign-extend to 16 x i16
                        // b_t contains u8 values that represent i8 (via XOR 0x80)
                        let b0_128 = _mm_loadu_si128(bt_ptr0.add(kk) as *const __m128i);
                        let b1_128 = _mm_loadu_si128(bt_ptr1.add(kk) as *const __m128i);
                        let b2_128 = _mm_loadu_si128(bt_ptr2.add(kk) as *const __m128i);
                        let b3_128 = _mm_loadu_si128(bt_ptr3.add(kk) as *const __m128i);

                        let vb0_lo = _mm256_cvtepi8_epi16(b0_128);
                        let vb1_lo = _mm256_cvtepi8_epi16(b1_128);
                        let vb2_lo = _mm256_cvtepi8_epi16(b2_128);
                        let vb3_lo = _mm256_cvtepi8_epi16(b3_128);

                        // VPMADDWD: i16 * i16 → i32 (adjacent pairs summed, NO saturation)
                        // Each produces 8 x i32 from 16 x i16 pairs
                        iacc0 = _mm256_add_epi32(iacc0, _mm256_madd_epi16(va_lo, vb0_lo));
                        iacc1 = _mm256_add_epi32(iacc1, _mm256_madd_epi16(va_lo, vb1_lo));
                        iacc2 = _mm256_add_epi32(iacc2, _mm256_madd_epi16(va_lo, vb2_lo));
                        iacc3 = _mm256_add_epi32(iacc3, _mm256_madd_epi16(va_lo, vb3_lo));

                        kk += 16;
                    }

                    // dot = sum(a_u8 * (b_u8 - 128))
                    let dot0 = hsum_epi32(iacc0);
                    let dot1 = hsum_epi32(iacc1);
                    let dot2 = hsum_epi32(iacc2);
                    let dot3 = hsum_epi32(iacc3);

                    // Full formula:
                    // (a - zp_a)(b - zp_b) = dot + row_sum_a*(128 - zp_b) - zp_a*(col_sum_b_u8 - K*zp_b)
                    let col_corr0 = zp_a * (col_sums_b_u8[j] - k_zp_b);
                    let col_corr1 = zp_a * (col_sums_b_u8[j + 1] - k_zp_b);
                    let col_corr2 = zp_a * (col_sums_b_u8[j + 2] - k_zp_b);
                    let col_corr3 = zp_a * (col_sums_b_u8[j + 3] - k_zp_b);

                    let mut f0 = (dot0 + row_corr - col_corr0) as f32;
                    let mut f1 = (dot1 + row_corr - col_corr1) as f32;
                    let mut f2 = (dot2 + row_corr - col_corr2) as f32;
                    let mut f3 = (dot3 + row_corr - col_corr3) as f32;

                    // Apply scale
                    if let Some(scale_data) = scale {
                        if scale_data.data.len() == 1 {
                            let sv = scale_data.data[0];
                            f0 *= sv; f1 *= sv; f2 *= sv; f3 *= sv;
                        } else {
                            f0 *= scale_data.data[j];
                            f1 *= scale_data.data[j + 1];
                            f2 *= scale_data.data[j + 2];
                            f3 *= scale_data.data[j + 3];
                        }
                    }
                    // Apply bias
                    if let Some(bias_data) = bias {
                        f0 += bias_data.data[j];
                        f1 += bias_data.data[j + 1];
                        f2 += bias_data.data[j + 2];
                        f3 += bias_data.data[j + 3];
                    }
                    // Apply ReLU
                    if apply_relu {
                        f0 = f0.max(0.0);
                        f1 = f1.max(0.0);
                        f2 = f2.max(0.0);
                        f3 = f3.max(0.0);
                    }

                    out_slice[i * n + j] = f0;
                    out_slice[i * n + j + 1] = f1;
                    out_slice[i * n + j + 2] = f2;
                    out_slice[i * n + j + 3] = f3;

                    j += 4;
                }

                // Handle remaining columns
                while j < n {
                    let bt_ptr = b_t.as_ptr().add(j * k_padded);
                    let mut iacc = _mm256_setzero_si256();
                    let mut kk = 0;
                    while kk + 16 <= k_padded {
                        let a_128 = _mm_loadu_si128(a_ptr.add(kk) as *const __m128i);
                        let va_lo = _mm256_cvtepu8_epi16(a_128);
                        let b_128 = _mm_loadu_si128(bt_ptr.add(kk) as *const __m128i);
                        let vb_lo = _mm256_cvtepi8_epi16(b_128);
                        iacc = _mm256_add_epi32(iacc, _mm256_madd_epi16(va_lo, vb_lo));
                        kk += 16;
                    }
                    let dot = hsum_epi32(iacc);
                    let col_corr = zp_a * (col_sums_b_u8[j] - k_zp_b);
                    let mut sum = (dot + row_corr - col_corr) as f32;
                    if let Some(scale_data) = scale {
                        if scale_data.data.len() == 1 { sum *= scale_data.data[0]; }
                        else { sum *= scale_data.data[j]; }
                    }
                    if let Some(bias_data) = bias { sum += bias_data.data[j]; }
                    if apply_relu && sum < 0.0 { sum = 0.0; }
                    out_slice[i * n + j] = sum;
                    j += 1;
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

/// Compute one row of the output matrix C = A_row × B_t (with zero-point corrections).
/// This is the core GEMM kernel, designed to be called from parallel or sequential paths.
/// `a_row`: u8 slice [K], `out_row`: f32 slice [N]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemm_row_avx2(
    a_row: &[u8],
    b_t_ptr: *const u8,     // Pre-transposed [N, K_padded] with XOR 0x80
    k: usize,
    n: usize,
    k_padded: usize,
    k_aligned: bool,
    col_sums_ptr: *const i32,
    zp_a: i32,
    _zp_b: i32,
    k_zp_b: i32,
    corr_128_minus_zpb: i32,
    scale_data_ptr: Option<*const f32>,
    scale_len: usize,
    bias_data_ptr: Option<*const f32>,
    apply_relu: bool,
    out_row: &mut [f32],
) {
    unsafe {
        // Vectorized row_sum_a using _mm256_sad_epu8
        let mut row_sum_a: i32;
        let zero_vec = _mm256_setzero_si256();
        {
            let mut sad_acc = _mm256_setzero_si256();
            let mut kk = 0;
            while kk + 32 <= k {
                let va = _mm256_loadu_si256(a_row.as_ptr().add(kk) as *const __m256i);
                sad_acc = _mm256_add_epi64(sad_acc, _mm256_sad_epu8(va, zero_vec));
                kk += 32;
            }
            let hi = _mm256_extracti128_si256(sad_acc, 1);
            let lo = _mm256_castsi256_si128(sad_acc);
            let s128 = _mm_add_epi64(lo, hi);
            row_sum_a = (_mm_extract_epi64(s128, 0) + _mm_extract_epi64(s128, 1)) as i32;
            while kk < k {
                row_sum_a += a_row[kk] as i32;
                kk += 1;
            }
        }

        // Use A directly if aligned, otherwise pad
        let a_ptr: *const u8;
        let mut a_padded_stack = [0u8; 2080];
        if k_aligned {
            a_ptr = a_row.as_ptr();
        } else {
            a_padded_stack[..k].copy_from_slice(a_row);
            a_ptr = a_padded_stack.as_ptr();
        }

        let row_corr = row_sum_a * corr_128_minus_zpb;

        let mut j = 0;
        // 8-column unrolling
        while j + 8 <= n {
            let bt0 = b_t_ptr.add(j * k_padded);
            let bt1 = b_t_ptr.add((j + 1) * k_padded);
            let bt2 = b_t_ptr.add((j + 2) * k_padded);
            let bt3 = b_t_ptr.add((j + 3) * k_padded);
            let bt4 = b_t_ptr.add((j + 4) * k_padded);
            let bt5 = b_t_ptr.add((j + 5) * k_padded);
            let bt6 = b_t_ptr.add((j + 6) * k_padded);
            let bt7 = b_t_ptr.add((j + 7) * k_padded);

            let mut iacc0 = _mm256_setzero_si256();
            let mut iacc1 = _mm256_setzero_si256();
            let mut iacc2 = _mm256_setzero_si256();
            let mut iacc3 = _mm256_setzero_si256();
            let mut iacc4 = _mm256_setzero_si256();
            let mut iacc5 = _mm256_setzero_si256();
            let mut iacc6 = _mm256_setzero_si256();
            let mut iacc7 = _mm256_setzero_si256();

            let mut kk = 0;
            let k32 = k_padded & !31;

            // Main loop: 32 bytes per K iteration (256-bit loads)
            while kk < k32 {
                let a_full = _mm256_loadu_si256(a_ptr.add(kk) as *const __m256i);
                let va_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a_full));
                let va_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(a_full, 1));

                macro_rules! acc_col {
                    ($iacc:ident, $bt:ident) => {{
                        let b_full = _mm256_loadu_si256($bt.add(kk) as *const __m256i);
                        let vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b_full));
                        let vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_full, 1));
                        $iacc = _mm256_add_epi32($iacc, _mm256_madd_epi16(va_lo, vb_lo));
                        $iacc = _mm256_add_epi32($iacc, _mm256_madd_epi16(va_hi, vb_hi));
                    }};
                }

                acc_col!(iacc0, bt0);
                acc_col!(iacc1, bt1);
                acc_col!(iacc2, bt2);
                acc_col!(iacc3, bt3);
                acc_col!(iacc4, bt4);
                acc_col!(iacc5, bt5);
                acc_col!(iacc6, bt6);
                acc_col!(iacc7, bt7);

                kk += 32;
            }

            // 16-byte remainder
            if kk < k_padded {
                let a_128 = _mm_loadu_si128(a_ptr.add(kk) as *const __m128i);
                let va = _mm256_cvtepu8_epi16(a_128);

                iacc0 = _mm256_add_epi32(iacc0, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt0.add(kk) as *const __m128i))));
                iacc1 = _mm256_add_epi32(iacc1, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt1.add(kk) as *const __m128i))));
                iacc2 = _mm256_add_epi32(iacc2, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt2.add(kk) as *const __m128i))));
                iacc3 = _mm256_add_epi32(iacc3, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt3.add(kk) as *const __m128i))));
                iacc4 = _mm256_add_epi32(iacc4, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt4.add(kk) as *const __m128i))));
                iacc5 = _mm256_add_epi32(iacc5, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt5.add(kk) as *const __m128i))));
                iacc6 = _mm256_add_epi32(iacc6, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt6.add(kk) as *const __m128i))));
                iacc7 = _mm256_add_epi32(iacc7, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt7.add(kk) as *const __m128i))));
            }

            // Vectorized horizontal sum using hadd
            let h01 = _mm256_hadd_epi32(iacc0, iacc1);
            let h23 = _mm256_hadd_epi32(iacc2, iacc3);
            let h0123 = _mm256_hadd_epi32(h01, h23);
            let h0123_hi = _mm256_extracti128_si256(h0123, 1);
            let h0123_lo = _mm256_castsi256_si128(h0123);
            let dots_0123 = _mm_add_epi32(h0123_lo, h0123_hi);

            let h45 = _mm256_hadd_epi32(iacc4, iacc5);
            let h67 = _mm256_hadd_epi32(iacc6, iacc7);
            let h4567 = _mm256_hadd_epi32(h45, h67);
            let h4567_hi = _mm256_extracti128_si256(h4567, 1);
            let h4567_lo = _mm256_castsi256_si128(h4567);
            let dots_4567 = _mm_add_epi32(h4567_lo, h4567_hi);

            let d0 = _mm_extract_epi32(dots_0123, 0);
            let d1 = _mm_extract_epi32(dots_0123, 1);
            let d2 = _mm_extract_epi32(dots_0123, 2);
            let d3 = _mm_extract_epi32(dots_0123, 3);
            let d4 = _mm_extract_epi32(dots_4567, 0);
            let d5 = _mm_extract_epi32(dots_4567, 1);
            let d6 = _mm_extract_epi32(dots_4567, 2);
            let d7 = _mm_extract_epi32(dots_4567, 3);

            let mut f0 = (d0 + row_corr - zp_a * (*col_sums_ptr.add(j) - k_zp_b)) as f32;
            let mut f1 = (d1 + row_corr - zp_a * (*col_sums_ptr.add(j+1) - k_zp_b)) as f32;
            let mut f2 = (d2 + row_corr - zp_a * (*col_sums_ptr.add(j+2) - k_zp_b)) as f32;
            let mut f3 = (d3 + row_corr - zp_a * (*col_sums_ptr.add(j+3) - k_zp_b)) as f32;
            let mut f4 = (d4 + row_corr - zp_a * (*col_sums_ptr.add(j+4) - k_zp_b)) as f32;
            let mut f5 = (d5 + row_corr - zp_a * (*col_sums_ptr.add(j+5) - k_zp_b)) as f32;
            let mut f6 = (d6 + row_corr - zp_a * (*col_sums_ptr.add(j+6) - k_zp_b)) as f32;
            let mut f7 = (d7 + row_corr - zp_a * (*col_sums_ptr.add(j+7) - k_zp_b)) as f32;

            if let Some(sp) = scale_data_ptr {
                if scale_len == 1 {
                    let sv = *sp;
                    f0 *= sv; f1 *= sv; f2 *= sv; f3 *= sv;
                    f4 *= sv; f5 *= sv; f6 *= sv; f7 *= sv;
                } else {
                    f0 *= *sp.add(j); f1 *= *sp.add(j+1);
                    f2 *= *sp.add(j+2); f3 *= *sp.add(j+3);
                    f4 *= *sp.add(j+4); f5 *= *sp.add(j+5);
                    f6 *= *sp.add(j+6); f7 *= *sp.add(j+7);
                }
            }
            if let Some(bp) = bias_data_ptr {
                f0 += *bp.add(j); f1 += *bp.add(j+1);
                f2 += *bp.add(j+2); f3 += *bp.add(j+3);
                f4 += *bp.add(j+4); f5 += *bp.add(j+5);
                f6 += *bp.add(j+6); f7 += *bp.add(j+7);
            }
            if apply_relu {
                f0 = f0.max(0.0); f1 = f1.max(0.0); f2 = f2.max(0.0); f3 = f3.max(0.0);
                f4 = f4.max(0.0); f5 = f5.max(0.0); f6 = f6.max(0.0); f7 = f7.max(0.0);
            }

            out_row[j] = f0; out_row[j+1] = f1; out_row[j+2] = f2; out_row[j+3] = f3;
            out_row[j+4] = f4; out_row[j+5] = f5; out_row[j+6] = f6; out_row[j+7] = f7;

            j += 8;
        }

        // 4-column remainder
        while j + 4 <= n {
            let bt0 = b_t_ptr.add(j * k_padded);
            let bt1 = b_t_ptr.add((j + 1) * k_padded);
            let bt2 = b_t_ptr.add((j + 2) * k_padded);
            let bt3 = b_t_ptr.add((j + 3) * k_padded);

            let mut iacc0 = _mm256_setzero_si256();
            let mut iacc1 = _mm256_setzero_si256();
            let mut iacc2 = _mm256_setzero_si256();
            let mut iacc3 = _mm256_setzero_si256();

            let mut kk = 0;
            let k32_4 = k_padded & !31;
            while kk < k32_4 {
                let a_full = _mm256_loadu_si256(a_ptr.add(kk) as *const __m256i);
                let va_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a_full));
                let va_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(a_full, 1));

                macro_rules! acc4 {
                    ($iacc:ident, $bt:ident) => {{
                        let b_full = _mm256_loadu_si256($bt.add(kk) as *const __m256i);
                        let vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b_full));
                        let vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_full, 1));
                        $iacc = _mm256_add_epi32($iacc, _mm256_madd_epi16(va_lo, vb_lo));
                        $iacc = _mm256_add_epi32($iacc, _mm256_madd_epi16(va_hi, vb_hi));
                    }};
                }
                acc4!(iacc0, bt0);
                acc4!(iacc1, bt1);
                acc4!(iacc2, bt2);
                acc4!(iacc3, bt3);
                kk += 32;
            }
            if kk < k_padded {
                let a_128 = _mm_loadu_si128(a_ptr.add(kk) as *const __m128i);
                let va = _mm256_cvtepu8_epi16(a_128);
                iacc0 = _mm256_add_epi32(iacc0, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt0.add(kk) as *const __m128i))));
                iacc1 = _mm256_add_epi32(iacc1, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt1.add(kk) as *const __m128i))));
                iacc2 = _mm256_add_epi32(iacc2, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt2.add(kk) as *const __m128i))));
                iacc3 = _mm256_add_epi32(iacc3, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt3.add(kk) as *const __m128i))));
            }

            let h01 = _mm256_hadd_epi32(iacc0, iacc1);
            let h23 = _mm256_hadd_epi32(iacc2, iacc3);
            let h0123 = _mm256_hadd_epi32(h01, h23);
            let hi = _mm256_extracti128_si256(h0123, 1);
            let lo = _mm256_castsi256_si128(h0123);
            let dots = _mm_add_epi32(lo, hi);

            let d0 = _mm_extract_epi32(dots, 0);
            let d1 = _mm_extract_epi32(dots, 1);
            let d2 = _mm_extract_epi32(dots, 2);
            let d3 = _mm_extract_epi32(dots, 3);

            let mut f0 = (d0 + row_corr - zp_a * (*col_sums_ptr.add(j) - k_zp_b)) as f32;
            let mut f1 = (d1 + row_corr - zp_a * (*col_sums_ptr.add(j+1) - k_zp_b)) as f32;
            let mut f2 = (d2 + row_corr - zp_a * (*col_sums_ptr.add(j+2) - k_zp_b)) as f32;
            let mut f3 = (d3 + row_corr - zp_a * (*col_sums_ptr.add(j+3) - k_zp_b)) as f32;

            if let Some(sp) = scale_data_ptr {
                if scale_len == 1 {
                    let sv = *sp;
                    f0 *= sv; f1 *= sv; f2 *= sv; f3 *= sv;
                } else {
                    f0 *= *sp.add(j); f1 *= *sp.add(j+1);
                    f2 *= *sp.add(j+2); f3 *= *sp.add(j+3);
                }
            }
            if let Some(bp) = bias_data_ptr {
                f0 += *bp.add(j); f1 += *bp.add(j+1);
                f2 += *bp.add(j+2); f3 += *bp.add(j+3);
            }
            if apply_relu {
                f0 = f0.max(0.0); f1 = f1.max(0.0);
                f2 = f2.max(0.0); f3 = f3.max(0.0);
            }

            out_row[j] = f0; out_row[j+1] = f1;
            out_row[j+2] = f2; out_row[j+3] = f3;

            j += 4;
        }

        // Scalar remainder columns
        while j < n {
            let bt = b_t_ptr.add(j * k_padded);
            let mut iacc = _mm256_setzero_si256();
            let mut kk = 0;
            let k32_s = k_padded & !31;
            while kk < k32_s {
                let a_full = _mm256_loadu_si256(a_ptr.add(kk) as *const __m256i);
                let va_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a_full));
                let va_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(a_full, 1));
                let b_full = _mm256_loadu_si256(bt.add(kk) as *const __m256i);
                let vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b_full));
                let vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_full, 1));
                iacc = _mm256_add_epi32(iacc, _mm256_madd_epi16(va_lo, vb_lo));
                iacc = _mm256_add_epi32(iacc, _mm256_madd_epi16(va_hi, vb_hi));
                kk += 32;
            }
            if kk < k_padded {
                let a_128 = _mm_loadu_si128(a_ptr.add(kk) as *const __m128i);
                let va = _mm256_cvtepu8_epi16(a_128);
                let b_128 = _mm_loadu_si128(bt.add(kk) as *const __m128i);
                iacc = _mm256_add_epi32(iacc, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(b_128)));
            }
            let dot = hsum_epi32(iacc);
            let col_corr = zp_a * (*col_sums_ptr.add(j) - k_zp_b);
            let mut sum = (dot + row_corr - col_corr) as f32;
            if let Some(sp) = scale_data_ptr {
                if scale_len == 1 { sum *= *sp; }
                else { sum *= *sp.add(j); }
            }
            if let Some(bp) = bias_data_ptr { sum += *bp.add(j); }
            if apply_relu && sum < 0.0 { sum = 0.0; }
            out_row[j] = sum;
            j += 1;
        }
    }
}

/// Multi-row GEMM kernel: processes 2 rows of A simultaneously, sharing B loads.
/// This halves B memory traffic compared to single-row, giving ~1.5x GEMM speedup.
/// Uses cvtepu8_epi16/cvtepi8_epi16 + madd_epi16 (saturation-free, same as gemm_row_avx2).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemm_2rows_avx2(
    a_row0: &[u8],
    a_row1: &[u8],
    b_t_ptr: *const u8,
    k: usize,
    n: usize,
    k_padded: usize,
    k_aligned: bool,
    col_sums_ptr: *const i32,
    zp_a: i32,
    _zp_b: i32,
    k_zp_b: i32,
    corr_128_minus_zpb: i32,
    scale_data_ptr: Option<*const f32>,
    scale_len: usize,
    bias_data_ptr: Option<*const f32>,
    apply_relu: bool,
    row_sum_a0: i32,
    row_sum_a1: i32,
    out_row0: &mut [f32],
    out_row1: &mut [f32],
) {
    unsafe {
        // Pad A rows if needed
        let a0_ptr: *const u8;
        let a1_ptr: *const u8;
        let mut a0_padded = [0u8; 2080];
        let mut a1_padded = [0u8; 2080];
        if k_aligned {
            a0_ptr = a_row0.as_ptr();
            a1_ptr = a_row1.as_ptr();
        } else {
            a0_padded[..k].copy_from_slice(a_row0);
            a1_padded[..k].copy_from_slice(a_row1);
            a0_ptr = a0_padded.as_ptr();
            a1_ptr = a1_padded.as_ptr();
        }

        let row_corr0 = row_sum_a0 * corr_128_minus_zpb;
        let row_corr1 = row_sum_a1 * corr_128_minus_zpb;

        let mut j = 0;
        // 4-column unrolling with 2 rows = 8 accumulators (fits in 16 regs)
        while j + 4 <= n {
            let bt0 = b_t_ptr.add(j * k_padded);
            let bt1 = b_t_ptr.add((j + 1) * k_padded);
            let bt2 = b_t_ptr.add((j + 2) * k_padded);
            let bt3 = b_t_ptr.add((j + 3) * k_padded);

            // Accumulators: row0×col0..3, row1×col0..3
            let mut r0c0 = _mm256_setzero_si256();
            let mut r0c1 = _mm256_setzero_si256();
            let mut r0c2 = _mm256_setzero_si256();
            let mut r0c3 = _mm256_setzero_si256();
            let mut r1c0 = _mm256_setzero_si256();
            let mut r1c1 = _mm256_setzero_si256();
            let mut r1c2 = _mm256_setzero_si256();
            let mut r1c3 = _mm256_setzero_si256();

            let mut kk = 0;
            let k32 = k_padded & !31; // largest multiple of 32 <= k_padded

            // Main loop: 32 bytes per K iteration (256-bit loads, split into two 16-byte halves)
            while kk < k32 {
                // Load 32 bytes of each A row, split into lo/hi 128-bit halves
                let a0_full = _mm256_loadu_si256(a0_ptr.add(kk) as *const __m256i);
                let a1_full = _mm256_loadu_si256(a1_ptr.add(kk) as *const __m256i);
                let a0_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a0_full));
                let a0_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(a0_full, 1));
                let a1_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a1_full));
                let a1_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(a1_full, 1));

                // Column 0: load 32 B bytes, shared by both A rows
                let b0_full = _mm256_loadu_si256(bt0.add(kk) as *const __m256i);
                let vb0_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b0_full));
                let vb0_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b0_full, 1));
                r0c0 = _mm256_add_epi32(r0c0, _mm256_madd_epi16(a0_lo, vb0_lo));
                r0c0 = _mm256_add_epi32(r0c0, _mm256_madd_epi16(a0_hi, vb0_hi));
                r1c0 = _mm256_add_epi32(r1c0, _mm256_madd_epi16(a1_lo, vb0_lo));
                r1c0 = _mm256_add_epi32(r1c0, _mm256_madd_epi16(a1_hi, vb0_hi));

                // Column 1
                let b1_full = _mm256_loadu_si256(bt1.add(kk) as *const __m256i);
                let vb1_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b1_full));
                let vb1_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b1_full, 1));
                r0c1 = _mm256_add_epi32(r0c1, _mm256_madd_epi16(a0_lo, vb1_lo));
                r0c1 = _mm256_add_epi32(r0c1, _mm256_madd_epi16(a0_hi, vb1_hi));
                r1c1 = _mm256_add_epi32(r1c1, _mm256_madd_epi16(a1_lo, vb1_lo));
                r1c1 = _mm256_add_epi32(r1c1, _mm256_madd_epi16(a1_hi, vb1_hi));

                // Column 2
                let b2_full = _mm256_loadu_si256(bt2.add(kk) as *const __m256i);
                let vb2_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b2_full));
                let vb2_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b2_full, 1));
                r0c2 = _mm256_add_epi32(r0c2, _mm256_madd_epi16(a0_lo, vb2_lo));
                r0c2 = _mm256_add_epi32(r0c2, _mm256_madd_epi16(a0_hi, vb2_hi));
                r1c2 = _mm256_add_epi32(r1c2, _mm256_madd_epi16(a1_lo, vb2_lo));
                r1c2 = _mm256_add_epi32(r1c2, _mm256_madd_epi16(a1_hi, vb2_hi));

                // Column 3
                let b3_full = _mm256_loadu_si256(bt3.add(kk) as *const __m256i);
                let vb3_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b3_full));
                let vb3_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b3_full, 1));
                r0c3 = _mm256_add_epi32(r0c3, _mm256_madd_epi16(a0_lo, vb3_lo));
                r0c3 = _mm256_add_epi32(r0c3, _mm256_madd_epi16(a0_hi, vb3_hi));
                r1c3 = _mm256_add_epi32(r1c3, _mm256_madd_epi16(a1_lo, vb3_lo));
                r1c3 = _mm256_add_epi32(r1c3, _mm256_madd_epi16(a1_hi, vb3_hi));

                kk += 32;
            }

            // 16-byte remainder (for K not a multiple of 32)
            if kk < k_padded {
                let va0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(a0_ptr.add(kk) as *const __m128i));
                let va1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(a1_ptr.add(kk) as *const __m128i));

                let vb0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(bt0.add(kk) as *const __m128i));
                r0c0 = _mm256_add_epi32(r0c0, _mm256_madd_epi16(va0, vb0));
                r1c0 = _mm256_add_epi32(r1c0, _mm256_madd_epi16(va1, vb0));

                let vb1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(bt1.add(kk) as *const __m128i));
                r0c1 = _mm256_add_epi32(r0c1, _mm256_madd_epi16(va0, vb1));
                r1c1 = _mm256_add_epi32(r1c1, _mm256_madd_epi16(va1, vb1));

                let vb2 = _mm256_cvtepi8_epi16(_mm_loadu_si128(bt2.add(kk) as *const __m128i));
                r0c2 = _mm256_add_epi32(r0c2, _mm256_madd_epi16(va0, vb2));
                r1c2 = _mm256_add_epi32(r1c2, _mm256_madd_epi16(va1, vb2));

                let vb3 = _mm256_cvtepi8_epi16(_mm_loadu_si128(bt3.add(kk) as *const __m128i));
                r0c3 = _mm256_add_epi32(r0c3, _mm256_madd_epi16(va0, vb3));
                r1c3 = _mm256_add_epi32(r1c3, _mm256_madd_epi16(va1, vb3));
            }

            // Horizontal reduce for row 0
            let h01_r0 = _mm256_hadd_epi32(r0c0, r0c1);
            let h23_r0 = _mm256_hadd_epi32(r0c2, r0c3);
            let h0123_r0 = _mm256_hadd_epi32(h01_r0, h23_r0);
            let hi_r0 = _mm256_extracti128_si256(h0123_r0, 1);
            let lo_r0 = _mm256_castsi256_si128(h0123_r0);
            let dots_r0 = _mm_add_epi32(lo_r0, hi_r0);

            // Horizontal reduce for row 1
            let h01_r1 = _mm256_hadd_epi32(r1c0, r1c1);
            let h23_r1 = _mm256_hadd_epi32(r1c2, r1c3);
            let h0123_r1 = _mm256_hadd_epi32(h01_r1, h23_r1);
            let hi_r1 = _mm256_extracti128_si256(h0123_r1, 1);
            let lo_r1 = _mm256_castsi256_si128(h0123_r1);
            let dots_r1 = _mm_add_epi32(lo_r1, hi_r1);

            // Extract and apply corrections for row 0
            let d0_r0 = _mm_extract_epi32(dots_r0, 0);
            let d1_r0 = _mm_extract_epi32(dots_r0, 1);
            let d2_r0 = _mm_extract_epi32(dots_r0, 2);
            let d3_r0 = _mm_extract_epi32(dots_r0, 3);

            let cs0 = *col_sums_ptr.add(j) - k_zp_b;
            let cs1 = *col_sums_ptr.add(j + 1) - k_zp_b;
            let cs2 = *col_sums_ptr.add(j + 2) - k_zp_b;
            let cs3 = *col_sums_ptr.add(j + 3) - k_zp_b;

            let mut f0_r0 = (d0_r0 + row_corr0 - zp_a * cs0) as f32;
            let mut f1_r0 = (d1_r0 + row_corr0 - zp_a * cs1) as f32;
            let mut f2_r0 = (d2_r0 + row_corr0 - zp_a * cs2) as f32;
            let mut f3_r0 = (d3_r0 + row_corr0 - zp_a * cs3) as f32;

            // Extract and apply corrections for row 1
            let d0_r1 = _mm_extract_epi32(dots_r1, 0);
            let d1_r1 = _mm_extract_epi32(dots_r1, 1);
            let d2_r1 = _mm_extract_epi32(dots_r1, 2);
            let d3_r1 = _mm_extract_epi32(dots_r1, 3);

            let mut f0_r1 = (d0_r1 + row_corr1 - zp_a * cs0) as f32;
            let mut f1_r1 = (d1_r1 + row_corr1 - zp_a * cs1) as f32;
            let mut f2_r1 = (d2_r1 + row_corr1 - zp_a * cs2) as f32;
            let mut f3_r1 = (d3_r1 + row_corr1 - zp_a * cs3) as f32;

            // Scale
            if let Some(sp) = scale_data_ptr {
                if scale_len == 1 {
                    let sv = *sp;
                    f0_r0 *= sv; f1_r0 *= sv; f2_r0 *= sv; f3_r0 *= sv;
                    f0_r1 *= sv; f1_r1 *= sv; f2_r1 *= sv; f3_r1 *= sv;
                } else {
                    let s0 = *sp.add(j); let s1 = *sp.add(j+1);
                    let s2 = *sp.add(j+2); let s3 = *sp.add(j+3);
                    f0_r0 *= s0; f1_r0 *= s1; f2_r0 *= s2; f3_r0 *= s3;
                    f0_r1 *= s0; f1_r1 *= s1; f2_r1 *= s2; f3_r1 *= s3;
                }
            }
            // Bias
            if let Some(bp) = bias_data_ptr {
                let b0 = *bp.add(j); let b1 = *bp.add(j+1);
                let b2 = *bp.add(j+2); let b3 = *bp.add(j+3);
                f0_r0 += b0; f1_r0 += b1; f2_r0 += b2; f3_r0 += b3;
                f0_r1 += b0; f1_r1 += b1; f2_r1 += b2; f3_r1 += b3;
            }
            // ReLU
            if apply_relu {
                f0_r0 = f0_r0.max(0.0); f1_r0 = f1_r0.max(0.0);
                f2_r0 = f2_r0.max(0.0); f3_r0 = f3_r0.max(0.0);
                f0_r1 = f0_r1.max(0.0); f1_r1 = f1_r1.max(0.0);
                f2_r1 = f2_r1.max(0.0); f3_r1 = f3_r1.max(0.0);
            }

            out_row0[j] = f0_r0; out_row0[j+1] = f1_r0;
            out_row0[j+2] = f2_r0; out_row0[j+3] = f3_r0;
            out_row1[j] = f0_r1; out_row1[j+1] = f1_r1;
            out_row1[j+2] = f2_r1; out_row1[j+3] = f3_r1;

            j += 4;
        }

        // Scalar remainder columns (same as single-row for both rows)
        while j < n {
            let bt = b_t_ptr.add(j * k_padded);
            let mut iacc0 = _mm256_setzero_si256();
            let mut iacc1 = _mm256_setzero_si256();
            let mut kk = 0;
            while kk + 16 <= k_padded {
                let vb = _mm256_cvtepi8_epi16(_mm_loadu_si128(bt.add(kk) as *const __m128i));
                let va0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(a0_ptr.add(kk) as *const __m128i));
                let va1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(a1_ptr.add(kk) as *const __m128i));
                iacc0 = _mm256_add_epi32(iacc0, _mm256_madd_epi16(va0, vb));
                iacc1 = _mm256_add_epi32(iacc1, _mm256_madd_epi16(va1, vb));
                kk += 16;
            }
            let dot0 = hsum_epi32(iacc0);
            let dot1 = hsum_epi32(iacc1);
            let col_corr = *col_sums_ptr.add(j) - k_zp_b;

            let mut sum0 = (dot0 + row_corr0 - zp_a * col_corr) as f32;
            let mut sum1 = (dot1 + row_corr1 - zp_a * col_corr) as f32;
            if let Some(sp) = scale_data_ptr {
                let sv = if scale_len == 1 { *sp } else { *sp.add(j) };
                sum0 *= sv; sum1 *= sv;
            }
            if let Some(bp) = bias_data_ptr {
                let bv = *bp.add(j);
                sum0 += bv; sum1 += bv;
            }
            if apply_relu { sum0 = sum0.max(0.0); sum1 = sum1.max(0.0); }
            out_row0[j] = sum0;
            out_row1[j] = sum1;
            j += 1;
        }
    }
}

/// Compute row_sum_a for a slice of u8 values using AVX2 sad_epu8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn row_sum_u8_avx2(a_row: &[u8]) -> i32 {
    let k = a_row.len();
    let zero_vec = _mm256_setzero_si256();
    let mut sad_acc = _mm256_setzero_si256();
    let mut kk = 0;
    while kk + 32 <= k {
        unsafe {
            let va = _mm256_loadu_si256(a_row.as_ptr().add(kk) as *const __m256i);
            sad_acc = _mm256_add_epi64(sad_acc, _mm256_sad_epu8(va, zero_vec));
        }
        kk += 32;
    }
    let hi = _mm256_extracti128_si256(sad_acc, 1);
    let lo = _mm256_castsi256_si128(sad_acc);
    let s128 = _mm_add_epi64(lo, hi);
    let mut sum = (_mm_extract_epi64(s128, 0) + _mm_extract_epi64(s128, 1)) as i32;
    while kk < k {
        sum += a_row[kk] as i32;
        kk += 1;
    }
    sum
}

/// VPMADDUBSW-based GEMM row: processes 32 K elements per iteration (2x throughput).
/// Uses saturating u8×i8→i16 horizontal add, then widens to i32 via madd_epi16(prod, ones).
/// Falls back to the 16-element path for remaining K when k_padded is not multiple of 32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemm_row_avx2_v2(
    a_row: &[u8],
    b_t_ptr: *const u8,
    k: usize,
    n: usize,
    k_padded: usize,
    col_sums_ptr: *const i32,
    zp_a: i32,
    _zp_b: i32,
    k_zp_b: i32,
    corr_128_minus_zpb: i32,
    scale_data_ptr: Option<*const f32>,
    scale_len: usize,
    bias_data_ptr: Option<*const f32>,
    apply_relu: bool,
    out_row: &mut [f32],
) {
    unsafe {
        // Vectorized row_sum_a
        let zero_vec = _mm256_setzero_si256();
        let mut row_sum_a: i32;
        {
            let mut sad_acc = _mm256_setzero_si256();
            let mut kk = 0;
            while kk + 32 <= k {
                let va = _mm256_loadu_si256(a_row.as_ptr().add(kk) as *const __m256i);
                sad_acc = _mm256_add_epi64(sad_acc, _mm256_sad_epu8(va, zero_vec));
                kk += 32;
            }
            let hi = _mm256_extracti128_si256(sad_acc, 1);
            let lo = _mm256_castsi256_si128(sad_acc);
            let s128 = _mm_add_epi64(lo, hi);
            row_sum_a = (_mm_extract_epi64(s128, 0) + _mm_extract_epi64(s128, 1)) as i32;
            while kk < k {
                row_sum_a += a_row[kk] as i32;
                kk += 1;
            }
        }

        // Pad A to k_padded if needed (ensure 32-byte alignment for VPMADDUBSW)
        let a_ptr: *const u8;
        let mut a_padded_stack = [0u8; 4096];
        if k == k_padded {
            a_ptr = a_row.as_ptr();
        } else {
            a_padded_stack[..k].copy_from_slice(a_row);
            a_ptr = a_padded_stack.as_ptr();
        }

        let row_corr = row_sum_a * corr_128_minus_zpb;
        let ones_16 = _mm256_set1_epi16(1);
        let k32 = k_padded & !31; // largest multiple of 32 <= k_padded

        let mut j = 0;
        // 8-column unrolling with VPMADDUBSW (32 elements per K iteration)
        while j + 8 <= n {
            let bt0 = b_t_ptr.add(j * k_padded);
            let bt1 = b_t_ptr.add((j + 1) * k_padded);
            let bt2 = b_t_ptr.add((j + 2) * k_padded);
            let bt3 = b_t_ptr.add((j + 3) * k_padded);
            let bt4 = b_t_ptr.add((j + 4) * k_padded);
            let bt5 = b_t_ptr.add((j + 5) * k_padded);
            let bt6 = b_t_ptr.add((j + 6) * k_padded);
            let bt7 = b_t_ptr.add((j + 7) * k_padded);

            let mut iacc0 = _mm256_setzero_si256();
            let mut iacc1 = _mm256_setzero_si256();
            let mut iacc2 = _mm256_setzero_si256();
            let mut iacc3 = _mm256_setzero_si256();
            let mut iacc4 = _mm256_setzero_si256();
            let mut iacc5 = _mm256_setzero_si256();
            let mut iacc6 = _mm256_setzero_si256();
            let mut iacc7 = _mm256_setzero_si256();

            // Main loop: 32 elements per iteration using VPMADDUBSW
            let mut kk = 0;
            while kk < k32 {
                let va = _mm256_loadu_si256(a_ptr.add(kk) as *const __m256i);

                iacc0 = _mm256_add_epi32(iacc0, _mm256_madd_epi16(_mm256_maddubs_epi16(va, _mm256_loadu_si256(bt0.add(kk) as *const __m256i)), ones_16));
                iacc1 = _mm256_add_epi32(iacc1, _mm256_madd_epi16(_mm256_maddubs_epi16(va, _mm256_loadu_si256(bt1.add(kk) as *const __m256i)), ones_16));
                iacc2 = _mm256_add_epi32(iacc2, _mm256_madd_epi16(_mm256_maddubs_epi16(va, _mm256_loadu_si256(bt2.add(kk) as *const __m256i)), ones_16));
                iacc3 = _mm256_add_epi32(iacc3, _mm256_madd_epi16(_mm256_maddubs_epi16(va, _mm256_loadu_si256(bt3.add(kk) as *const __m256i)), ones_16));
                iacc4 = _mm256_add_epi32(iacc4, _mm256_madd_epi16(_mm256_maddubs_epi16(va, _mm256_loadu_si256(bt4.add(kk) as *const __m256i)), ones_16));
                iacc5 = _mm256_add_epi32(iacc5, _mm256_madd_epi16(_mm256_maddubs_epi16(va, _mm256_loadu_si256(bt5.add(kk) as *const __m256i)), ones_16));
                iacc6 = _mm256_add_epi32(iacc6, _mm256_madd_epi16(_mm256_maddubs_epi16(va, _mm256_loadu_si256(bt6.add(kk) as *const __m256i)), ones_16));
                iacc7 = _mm256_add_epi32(iacc7, _mm256_madd_epi16(_mm256_maddubs_epi16(va, _mm256_loadu_si256(bt7.add(kk) as *const __m256i)), ones_16));

                kk += 32;
            }

            // Remainder: 16 elements using cvtepu8/cvtepi8 + madd_epi16
            if kk < k_padded {
                let a_128 = _mm_loadu_si128(a_ptr.add(kk) as *const __m128i);
                let va = _mm256_cvtepu8_epi16(a_128);

                iacc0 = _mm256_add_epi32(iacc0, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt0.add(kk) as *const __m128i))));
                iacc1 = _mm256_add_epi32(iacc1, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt1.add(kk) as *const __m128i))));
                iacc2 = _mm256_add_epi32(iacc2, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt2.add(kk) as *const __m128i))));
                iacc3 = _mm256_add_epi32(iacc3, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt3.add(kk) as *const __m128i))));
                iacc4 = _mm256_add_epi32(iacc4, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt4.add(kk) as *const __m128i))));
                iacc5 = _mm256_add_epi32(iacc5, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt5.add(kk) as *const __m128i))));
                iacc6 = _mm256_add_epi32(iacc6, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt6.add(kk) as *const __m128i))));
                iacc7 = _mm256_add_epi32(iacc7, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt7.add(kk) as *const __m128i))));
            }

            // Vectorized horizontal sum (same reduction as gemm_row_avx2)
            let h01 = _mm256_hadd_epi32(iacc0, iacc1);
            let h23 = _mm256_hadd_epi32(iacc2, iacc3);
            let h0123 = _mm256_hadd_epi32(h01, h23);
            let h0123_hi = _mm256_extracti128_si256(h0123, 1);
            let h0123_lo = _mm256_castsi256_si128(h0123);
            let dots_0123 = _mm_add_epi32(h0123_lo, h0123_hi);

            let h45 = _mm256_hadd_epi32(iacc4, iacc5);
            let h67 = _mm256_hadd_epi32(iacc6, iacc7);
            let h4567 = _mm256_hadd_epi32(h45, h67);
            let h4567_hi = _mm256_extracti128_si256(h4567, 1);
            let h4567_lo = _mm256_castsi256_si128(h4567);
            let dots_4567 = _mm_add_epi32(h4567_lo, h4567_hi);

            let d0 = _mm_extract_epi32(dots_0123, 0);
            let d1 = _mm_extract_epi32(dots_0123, 1);
            let d2 = _mm_extract_epi32(dots_0123, 2);
            let d3 = _mm_extract_epi32(dots_0123, 3);
            let d4 = _mm_extract_epi32(dots_4567, 0);
            let d5 = _mm_extract_epi32(dots_4567, 1);
            let d6 = _mm_extract_epi32(dots_4567, 2);
            let d7 = _mm_extract_epi32(dots_4567, 3);

            let mut f0 = (d0 + row_corr - zp_a * (*col_sums_ptr.add(j) - k_zp_b)) as f32;
            let mut f1 = (d1 + row_corr - zp_a * (*col_sums_ptr.add(j+1) - k_zp_b)) as f32;
            let mut f2 = (d2 + row_corr - zp_a * (*col_sums_ptr.add(j+2) - k_zp_b)) as f32;
            let mut f3 = (d3 + row_corr - zp_a * (*col_sums_ptr.add(j+3) - k_zp_b)) as f32;
            let mut f4 = (d4 + row_corr - zp_a * (*col_sums_ptr.add(j+4) - k_zp_b)) as f32;
            let mut f5 = (d5 + row_corr - zp_a * (*col_sums_ptr.add(j+5) - k_zp_b)) as f32;
            let mut f6 = (d6 + row_corr - zp_a * (*col_sums_ptr.add(j+6) - k_zp_b)) as f32;
            let mut f7 = (d7 + row_corr - zp_a * (*col_sums_ptr.add(j+7) - k_zp_b)) as f32;

            if let Some(sp) = scale_data_ptr {
                if scale_len == 1 {
                    let sv = *sp;
                    f0 *= sv; f1 *= sv; f2 *= sv; f3 *= sv;
                    f4 *= sv; f5 *= sv; f6 *= sv; f7 *= sv;
                } else {
                    f0 *= *sp.add(j); f1 *= *sp.add(j+1);
                    f2 *= *sp.add(j+2); f3 *= *sp.add(j+3);
                    f4 *= *sp.add(j+4); f5 *= *sp.add(j+5);
                    f6 *= *sp.add(j+6); f7 *= *sp.add(j+7);
                }
            }
            if let Some(bp) = bias_data_ptr {
                f0 += *bp.add(j); f1 += *bp.add(j+1);
                f2 += *bp.add(j+2); f3 += *bp.add(j+3);
                f4 += *bp.add(j+4); f5 += *bp.add(j+5);
                f6 += *bp.add(j+6); f7 += *bp.add(j+7);
            }
            if apply_relu {
                f0 = f0.max(0.0); f1 = f1.max(0.0); f2 = f2.max(0.0); f3 = f3.max(0.0);
                f4 = f4.max(0.0); f5 = f5.max(0.0); f6 = f6.max(0.0); f7 = f7.max(0.0);
            }

            out_row[j] = f0; out_row[j+1] = f1; out_row[j+2] = f2; out_row[j+3] = f3;
            out_row[j+4] = f4; out_row[j+5] = f5; out_row[j+6] = f6; out_row[j+7] = f7;

            j += 8;
        }

        // 4-column remainder
        while j + 4 <= n {
            let bt0 = b_t_ptr.add(j * k_padded);
            let bt1 = b_t_ptr.add((j + 1) * k_padded);
            let bt2 = b_t_ptr.add((j + 2) * k_padded);
            let bt3 = b_t_ptr.add((j + 3) * k_padded);

            let mut iacc0 = _mm256_setzero_si256();
            let mut iacc1 = _mm256_setzero_si256();
            let mut iacc2 = _mm256_setzero_si256();
            let mut iacc3 = _mm256_setzero_si256();

            let mut kk = 0;
            while kk < k32 {
                let va = _mm256_loadu_si256(a_ptr.add(kk) as *const __m256i);
                iacc0 = _mm256_add_epi32(iacc0, _mm256_madd_epi16(_mm256_maddubs_epi16(va, _mm256_loadu_si256(bt0.add(kk) as *const __m256i)), ones_16));
                iacc1 = _mm256_add_epi32(iacc1, _mm256_madd_epi16(_mm256_maddubs_epi16(va, _mm256_loadu_si256(bt1.add(kk) as *const __m256i)), ones_16));
                iacc2 = _mm256_add_epi32(iacc2, _mm256_madd_epi16(_mm256_maddubs_epi16(va, _mm256_loadu_si256(bt2.add(kk) as *const __m256i)), ones_16));
                iacc3 = _mm256_add_epi32(iacc3, _mm256_madd_epi16(_mm256_maddubs_epi16(va, _mm256_loadu_si256(bt3.add(kk) as *const __m256i)), ones_16));
                kk += 32;
            }
            if kk < k_padded {
                let a_128 = _mm_loadu_si128(a_ptr.add(kk) as *const __m128i);
                let va = _mm256_cvtepu8_epi16(a_128);
                iacc0 = _mm256_add_epi32(iacc0, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt0.add(kk) as *const __m128i))));
                iacc1 = _mm256_add_epi32(iacc1, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt1.add(kk) as *const __m128i))));
                iacc2 = _mm256_add_epi32(iacc2, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt2.add(kk) as *const __m128i))));
                iacc3 = _mm256_add_epi32(iacc3, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt3.add(kk) as *const __m128i))));
            }

            let h01 = _mm256_hadd_epi32(iacc0, iacc1);
            let h23 = _mm256_hadd_epi32(iacc2, iacc3);
            let h0123 = _mm256_hadd_epi32(h01, h23);
            let hi = _mm256_extracti128_si256(h0123, 1);
            let lo = _mm256_castsi256_si128(h0123);
            let dots = _mm_add_epi32(lo, hi);

            let d0 = _mm_extract_epi32(dots, 0);
            let d1 = _mm_extract_epi32(dots, 1);
            let d2 = _mm_extract_epi32(dots, 2);
            let d3 = _mm_extract_epi32(dots, 3);

            let mut f0 = (d0 + row_corr - zp_a * (*col_sums_ptr.add(j) - k_zp_b)) as f32;
            let mut f1 = (d1 + row_corr - zp_a * (*col_sums_ptr.add(j+1) - k_zp_b)) as f32;
            let mut f2 = (d2 + row_corr - zp_a * (*col_sums_ptr.add(j+2) - k_zp_b)) as f32;
            let mut f3 = (d3 + row_corr - zp_a * (*col_sums_ptr.add(j+3) - k_zp_b)) as f32;

            if let Some(sp) = scale_data_ptr {
                if scale_len == 1 {
                    let sv = *sp;
                    f0 *= sv; f1 *= sv; f2 *= sv; f3 *= sv;
                } else {
                    f0 *= *sp.add(j); f1 *= *sp.add(j+1);
                    f2 *= *sp.add(j+2); f3 *= *sp.add(j+3);
                }
            }
            if let Some(bp) = bias_data_ptr {
                f0 += *bp.add(j); f1 += *bp.add(j+1);
                f2 += *bp.add(j+2); f3 += *bp.add(j+3);
            }
            if apply_relu {
                f0 = f0.max(0.0); f1 = f1.max(0.0);
                f2 = f2.max(0.0); f3 = f3.max(0.0);
            }

            out_row[j] = f0; out_row[j+1] = f1;
            out_row[j+2] = f2; out_row[j+3] = f3;

            j += 4;
        }

        // Scalar remainder
        while j < n {
            let bt = b_t_ptr.add(j * k_padded);
            let mut iacc = _mm256_setzero_si256();
            let mut kk = 0;
            while kk < k32 {
                let va = _mm256_loadu_si256(a_ptr.add(kk) as *const __m256i);
                iacc = _mm256_add_epi32(iacc, _mm256_madd_epi16(_mm256_maddubs_epi16(va, _mm256_loadu_si256(bt.add(kk) as *const __m256i)), ones_16));
                kk += 32;
            }
            if kk < k_padded {
                let a_128 = _mm_loadu_si128(a_ptr.add(kk) as *const __m128i);
                let va = _mm256_cvtepu8_epi16(a_128);
                iacc = _mm256_add_epi32(iacc, _mm256_madd_epi16(va, _mm256_cvtepi8_epi16(_mm_loadu_si128(bt.add(kk) as *const __m128i))));
            }
            let dot = hsum_epi32(iacc);
            let col_corr = zp_a * (*col_sums_ptr.add(j) - k_zp_b);
            let mut sum = (dot + row_corr - col_corr) as f32;
            if let Some(sp) = scale_data_ptr {
                if scale_len == 1 { sum *= *sp; }
                else { sum *= *sp.add(j); }
            }
            if let Some(bp) = bias_data_ptr { sum += *bp.add(j); }
            if apply_relu && sum < 0.0 { sum = 0.0; }
            out_row[j] = sum;
            j += 1;
        }
    }
}

/// Horizontal sum of 8 x i32 in __m256i -> single i32
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_epi32(v: __m256i) -> i32 {
    let hi128 = _mm256_extracti128_si256(v, 1);
    let lo128 = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(lo128, hi128);
    let hi64 = _mm_unpackhi_epi64(sum128, sum128);
    let sum64 = _mm_add_epi32(sum128, hi64);
    let hi32 = _mm_shuffle_epi32(sum64, 1);
    let sum32 = _mm_add_epi32(sum64, hi32);
    _mm_cvtsi128_si32(sum32)
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
