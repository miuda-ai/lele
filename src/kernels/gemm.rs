use crate::kernels::utils;
#[cfg(target_arch = "wasm32")]
use crate::kernels::wasm_matmul::{Accum, MatMut, MatRef, Par, matmul as faer_matmul};
use crate::tensor::TensorView;
#[cfg(not(any(
    target_arch = "wasm32",
    all(target_arch = "aarch64", target_os = "macos")
)))]
use faer::linalg::matmul::matmul as faer_matmul;
#[cfg(not(any(
    target_arch = "wasm32",
    all(target_arch = "aarch64", target_os = "macos")
)))]
use faer::mat::{MatMut, MatRef};
#[cfg(not(any(
    target_arch = "wasm32",
    all(target_arch = "aarch64", target_os = "macos")
)))]
use faer::{Accum, Par};
use std::borrow::Cow;

// Apple Accelerate framework bindings for AMX-accelerated GEMM
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
mod accelerate {
    // CBLAS enums
    pub const CBLAS_ROW_MAJOR: i32 = 101;
    pub const CBLAS_NO_TRANS: i32 = 111;
    pub const CBLAS_TRANS: i32 = 112;

    unsafe extern "C" {
        pub fn cblas_sgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }

    unsafe extern "C" {
        fn setenv(name: *const i8, value: *const i8, overwrite: i32) -> i32;
    }

    /// Ensure Accelerate uses single thread (avoid thread-spawning overhead for small matrices)
    pub fn init() {
        use std::sync::Once;
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            unsafe {
                setenv(
                    c"VECLIB_MAXIMUM_THREADS".as_ptr(),
                    c"1".as_ptr(),
                    1, // force overwrite
                );
            }
        });
    }
}

/// Public cblas_sgemm wrapper (used by quantization module)
/// C = alpha * A * B + beta * C
/// A: [M, K] row-major, B: [K, N] row-major, C: [M, N] row-major
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub unsafe fn accelerate_sgemm(
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    b: *const f32,
    ldb: i32,
    beta: f32,
    c: *mut f32,
    ldc: i32,
) {
    unsafe {
        accelerate::cblas_sgemm(
            accelerate::CBLAS_ROW_MAJOR,
            accelerate::CBLAS_NO_TRANS,
            accelerate::CBLAS_NO_TRANS,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
        );
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub fn accelerate_init() {
    accelerate::init();
}

pub fn matmul<'a>(
    a: &TensorView<'_>,
    b: &TensorView<'_>,
    out_buf: &'a mut Vec<f32>,
) -> TensorView<'a> {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    accelerate::init();

    let a_dims = a.shape.len();
    let b_dims = b.shape.len();
    assert!(a_dims >= 2);
    assert!(b_dims >= 2);
    let m = a.shape[a_dims - 2];
    let k = a.shape[a_dims - 1];
    let k_b = b.shape[b_dims - 2];
    let n = b.shape[b_dims - 1];

    assert_eq!(k, k_b, "MatMul K dim mismatch: {} vs {}", k, k_b);
    let batch_a: usize = a.shape[..a_dims - 2].iter().product();
    let batch_b: usize = b.shape[..b_dims - 2].iter().product();
    let final_batch = batch_a.max(batch_b);

    assert!(
        batch_b == 1 || batch_b == batch_a,
        "MatMul broadcast not fully supported yet"
    );
    let mut out_shape = if batch_a >= batch_b {
        a.shape[..a_dims - 2].to_vec()
    } else {
        b.shape[..b_dims - 2].to_vec()
    };
    out_shape.push(m);
    out_shape.push(n);

    let output_len = final_batch * m * n;
    utils::ensure_capacity(out_buf, output_len);

    let out_slice: &mut [f32] =
        unsafe { std::slice::from_raw_parts_mut(out_buf.as_mut_ptr(), output_len) };
    let stride_a = m * k;
    let stride_b = k * n;
    let stride_out = m * n;

    for b_i in 0..final_batch {
        let a_offset = if batch_a == 1 { 0 } else { b_i * stride_a };
        let b_offset = if batch_b == 1 { 0 } else { b_i * stride_b };
        let out_offset = b_i * stride_out;

        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        unsafe {
            accelerate::cblas_sgemm(
                accelerate::CBLAS_ROW_MAJOR,
                accelerate::CBLAS_NO_TRANS,
                accelerate::CBLAS_NO_TRANS,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.data.as_ptr().add(a_offset),
                k as i32,
                b.data.as_ptr().add(b_offset),
                n as i32,
                0.0,
                out_slice.as_mut_ptr().add(out_offset),
                n as i32,
            );
        }

        #[cfg(not(any(
            target_arch = "wasm32",
            all(target_arch = "aarch64", target_os = "macos")
        )))]
        unsafe {
            let a_mat =
                MatRef::<f32>::from_raw_parts(a.data.as_ptr().add(a_offset), m, k, k as isize, 1);
            let b_mat =
                MatRef::<f32>::from_raw_parts(b.data.as_ptr().add(b_offset), k, n, n as isize, 1);
            let out_mat = MatMut::<f32>::from_raw_parts_mut(
                out_slice.as_mut_ptr().add(out_offset),
                m,
                n,
                n as isize,
                1,
            );
            faer_matmul(out_mat, Accum::Replace, a_mat, b_mat, 1.0, Par::Seq);
        }

        #[cfg(target_arch = "wasm32")]
        unsafe {
            use crate::kernels::wasm_matmul::{Accum, MatMut, MatRef, Par};

            let a_mat =
                MatRef::<f32>::from_raw_parts(a.data.as_ptr().add(a_offset), m, k, k as isize, 1);
            let b_mat =
                MatRef::<f32>::from_raw_parts(b.data.as_ptr().add(b_offset), k, n, n as isize, 1);
            let out_mat = MatMut::<f32>::from_raw_parts_mut(
                out_slice.as_mut_ptr().add(out_offset),
                m,
                n,
                n as isize,
                1,
            );

            faer_matmul(out_mat, Accum::Replace, a_mat, b_mat, 1.0, Par::Seq);
        }
    }
    TensorView {
        data: Cow::Borrowed(out_slice),
        shape: Cow::Owned(out_shape),
    }
}
pub fn matmul_fused_add<'a>(
    a: &TensorView<'_>,
    b: &TensorView<'_>,
    bias: &TensorView<'_>,
    out_buf: &'a mut Vec<f32>,
) -> TensorView<'a> {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    accelerate::init();

    let a_dims = a.shape.len();
    let b_dims = b.shape.len();
    let batch_a: usize = a.shape[..a_dims - 2].iter().product::<usize>().max(1);
    let batch_b: usize = b.shape[..b_dims - 2].iter().product::<usize>().max(1);
    let final_batch = batch_a.max(batch_b);

    let m = a.shape[a_dims - 2];
    let k = a.shape[a_dims - 1];
    let n = b.shape[b_dims - 1];
    let out_numel = final_batch * m * n;

    utils::ensure_capacity(out_buf, out_numel);
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_buf.as_mut_ptr(), out_numel) };

    let stride_a = m * k;
    let stride_b = k * n;
    let stride_out = m * n;

    // Check if bias matches output columns for fused operation
    if bias.data.len() == n {
        for b_i in 0..final_batch {
            let a_offset = if batch_a == 1 { 0 } else { b_i * stride_a };
            let b_offset = if batch_b == 1 { 0 } else { b_i * stride_b };
            let out_offset = b_i * stride_out;

            // Pre-fill output rows with bias
            for i in 0..m {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bias.data.as_ptr(),
                        out_slice.as_mut_ptr().add(out_offset + i * n),
                        n,
                    );
                }
            }

            // GEMM: C = 1.0 * A * B + 1.0 * C (where C is pre-filled with bias)
            #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
            unsafe {
                accelerate::cblas_sgemm(
                    accelerate::CBLAS_ROW_MAJOR,
                    accelerate::CBLAS_NO_TRANS,
                    accelerate::CBLAS_NO_TRANS,
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    a.data.as_ptr().add(a_offset),
                    k as i32,
                    b.data.as_ptr().add(b_offset),
                    n as i32,
                    1.0,
                    out_slice.as_mut_ptr().add(out_offset),
                    n as i32,
                );
            }

            #[cfg(not(any(
                target_arch = "wasm32",
                all(target_arch = "aarch64", target_os = "macos")
            )))]
            unsafe {
                let a_mat = MatRef::<f32>::from_raw_parts(
                    a.data.as_ptr().add(a_offset),
                    m,
                    k,
                    k as isize,
                    1,
                );
                let b_mat = MatRef::<f32>::from_raw_parts(
                    b.data.as_ptr().add(b_offset),
                    k,
                    n,
                    n as isize,
                    1,
                );
                let out_mat = MatMut::<f32>::from_raw_parts_mut(
                    out_slice.as_mut_ptr().add(out_offset),
                    m,
                    n,
                    n as isize,
                    1,
                );
                faer_matmul(out_mat, Accum::Add, a_mat, b_mat, 1.0, Par::Seq);
            }

            #[cfg(target_arch = "wasm32")]
            unsafe {
                use crate::kernels::wasm_matmul::{
                    Accum as WAccum, MatMut as WMatMut, MatRef as WMatRef, Par as WPar,
                };
                let a_mat = WMatRef::<f32>::from_raw_parts(
                    a.data.as_ptr().add(a_offset),
                    m,
                    k,
                    k as isize,
                    1,
                );
                let b_mat = WMatRef::<f32>::from_raw_parts(
                    b.data.as_ptr().add(b_offset),
                    k,
                    n,
                    n as isize,
                    1,
                );
                let out_mat = WMatMut::<f32>::from_raw_parts_mut(
                    out_slice.as_mut_ptr().add(out_offset),
                    m,
                    n,
                    n as isize,
                    1,
                );
                crate::kernels::wasm_matmul::matmul(
                    out_mat,
                    WAccum::Add,
                    a_mat,
                    b_mat,
                    1.0,
                    WPar::Seq,
                );
            }
        }
    } else {
        // Fallback: compute GEMM first, then add bias
        let view = matmul(a, b, out_buf);
        let len = view.data.len();
        let bias_data = &bias.data;

        if bias_data.len() == 1 {
            let b_val = bias_data[0];
            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                unsafe {
                    let b_vec = vdupq_n_f32(b_val);
                    let mut i = 0;
                    while i + 4 <= len {
                        let v = vld1q_f32(out_slice.as_ptr().add(i));
                        vst1q_f32(out_slice.as_mut_ptr().add(i), vaddq_f32(v, b_vec));
                        i += 4;
                    }
                    while i < len {
                        out_slice[i] += b_val;
                        i += 1;
                    }
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                for i in 0..len {
                    out_slice[i] += b_val;
                }
            }
        } else if bias_data.len() == len {
            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                unsafe {
                    let mut i = 0;
                    while i + 4 <= len {
                        let v = vld1q_f32(out_slice.as_ptr().add(i));
                        let b = vld1q_f32(bias_data.as_ptr().add(i));
                        vst1q_f32(out_slice.as_mut_ptr().add(i), vaddq_f32(v, b));
                        i += 4;
                    }
                    while i < len {
                        out_slice[i] += bias_data[i];
                        i += 1;
                    }
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                for i in 0..len {
                    out_slice[i] += bias_data[i];
                }
            }
        } else {
            let b_len = bias_data.len();
            for i in 0..len {
                out_slice[i] += bias_data[i % b_len];
            }
        }
    }

    let mut out_shape = if batch_a >= batch_b {
        a.shape[..a_dims - 2].to_vec()
    } else {
        b.shape[..b_dims - 2].to_vec()
    };
    if out_shape.is_empty() {
        // Both inputs are 2D, no batch dims
    }
    out_shape.push(m);
    out_shape.push(n);

    TensorView {
        data: Cow::Borrowed(out_slice),
        shape: Cow::Owned(out_shape),
    }
}

pub struct Int8Weight {
    pub data: Vec<u16>,
}

pub struct I8TiledWeight {
    pub data: Vec<i8>,
    pub scales: Vec<f32>,
}

pub fn quantize_f32_to_i8_tiled(data: &[f32], k: usize, n: usize) -> I8TiledWeight {
    let n4 = (n + 3) & !3;
    let k4 = (k + 3) & !3;
    let n_blocks = n4 / 4;
    let k_blocks = k4 / 4;
    let jb_groups = n_blocks / 4;
    let jb_remainder = n_blocks % 4;
    
    // New layout: for each 16-column group (4 j-blocks), all k-blocks are contiguous
    // tile(kb, jb) where jb = jbg*4 + jb_in: offset = (jbg * k_blocks + kb) * 64 + jb_in * 16
    // Remainder blocks at the end
    let total_size = k_blocks * n_blocks * 16;
    let mut tiled = vec![0i8; total_size];
    let mut scales = vec![0f32; n4];
    
    for j in 0..n {
        let mut max_abs = 1e-8f32;
        for ki in 0..k {
            if ki < data.len() / n {
                max_abs = max_abs.max(data[ki * n + j].abs());
            }
        }
        scales[j] = max_abs / 127.0;
    }
    for j in (n..n4).step_by(1) { scales[j] = 1.0; }
    
    fn write_tile(tiled: &mut [i8], data: &[f32], scales: &[f32], kb: usize, jb: usize, k: usize, n: usize, offset: usize) {
        for jc in 0..4 {
            for kr in 0..4 {
                let ki = kb * 4 + kr;
                let ji = jb * 4 + jc;
                let tile_idx = offset + jc * 4 + kr;
                if ki < k && ji < n {
                    let raw = data[ki * n + ji] / scales[ji];
                    tiled[tile_idx] = raw.round().clamp(-128.0, 127.0) as i8;
                } else {
                    tiled[tile_idx] = 0;
                }
            }
        }
    }
    
    // Full 16-column groups
    for jbg in 0..jb_groups {
        for kb in 0..k_blocks {
            for jb_in in 0..4 {
                let jb = jbg * 4 + jb_in;
                let offset = (jbg * k_blocks + kb) * 64 + jb_in * 16;
                write_tile(&mut tiled, data, &scales, kb, jb, k, n, offset);
            }
        }
    }
    
    // Remainder blocks (old layout)
    let rem_base = jb_groups * k_blocks * 64;
    for (ri, jb) in (jb_groups * 4..n_blocks).enumerate() {
        for kb in 0..k_blocks {
            let offset = rem_base + (kb * jb_remainder + ri) * 16;
            write_tile(&mut tiled, data, &scales, kb, jb, k, n, offset);
        }
    }
    
    I8TiledWeight { data: tiled, scales }
}

pub fn quantize_f32_to_i8(data: &[f32], k: usize, n: usize) -> Int8Weight {
    let mut out = Vec::with_capacity(k * n);
    for &v in data {
        out.push(f32_to_f16_bits(v));
    }
    Int8Weight { data: out }
}

#[inline(always)]
pub fn f32_to_f16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = (bits & 0x7FFFFF) as i32;
    if exp == 0xFF {
        return sign | 0x7C00 | ((mant >> 13) as u16 & 0x3FF);
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1F {
        return sign | 0x7C00;
    }
    if new_exp <= 0 {
        if new_exp < -10 { return sign; }
        let mant2 = mant | 0x800000;
        let shift = 14 - new_exp;
        let rounding = 1 << (shift - 1);
        return sign | (((mant2 + rounding) >> shift) as u16 & 0x7FF);
    }
    sign | ((new_exp as u16) << 10) | (((mant + 0x1000) >> 13) as u16)
}

#[inline(always)]
pub fn f16_to_f32_scalar(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h as u32) >> 10) & 0x1F;
    let mant = (h as u32) & 0x3FF;
    if exp == 0 {
        if mant == 0 { return f32::from_bits(sign); }
        let mut m = mant;
        let mut e = 1i32;
        while (m & 0x400) == 0 { m <<= 1; e -= 1; }
        m &= 0x3FF;
        return f32::from_bits(sign | ((127 + e - 15 + 1 - 1) as u32) << 23 | (m << 13));
    }
    if exp == 0x1F { return f32::from_bits(sign | 0x7F800000 | (mant << 13)); }
    f32::from_bits(sign | ((exp + 112) as u32) << 23 | (mant << 13))
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn gemv_i8_f32_neon(
    x: *const f32,
    w: *const u16,
    _scales: *const f32,
    bias: Option<*const f32>,
    out: *mut f32,
    k: usize,
    n: usize,
) {
    use core::arch::aarch64::*;

    if let Some(bp) = bias {
        let mut j = 0;
        while j + 4 <= n { vst1q_f32(out.add(j), vld1q_f32(bp.add(j))); j += 4; }
        while j < n { *out.add(j) = *bp.add(j); j += 1; }
    } else {
        let mut j = 0;
        while j + 4 <= n { vst1q_f32(out.add(j), vdupq_n_f32(0.0)); j += 4; }
        while j < n { *out.add(j) = 0.0; j += 1; }
    }

    for kk in 0..k {
        let x_val = *x.add(kk);
        let wptr = w.add(kk * n);
        let mut j = 0;
        while j + 16 <= n {
            let w0 = vld1q_u16(wptr.add(j));
            let w1 = vld1q_u16(wptr.add(j + 8));
            let f0 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(w0)));
            let f1 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(w0)));
            let f2 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(w1)));
            let f3 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(w1)));
            let c0 = vld1q_f32(out.add(j));
            let c1 = vld1q_f32(out.add(j + 4));
            let c2 = vld1q_f32(out.add(j + 8));
            let c3 = vld1q_f32(out.add(j + 12));
            vst1q_f32(out.add(j), vfmaq_n_f32(c0, f0, x_val));
            vst1q_f32(out.add(j + 4), vfmaq_n_f32(c1, f1, x_val));
            vst1q_f32(out.add(j + 8), vfmaq_n_f32(c2, f2, x_val));
            vst1q_f32(out.add(j + 12), vfmaq_n_f32(c3, f3, x_val));
            j += 16;
        }
        while j + 4 <= n {
            let w0 = vld1_u16(wptr.add(j));
            let f0 = vcvt_f32_f16(vreinterpret_f16_u16(w0));
            let c0 = vld1q_f32(out.add(j));
            vst1q_f32(out.add(j), vfmaq_n_f32(c0, f0, x_val));
            j += 4;
        }
        while j < n {
            *out.add(j) += x_val * f16_to_f32_scalar(*wptr.add(j));
            j += 1;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn gemv_i8_f32_neon(
    x: *const f32,
    w: *const u16,
    _scales: *const f32,
    bias: Option<*const f32>,
    out: *mut f32,
    k: usize,
    n: usize,
) {
    if let Some(bp) = bias {
        for j in 0..n { *out.add(j) = *bp.add(j); }
    } else {
        for j in 0..n { *out.add(j) = 0.0; }
    }
    for kk in 0..k {
        let xv = *x.add(kk);
        let wptr = w.add(kk * n);
        for j in 0..n {
            *out.add(j) += xv * f16_to_f32_scalar(*wptr.add(j));
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn gemv_i8mm_sdot(
    x: *const f32,
    tiled_w: *const i8,
    scales: *const f32,
    bias: Option<*const f32>,
    out: *mut f32,
    k: usize,
    n: usize,
) {
    use core::arch::aarch64::*;

    let n4 = (n + 3) & !3;
    let k4 = (k + 3) & !3;
    let n_blocks = n4 / 4;
    let k_blocks = k4 / 4;

    // Compute x_scale via NEON
    let x_scale = {
        let mut max_abs = vdupq_n_f32(1e-8);
        let mut kk = 0;
        while kk + 4 <= k {
            let v = vld1q_f32(x.add(kk));
            max_abs = vmaxq_f32(max_abs, vabsq_f32(v));
            kk += 4;
        }
        let mut m = vmaxvq_f32(max_abs);
        while kk < k {
            m = m.max((*x.add(kk)).abs());
            kk += 1;
        }
        m / 127.0
    };

    // Pre-quantize x using scalar (matches original exactly) into stack buffer
    const MAX_K_BLOCKS: usize = 1024;
    let mut x_bcast_buf: [std::mem::MaybeUninit<[i8; 16]>; MAX_K_BLOCKS] =
        unsafe { std::mem::MaybeUninit::uninit().assume_init() };
    for kb in 0..k_blocks {
        let k0 = kb * 4;
        let vals: [i8; 4] = if k0 + 3 < k {
            [
                (*x.add(k0) / x_scale).round().clamp(-128.0, 127.0) as i8,
                (*x.add(k0 + 1) / x_scale).round().clamp(-128.0, 127.0) as i8,
                (*x.add(k0 + 2) / x_scale).round().clamp(-128.0, 127.0) as i8,
                (*x.add(k0 + 3) / x_scale).round().clamp(-128.0, 127.0) as i8,
            ]
        } else {
            let mut tmp = [0i8; 4];
            for i in 0..4 {
                if k0 + i < k {
                    tmp[i] = (*x.add(k0 + i) / x_scale).round().clamp(-128.0, 127.0) as i8;
                }
            }
            tmp
        };
        unsafe {
            std::ptr::write(x_bcast_buf[kb].as_mut_ptr(), [
                vals[0], vals[1], vals[2], vals[3],
                vals[0], vals[1], vals[2], vals[3],
                vals[0], vals[1], vals[2], vals[3],
                vals[0], vals[1], vals[2], vals[3],
            ]);
        }
    }

    let x_scale_vec = vdupq_n_f32(x_scale);
    let buf_base = x_bcast_buf.as_ptr() as *const i8;

    // Process 16 columns at a time with 4 independent accumulator chains
    let jb_groups = n_blocks / 4;
    let jb_remainder = n_blocks % 4;
    let rem_base = jb_groups * k_blocks * 64;
    
    for jbg in 0..jb_groups {
        let j_start = jbg * 16;
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        let mut acc2 = vdupq_n_s32(0);
        let mut acc3 = vdupq_n_s32(0);

        for kb in 0..k_blocks {
            let x_bcast = vld1q_s8(buf_base.add(kb * 16));

            let w_base = (jbg * k_blocks + kb) * 64;
            let w0 = vld1q_s8(tiled_w.add(w_base));
            let w1 = vld1q_s8(tiled_w.add(w_base + 16));
            let w2 = vld1q_s8(tiled_w.add(w_base + 32));
            let w3 = vld1q_s8(tiled_w.add(w_base + 48));

            // Prefetch next k-block's weights
            if kb + 2 < k_blocks {
                let pf_base = (jbg * k_blocks + kb + 2) * 64;
                core::arch::asm!(
                    "prfm pldl1keep, [{0}]",
                    in(reg) tiled_w.add(pf_base),
                    options(nostack, preserves_flags, readonly),
                );
            }

            core::arch::asm!(
                ".inst 0x4e809400 | (2 << 16) | (1 << 5) | 0",
                inout("v0") acc0,
                in("v1") w0,
                in("v2") x_bcast,
                options(pure, nomem, nostack, preserves_flags),
            );
            core::arch::asm!(
                ".inst 0x4e809400 | (5 << 16) | (4 << 5) | 3",
                inout("v3") acc1,
                in("v4") w1,
                in("v5") x_bcast,
                options(pure, nomem, nostack, preserves_flags),
            );
            core::arch::asm!(
                ".inst 0x4e809400 | (8 << 16) | (7 << 5) | 6",
                inout("v6") acc2,
                in("v7") w2,
                in("v8") x_bcast,
                options(pure, nomem, nostack, preserves_flags),
            );
            core::arch::asm!(
                ".inst 0x4e809400 | (11 << 16) | (10 << 5) | 9",
                inout("v9") acc3,
                in("v10") w3,
                in("v11") x_bcast,
                options(pure, nomem, nostack, preserves_flags),
            );
        }

        let sc0 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(acc0), vld1q_f32(scales.add(j_start))), x_scale_vec);
        let sc1 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(acc1), vld1q_f32(scales.add(j_start + 4))), x_scale_vec);
        let sc2 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(acc2), vld1q_f32(scales.add(j_start + 8))), x_scale_vec);
        let sc3 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(acc3), vld1q_f32(scales.add(j_start + 12))), x_scale_vec);

        if let Some(bp) = bias {
            vst1q_f32(out.add(j_start), vaddq_f32(sc0, vld1q_f32(bp.add(j_start))));
            vst1q_f32(out.add(j_start + 4), vaddq_f32(sc1, vld1q_f32(bp.add(j_start + 4))));
            vst1q_f32(out.add(j_start + 8), vaddq_f32(sc2, vld1q_f32(bp.add(j_start + 8))));
            vst1q_f32(out.add(j_start + 12), vaddq_f32(sc3, vld1q_f32(bp.add(j_start + 12))));
        } else {
            vst1q_f32(out.add(j_start), sc0);
            vst1q_f32(out.add(j_start + 4), sc1);
            vst1q_f32(out.add(j_start + 8), sc2);
            vst1q_f32(out.add(j_start + 12), sc3);
        }
    }

    // Handle remaining 4-column blocks (old layout)
    for ri in 0..jb_remainder {
        let jb = jb_groups * 4 + ri;
        let j_start = jb * 4;
        let mut acc = vdupq_n_s32(0);

        for kb in 0..k_blocks {
            let x_bcast = vld1q_s8(buf_base.add(kb * 16));
            let w = vld1q_s8(tiled_w.add(rem_base + (kb * jb_remainder + ri) * 16));
            core::arch::asm!(
                ".inst 0x4e809400 | (2 << 16) | (1 << 5) | 0",
                inout("v0") acc,
                in("v1") w,
                in("v2") x_bcast,
                options(pure, nomem, nostack, preserves_flags),
            );
        }

        let result = vmulq_f32(vmulq_f32(vcvtq_f32_s32(acc), vld1q_f32(scales.add(j_start))), x_scale_vec);
        if let Some(bp) = bias {
            vst1q_f32(out.add(j_start), vaddq_f32(result, vld1q_f32(bp.add(j_start))));
        } else {
            vst1q_f32(out.add(j_start), result);
        }
    }

    let mut j = n4;
    while j < n {
        *out.add(j) = if let Some(bp) = bias { *bp.add(j) } else { 0.0 };
        j += 1;
    }
}

pub fn gemm<'a>(
    a: &TensorView<'_>,
    b: &TensorView<'_>,
    c: Option<&TensorView<'_>>,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
    out_buf: &'a mut Vec<f32>,
) -> TensorView<'a> {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    accelerate::init();

    let m = if trans_a {
        a.shape[a.shape.len() - 1]
    } else {
        a.shape[a.shape.len() - 2]
    };
    let k = if trans_a {
        a.shape[a.shape.len() - 2]
    } else {
        a.shape[a.shape.len() - 1]
    };
    let n = if trans_b {
        b.shape[b.shape.len() - 2]
    } else {
        b.shape[b.shape.len() - 1]
    };
    let k2 = if trans_b {
        b.shape[b.shape.len() - 1]
    } else {
        b.shape[b.shape.len() - 2]
    };
    assert_eq!(k, k2, "Gemm K dim mismatch");

    #[cfg(target_arch = "aarch64")]
    {
        if !trans_a && !trans_b {
            return gemm_neon_path(a, b, c, alpha, beta, m, k, n, out_buf);
        } else {
            return gemm_transposed_path(a, b, c, alpha, beta, m, k, n, trans_a, trans_b, out_buf);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let output_len = m * n;
        utils::ensure_capacity(out_buf, output_len);
        unsafe {
            out_buf.set_len(output_len);
        }
        if let Some(cv) = c {
            if beta == 0.0 {
                out_buf.fill(0.0);
            } else {
                if cv.data.len() == output_len {
                    for i in 0..output_len {
                        out_buf[i] = cv.data[i] * beta;
                    }
                } else if cv.data.len() == n {
                    for i in 0..m {
                        for j in 0..n {
                            out_buf[i * n + j] = cv.data[j] * beta;
                        }
                    }
                } else if cv.data.len() == m {
                    for i in 0..m {
                        for j in 0..n {
                            out_buf[i * n + j] = cv.data[i] * beta;
                        }
                    }
                } else if cv.data.len() == 1 {
                    let v = cv.data[0] * beta;
                    out_buf.fill(v);
                } else {
                    for i in 0..output_len {
                        out_buf[i] = cv.data[i % cv.data.len()] * beta;
                    }
                }
            }
        } else {
            out_buf.fill(0.0);
        }

        let rsa = if trans_a { 1 } else { k as isize };
        let csa = if trans_a { m as isize } else { 1 };
        let rsb = if trans_b { 1 } else { n as isize };
        let csb = if trans_b { k as isize } else { 1 };
        unsafe {
            let a_mat = MatRef::<f32>::from_raw_parts(a.data.as_ptr(), m, k, rsa, csa);
            let b_mat = MatRef::<f32>::from_raw_parts(b.data.as_ptr(), k, n, rsb, csb);
            let out_mat =
                MatMut::<f32>::from_raw_parts_mut(out_buf.as_mut_ptr(), m, n, n as isize, 1);

            faer_matmul(out_mat, Accum::Add, a_mat, b_mat, alpha, Par::Seq);
        }

        TensorView {
            data: Cow::Borrowed(out_buf),
            shape: Cow::Owned(vec![m, n]),
        }
    }
}

/// High-performance GEMM path for non-transposed matrices via Apple Accelerate AMX
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn gemm_neon_path<'a>(
    a: &TensorView<'_>,
    b: &TensorView<'_>,
    c: Option<&TensorView<'_>>,
    alpha: f32,
    beta: f32,
    m: usize,
    k: usize,
    n: usize,
    out_buf: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let output_len = m * n;
    utils::ensure_capacity(out_buf, output_len);
    unsafe {
        out_buf.set_len(output_len);
    }

    // Initialize output with C * beta if needed
    let actual_beta = if let Some(cv) = c {
        if beta == 0.0 {
            out_buf.fill(0.0);
            0.0
        } else {
            if cv.data.len() == output_len {
                out_buf.copy_from_slice(&cv.data[..output_len]);
            } else if cv.data.len() == n {
                for i in 0..m {
                    out_buf[i * n..i * n + n].copy_from_slice(&cv.data[..n]);
                }
            } else if cv.data.len() == 1 {
                out_buf.fill(cv.data[0]);
            } else {
                for i in 0..output_len {
                    out_buf[i] = cv.data[i % cv.data.len()];
                }
            }
            beta
        }
    } else {
        out_buf.fill(0.0);
        0.0
    };

    unsafe {
        accelerate::cblas_sgemm(
            accelerate::CBLAS_ROW_MAJOR,
            accelerate::CBLAS_NO_TRANS,
            accelerate::CBLAS_NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a.data.as_ptr(),
            k as i32,
            b.data.as_ptr(),
            n as i32,
            actual_beta,
            out_buf.as_mut_ptr(),
            n as i32,
        );
    }

    TensorView {
        data: Cow::Borrowed(out_buf),
        shape: Cow::Owned(vec![m, n]),
    }
}

/// GEMM path for transposed matrices via Apple Accelerate AMX
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn gemm_transposed_path<'a>(
    a: &TensorView<'_>,
    b: &TensorView<'_>,
    c: Option<&TensorView<'_>>,
    alpha: f32,
    beta: f32,
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
    out_buf: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let output_len = m * n;
    utils::ensure_capacity(out_buf, output_len);
    unsafe {
        out_buf.set_len(output_len);
    }

    let actual_beta = if let Some(cv) = c {
        if beta == 0.0 {
            out_buf.fill(0.0);
            0.0
        } else {
            if cv.data.len() == output_len {
                out_buf.copy_from_slice(&cv.data[..output_len]);
            } else if cv.data.len() == n {
                for i in 0..m {
                    out_buf[i * n..i * n + n].copy_from_slice(&cv.data[..n]);
                }
            } else if cv.data.len() == 1 {
                out_buf.fill(cv.data[0]);
            } else {
                for i in 0..output_len {
                    out_buf[i] = cv.data[i % cv.data.len()];
                }
            }
            beta
        }
    } else {
        out_buf.fill(0.0);
        0.0
    };

    let cblas_trans_a = if trans_a {
        accelerate::CBLAS_TRANS
    } else {
        accelerate::CBLAS_NO_TRANS
    };
    let cblas_trans_b = if trans_b {
        accelerate::CBLAS_TRANS
    } else {
        accelerate::CBLAS_NO_TRANS
    };
    // lda: leading dimension of A in row-major order
    let lda = if trans_a { m } else { k };
    let ldb = if trans_b { k } else { n };

    unsafe {
        accelerate::cblas_sgemm(
            accelerate::CBLAS_ROW_MAJOR,
            cblas_trans_a,
            cblas_trans_b,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a.data.as_ptr(),
            lda as i32,
            b.data.as_ptr(),
            ldb as i32,
            actual_beta,
            out_buf.as_mut_ptr(),
            n as i32,
        );
    }

    TensorView {
        data: Cow::Borrowed(out_buf),
        shape: Cow::Owned(vec![m, n]),
    }
}

/// High-performance GEMM path for non-transposed matrices via faer (non-macOS)
#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
fn gemm_neon_path<'a>(
    a: &TensorView<'_>,
    b: &TensorView<'_>,
    c: Option<&TensorView<'_>>,
    alpha: f32,
    beta: f32,
    m: usize,
    k: usize,
    n: usize,
    out_buf: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let output_len = m * n;
    utils::ensure_capacity(out_buf, output_len);
    unsafe {
        out_buf.set_len(output_len);
    }

    // Initialize output with C * beta
    let use_accum = if let Some(cv) = c {
        if beta == 0.0 {
            false
        } else {
            // Pre-fill output with C * beta
            unsafe {
                let beta_vec = core::arch::aarch64::vdupq_n_f32(beta);
                if cv.data.len() == output_len {
                    let mut i = 0;
                    while i + 4 <= output_len {
                        let v = core::arch::aarch64::vld1q_f32(cv.data.as_ptr().add(i));
                        core::arch::aarch64::vst1q_f32(
                            out_buf.as_mut_ptr().add(i),
                            core::arch::aarch64::vmulq_f32(v, beta_vec),
                        );
                        i += 4;
                    }
                    while i < output_len {
                        out_buf[i] = cv.data[i] * beta;
                        i += 1;
                    }
                } else if cv.data.len() == n {
                    for i in 0..m {
                        let mut j = 0;
                        while j + 4 <= n {
                            let v = core::arch::aarch64::vld1q_f32(cv.data.as_ptr().add(j));
                            core::arch::aarch64::vst1q_f32(
                                out_buf.as_mut_ptr().add(i * n + j),
                                core::arch::aarch64::vmulq_f32(v, beta_vec),
                            );
                            j += 4;
                        }
                        while j < n {
                            out_buf[i * n + j] = cv.data[j] * beta;
                            j += 1;
                        }
                    }
                } else if cv.data.len() == 1 {
                    let v = cv.data[0] * beta;
                    out_buf.fill(v);
                } else {
                    for i in 0..output_len {
                        out_buf[i] = cv.data[i % cv.data.len()] * beta;
                    }
                }
            }
            true
        }
    } else {
        false
    };

    // Use faer for GEMM: C = alpha * A * B [+ C]
    unsafe {
        let a_mat = MatRef::<f32>::from_raw_parts(a.data.as_ptr(), m, k, k as isize, 1);
        let b_mat = MatRef::<f32>::from_raw_parts(b.data.as_ptr(), k, n, n as isize, 1);
        let out_mat = MatMut::<f32>::from_raw_parts_mut(out_buf.as_mut_ptr(), m, n, n as isize, 1);
        let accum = if use_accum {
            Accum::Add
        } else {
            Accum::Replace
        };
        faer_matmul(out_mat, accum, a_mat, b_mat, alpha, Par::Seq);
    }

    TensorView {
        data: Cow::Borrowed(out_buf),
        shape: Cow::Owned(vec![m, n]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_wrapper() {
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = TensorView::from_slice(&a_data, vec![2, 3]);
        let b = TensorView::from_slice(&b_data, vec![3, 2]);
        let mut out = Vec::new();
        let result = matmul(&a, &b, &mut out);
        eprintln!("matmul result: {:?}", result.data.as_ref());
        let expected = vec![22.0f32, 28.0, 49.0, 64.0];
        for (i, (r, e)) in result.data.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 0.01, "Mismatch at {i}: {r} vs {e}");
        }
    }

    #[test]
    fn test_matmul_fused_add_wrapper() {
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bias_data = vec![100.0f32, 200.0];
        let a = TensorView::from_slice(&a_data, vec![2, 3]);
        let b = TensorView::from_slice(&b_data, vec![3, 2]);
        let bias = TensorView::from_slice(&bias_data, vec![2]);
        let mut out = Vec::new();
        let result = matmul_fused_add(&a, &b, &bias, &mut out);
        eprintln!("matmul_fused_add result: {:?}", result.data.as_ref());
        let expected = vec![122.0f32, 228.0, 149.0, 264.0];
        for (i, (r, e)) in result.data.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 0.01, "Mismatch at {i}: {r} vs {e}");
        }
    }

    #[test]
    fn test_i8mm_vs_fp16_gemv() {
        let k = 768usize;
        let n = 768usize;
        let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.7) % 3.0 - 1.5).collect();
        let w: Vec<f32> = (0..k * n).map(|i| (i as f32 * 1.3) % 4.0 - 2.0).collect();
        let bias: Vec<f32> = (0..n).map(|j| j as f32 * 0.01).collect();

        let mut ref_out = vec![0f32; n];
        for j in 0..n {
            let mut acc = 0.0f64;
            for ki in 0..k {
                acc += (x[ki] * w[ki * n + j]) as f64;
            }
            ref_out[j] = acc as f32 + bias[j];
        }

        let fp16_w = quantize_f32_to_i8(&w, k, n);
        let mut fp16_out = vec![0f32; n];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            gemv_i8_f32_neon(
                x.as_ptr(),
                fp16_w.data.as_ptr(),
                std::ptr::null(),
                Some(bias.as_ptr()),
                fp16_out.as_mut_ptr(),
                k,
                n,
            );
        }

        let tiled_w = quantize_f32_to_i8_tiled(&w, k, n);
        let mut i8mm_out = vec![0f32; n];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            gemv_i8mm_sdot(
                x.as_ptr(),
                tiled_w.data.as_ptr(),
                tiled_w.scales.as_ptr(),
                Some(bias.as_ptr()),
                i8mm_out.as_mut_ptr(),
                k,
                n,
            );
        }

        let max_ref = ref_out.iter().fold(0f32, |a, &v| a.max(v.abs()));
        #[cfg(target_arch = "aarch64")]
        {
            let mut max_err_fp16 = 0f32;
            let mut max_err_i8mm = 0f32;
            for j in 0..n {
                max_err_fp16 = max_err_fp16.max((fp16_out[j] - ref_out[j]).abs());
                max_err_i8mm = max_err_i8mm.max((i8mm_out[j] - ref_out[j]).abs());
            }
            eprintln!("ref max_abs: {:.2}", max_ref);
            eprintln!("fp16 max err: {:.4} ({:.2}%)", max_err_fp16, max_err_fp16 / max_ref * 100.0);
            eprintln!("i8mm max err: {:.4} ({:.2}%)", max_err_i8mm, max_err_i8mm / max_ref * 100.0);
        }
    }

    #[test]
    fn bench_gemv_model_sizes() {
        let cases: &[(usize, usize, &str)] = &[
            (768, 2304, "c_attn"),
            (768, 768, "c_proj"),
            (768, 3072, "mlp_fc_in"),
            (3072, 768, "mlp_fc_out"),
        ];
        let iters = 5000;

        for &(k, n, name) in cases {
            let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.7) % 3.0 - 1.5).collect();
            let w: Vec<f32> = (0..k * n).map(|i| (i as f32 * 1.3) % 4.0 - 2.0).collect();
            let tiled_w = quantize_f32_to_i8_tiled(&w, k, n);
            let mut out = vec![0f32; n];

            #[cfg(target_arch = "aarch64")]
            {
                for _ in 0..100 {
                    unsafe { gemv_i8mm_sdot(x.as_ptr(), tiled_w.data.as_ptr(), tiled_w.scales.as_ptr(), None, out.as_mut_ptr(), k, n); }
                }
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    unsafe { gemv_i8mm_sdot(x.as_ptr(), tiled_w.data.as_ptr(), tiled_w.scales.as_ptr(), None, out.as_mut_ptr(), k, n); }
                }
                let dt = t0.elapsed();
                let flops = 2 * k * n * iters;
                let gflops = flops as f64 / dt.as_secs_f64() / 1e9;
                let per_call_us = dt.as_secs_f64() * 1e6 / iters as f64;
                let wsize_mb = tiled_w.data.len() as f64 / 1e6;
                eprintln!("  {:>12} k={:>4} n={:>4} w={:.2}MB: {:.1}μs/call, {:.0} GFLOPS", name, k, n, wsize_mb, per_call_us, gflops);
            }
        }
    }
}
