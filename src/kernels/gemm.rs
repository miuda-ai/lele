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

/// Public init wrapper for Accelerate (used by quantization module)
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub fn accelerate_init() {
    accelerate::init();
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
}
