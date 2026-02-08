use crate::kernels::utils;
use crate::tensor::TensorView;
use faer::linalg::matmul::matmul as faer_matmul;
use faer::mat::{MatMut, MatRef};
use faer::{Accum, Par};
use std::borrow::Cow;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn matmul<'a>(
    a: &TensorView<'_>,
    b: &TensorView<'_>,
    out_buf: &'a mut Vec<f32>,
) -> TensorView<'a> {
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

            faer_matmul(out_mat, Accum::Replace, a_mat, b_mat, 1.0, utils::get_parallelism(m, n, k));
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
    let view = matmul(a, b, out_buf);
    let len = view.data.len();
    let n = b.shape[b.shape.len() - 1];
    let bias_data = &bias.data;
    let out_slice = unsafe { std::slice::from_raw_parts_mut(view.data.as_ptr() as *mut f32, len) };
    if bias_data.len() == n {
        // Optimized broadcasting loop (row-wise) to avoid expensive modulo
        for chunk in out_slice.chunks_exact_mut(n) {
            for (x, &b) in chunk.iter_mut().zip(bias_data.iter()) {
                *x += b;
            }
        }
    } else if bias_data.len() == 1 {
        let b_val = bias_data[0];
        for i in 0..len {
            out_slice[i] += b_val;
        }
    } else if bias_data.len() == len {
        for i in 0..len {
            out_slice[i] += bias_data[i];
        }
    } else {
        let b_len = bias_data.len();
        for i in 0..len {
            out_slice[i] += bias_data[i % b_len];
        }
    }
    view
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
        let out_mat = MatMut::<f32>::from_raw_parts_mut(out_buf.as_mut_ptr(), m, n, n as isize, 1);

        // if use beta (=1.0 usually), we need accumulation to be Some(1.0).
        // However, here we already filled out_buf with C*beta.
        // So we want out = alpha * A * B + 1.0 * out_buf.
        // faer: out <- alpha * A * B + beta * out.
        // We set faer's beta to 1.0 because out_buf already contains the previous C term scaled.

        faer_matmul(
            out_mat,
            Accum::Add, // accumulate into existing content
            a_mat,
            b_mat,
            alpha,
            utils::get_parallelism(m, n, k),
        );
    }

    TensorView {
        data: Cow::Borrowed(out_buf),
        shape: Cow::Owned(vec![m, n]),
    }
}
