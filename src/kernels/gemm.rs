use crate::kernels::utils;
use crate::tensor::TensorView;
use matrixmultiply::sgemm;
use std::borrow::Cow;
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
            sgemm(
                m,
                k,
                n,
                1.0,
                a.data.as_ptr().add(a_offset),
                k as isize,
                1,
                b.data.as_ptr().add(b_offset),
                n as isize,
                1,
                0.0,
                out_slice.as_mut_ptr().add(out_offset),
                n as isize,
                1,
            );
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
        for i in 0..len {
            unsafe {
                *out_slice.get_unchecked_mut(i) += *bias_data.get_unchecked(i % n);
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
        sgemm(
            m,
            k,
            n,
            alpha,
            a.data.as_ptr(),
            rsa,
            csa,
            b.data.as_ptr(),
            rsb,
            csb,
            1.0,
            out_buf.as_mut_ptr(),
            n as isize,
            1,
        );
    }
    TensorView {
        data: Cow::Borrowed(out_buf),
        shape: Cow::Owned(vec![m, n]),
    }
}
