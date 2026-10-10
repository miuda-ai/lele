use crate::kernels::utils;
use crate::kernels::matmul::{Accum, MatMut, MatRef, Par, matmul as strided_matmul};
use crate::tensor::TensorView;
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
            strided_matmul(out_mat, Accum::Replace, a_mat, b_mat, 1.0, Par::Seq);
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
                strided_matmul(out_mat, Accum::Add, a_mat, b_mat, 1.0, Par::Seq);
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
    // With no C to add, the product overwrites the buffer, so it is not filled first.
    let accum = match c {
        Some(_) if beta == 0.0 => Accum::Replace,
        None => Accum::Replace,
        Some(cv) => {
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
            Accum::Add
        }
    };

    let rsa = if trans_a { 1 } else { k as isize };
    let csa = if trans_a { m as isize } else { 1 };
    let rsb = if trans_b { 1 } else { n as isize };
    let csb = if trans_b { k as isize } else { 1 };
    unsafe {
        let a_mat = MatRef::<f32>::from_raw_parts(a.data.as_ptr(), m, k, rsa, csa);
        let b_mat = MatRef::<f32>::from_raw_parts(b.data.as_ptr(), k, n, rsb, csb);
        let out_mat = MatMut::<f32>::from_raw_parts_mut(out_buf.as_mut_ptr(), m, n, n as isize, 1);

        strided_matmul(out_mat, accum, a_mat, b_mat, alpha, Par::Seq);
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
