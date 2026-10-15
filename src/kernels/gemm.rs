use crate::kernels::bias_act::bias_act_inplace;
use crate::kernels::conv2d::Activation;
use crate::kernels::matmul::{
    Accum, Codes, MatMut, MatRef, Par, matmul as strided_matmul, matmul_codes, matmul_f16,
    matmul_f16_a, matmul_f16_packed, matmul_f16_packed_a,
};
use crate::kernels::utils;
use crate::tensor::TensorView;
use std::borrow::Cow;

/// Element type of a weight operand of the matrix products ([`matmul`],
/// [`matmul_fused_add`], pointwise `conv1d`): f32, or f16, which the kernels widen as they
/// read it, so that the product is that of the widened matrix, bit for bit.
pub trait WeightElem: Clone + std::fmt::Debug + 'static {
    /// `out (+)= a * b`, with `b` the row-major `k x n` matrix at `b`.
    ///
    /// # Safety
    /// `b` must be valid for reads of `k * n` elements.
    #[doc(hidden)]
    unsafe fn matmul_into(
        out: MatMut<f32>,
        accum: Accum,
        a: MatRef<f32>,
        b: *const Self,
        k: usize,
        n: usize,
    );

    /// `out (+)= a * b`, with `a` the row-major `m x k` matrix at `a`.
    ///
    /// # Safety
    /// `a` must be valid for reads of `m * k` elements.
    #[doc(hidden)]
    unsafe fn matmul_left_into(
        out: MatMut<f32>,
        accum: Accum,
        a: *const Self,
        m: usize,
        k: usize,
        b: MatRef<f32>,
    );
}

impl WeightElem for f32 {
    unsafe fn matmul_into(
        out: MatMut<f32>,
        accum: Accum,
        a: MatRef<f32>,
        b: *const f32,
        k: usize,
        n: usize,
    ) {
        // SAFETY: the caller promised `k * n` readable elements.
        let b = unsafe { MatRef::<f32>::from_raw_parts(b, k, n, n as isize, 1) };
        strided_matmul(out, accum, a, b, 1.0, Par::Seq);
    }

    unsafe fn matmul_left_into(
        out: MatMut<f32>,
        accum: Accum,
        a: *const f32,
        m: usize,
        k: usize,
        b: MatRef<f32>,
    ) {
        // SAFETY: the caller promised `m * k` readable elements.
        let a = unsafe { MatRef::<f32>::from_raw_parts(a, m, k, k as isize, 1) };
        strided_matmul(out, accum, a, b, 1.0, Par::Seq);
    }
}

impl WeightElem for half::f16 {
    unsafe fn matmul_into(
        out: MatMut<f32>,
        accum: Accum,
        a: MatRef<f32>,
        b: *const half::f16,
        k: usize,
        n: usize,
    ) {
        // SAFETY: the caller promised `k * n` readable elements.
        let b = unsafe { std::slice::from_raw_parts(b, k * n) };
        matmul_f16(out, accum, a, b, n, 1.0);
    }

    unsafe fn matmul_left_into(
        out: MatMut<f32>,
        accum: Accum,
        a: *const half::f16,
        m: usize,
        k: usize,
        b: MatRef<f32>,
    ) {
        // SAFETY: the caller promised `m * k` readable elements.
        let a = unsafe { std::slice::from_raw_parts(a, m * k) };
        matmul_f16_a(out, accum, a, k, b, 1.0);
    }
}

/// An element of an f16 weight stored as [`pack_f16`](crate::kernels::matmul::pack_f16)
/// lays it out. A `TensorView<PackedF16>` of shape `[k, n]` is a matrix packed that way,
/// taken as the right operand of [`matmul`] and [`matmul_fused_add`]. As the weights
/// `[m, k, 1]` of `conv1d_pointwise`, the left operand, it is the transpose `[k, m]`
/// packed.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(transparent)]
pub struct PackedF16(pub half::f16);

impl WeightElem for PackedF16 {
    unsafe fn matmul_into(
        out: MatMut<f32>,
        accum: Accum,
        a: MatRef<f32>,
        b: *const PackedF16,
        k: usize,
        n: usize,
    ) {
        // SAFETY: the caller promised `k * n` readable elements, and `PackedF16` is a
        // transparent `f16`.
        let b = unsafe { std::slice::from_raw_parts(b as *const half::f16, k * n) };
        matmul_f16_packed(out, accum, a, b, 1.0);
    }

    unsafe fn matmul_left_into(
        out: MatMut<f32>,
        accum: Accum,
        a: *const PackedF16,
        m: usize,
        k: usize,
        b: MatRef<f32>,
    ) {
        // SAFETY: the caller promised `m * k` readable elements, and `PackedF16` is a
        // transparent `f16`.
        let a = unsafe { std::slice::from_raw_parts(a as *const half::f16, m * k) };
        matmul_f16_packed_a(out, accum, a, b, 1.0);
    }
}

/// Shapes of a batched `a x b`: `(m, k, n)`, the batch counts of `a` and `b`, and
/// the output shape. Batch dims broadcast numpy-style, so `[M,K] x [1,K,N]` is
/// `[1,M,N]`: the lower-rank operand's missing leading dims still appear in the output.
fn matmul_shape<B>(
    a: &TensorView<'_>,
    b: &TensorView<'_, B>,
) -> ((usize, usize, usize), usize, usize, Vec<usize>)
where
    B: Clone + std::fmt::Debug,
{
    let (a_dims, b_dims) = (a.shape.len(), b.shape.len());
    assert!(a_dims >= 2 && b_dims >= 2, "MatMul of {:?} and {:?}", a.shape, b.shape);
    let (m, k) = (a.shape[a_dims - 2], a.shape[a_dims - 1]);
    let (k_b, n) = (b.shape[b_dims - 2], b.shape[b_dims - 1]);
    assert_eq!(k, k_b, "MatMul K dim mismatch: {} vs {}", k, k_b);
    let batch_a: usize = a.shape[..a_dims - 2].iter().product();
    let batch_b: usize = b.shape[..b_dims - 2].iter().product();
    let mut out_shape = utils::broadcast_shapes(&a.shape[..a_dims - 2], &b.shape[..b_dims - 2])
        .expect("MatMul batch dims do not broadcast");
    // An empty batch broadcasts to an empty output.
    let batches = if batch_a == 0 || batch_b == 0 { 0 } else { batch_a.max(batch_b) };
    assert!(
        (batch_a == 1 || batch_b == 1 || batch_a == batch_b)
            && out_shape.iter().product::<usize>() == batches,
        "MatMul broadcast not fully supported yet"
    );
    assert!(
        a.data.len() >= batch_a * m * k && b.data.len() >= batch_b * k * n,
        "MatMul operands shorter than their shapes {:?} and {:?}",
        a.shape,
        b.shape
    );
    out_shape.push(m);
    out_shape.push(n);
    ((m, k, n), batch_a, batch_b, out_shape)
}

/// Each batch of `out` (`m x n`, as [`matmul_shape`] gives them) `= a * b`, or `+=`
/// with [`Accum::Add`].
fn matmul_batches<B: WeightElem>(
    a: &TensorView<'_>,
    b: &TensorView<'_, B>,
    ((m, k, n), batch_a, batch_b): ((usize, usize, usize), usize, usize),
    accum: Accum,
    out: &mut [f32],
) {
    for (b_i, out) in out.chunks_exact_mut(m * n).enumerate() {
        let a_offset = if batch_a == 1 { 0 } else { b_i * m * k };
        let b_offset = if batch_b == 1 { 0 } else { b_i * k * n };
        // SAFETY: `matmul_shape` checked that the operands hold every batch, and
        // `out` is one batch.
        unsafe {
            let a_mat =
                MatRef::<f32>::from_raw_parts(a.data.as_ptr().add(a_offset), m, k, k as isize, 1);
            let out_mat = MatMut::<f32>::from_raw_parts_mut(out.as_mut_ptr(), m, n, n as isize, 1);
            B::matmul_into(out_mat, accum, a_mat, b.data.as_ptr().add(b_offset), k, n);
        }
    }
}

pub fn matmul<'a, B: WeightElem>(
    a: &TensorView<'_>,
    b: &TensorView<'_, B>,
    out_buf: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let (dims, batch_a, batch_b, out_shape) = matmul_shape(a, b);
    let output_len = out_shape.iter().product();
    // SAFETY: the product replaces every element.
    unsafe { utils::ensure_capacity(out_buf, output_len) };
    let out = &mut out_buf[..output_len];
    if output_len > 0 {
        matmul_batches(a, b, (dims, batch_a, batch_b), Accum::Replace, out);
    }
    TensorView {
        data: Cow::Borrowed(out),
        shape: Cow::Owned(out_shape),
    }
}

/// `a * b` with `b` a quantized `[k, n]` matrix of integer `codes`, whose values are
/// `scale * (code - zero_point)`: the product of [`matmul`] with it widened to f32 (see
/// [`crate::kernels::matmul::matmul_codes`]), without ever holding it in f32. `a` is
/// `[.., k]`, its leading dims are the rows of the product.
pub fn weight_only_matmul<'a>(
    a: &TensorView<'_>,
    codes: Codes<'_>,
    (k, n): (usize, usize),
    zero_point: f32,
    scale: f32,
    out_buf: &'a mut Vec<f32>,
) -> TensorView<'a> {
    assert_eq!(
        a.shape.last(),
        Some(&k),
        "MatMul of {:?} and quantized weights [{k}, {n}]",
        a.shape
    );
    assert!(
        codes.len() >= k * n,
        "quantized weights shorter than [{k}, {n}]"
    );
    let m = a.data.len() / k.max(1);
    let mut out_shape = a.shape[..a.shape.len() - 1].to_vec();
    out_shape.push(n);
    // SAFETY: the product replaces every element.
    unsafe { utils::ensure_capacity(out_buf, m * n) };
    let out = &mut out_buf[..m * n];
    if m * n > 0 {
        // SAFETY: `a` holds `m * k` elements, `out` `m * n`, and `codes` `k * n`.
        unsafe {
            let a_mat = MatRef::<f32>::from_raw_parts(a.data.as_ptr(), m, k, k as isize, 1);
            let out_mat = MatMut::<f32>::from_raw_parts_mut(out.as_mut_ptr(), m, n, n as isize, 1);
            matmul_codes(out_mat, Accum::Replace, a_mat, codes, n, zero_point, scale);
        }
    }
    TensorView {
        data: Cow::Borrowed(out),
        shape: Cow::Owned(out_shape),
    }
}

pub fn matmul_fused_add<'a, B: WeightElem>(
    a: &TensorView<'_>,
    b: &TensorView<'_, B>,
    bias: &TensorView<'_>,
    out_buf: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let (dims, batch_a, batch_b, out_shape) = matmul_shape(a, b);
    let n = dims.2;
    let out_numel: usize = out_shape.iter().product();
    // SAFETY: the bias or the product is written to every element before any
    // is read.
    unsafe { utils::ensure_capacity(out_buf, out_numel) };
    let out = &mut out_buf[..out_numel];
    if out_numel == 0 {
        return TensorView {
            data: Cow::Borrowed(out),
            shape: Cow::Owned(out_shape),
        };
    }
    let bias_data = &bias.data[..];
    // How the bias broadcasts against the output.
    let bias_shape = utils::broadcast_shapes(&out_shape, &bias.shape);
    assert!(
        bias_shape.as_ref() == Some(&out_shape) && bias_data.len() == bias.shape.iter().product(),
        "MatMul output {out_shape:?} and bias {:?} do not broadcast",
        bias.shape
    );
    if bias_data.len() == n && bias.shape.last() == Some(&n) {
        // One bias per column: written first, then the product added to it.
        for row in out.chunks_exact_mut(n) {
            row.copy_from_slice(bias_data);
        }
        matmul_batches(a, b, (dims, batch_a, batch_b), Accum::Add, out);
    } else {
        matmul_batches(a, b, (dims, batch_a, batch_b), Accum::Replace, out);
        // The bias repeats whole over the output when its dimensions, past any
        // leading ones, are the output's last ones.
        let first = bias.shape.iter().position(|&d| d != 1);
        let trailing = first.map_or(&[][..], |f| &bias.shape[f..]);
        if bias_data.len() == 1 {
            bias_act_inplace(out, bias_data[0], Activation::None);
        } else if out_shape.ends_with(trailing) {
            for chunk in out.chunks_exact_mut(bias_data.len()) {
                for (o, &b) in chunk.iter_mut().zip(bias_data) {
                    *o += b;
                }
            }
        } else {
            // Coordinates of each output element, and the bias element they take.
            let rank = out_shape.len();
            let mut strides = vec![0; rank];
            let mut s = 1;
            for (i, &d) in bias.shape.iter().enumerate().rev() {
                if d != 1 {
                    strides[rank - bias.shape.len() + i] = s;
                }
                s *= d;
            }
            let mut coords = vec![0; rank];
            for o in out.iter_mut() {
                let at: usize = coords.iter().zip(&strides).map(|(c, s)| c * s).sum();
                *o += bias_data[at];
                for d in (0..rank).rev() {
                    coords[d] += 1;
                    if coords[d] < out_shape[d] {
                        break;
                    }
                    coords[d] = 0;
                }
            }
        }
    }
    TensorView {
        data: Cow::Borrowed(out),
        shape: Cow::Owned(out_shape),
    }
}

/// The gated product of a SwiGLU feed-forward, `b` `[k, 2n]` holding the gate's
/// weights then the value's, and `bias` `[2n]` their biases likewise:
/// `silu(a · b[:, :n] + bias[:n]) * (a · b[:, n:] + bias[n:])`, `[.., n]`.
///
/// Both halves are one [`matmul_fused_add`], and the gate goes through
/// [`silu`](crate::kernels::silu)'s arithmetic. With `n` a multiple of
/// [`PACK_COLS`](crate::kernels::matmul::PACK_COLS), each column of the product
/// comes out as it would from a product with that half alone, so the result is
/// that of the two products, `silu` and `mul`, bit for bit; with another `n`,
/// the columns of the last, narrower panel of a half may round differently.
pub fn linear_swiglu<'a, B: WeightElem>(
    a: &TensorView<'_>,
    b: &TensorView<'_, B>,
    bias: &TensorView<'_>,
    out_buf: &'a mut Vec<f32>,
) -> TensorView<'a> {
    thread_local! {
        static BOTH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let two_n = *b.shape.last().expect("linear_swiglu: b has no columns");
    assert!(
        two_n % 2 == 0,
        "linear_swiglu: b has {two_n} columns, an odd number"
    );
    let n = two_n / 2;
    let (out_shape, len) = BOTH.with_borrow_mut(|both| {
        let both = matmul_fused_add(a, b, bias, both);
        let mut out_shape = both.shape.to_vec();
        *out_shape.last_mut().unwrap() = n;
        let len = both.data.len() / 2;
        // SAFETY: every element is written below before any is read.
        unsafe { utils::ensure_capacity(out_buf, len) };
        if n > 0 {
            let rows = both.data.chunks_exact(two_n);
            for (row, out) in rows.zip(out_buf[..len].chunks_exact_mut(n)) {
                let (gate, value) = row.split_at(n);
                crate::kernels::math::silu_slice(gate, out);
                for (o, &v) in out.iter_mut().zip(value) {
                    *o *= v;
                }
            }
        }
        (out_shape, len)
    });
    TensorView {
        data: Cow::Borrowed(&out_buf[..len]),
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
    // SAFETY: the product replaces every element.
    unsafe { utils::ensure_capacity(out_buf, output_len) };
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

    /// `linear_swiglu` is the two products, `silu` and `mul` it replaces, bit for
    /// bit, with f32, f16 and packed f16 weights, and rows of C below and above
    /// the small-M kernel's, for gates a multiple of `PACK_COLS` wide.
    #[test]
    fn test_linear_swiglu_is_the_chain_it_replaces() {
        use crate::kernels::matmul::{PACK_COLS, pack_f16};
        let k = 40;
        let value = |i: usize, scale: f32| ((i * 7919) % 23) as f32 * scale - 1.0;
        for m in [1, 3, 5, 10] {
            for n in [PACK_COLS, 3 * PACK_COLS] {
                let a: Vec<f32> = (0..m * k).map(|i| value(i, 0.11)).collect();
                // The gate's and the value's weights side by side, and biases.
                let both: Vec<f32> = (0..k * 2 * n).map(|i| value(i, 0.07)).collect();
                let bias: Vec<f32> = (0..2 * n).map(|i| value(i, 0.05)).collect();
                let half_of = |h: usize| -> Vec<f32> {
                    (0..k * n)
                        .map(|i| both[i / n * 2 * n + h * n + i % n])
                        .collect()
                };
                let at = TensorView::from_slice(&a, vec![m, k]);
                let reference = |gate: TensorView<'_>, value: TensorView<'_>| {
                    let (mut s, mut y) = (Vec::new(), Vec::new());
                    let s = crate::kernels::silu(&gate, &mut s);
                    crate::kernels::mul(&s, &value, &mut y).data.to_vec()
                };
                let bias_of = |h: usize| TensorView::from_slice(&bias[h * n..][..n], vec![n]);
                let bias_both = TensorView::from_slice(&bias, vec![2 * n]);
                // f32 weights.
                let (mut g, mut v, mut out) = (Vec::new(), Vec::new(), Vec::new());
                let (wg, wv) = (half_of(0), half_of(1));
                let want = reference(
                    matmul_fused_add(
                        &at,
                        &TensorView::from_slice(&wg, vec![k, n]),
                        &bias_of(0),
                        &mut g,
                    ),
                    matmul_fused_add(
                        &at,
                        &TensorView::from_slice(&wv, vec![k, n]),
                        &bias_of(1),
                        &mut v,
                    ),
                );
                let got = linear_swiglu(
                    &at,
                    &TensorView::from_slice(&both, vec![k, 2 * n]),
                    &bias_both,
                    &mut out,
                );
                assert_eq!(got.shape.as_ref(), &[m, n]);
                assert_eq!(got.data.as_ref(), want, "f32, m {m}, n {n}");
                // f16 weights, as stored and packed.
                let f16 = |x: &[f32]| -> Vec<half::f16> {
                    x.iter().map(|&x| half::f16::from_f32(x)).collect()
                };
                let (hg, hv, hb) = (f16(&wg), f16(&wv), f16(&both));
                let want = reference(
                    matmul_fused_add(
                        &at,
                        &TensorView::from_slice(&hg, vec![k, n]),
                        &bias_of(0),
                        &mut g,
                    ),
                    matmul_fused_add(
                        &at,
                        &TensorView::from_slice(&hv, vec![k, n]),
                        &bias_of(1),
                        &mut v,
                    ),
                );
                let got = linear_swiglu(
                    &at,
                    &TensorView::from_slice(&hb, vec![k, 2 * n]),
                    &bias_both,
                    &mut out,
                );
                assert_eq!(got.data.as_ref(), want, "f16, m {m}, n {n}");
                let packed = |h: &[half::f16], cols: usize| -> Vec<PackedF16> {
                    pack_f16(h, k, cols).into_iter().map(PackedF16).collect()
                };
                let (pg, pv, pb) = (packed(&hg, n), packed(&hv, n), packed(&hb, 2 * n));
                let want = reference(
                    matmul_fused_add(
                        &at,
                        &TensorView::from_slice(&pg, vec![k, n]),
                        &bias_of(0),
                        &mut g,
                    ),
                    matmul_fused_add(
                        &at,
                        &TensorView::from_slice(&pv, vec![k, n]),
                        &bias_of(1),
                        &mut v,
                    ),
                );
                let got = linear_swiglu(
                    &at,
                    &TensorView::from_slice(&pb, vec![k, 2 * n]),
                    &bias_both,
                    &mut out,
                );
                assert_eq!(got.data.as_ref(), want, "packed f16, m {m}, n {n}");
            }
        }
    }

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

    /// Biases broadcast as ONNX `Add` does: one per column (added to), and the
    /// others added after the product: one value, one per output, repeating over
    /// the batch, or one per row (a middle dimension).
    #[test]
    fn test_matmul_fused_add_other_bias_shapes() {
        // [2, 3, 5] x [5, 17]: long enough rows for whole vectors and tails.
        let (batch, m, k, n) = (2, 3, 5, 17);
        let a: Vec<f32> = (0..batch * m * k).map(|i| (i % 7) as f32 - 3.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 * 0.5 - 1.0).collect();
        let product: Vec<f32> = (0..batch * m * n)
            .map(|idx| {
                let (bi, i, j) = (idx / (m * n), idx / n % m, idx % n);
                (0..k).map(|p| a[(bi * m + i) * k + p] * b[p * n + j]).sum()
            })
            .collect();
        let at = TensorView::from_slice(&a, vec![batch, m, k]);
        let bt = TensorView::from_slice(&b, vec![k, n]);
        for bias_shape in [vec![1], vec![batch, m, n], vec![1, m, n], vec![m, 1], vec![n]] {
            let len: usize = bias_shape.iter().product();
            let bias: Vec<f32> = (0..len).map(|i| i as f32 * 0.25 - 2.0).collect();
            let mut out = Vec::new();
            let got = matmul_fused_add(
                &at,
                &bt,
                &TensorView::from_slice(&bias, bias_shape.clone()),
                &mut out,
            );
            assert_eq!(got.shape.as_ref(), &[batch, m, n]);
            for (idx, (&g, &p)) in got.data.iter().zip(&product).enumerate() {
                let co = [idx / (m * n), idx / n % m, idx % n];
                // The bias element at `co`, broadcast.
                let skip = 3 - bias_shape.len();
                let at = bias_shape
                    .iter()
                    .zip(&co[skip..])
                    .fold(0, |at, (&d, &c)| at * d + if d == 1 { 0 } else { c });
                let want = p + bias[at];
                assert!((g - want).abs() < 1e-4, "bias {bias_shape:?} at {co:?}: {g} vs {want}");
            }
        }
    }

    /// An empty batch gives an empty output, and a bias that does not broadcast to
    /// the product is refused.
    #[test]
    fn test_matmul_fused_add_empty_batch_and_bad_bias() {
        let (a, b, bias) = (Vec::<f32>::new(), vec![1.0f32; 6], vec![1.0f32; 3]);
        let mut out = Vec::new();
        let got = matmul_fused_add(
            &TensorView::from_slice(&a, vec![0, 2, 2]),
            &TensorView::from_slice(&b, vec![2, 3]),
            &TensorView::from_slice(&bias, vec![3]),
            &mut out,
        );
        assert_eq!(got.shape.as_ref(), &[0, 2, 3]);
        assert!(got.data.is_empty());
        let a = vec![1.0f32; 4];
        let bad = std::panic::catch_unwind(|| {
            let mut out = Vec::new();
            matmul_fused_add(
                &TensorView::from_slice(&a, vec![2, 2]),
                &TensorView::from_slice(&b, vec![2, 3]),
                &TensorView::from_slice(&bias[..2], vec![2]),
                &mut out,
            );
        });
        assert!(bad.is_err());
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
