use crate::kernels::utils;
use crate::tensor::TensorView;
use crate::kernels::simd::simd_call;
use crate::kernels::simd_math::exp;
use fearless_simd::{Level, Simd, f32x16, prelude::*};
use fearless_simd_macros::simd;
use std::borrow::Cow;

pub fn softmax<'b, 'a>(
    input: &TensorView<'b>,
    axis: i32,
    out_buf: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let ndim = input.shape.len();
    let axis = if axis < 0 { ndim as i32 + axis } else { axis } as usize;
    assert!(axis < ndim);
    let numel = input.data.len();
    utils::ensure_capacity(out_buf, numel);
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_buf.as_mut_ptr(), numel) };
    let inner_size: usize = input.shape[axis + 1..].iter().product();
    let axis_size = input.shape[axis];
    let outer_size: usize = input.shape[..axis].iter().product();
    let data = &input.data;
    if inner_size == 1 {
        softmax_rows(Level::new(), outer_size, data, out_slice, axis_size);
    } else {
        unimplemented!("Softmax only supported on last dimension for now");
    }
    TensorView {
        data: Cow::Borrowed(out_slice),
        shape: std::borrow::Cow::Owned(input.shape.to_vec()),
    }
}

/// Softmax of the first `rows` `n`-wide rows of `src` into `out`.
///
/// Spends more per element on `exp` than the other norms do on memory, but
/// still runs AVX-512 machines at AVX2 until measured otherwise; see
/// `simd_call!`.
fn softmax_rows(level: Level, rows: usize, src: &[f32], out: &mut [f32], n: usize) {
    if n < LANES {
        return softmax_short_rows(rows, n, src, out);
    }
    simd_call!(level, max = Avx2, softmax_rows_simd(rows, n, src, out))
}

/// Rows narrower than a vector, in scalar code. Padding one into a vector
/// costs more than the row: a variable-length `copy_from_slice` is a `memcpy`
/// call on each side, and filling the vector with scalar stores before a
/// wide load stalls on store forwarding (80 ns against 51 for the scalar
/// loop, on a row of 10).
fn softmax_short_rows(rows: usize, n: usize, src: &[f32], out: &mut [f32]) {
    for (row, out_row) in src[..rows * n].chunks_exact(n).zip(out[..rows * n].chunks_exact_mut(n)) {
        let max = row.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
        let mut sum = 0.0;
        for (&x, o) in row.iter().zip(&mut *out_row) {
            *o = (x - max).exp();
            sum += *o;
        }
        let inv_sum = 1.0 / sum;
        for o in out_row {
            *o *= inv_sum;
        }
    }
}

/// The row's last `LANES` elements, which overlap the last full vector when
/// the width is not a multiple of `LANES`.
///
/// Redoing elements is harmless for max, and for storing outputs that depend
/// only on their own input; the sum must skip the overlap.
#[inline(always)]
fn last_vector<S: Simd>(simd: S, row: &[f32]) -> f32x16<S> {
    f32x16::load_array_ref(simd, row.last_chunk::<LANES>().expect("row at least a vector wide"))
}

/// Stores the lanes of `v` that `last_vector` took from the end of `row`.
#[inline(always)]
fn store_last_vector<S: Simd>(v: f32x16<S>, out_row: &mut [f32]) {
    v.store_array(out_row.last_chunk_mut::<LANES>().expect("row at least a vector wide"));
}

/// `v` with the lanes `last_vector` shares with the last full vector zeroed,
/// leaving the `tail_len` it adds to a sum.
#[inline(always)]
fn tail_lanes<S: Simd>(simd: S, v: f32x16<S>, tail_len: usize) -> f32x16<S> {
    let lane = f32x16::load_array_ref(simd, &LANE_INDEX);
    let new = lane.simd_ge(f32x16::splat(simd, (LANES - tail_len) as f32));
    new.select(v, f32x16::splat(simd, 0.0))
}

/// Largest element of a row at least `LANES` wide.
#[inline(always)]
fn row_max<S: Simd>(simd: S, row: &[f32]) -> f32 {
    let (x, x_tail) = row.as_chunks::<LANES>();
    let (x2, x1) = x.as_chunks::<2>();
    let mut max0 = f32x16::splat(simd, f32::NEG_INFINITY);
    let mut max1 = max0;
    for [a, b] in x2 {
        max0 = max0.max(f32x16::load_array_ref(simd, a));
        max1 = max1.max(f32x16::load_array_ref(simd, b));
    }
    for a in x1 {
        max0 = max0.max(f32x16::load_array_ref(simd, a));
    }
    if !x_tail.is_empty() {
        max0 = max0.max(last_vector(simd, row));
    }
    max0.max(max1).reduce_max()
}

/// Rows at least `LANES` wide.
#[simd]
fn softmax_rows_simd<S: Simd>(simd: S, rows: usize, n: usize, src: &[f32], out: &mut [f32]) {
    let (mut src, mut out) = (&src[..rows * n], &mut out[..rows * n]);
    for _ in 0..rows {
        let row;
        (row, src) = src.split_at(n);
        let out_row;
        (out_row, out) = std::mem::take(&mut out).split_at_mut(n);
        let (x, x_tail) = row.as_chunks::<LANES>();
        let (x2, x1) = x.as_chunks::<2>();
        let load = |x| f32x16::load_array_ref(simd, x);
        let max_v = f32x16::splat(simd, row_max(simd, row));

        // exp(x - max) goes to `out` and is scaled in place afterwards: the
        // row is still in cache, and the sum is needed before any output is final.
        let (o, _) = out_row.as_chunks_mut::<LANES>();
        let (o2, o1) = o.as_chunks_mut::<2>();
        let mut sum0 = f32x16::splat(simd, 0.0);
        let mut sum1 = sum0;
        for ([a, b], [oa, ob]) in x2.iter().zip(o2) {
            let (ea, eb) = (exp(load(a) - max_v), exp(load(b) - max_v));
            ea.store_array(oa);
            eb.store_array(ob);
            sum0 += ea;
            sum1 += eb;
        }
        for (a, oa) in x1.iter().zip(o1) {
            let ea = exp(load(a) - max_v);
            ea.store_array(oa);
            sum0 += ea;
        }
        // The last vector's exponentials stay in a register until the row is
        // scaled, instead of taking a round trip through `out`. Only its
        // last `x_tail.len()` lanes are new to the sum.
        let tail_e = (!x_tail.is_empty()).then(|| exp(last_vector(simd, row) - max_v));
        if let Some(e) = tail_e {
            sum0 += tail_lanes(simd, e, x_tail.len());
        }
        let inv_sum = 1.0 / (sum0 + sum1).reduce_sum();

        let inv_v = f32x16::splat(simd, inv_sum);
        let (o, _) = out_row.as_chunks_mut::<LANES>();
        let (o2, o1) = o.as_chunks_mut::<2>();
        for [a, b] in o2 {
            (f32x16::load_array_ref(simd, a) * inv_v).store_array(a);
            (f32x16::load_array_ref(simd, b) * inv_v).store_array(b);
        }
        for a in o1 {
            (f32x16::load_array_ref(simd, a) * inv_v).store_array(a);
        }
        // Overlapped lanes get the value they already hold: `e * inv_v` is
        // the same product the loops above took from the stored `e`.
        if let Some(e) = tail_e {
            store_last_vector(e * inv_v, out_row);
        }
    }
}

const LANE_INDEX: [f32; LANES] =
    [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.];

/// `x - max - ln(sum(exp(x - max)))` along `axis`.
///
/// Outputs are `(x - max) - ln(sum)`, never an exponential, so logits far
/// below the max keep their exact distance from it instead of underflowing
/// through `exp`; that tail is where a CTC beam search reads them. Taking
/// `x - max` first keeps it exact for logits near the max, where adding `ln(sum)`
/// to a large `max` would round away most of a small result.
pub fn log_softmax<'b, 'a>(
    input: &TensorView<'b>,
    axis: i32,
    out_buf: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let ndim = input.shape.len();
    let axis = if axis < 0 { ndim as i32 + axis } else { axis } as usize;
    assert!(axis < ndim);
    let numel = input.data.len();
    utils::ensure_capacity(out_buf, numel);
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_buf.as_mut_ptr(), numel) };
    let inner_size: usize = input.shape[axis + 1..].iter().product();
    let axis_size = input.shape[axis];
    let outer_size: usize = input.shape[..axis].iter().product();
    if inner_size == 1 {
        log_softmax_rows(Level::new(), outer_size, &input.data, out_slice, axis_size);
    } else {
        log_softmax_strided(outer_size, axis_size, inner_size, &input.data, out_slice);
    }
    TensorView {
        data: Cow::Borrowed(out_slice),
        shape: std::borrow::Cow::Owned(input.shape.to_vec()),
    }
}

/// Log-softmax of the first `rows` `n`-wide rows of `src` into `out`.
fn log_softmax_rows(level: Level, rows: usize, src: &[f32], out: &mut [f32], n: usize) {
    if n < LANES {
        return log_softmax_strided(rows, n, 1, src, out);
    }
    simd_call!(level, max = Avx2, log_softmax_rows_simd(rows, n, src, out))
}

/// Log-softmax over the middle axis of `[outer, axis_size, inner]`, in scalar
/// code: for an axis other than the last, and for rows narrower than a vector.
fn log_softmax_strided(outer: usize, axis_size: usize, inner: usize, src: &[f32], out: &mut [f32]) {
    for o in 0..outer {
        for i in 0..inner {
            let base = o * axis_size * inner + i;
            let at = |k: usize| base + k * inner;
            let max_val = (0..axis_size).fold(f32::NEG_INFINITY, |m, k| m.max(src[at(k)]));
            let sum: f32 = (0..axis_size).map(|k| (src[at(k)] - max_val).exp()).sum();
            let log_sum = sum.ln();
            for k in 0..axis_size {
                out[at(k)] = (src[at(k)] - max_val) - log_sum;
            }
        }
    }
}

/// Rows at least `LANES` wide.
#[simd]
fn log_softmax_rows_simd<S: Simd>(simd: S, rows: usize, n: usize, src: &[f32], out: &mut [f32]) {
    let rows_out = out[..rows * n].chunks_exact_mut(n);
    for (row, out_row) in src[..rows * n].chunks_exact(n).zip(rows_out) {
        let (x, x_tail) = row.as_chunks::<LANES>();
        let (x2, x1) = x.as_chunks::<2>();
        let max = row_max(simd, row);
        let max_v = f32x16::splat(simd, max);

        let mut sum0 = f32x16::splat(simd, 0.0);
        let mut sum1 = sum0;
        for [a, b] in x2 {
            sum0 += exp(f32x16::load_array_ref(simd, a) - max_v);
            sum1 += exp(f32x16::load_array_ref(simd, b) - max_v);
        }
        for a in x1 {
            sum0 += exp(f32x16::load_array_ref(simd, a) - max_v);
        }
        if !x_tail.is_empty() {
            let e = exp(last_vector(simd, row) - max_v);
            sum0 += tail_lanes(simd, e, x_tail.len());
        }
        let log_sum_v = f32x16::splat(simd, (sum0 + sum1).reduce_sum().ln());

        let (o, _) = out_row.as_chunks_mut::<LANES>();
        for (a, oa) in x.iter().zip(o) {
            (f32x16::load_array_ref(simd, a) - max_v - log_sum_v).store_array(oa);
        }
        if !x_tail.is_empty() {
            store_last_vector(last_vector(simd, row) - max_v - log_sum_v, out_row);
        }
    }
}

pub fn layer_norm<'b, 'a>(
    input: &TensorView<'b>,
    scale: &TensorView<'b>,
    bias: &TensorView<'b>,
    axis: i32,
    epsilon: f32,
    out_buf: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let ndim = input.shape.len();
    let axis = if axis < 0 { ndim as i32 + axis } else { axis } as usize;
    let outer_size: usize = input.shape[..axis].iter().product();
    let norm_size: usize = input.shape[axis..].iter().product();
    utils::ensure_capacity(out_buf, input.data.len());
    let out_slice =
        unsafe { std::slice::from_raw_parts_mut(out_buf.as_mut_ptr(), input.data.len()) };
    debug_assert_eq!(outer_size * norm_size, input.data.len());
    layer_norm_rows(
        Level::new(),
        outer_size,
        &input.data,
        &scale.data[..norm_size],
        &bias.data[..norm_size],
        out_slice,
        epsilon,
    );
    TensorView {
        data: Cow::Borrowed(out_slice),
        shape: std::borrow::Cow::Owned(input.shape.to_vec()),
    }
}
pub fn batch_norm<'b, 'a>(
    input: &TensorView<'b>,
    scale: &TensorView<'b>,
    bias: &TensorView<'b>,
    mean: &TensorView<'b>,
    var: &TensorView<'b>,
    epsilon: f32,
    out_buf: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let shape = &input.shape;
    let numel = input.data.len();
    utils::ensure_capacity(out_buf, numel);
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_buf.as_mut_ptr(), numel) };
    // [outer, channels, inner...]: every rank is the 4D NCHW case with the
    // trailing dimensions flattened.
    let (outer, c) = if shape.len() > 1 { (shape[0], shape[1]) } else { (1, shape[0]) };
    let inner: usize = shape.iter().skip(2).product();
    batch_norm_channels(
        Level::new(),
        outer,
        inner,
        &input.data,
        [&scale.data[..c], &bias.data[..c], &mean.data[..c], &var.data[..c]],
        epsilon,
        out_slice,
    );
    TensorView {
        data: Cow::Borrowed(out_slice),
        shape: Cow::Owned(shape.to_vec()),
    }
}

/// BatchNorm of `outer` samples of `[channels, inner]` into `out`, with
/// per-channel `[scale, bias, mean, var]`.
///
/// Streams memory like the other norms, so it runs AVX-512 machines at AVX2;
/// see `simd_call!`.
fn batch_norm_channels(
    level: Level,
    outer: usize,
    inner: usize,
    src: &[f32],
    params: [&[f32]; 4],
    epsilon: f32,
    out: &mut [f32],
) {
    simd_call!(level, max = Avx2, batch_norm_channels_simd(outer, inner, src, params, epsilon, out))
}

#[simd]
fn batch_norm_channels_simd<S: Simd>(
    simd: S,
    outer: usize,
    inner: usize,
    src: &[f32],
    [scale, bias, mean, var]: [&[f32]; 4],
    epsilon: f32,
    out: &mut [f32],
) {
    let c = scale.len();
    let (mut src, mut out) = (&src[..outer * c * inner], &mut out[..outer * c * inner]);
    for _ in 0..outer {
        for ch in 0..c {
            let row;
            (row, src) = src.split_at(inner);
            let out_row;
            (out_row, out) = std::mem::take(&mut out).split_at_mut(inner);
            // out = x * scale / sqrt(var + eps) + (bias - mean * scale / sqrt(var + eps))
            let k = scale[ch] / (var[ch] + epsilon).sqrt();
            let shift = bias[ch] - mean[ch] * k;

            let k_v = f32x16::splat(simd, k);
            let shift_v = f32x16::splat(simd, shift);
            let (x, x_tail) = row.as_chunks::<LANES>();
            let (o, _) = out_row.as_chunks_mut::<LANES>();
            let (x4, x1) = x.as_chunks::<4>();
            let (o4, o1) = o.as_chunks_mut::<4>();
            for (x, o) in x4.iter().zip(o4) {
                for (x, o) in x.iter().zip(o) {
                    f32x16::load_array_ref(simd, x).mul_add(k_v, shift_v).store_array(o);
                }
            }
            for (x, o) in x1.iter().zip(o1) {
                f32x16::load_array_ref(simd, x).mul_add(k_v, shift_v).store_array(o);
            }
            if !x_tail.is_empty() {
                if let (Some(x), Some(o)) =
                    (row.last_chunk::<LANES>(), out_row.last_chunk_mut::<LANES>())
                {
                    // Spatial widths are rarely multiples of 16, so the tail
                    // is common here: redo the last full vector, overlapping
                    // the previous one, instead of finishing with scalars.
                    // Each output depends only on its own input, and `src`
                    // and `out` are distinct, so recomputing is harmless.
                    f32x16::load_array_ref(simd, x).mul_add(k_v, shift_v).store_array(o);
                } else {
                    let from = inner - x_tail.len();
                    scale_shift_tail(x_tail, &mut out_row[from..], k, shift);
                }
            }
        }
    }
}

#[cold]
#[inline(never)]
fn scale_shift_tail(x: &[f32], out: &mut [f32], k: f32, shift: f32) {
    for (&x, o) in x.iter().zip(out) {
        *o = x * k + shift;
    }
}

pub fn rms_norm<'b, 'a>(
    input: &TensorView<'b>,
    weight: &TensorView<'b>,
    axis: i32,
    epsilon: f32,
    out_buf: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let ndim = input.shape.len();
    let axis = if axis < 0 { ndim as i32 + axis } else { axis } as usize;
    let outer_size: usize = input.shape[..axis].iter().product();
    let norm_size: usize = input.shape[axis..].iter().product();
    utils::ensure_capacity(out_buf, input.data.len());
    let out_slice =
        unsafe { std::slice::from_raw_parts_mut(out_buf.as_mut_ptr(), input.data.len()) };
    debug_assert_eq!(outer_size * norm_size, input.data.len());
    rms_norm_rows(Level::new(), outer_size, &input.data, &weight.data[..norm_size], out_slice, epsilon);

    TensorView {
        data: Cow::Borrowed(out_slice),
        shape: std::borrow::Cow::Owned(input.shape.to_vec()),
    }
}

/// Lanes per vector in the norm kernels. Fixed rather than native: AVX2 runs
/// it as two registers and NEON or WASM as four, which gives each loop
/// independent work to overlap.
const LANES: usize = 16;

/// LayerNorm of the first `rows` `gamma.len()`-wide rows of `src` into `out`.
///
/// The row count is passed rather than derived: deriving it, as
/// `chunks_exact` does, costs an integer division per call, which is a
/// noticeable share of a call on one short row.
///
/// Both norms stream memory, so they run AVX-512 machines at AVX2; see
/// `simd_call!`.
fn layer_norm_rows(
    level: Level,
    rows: usize,
    src: &[f32],
    gamma: &[f32],
    beta: &[f32],
    out: &mut [f32],
    epsilon: f32,
) {
    simd_call!(level, max = Avx2, layer_norm_rows_simd(rows, src, gamma, beta, out, epsilon))
}

#[simd]
fn layer_norm_rows_simd<S: Simd>(
    simd: S,
    rows: usize,
    src: &[f32],
    gamma: &[f32],
    beta: &[f32],
    out: &mut [f32],
    epsilon: f32,
) {
    let n = gamma.len();
    // Multiplying by 1/n keeps a division off each row's chain from its sums
    // to its first output, which short rows spend most of their time on.
    let inv_n = 1.0 / n as f32;
    // Peeling rows off the front checks each split once, where indexing by
    // `r * n` checks both ends of every row.
    let (mut src, mut out) = (&src[..rows * n], &mut out[..rows * n]);
    for _ in 0..rows {
        let row;
        (row, src) = src.split_at(n);
        let out_row;
        (out_row, out) = std::mem::take(&mut out).split_at_mut(n);
        let (sum, sumsq) = sums(simd, row, 0.0);
        let mut mean = sum * inv_n;
        let mut var = sumsq * inv_n - mean * mean;
        // E[x²] - E[x]² loses log2(E[x²] / var) bits of the variance to
        // cancellation. Rows centred near zero lose almost none; for the rest
        // a second pass over the deviations is cheaper than the error. Its
        // deviations also sum to the rounding error of `mean`, which corrects
        // both.
        if mean * mean > VAR_CANCEL_LIMIT * var {
            std::hint::cold_path();
            let (dev, devsq) = sums(simd, row, mean);
            let shift = dev * inv_n;
            mean += shift;
            var = devsq * inv_n - shift * shift;
        }
        let inv_std = 1.0 / (var + epsilon).sqrt();

        let mean_v = f32x16::splat(simd, mean);
        let inv_std_v = f32x16::splat(simd, inv_std);
        let norm = |x, g, b, o| {
            let normed = (f32x16::load_array_ref(simd, x) - mean_v) * inv_std_v;
            normed
                .mul_add(f32x16::load_array_ref(simd, g), f32x16::load_array_ref(simd, b))
                .store_array(o);
        };
        let (x, x_tail) = row.as_chunks::<LANES>();
        let (g, g_tail) = gamma.as_chunks::<LANES>();
        let (b, b_tail) = beta.as_chunks::<LANES>();
        let (o, o_tail) = out_row.as_chunks_mut::<LANES>();
        // Two vectors per step, like the reductions: this loop is bound by
        // stores, and halving its steps halves its loop overhead.
        let ((x2, x1), (g2, g1), (b2, b1)) = (x.as_chunks::<2>(), g.as_chunks::<2>(), b.as_chunks::<2>());
        let (o2, o1) = o.as_chunks_mut::<2>();
        for ((([x0, x1], [g0, g1]), [b0, b1]), [o0, o1]) in x2.iter().zip(g2).zip(b2).zip(o2) {
            norm(x0, g0, b0, o0);
            norm(x1, g1, b1, o1);
        }
        for (((x, g), b), o) in x1.iter().zip(g1).zip(b1).zip(o1) {
            norm(x, g, b, o);
        }
        if !x_tail.is_empty() {
            layer_norm_tail(x_tail, g_tail, b_tail, o_tail, mean, inv_std);
        }
    }
}

// The scalar tails past the last full vector live out of line. Widths are
// almost always multiples of 16, and inlined, each tail's unrolled loop costs
// its setup on every call whether or not it runs; a cold call costs nothing
// until it does.

#[cold]
#[inline(never)]
fn layer_norm_tail(x: &[f32], g: &[f32], b: &[f32], out: &mut [f32], mean: f32, inv_std: f32) {
    for (((&x, &g), &b), o) in x.iter().zip(g).zip(b).zip(out) {
        *o = (x - mean) * inv_std * g + b;
    }
}

#[cold]
#[inline(never)]
fn rms_norm_tail(x: &[f32], w: &[f32], out: &mut [f32], inv_rms: f32) {
    for ((&x, &w), o) in x.iter().zip(w).zip(out) {
        *o = x * inv_rms * w;
    }
}

/// Sum and sum of squares of `tail - shift`.
#[cold]
#[inline(never)]
fn tail_sums(tail: &[f32], shift: f32) -> (f32, f32) {
    tail.iter().fold((0.0, 0.0), |(sum, sumsq), &x| {
        let d = x - shift;
        (sum + d, sumsq + d * d)
    })
}

/// How far E[x]² may exceed the variance before LayerNorm recomputes the
/// variance from deviations: past 16, E[x²] - E[x]² has lost over 4 bits.
const VAR_CANCEL_LIMIT: f32 = 16.0;

/// Sum and sum of squares of `row - shift`. Each sum is split across two
/// accumulators so consecutive adds need not wait on one another.
#[inline(always)]
fn sums<S: Simd>(simd: S, row: &[f32], shift: f32) -> (f32, f32) {
    let shift_v = f32x16::splat(simd, shift);
    let zero = f32x16::splat(simd, 0.0);
    let (mut sum0, mut sum1, mut sq0, mut sq1) = (zero, zero, zero, zero);
    let (vectors, tail) = row.as_chunks::<LANES>();
    let (pairs, single) = vectors.as_chunks::<2>();
    for [a, b] in pairs {
        let a = f32x16::load_array_ref(simd, a) - shift_v;
        let b = f32x16::load_array_ref(simd, b) - shift_v;
        sum0 = sum0 + a;
        sum1 = sum1 + b;
        sq0 = a.mul_add(a, sq0);
        sq1 = b.mul_add(b, sq1);
    }
    for x in single {
        let x = f32x16::load_array_ref(simd, x) - shift_v;
        sum0 = sum0 + x;
        sq0 = x.mul_add(x, sq0);
    }
    let (mut sum, mut sumsq) = ((sum0 + sum1).reduce_sum(), (sq0 + sq1).reduce_sum());
    if !tail.is_empty() {
        let (s, q) = tail_sums(tail, shift);
        sum += s;
        sumsq += q;
    }
    (sum, sumsq)
}

/// RMSNorm of the first `rows` `weight.len()`-wide rows of `src` into `out`;
/// see [`layer_norm_rows`] for why the row count is passed.
fn rms_norm_rows(level: Level, rows: usize, src: &[f32], weight: &[f32], out: &mut [f32], epsilon: f32) {
    simd_call!(level, max = Avx2, rms_norm_rows_simd(rows, src, weight, out, epsilon))
}

#[simd]
fn rms_norm_rows_simd<S: Simd>(
    simd: S,
    rows: usize,
    src: &[f32],
    weight: &[f32],
    out: &mut [f32],
    epsilon: f32,
) {
    let n = weight.len();
    let inv_n = 1.0 / n as f32;
    // Peeling rows off the front checks each split once, where indexing by
    // `r * n` checks both ends of every row.
    let (mut src, mut out) = (&src[..rows * n], &mut out[..rows * n]);
    for _ in 0..rows {
        let row;
        (row, src) = src.split_at(n);
        let out_row;
        (out_row, out) = std::mem::take(&mut out).split_at_mut(n);
        let zero = f32x16::splat(simd, 0.0);
        let (mut sq0, mut sq1) = (zero, zero);
        let (vectors, tail) = row.as_chunks::<LANES>();
        let (pairs, single) = vectors.as_chunks::<2>();
        for [a, b] in pairs {
            let a = f32x16::load_array_ref(simd, a);
            let b = f32x16::load_array_ref(simd, b);
            sq0 = a.mul_add(a, sq0);
            sq1 = b.mul_add(b, sq1);
        }
        for x in single {
            let x = f32x16::load_array_ref(simd, x);
            sq0 = x.mul_add(x, sq0);
        }
        let mut sumsq = (sq0 + sq1).reduce_sum();
        if !tail.is_empty() {
            sumsq += tail_sums(tail, 0.0).1;
        }
        let inv_rms = 1.0 / (sumsq * inv_n + epsilon).sqrt();

        let inv_rms_v = f32x16::splat(simd, inv_rms);
        let (w, w_tail) = weight.as_chunks::<LANES>();
        let (o, o_tail) = out_row.as_chunks_mut::<LANES>();
        let scale = |x, w, o| {
            (f32x16::load_array_ref(simd, x) * inv_rms_v * f32x16::load_array_ref(simd, w)).store_array(o);
        };
        // Two vectors per step; see layer_norm_rows_simd.
        let ((x2, x1), (w2, w1)) = (vectors.as_chunks::<2>(), w.as_chunks::<2>());
        let (o2, o1) = o.as_chunks_mut::<2>();
        for (([x0, x1], [w0, w1]), [o0, o1]) in x2.iter().zip(w2).zip(o2) {
            scale(x0, w0, o0);
            scale(x1, w1, o1);
        }
        for ((x, w), o) in x1.iter().zip(w1).zip(o1) {
            scale(x, w, o);
        }
        if !tail.is_empty() {
            rms_norm_tail(tail, w_tail, o_tail, inv_rms);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::test_util::{AWKWARD_LENS, Rng, assert_close, levels};

    const EPS: f32 = 1e-5;
    const ROWS: usize = 3;

    fn layer_norm_ref(x: &[f32], gamma: &[f32], beta: &[f32]) -> Vec<f64> {
        let n = gamma.len();
        x.chunks_exact(n)
            .flat_map(|row| {
                let mean = row.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
                let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
                let inv_std = 1.0 / (var + EPS as f64).sqrt();
                row.iter()
                    .zip(gamma.iter().zip(beta))
                    .map(move |(&v, (&g, &b))| (v as f64 - mean) * inv_std * g as f64 + b as f64)
            })
            .collect()
    }

    fn rms_norm_ref(x: &[f32], weight: &[f32]) -> Vec<f64> {
        let n = weight.len();
        x.chunks_exact(n)
            .flat_map(|row| {
                let mean_sq = row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / n as f64;
                let inv_rms = 1.0 / (mean_sq + EPS as f64).sqrt();
                row.iter()
                    .zip(weight)
                    .map(move |(&v, &w)| v as f64 * inv_rms * w as f64)
            })
            .collect()
    }

    #[test]
    fn test_layer_norm_matches_reference_at_every_length_and_level() {
        // Rows centred on zero take the single pass; rows centred on 3 have
        // E[x]² 27 times their variance, so they take the second pass.
        for (lo, hi) in [(-1.0, 1.0), (2.0, 4.0)] {
            for level in levels() {
                let mut rng = Rng::new(1);
                for &n in AWKWARD_LENS {
                    let x = rng.vec(ROWS * n, lo, hi);
                    let gamma = rng.vec(n, 0.5, 1.5);
                    let beta = rng.vec(n, -0.5, 0.5);
                    let mut out = vec![0.0; x.len()];
                    layer_norm_rows(level, ROWS, &x, &gamma, &beta, &mut out, EPS);
                    let want = layer_norm_ref(&x, &gamma, &beta);
                    assert_close(&out, &want, 1e-5, &format!("{level:?} x in [{lo}, {hi}) n={n}"));
                }
            }
        }
    }

    #[test]
    fn test_layer_norm_keeps_variance_under_large_mean() {
        // Spread 1 around 1000: E[x²] - E[x]² on raw values keeps only a few
        // bits of the variance here.
        let mut rng = Rng::new(4);
        let n = 384;
        let x: Vec<f32> = rng.vec(ROWS * n, -1.0, 1.0).iter().map(|v| v + 1000.0).collect();
        let gamma = vec![1.0; n];
        let beta = vec![0.0; n];
        for level in levels() {
            let mut out = vec![0.0; x.len()];
            layer_norm_rows(level, ROWS, &x, &gamma, &beta, &mut out, EPS);
            assert_close(&out, &layer_norm_ref(&x, &gamma, &beta), 1e-4, &format!("{level:?}"));
        }
    }

    #[test]
    fn test_rms_norm_matches_reference_at_every_length_and_level() {
        for level in levels() {
            let mut rng = Rng::new(2);
            for &n in AWKWARD_LENS {
                let x = rng.vec(ROWS * n, -3.0, 3.0);
                let weight = rng.vec(n, 0.5, 1.5);
                let mut out = vec![0.0; x.len()];
                rms_norm_rows(level, ROWS, &x, &weight, &mut out, EPS);
                let want = rms_norm_ref(&x, &weight);
                assert_close(&out, &want, 1e-6, &format!("{level:?} n={n}"));
            }
        }
    }

    #[test]
    fn test_norms_accept_unaligned_rows() {
        // Slicing one element in leaves every row and weight off the vector alignment.
        let mut rng = Rng::new(3);
        let n = 67;
        let x = rng.vec(ROWS * n + 1, -2.0, 2.0);
        let w = rng.vec(n + 1, 0.5, 1.5);
        let b = rng.vec(n + 1, -0.5, 0.5);
        let (x, w, b) = (&x[1..], &w[1..], &b[1..]);
        let input = TensorView::from_slice(x, vec![ROWS, n]);
        let weight = TensorView::from_slice(w, vec![n]);
        let mut out = Vec::new();
        let got = layer_norm(&input, &weight, &TensorView::from_slice(b, vec![n]), -1, EPS, &mut out);
        assert_close(&got.data, &layer_norm_ref(x, w, b), 1e-5, "layer_norm");
        let got = rms_norm(&input, &weight, -1, EPS, &mut out);
        assert_close(&got.data, &rms_norm_ref(x, w), 1e-6, "rms_norm");
    }

    fn softmax_ref(x: &[f32], n: usize) -> Vec<f64> {
        x.chunks_exact(n)
            .flat_map(|row| {
                let max = row.iter().fold(f64::MIN, |m, &v| m.max(v as f64));
                let exps: Vec<f64> = row.iter().map(|&v| (v as f64 - max).exp()).collect();
                let sum: f64 = exps.iter().sum();
                exps.into_iter().map(move |e| e / sum)
            })
            .collect()
    }

    fn softmax_of(x: &[f32], n: usize) -> Vec<f32> {
        let input = TensorView::from_slice(x, vec![x.len() / n, n]);
        let mut out = Vec::new();
        softmax(&input, -1, &mut out).data.into_owned()
    }

    #[test]
    fn test_softmax_matches_reference_at_every_length() {
        // Probabilities are at most 1, so the check is effectively absolute.
        for (lo, hi) in [(-5.0, 5.0), (-80.0, 80.0)] {
            let mut rng = Rng::new(5);
            for &n in AWKWARD_LENS {
                let x = rng.vec(ROWS * n, lo, hi);
                let got = softmax_of(&x, n);
                assert_close(&got, &softmax_ref(&x, n), 1e-5, &format!("x in [{lo}, {hi}) n={n}"));
                for (r, row) in got.chunks_exact(n).enumerate() {
                    let sum: f64 = row.iter().map(|&v| v as f64).sum();
                    assert!((sum - 1.0).abs() < 1e-5, "n={n} row {r} sums to {sum}");
                }
            }
        }
    }

    #[test]
    fn test_softmax_matches_reference_at_every_level() {
        for level in levels() {
            let mut rng = Rng::new(8);
            for &n in AWKWARD_LENS {
                let x = rng.vec(ROWS * n, -20.0, 20.0);
                let mut out = vec![0.0; x.len()];
                softmax_rows(level, ROWS, &x, &mut out, n);
                assert_close(&out, &softmax_ref(&x, n), 1e-6, &format!("{level:?} n={n}"));
            }
        }
    }

    #[test]
    fn test_softmax_of_equal_logits_is_uniform() {
        for &n in AWKWARD_LENS {
            for v in [0.0, -30.0, 70.0] {
                let got = softmax_of(&vec![v; n], n);
                assert_close(&got, &vec![1.0 / n as f64; n], 1e-6, &format!("v={v} n={n}"));
            }
        }
    }

    #[test]
    fn test_softmax_keeps_far_below_max_logits_finite() {
        // Everything but the first logit sits past exp's underflow point.
        for &n in AWKWARD_LENS.iter().filter(|&&n| n > 1) {
            let mut x = vec![-200.0; n];
            x[0] = 50.0;
            let got = softmax_of(&x, n);
            assert_close(&got, &softmax_ref(&x, n), 1e-6, &format!("n={n}"));
        }
    }

    fn log_softmax_ref(x: &[f32], n: usize) -> Vec<f64> {
        x.chunks_exact(n)
            .flat_map(|row| {
                let max = row.iter().fold(f64::MIN, |m, &v| m.max(v as f64));
                let sum: f64 = row.iter().map(|&v| (v as f64 - max).exp()).sum();
                row.iter().map(move |&v| v as f64 - max - sum.ln())
            })
            .collect()
    }

    #[test]
    fn test_log_softmax_matches_reference_at_every_length_and_level() {
        for level in levels() {
            for (lo, hi) in [(-5.0, 5.0), (-80.0, 80.0)] {
                let mut rng = Rng::new(9);
                for &n in AWKWARD_LENS {
                    let x = rng.vec(ROWS * n, lo, hi);
                    let mut out = vec![0.0; x.len()];
                    log_softmax_rows(level, ROWS, &x, &mut out, n);
                    let what = format!("{level:?} x in [{lo}, {hi}) n={n}");
                    assert_close(&out, &log_softmax_ref(&x, n), 1e-6, &what);
                }
            }
        }
    }

    #[test]
    fn test_log_softmax_keeps_far_below_max_logits_exact() {
        // Everything but the first logit sits past exp's underflow point.
        for level in levels() {
            for &n in AWKWARD_LENS.iter().filter(|&&n| n > 1) {
                let mut x = vec![-200.0; n];
                x[0] = 50.0;
                let mut out = vec![0.0; n];
                log_softmax_rows(level, 1, &x, &mut out, n);
                assert_close(&out, &log_softmax_ref(&x, n), 1e-7, &format!("{level:?} n={n}"));
            }
        }
    }

    fn batch_norm_ref(x: &[f32], c: usize, inner: usize, p: &BnParams) -> Vec<f64> {
        x.iter()
            .enumerate()
            .map(|(i, &v)| {
                let ch = (i / inner) % c;
                let inv_std = 1.0 / (p.var[ch] as f64 + EPS as f64).sqrt();
                (v as f64 - p.mean[ch] as f64) * inv_std * p.scale[ch] as f64 + p.bias[ch] as f64
            })
            .collect()
    }

    struct BnParams {
        scale: Vec<f32>,
        bias: Vec<f32>,
        mean: Vec<f32>,
        var: Vec<f32>,
    }

    fn check_batch_norm(shape: Vec<usize>, rng: &mut Rng) {
        let c = shape[1];
        let inner: usize = shape[2..].iter().product();
        let x = rng.vec(shape.iter().product(), -3.0, 3.0);
        let p = BnParams {
            scale: rng.vec(c, 0.5, 1.5),
            bias: rng.vec(c, -0.5, 0.5),
            mean: rng.vec(c, -1.0, 1.0),
            var: rng.vec(c, 0.5, 2.0),
        };
        let input = TensorView::from_slice(&x, shape.clone());
        fn param(v: &[f32]) -> TensorView<'_> {
            TensorView::from_slice(v, vec![v.len()])
        }
        let mut out = Vec::new();
        let got = batch_norm(
            &input,
            &param(&p.scale),
            &param(&p.bias),
            &param(&p.mean),
            &param(&p.var),
            EPS,
            &mut out,
        );
        assert_close(&got.data, &batch_norm_ref(&x, c, inner, &p), 1e-6, &format!("{shape:?}"));
    }

    #[test]
    fn test_batch_norm_matches_reference_at_every_level() {
        let c = 3;
        for level in levels() {
            let mut rng = Rng::new(7);
            for &inner in AWKWARD_LENS {
                let x = rng.vec(2 * c * inner, -3.0, 3.0);
                let p = BnParams {
                    scale: rng.vec(c, 0.5, 1.5),
                    bias: rng.vec(c, -0.5, 0.5),
                    mean: rng.vec(c, -1.0, 1.0),
                    var: rng.vec(c, 0.5, 2.0),
                };
                let mut out = vec![0.0; x.len()];
                batch_norm_channels(level, 2, inner, &x, [&p.scale, &p.bias, &p.mean, &p.var], EPS, &mut out);
                let want = batch_norm_ref(&x, c, inner, &p);
                assert_close(&out, &want, 1e-6, &format!("{level:?} inner={inner}"));
            }
        }
    }

    #[test]
    fn test_batch_norm_matches_reference_at_every_spatial_size() {
        let mut rng = Rng::new(6);
        for &n in AWKWARD_LENS {
            // 4D takes the NCHW path, 3D and 2D the generic one.
            check_batch_norm(vec![2, 3, 1, n], &mut rng);
            check_batch_norm(vec![2, 3, n], &mut rng);
        }
        check_batch_norm(vec![2, 3, 5, 7], &mut rng);
        check_batch_norm(vec![4, 67], &mut rng);
    }
}
