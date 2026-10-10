//! LSTM and GRU, one direction and batch 1, on `fearless_simd`.
//!
//! The input's contribution to the gates does not depend on the state, so it
//! is one GEMM over every time step up front, with the biases added once.
//! Each step then adds the recurrent matrix times the previous hidden state
//! and applies the gates.

use crate::kernels::matmul::{Accum, MatMut, MatRef, matmul_at};
use crate::kernels::simd::simd_call;
use crate::kernels::simd_math as vm;
use crate::tensor::TensorView;
use fearless_simd::{Level, Simd, f32x16, prelude::*};
use fearless_simd_macros::simd;

/// LSTM forward pass, gates in ONNX order `[i, o, f, c]`.
///
/// `input` is `[seq_len, 1, input_size]`, `w` `[1, 4*hidden, input_size]`,
/// `r` `[1, 4*hidden, hidden]`, `bias` `[1, 8*hidden]` (`Wb` then `Rb`).
/// Returns `(Y, Y_h, Y_c)`: `[seq_len, 1, 1, hidden]`, `[1, 1, hidden]`,
/// `[1, 1, hidden]`. Sequence lengths and peepholes are not supported.
pub fn lstm<'b, 'a>(
    input: &TensorView<'b>,
    w: &TensorView<'b>,
    r: &TensorView<'b>,
    bias: Option<&TensorView<'b>>,
    _sequence_lens: Option<&TensorView<'b>>,
    initial_h: Option<&TensorView<'b>>,
    initial_c: Option<&TensorView<'b>>,
    out_y: &'a mut Vec<f32>,
    out_h: &'a mut Vec<f32>,
    out_c: &'a mut Vec<f32>,
) -> (TensorView<'a>, TensorView<'a>, TensorView<'a>) {
    lstm_at(Level::new(), input, w, r, bias, initial_h, initial_c, out_y, out_h, out_c);
    let hs = out_h.len();
    let seq_len = out_y.len() / hs.max(1);
    (
        TensorView::from_slice(out_y, vec![seq_len, 1, 1, hs]),
        TensorView::from_slice(out_h, vec![1, 1, hs]),
        TensorView::from_slice(out_c, vec![1, 1, hs]),
    )
}

fn lstm_at(
    level: Level,
    input: &TensorView,
    w: &TensorView,
    r: &TensorView,
    bias: Option<&TensorView>,
    initial_h: Option<&TensorView>,
    initial_c: Option<&TensorView>,
    out_y: &mut Vec<f32>,
    out_h: &mut Vec<f32>,
    out_c: &mut Vec<f32>,
) {
    let (seq_len, input_size) = check_shapes("LSTM", input, w);
    let hs = w.shape[1] / 4;
    let g = 4 * hs;

    // xw[t] = W·x_t + Wb + Rb.
    let mut xw = vec![0.0f32; seq_len * g];
    mul_transposed(level, &mut xw, &input.data, seq_len, &w.data, g, input_size);
    if let Some(b) = bias {
        let (bw, br) = b.data.split_at(g);
        add_bias(&mut xw, bw, br);
    }

    init_state(out_h, hs, initial_h);
    init_state(out_c, hs, initial_c);
    out_y.resize(seq_len * hs, 0.0);
    for (gates, y) in xw.chunks_exact_mut(g).zip(out_y.chunks_exact_mut(hs)) {
        matvec(level, gates, Accum::Add, &r.data, out_h, hs);
        simd_call!(level, lstm_cell(gates, out_c, out_h));
        y.copy_from_slice(out_h);
    }
}

/// GRU forward pass, gates in ONNX order `[z, r, h]`.
///
/// `input` is `[seq_len, 1, input_size]`, `w` `[1, 3*hidden, input_size]`,
/// `r` `[1, 3*hidden, hidden]`, `bias` `[1, 6*hidden]` (`Wb` then `Rb`).
/// With `linear_before_reset` the reset gate scales `R_h·h + Rb_h`, otherwise
/// it scales `h` before `R_h` multiplies it.
/// Returns `(Y, Y_h)`: `[seq_len, 1, 1, hidden]` and `[1, 1, hidden]`.
pub fn gru<'b, 'a>(
    input: &TensorView<'b>,
    w: &TensorView<'b>,
    r: &TensorView<'b>,
    bias: Option<&TensorView<'b>>,
    initial_h: Option<&TensorView<'b>>,
    linear_before_reset: bool,
    out_y: &'a mut Vec<f32>,
    out_h: &'a mut Vec<f32>,
) -> (TensorView<'a>, TensorView<'a>) {
    gru_at(Level::new(), input, w, r, bias, initial_h, linear_before_reset, out_y, out_h);
    let hs = out_h.len();
    let seq_len = out_y.len() / hs.max(1);
    (
        TensorView::from_slice(out_y, vec![seq_len, 1, 1, hs]),
        TensorView::from_slice(out_h, vec![1, 1, hs]),
    )
}

fn gru_at(
    level: Level,
    input: &TensorView,
    w: &TensorView,
    r: &TensorView,
    bias: Option<&TensorView>,
    initial_h: Option<&TensorView>,
    linear_before_reset: bool,
    out_y: &mut Vec<f32>,
    out_h: &mut Vec<f32>,
) {
    let (seq_len, input_size) = check_shapes("GRU", input, w);
    let hs = w.shape[1] / 3;
    let g = 3 * hs;

    // xw[t] = W·x_t + Wb, plus Rb for z and r, and for h too unless the
    // reset gate has to scale it.
    let mut xw = vec![0.0f32; seq_len * g];
    mul_transposed(level, &mut xw, &input.data, seq_len, &w.data, g, input_size);
    let zeros = vec![0.0f32; g];
    let (bw, br) = bias.map_or((&zeros[..], &zeros[..]), |b| b.data.split_at(g));
    let (br_zr, br_h) = br.split_at(2 * hs);
    if linear_before_reset {
        let br = [br_zr, &zeros[..hs]].concat();
        add_bias(&mut xw, bw, &br);
    } else {
        add_bias(&mut xw, bw, br);
    }

    let (r_zr, r_h) = r.data.split_at(2 * hs * hs);
    init_state(out_h, hs, initial_h);
    out_y.resize(seq_len * hs, 0.0);
    let mut scratch = vec![0.0f32; hs];
    for (gates, y) in xw.chunks_exact_mut(g).zip(out_y.chunks_exact_mut(hs)) {
        let (zr, xh) = gates.split_at_mut(2 * hs);
        matvec(level, zr, Accum::Add, r_zr, out_h, hs);
        if linear_before_reset {
            matvec(level, &mut scratch, Accum::Replace, r_h, out_h, hs);
            simd_call!(level, gru_cell_linear(zr, &scratch, br_h, xh, out_h));
        } else {
            simd_call!(level, gru_reset(zr, out_h, &mut scratch));
            matvec(level, xh, Accum::Add, r_h, &scratch, hs);
            simd_call!(level, gru_update(&zr[..hs], xh, out_h));
        }
        y.copy_from_slice(out_h);
    }
}

/// Checks what the kernels support; returns `(seq_len, input_size)`.
fn check_shapes(op: &str, input: &TensorView, w: &TensorView) -> (usize, usize) {
    assert_eq!(w.shape[0], 1, "{op}: only one direction is supported");
    assert_eq!(input.shape[1], 1, "{op}: only batch size 1 is supported");
    (input.shape[0], input.shape[2])
}

fn init_state(state: &mut Vec<f32>, hs: usize, initial: Option<&TensorView>) {
    state.resize(hs, 0.0);
    match initial {
        Some(v) => state.copy_from_slice(&v.data),
        None => state.fill(0.0),
    }
}

/// Adds `bw + br` to every row of `x`.
fn add_bias(x: &mut [f32], bw: &[f32], br: &[f32]) {
    let b: Vec<f32> = bw.iter().zip(br).map(|(w, r)| w + r).collect();
    for row in x.chunks_exact_mut(b.len()) {
        for (x, b) in row.iter_mut().zip(&b) {
            *x += b;
        }
    }
}

/// `out = x·wᵀ` for row-major `x` (`rows x k`), `w` (`n x k`) and `out`.
fn mul_transposed(level: Level, out: &mut [f32], x: &[f32], rows: usize, w: &[f32], n: usize, k: usize) {
    assert!(x.len() >= rows * k && w.len() >= n * k && out.len() >= rows * n);
    // SAFETY: the asserts above keep every view inside its slice.
    let (a, b, c) = unsafe {
        (
            MatRef::from_raw_parts(x.as_ptr(), rows, k, k as isize, 1),
            MatRef::from_raw_parts(w.as_ptr(), k, n, 1, k as isize),
            MatMut::from_raw_parts_mut(out.as_mut_ptr(), rows, n, n as isize, 1),
        )
    };
    matmul_at(level, c, Accum::Replace, a, b, 1.0);
}

/// `y = m·x` or `y += m·x`, for row-major `m` with `y.len()` rows and `cols`
/// columns.
fn matvec(level: Level, y: &mut [f32], accum: Accum, m: &[f32], x: &[f32], cols: usize) {
    let rows = y.len();
    assert!(m.len() >= rows * cols && x.len() >= cols);
    // SAFETY: the assert above keeps every view inside its slice.
    let (a, b, c) = unsafe {
        (
            MatRef::from_raw_parts(m.as_ptr(), rows, cols, cols as isize, 1),
            MatRef::from_raw_parts(x.as_ptr(), cols, 1, 1, 1),
            MatMut::from_raw_parts_mut(y.as_mut_ptr(), rows, 1, 1, 1),
        )
    };
    matmul_at(level, c, accum, a, b, 1.0);
}

/// `c = σ(f)·c + σ(i)·tanh(g)`, `h = σ(o)·tanh(c)`, from the pre-activations
/// `gates = [i, o, f, g]`, each `c.len()` long.
#[simd]
fn lstm_cell<S: Simd>(simd: S, gates: &[f32], c: &mut [f32], h: &mut [f32]) {
    let n = c.len();
    let (i, rest) = gates.split_at(n);
    let (o, rest) = rest.split_at(n);
    let (f, g) = rest.split_at(n);
    for k in (0..n).step_by(16) {
        let e = n.min(k + 16);
        let input = vm::sigmoid(load(simd, &i[k..e])) * vm::tanh(load(simd, &g[k..e]));
        let ct = vm::sigmoid(load(simd, &f[k..e])).mul_add(load(simd, &c[k..e]), input);
        let ht = vm::sigmoid(load(simd, &o[k..e])) * vm::tanh(ct);
        store(ct, &mut c[k..e]);
        store(ht, &mut h[k..e]);
    }
}

/// The GRU gates without `linear_before_reset`: `z = σ(z)` in place and
/// `rh = σ(r)·h`, from `zr = [z, r]`.
#[simd]
fn gru_reset<S: Simd>(simd: S, zr: &mut [f32], h: &[f32], rh: &mut [f32]) {
    let n = h.len();
    let (z, r) = zr.split_at_mut(n);
    for k in (0..n).step_by(16) {
        let e = n.min(k + 16);
        store(vm::sigmoid(load(simd, &z[k..e])), &mut z[k..e]);
        store(vm::sigmoid(load(simd, &r[k..e])) * load(simd, &h[k..e]), &mut rh[k..e]);
    }
}

/// `h = (1 - z)·tanh(x) + z·h`, as `tanh(x) + z·(h - tanh(x))`, where `z` is
/// already through its sigmoid.
#[simd]
fn gru_update<S: Simd>(simd: S, z: &[f32], x: &[f32], h: &mut [f32]) {
    let n = h.len();
    for k in (0..n).step_by(16) {
        let e = n.min(k + 16);
        let cand = vm::tanh(load(simd, &x[k..e]));
        let ht = load(simd, &z[k..e]).mul_add(load(simd, &h[k..e]) - cand, cand);
        store(ht, &mut h[k..e]);
    }
}

/// One GRU step with `linear_before_reset`, from the pre-activations
/// `zr = [z, r]`, `rh = R_h·h` and `x = W_h·x_t + Wb_h`:
/// `h = (1 - σ(z))·tanh(x + σ(r)·(rh + rb)) + σ(z)·h`.
#[simd]
fn gru_cell_linear<S: Simd>(simd: S, zr: &[f32], rh: &[f32], rb: &[f32], x: &[f32], h: &mut [f32]) {
    let n = h.len();
    let (z, r) = zr.split_at(n);
    for k in (0..n).step_by(16) {
        let e = n.min(k + 16);
        let reset = vm::sigmoid(load(simd, &r[k..e]));
        let pre = reset.mul_add(load(simd, &rh[k..e]) + load(simd, &rb[k..e]), load(simd, &x[k..e]));
        let cand = vm::tanh(pre);
        let z = vm::sigmoid(load(simd, &z[k..e]));
        store(z.mul_add(load(simd, &h[k..e]) - cand, cand), &mut h[k..e]);
    }
}

/// Up to 16 elements of `x`, the missing lanes 0.
#[inline(always)]
fn load<S: Simd>(simd: S, x: &[f32]) -> f32x16<S> {
    match x.first_chunk::<16>() {
        Some(a) => f32x16::load_array_ref(simd, a),
        None => {
            let mut padded = [0.0f32; 16];
            padded[..x.len()].copy_from_slice(x);
            f32x16::load_array_ref(simd, &padded)
        }
    }
}

/// The first `out.len()` (at most 16) lanes of `v`.
#[inline(always)]
fn store<S: Simd>(v: f32x16<S>, out: &mut [f32]) {
    match out.first_chunk_mut::<16>() {
        Some(a) => v.store_array(a),
        None => {
            let mut padded = [0.0f32; 16];
            v.store_array(&mut padded);
            let n = out.len();
            out.copy_from_slice(&padded[..n]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::test_util::{Rng, levels};

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// `m·x` for row-major `m` with `x.len()` columns, in f64.
    fn mv(m: &[f32], x: &[f64]) -> Vec<f64> {
        m.chunks_exact(x.len())
            .map(|row| row.iter().zip(x).map(|(&a, &b)| a as f64 * b).sum())
            .collect()
    }

    fn lstm_ref(x: &[f32], w: &[f32], r: &[f32], b: &[f32], h0: &[f32], c0: &[f32]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let hs = h0.len();
        let is = w.len() / (4 * hs);
        let mut h: Vec<f64> = h0.iter().map(|&v| v as f64).collect();
        let mut c: Vec<f64> = c0.iter().map(|&v| v as f64).collect();
        let mut y = Vec::new();
        for xt in x.chunks_exact(is) {
            let xt: Vec<f64> = xt.iter().map(|&v| v as f64).collect();
            let (wx, rh) = (mv(w, &xt), mv(r, &h));
            let pre = |gate: usize, k: usize| {
                let j = gate * hs + k;
                wx[j] + rh[j] + b[j] as f64 + b[4 * hs + j] as f64
            };
            for k in 0..hs {
                let (i, o, f, g) = (sigmoid(pre(0, k)), sigmoid(pre(1, k)), sigmoid(pre(2, k)), pre(3, k).tanh());
                c[k] = f * c[k] + i * g;
                h[k] = o * c[k].tanh();
            }
            y.extend_from_slice(&h);
        }
        (y, h, c)
    }

    fn gru_ref(x: &[f32], w: &[f32], r: &[f32], b: &[f32], h0: &[f32], linear: bool) -> Vec<f64> {
        let hs = h0.len();
        let is = w.len() / (3 * hs);
        let (r_zr, r_h) = r.split_at(2 * hs * hs);
        let mut h: Vec<f64> = h0.iter().map(|&v| v as f64).collect();
        let mut y = Vec::new();
        for xt in x.chunks_exact(is) {
            let xt: Vec<f64> = xt.iter().map(|&v| v as f64).collect();
            let (wx, rh) = (mv(w, &xt), mv(r_zr, &h));
            let bw = |j: usize| b[j] as f64;
            let br = |j: usize| b[3 * hs + j] as f64;
            let z: Vec<f64> = (0..hs).map(|k| sigmoid(wx[k] + rh[k] + bw(k) + br(k))).collect();
            let rg: Vec<f64> = (0..hs).map(|k| sigmoid(wx[hs + k] + rh[hs + k] + bw(hs + k) + br(hs + k))).collect();
            let rec = if linear {
                let rh = mv(r_h, &h);
                (0..hs).map(|k| rg[k] * (rh[k] + br(2 * hs + k))).collect::<Vec<_>>()
            } else {
                let scaled: Vec<f64> = (0..hs).map(|k| rg[k] * h[k]).collect();
                let rh = mv(r_h, &scaled);
                (0..hs).map(|k| rh[k] + br(2 * hs + k)).collect()
            };
            for k in 0..hs {
                let cand = (wx[2 * hs + k] + bw(2 * hs + k) + rec[k]).tanh();
                h[k] = (1.0 - z[k]) * cand + z[k] * h[k];
            }
            y.extend_from_slice(&h);
        }
        y
    }

    fn assert_close(got: &[f32], want: &[f64], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert!((g as f64 - w).abs() < 2e-5, "{what}[{i}]: {g} vs {w}");
        }
    }

    /// Hidden sizes below, at and around the vector width.
    const CASES: &[(usize, usize, usize)] = &[(1, 3, 1), (4, 5, 7), (3, 16, 16), (5, 9, 33), (2, 40, 20)];

    #[test]
    fn test_lstm_matches_reference_at_every_level() {
        let mut rng = Rng::new(1);
        for &(seq, is, hs) in CASES {
            let x = rng.vec(seq * is, -1.0, 1.0);
            let w = rng.vec(4 * hs * is, -0.5, 0.5);
            let r = rng.vec(4 * hs * hs, -0.5, 0.5);
            let b = rng.vec(8 * hs, -0.5, 0.5);
            let h0 = rng.vec(hs, -1.0, 1.0);
            let c0 = rng.vec(hs, -1.0, 1.0);
            let (y_ref, h_ref, c_ref) = lstm_ref(&x, &w, &r, &b, &h0, &c0);
            let views = (
                TensorView::from_slice(&x, vec![seq, 1, is]),
                TensorView::from_slice(&w, vec![1, 4 * hs, is]),
                TensorView::from_slice(&r, vec![1, 4 * hs, hs]),
                TensorView::from_slice(&b, vec![1, 8 * hs]),
                TensorView::from_slice(&h0, vec![1, 1, hs]),
                TensorView::from_slice(&c0, vec![1, 1, hs]),
            );
            for level in levels() {
                let (mut y, mut h, mut c) = (Vec::new(), Vec::new(), Vec::new());
                let (xv, wv, rv, bv, hv, cv) = &views;
                lstm_at(level, xv, wv, rv, Some(bv), Some(hv), Some(cv), &mut y, &mut h, &mut c);
                let what = format!("{level:?} seq={seq} hidden={hs}");
                assert_close(&y, &y_ref, &format!("{what} Y"));
                assert_close(&h, &h_ref, &format!("{what} Y_h"));
                assert_close(&c, &c_ref, &format!("{what} Y_c"));
            }
        }
    }

    #[test]
    fn test_gru_matches_reference_at_every_level() {
        let mut rng = Rng::new(2);
        for &(seq, is, hs) in CASES {
            let x = rng.vec(seq * is, -1.0, 1.0);
            let w = rng.vec(3 * hs * is, -0.5, 0.5);
            let r = rng.vec(3 * hs * hs, -0.5, 0.5);
            let b = rng.vec(6 * hs, -0.5, 0.5);
            let h0 = rng.vec(hs, -1.0, 1.0);
            let views = (
                TensorView::from_slice(&x, vec![seq, 1, is]),
                TensorView::from_slice(&w, vec![1, 3 * hs, is]),
                TensorView::from_slice(&r, vec![1, 3 * hs, hs]),
                TensorView::from_slice(&b, vec![1, 6 * hs]),
                TensorView::from_slice(&h0, vec![1, 1, hs]),
            );
            for linear in [false, true] {
                let y_ref = gru_ref(&x, &w, &r, &b, &h0, linear);
                for level in levels() {
                    let (mut y, mut h) = (Vec::new(), Vec::new());
                    let (xv, wv, rv, bv, hv) = &views;
                    gru_at(level, xv, wv, rv, Some(bv), Some(hv), linear, &mut y, &mut h);
                    let what = format!("{level:?} seq={seq} hidden={hs} linear_before_reset={linear}");
                    assert_close(&y, &y_ref, &format!("{what} Y"));
                    assert_close(&h, &y_ref[(seq - 1) * hs..], &format!("{what} Y_h"));
                }
            }
        }
    }
}
