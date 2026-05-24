#![allow(unsafe_op_in_unsafe_fn)]
use crate::kernels::activations::{sigmoid, tanh};
#[cfg(target_arch = "wasm32")]
use crate::kernels::wasm_matmul::{Accum, MatMut, MatRef, Par, matmul};
use crate::tensor::TensorView;
#[cfg(not(target_arch = "wasm32"))]
use faer::linalg::matmul::matmul;
#[cfg(not(target_arch = "wasm32"))]
use faer::mat::{MatMut, MatRef};
#[cfg(not(target_arch = "wasm32"))]
use faer::{Accum, Par};

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn lstm_gates_avx2(
    gates: &[f32],
    out_c: &mut [f32],
    out_h: &mut [f32],
    out_y: &mut [f32],
    hidden_size: usize,
    t: usize,
) {
    use std::arch::x86_64::*;
    unsafe {
        let mut k = 0;
        while k + 8 <= hidden_size {
            // Load gate values directly and compute sigmoid/tanh with AVX2 SIMD
            let i_raw = _mm256_loadu_ps(gates.as_ptr().add(k));
            let o_raw = _mm256_loadu_ps(gates.as_ptr().add(hidden_size + k));
            let f_raw = _mm256_loadu_ps(gates.as_ptr().add(2 * hidden_size + k));
            let c_raw = _mm256_loadu_ps(gates.as_ptr().add(3 * hidden_size + k));

            let i_gate = crate::kernels::avx::math::avx2_sigmoid_ps(i_raw);
            let o_gate = crate::kernels::avx::math::avx2_sigmoid_ps(o_raw);
            let f_gate = crate::kernels::avx::math::avx2_sigmoid_ps(f_raw);
            let c_gate = crate::kernels::avx::math::avx2_tanh_ps(c_raw);

            let prev_c = _mm256_loadu_ps(out_c.as_ptr().add(k));
            // ct = f_gate * prev_c + i_gate * c_gate
            let ct = _mm256_fmadd_ps(f_gate, prev_c, _mm256_mul_ps(i_gate, c_gate));

            // ht = o_gate * tanh(ct)
            let tanh_ct = crate::kernels::avx::math::avx2_tanh_ps(ct);
            let ht = _mm256_mul_ps(o_gate, tanh_ct);

            _mm256_storeu_ps(out_c.as_mut_ptr().add(k), ct);
            _mm256_storeu_ps(out_h.as_mut_ptr().add(k), ht);
            _mm256_storeu_ps(out_y.as_mut_ptr().add(t * hidden_size + k), ht);

            k += 8;
        }
        while k < hidden_size {
            let i_gate = sigmoid(gates[k]);
            let o_gate = sigmoid(gates[hidden_size + k]);
            let f_gate = sigmoid(gates[2 * hidden_size + k]);
            let c_gate = tanh(gates[3 * hidden_size + k]);
            let ct = f_gate * out_c[k] + i_gate * c_gate;
            let ht = o_gate * tanh(ct);
            out_c[k] = ct;
            out_h[k] = ht;
            out_y[t * hidden_size + k] = ht;
            k += 1;
        }
    }
}

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
    let seq_len = input.shape[0];
    let batch_size = input.shape[1];
    let input_size = input.shape[2];

    let num_directions = w.shape[0];
    let hidden_size = w.shape[1] / 4;
    if num_directions != 1 {
        panic!("LSTM: Only num_directions=1 supported");
    }
    if batch_size != 1 {
        panic!("LSTM: Only batch_size=1 supported");
    }
    let w_data = &w.data;
    let r_data = &r.data;
    let default_bias = vec![0.0; 8 * hidden_size];
    let bias_data: &[f32] = if let Some(b) = bias {
        b.data.as_ref()
    } else {
        default_bias.as_slice()
    };
    let (bias_w_slice, bias_r_slice) = bias_data.split_at(4 * hidden_size);

    out_h.resize(hidden_size, 0.0);
    if let Some(h) = initial_h {
        out_h.copy_from_slice(&h.data);
    } else {
        out_h.fill(0.0);
    }
    out_c.resize(hidden_size, 0.0);
    if let Some(c) = initial_c {
        out_c.copy_from_slice(&c.data);
    } else {
        out_c.fill(0.0);
    }
    out_y.resize(seq_len * hidden_size, 0.0);

    let mut gates = vec![0.0; 4 * hidden_size];
    let mut w_contribution = vec![0.0; 4 * hidden_size];
    let mut r_contribution = vec![0.0; 4 * hidden_size];

    for t in 0..seq_len {
        let input_offset = t * input_size;
        let x_t = &input.data[input_offset..input_offset + input_size];

        unsafe {
            // W * x_t
            // W: [4*H, I] row-major.
            // x_t: [I]
            // out: [4*H]
            let m = 4 * hidden_size;
            let k = input_size;
            let a = MatRef::<f32>::from_raw_parts(w_data.as_ptr(), m, k, k as isize, 1);
            let b = MatRef::<f32>::from_raw_parts(x_t.as_ptr(), k, 1, 1, 1);
            let c = MatMut::<f32>::from_raw_parts_mut(w_contribution.as_mut_ptr(), m, 1, 1, 1);

            matmul(c, Accum::Replace, a, b, 1.0, Par::Seq);

            // R * h_{t-1}
            // R: [4*H, H]
            // out_h: [H]
            let a = MatRef::<f32>::from_raw_parts(
                r_data.as_ptr(),
                m,
                hidden_size,
                hidden_size as isize,
                1,
            );
            let b = MatRef::<f32>::from_raw_parts(out_h.as_ptr(), hidden_size, 1, 1, 1);
            let c = MatMut::<f32>::from_raw_parts_mut(r_contribution.as_mut_ptr(), m, 1, 1, 1);

            matmul(c, Accum::Replace, a, b, 1.0, Par::Seq);
        }

        for g in 0..(4 * hidden_size) {
            gates[g] = w_contribution[g] + r_contribution[g] + bias_w_slice[g] + bias_r_slice[g];
        }

        #[cfg(all(target_arch = "aarch64", nightly_build))]
        {
            use crate::kernels::neon::math::{simd_sigmoid, simd_tanh};
            use std::simd::f32x4;

            let mut k = 0;
            while k + 4 <= hidden_size {
                let i_gate_in = f32x4::from_slice(&gates[k..k + 4]);
                let o_gate_in = f32x4::from_slice(&gates[hidden_size + k..hidden_size + k + 4]);
                let f_gate_in =
                    f32x4::from_slice(&gates[2 * hidden_size + k..2 * hidden_size + k + 4]);
                let c_gate_in =
                    f32x4::from_slice(&gates[3 * hidden_size + k..3 * hidden_size + k + 4]);

                let i_gate = simd_sigmoid(i_gate_in);
                let o_gate = simd_sigmoid(o_gate_in);
                let f_gate = simd_sigmoid(f_gate_in);
                let c_gate = simd_tanh(c_gate_in);

                let prev_c = f32x4::from_slice(&out_c[k..k + 4]);
                let ct = f_gate * prev_c + i_gate * c_gate;
                let ht = o_gate * simd_tanh(ct);

                ct.copy_to_slice(&mut out_c[k..k + 4]);
                ht.copy_to_slice(&mut out_h[k..k + 4]);
                ht.copy_to_slice(&mut out_y[t * hidden_size + k..t * hidden_size + k + 4]);

                k += 4;
            }
            // Remainder
            while k < hidden_size {
                let i_gate = sigmoid(gates[k]);
                let o_gate = sigmoid(gates[hidden_size + k]);
                let f_gate = sigmoid(gates[2 * hidden_size + k]);
                let c_gate = tanh(gates[3 * hidden_size + k]);
                let ct = f_gate * out_c[k] + i_gate * c_gate;
                let ht = o_gate * tanh(ct);
                out_c[k] = ct;
                out_h[k] = ht;
                out_y[t * hidden_size + k] = ht;
                k += 1;
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                lstm_gates_avx2(&gates, out_c, out_h, out_y, hidden_size, t);
            }
        }

        #[cfg(any(
            all(target_arch = "aarch64", not(nightly_build)),
            not(any(target_arch = "aarch64", target_arch = "x86_64"))
        ))]
        for k in 0..hidden_size {
            let i_gate = sigmoid(gates[k]);
            let o_gate = sigmoid(gates[hidden_size + k]);
            let f_gate = sigmoid(gates[2 * hidden_size + k]);
            let c_gate = tanh(gates[3 * hidden_size + k]);
            let ct = f_gate * out_c[k] + i_gate * c_gate;
            let ht = o_gate * tanh(ct);
            out_c[k] = ct;
            out_h[k] = ht;
            out_y[t * hidden_size + k] = ht;
        }
    }
    let output_y_shape = vec![seq_len, 1, 1, hidden_size];
    let output_y = TensorView::from_slice(out_y, output_y_shape);
    let output_h_shape = vec![1, 1, hidden_size];
    let output_h = TensorView::from_slice(out_h, output_h_shape);
    let output_c_shape = vec![1, 1, hidden_size];
    let output_c = TensorView::from_slice(out_c, output_c_shape);
    (output_y, output_h, output_c)
}

/// GRU (Gated Recurrent Unit) forward pass.
///
/// Inputs:
///   input:  [seq_len, batch_size, input_size]
///   w:      [1, 3*hidden_size, input_size]  (Weights for update/reset/candidate gates)
///   r:      [1, 3*hidden_size, hidden_size] (Recurrent weights)
///   bias:   optional [1, 6*hidden_size]     (bias_w + bias_r concatenated)
///   initial_h: optional [1, 1, hidden_size]
///
/// linear_before_reset: if true, apply bias_r before reset gate multiplication
///
/// Outputs: (Y, H_n)
///   Y:     [seq_len, 1, 1, hidden_size]
///   H_n:   [1, 1, hidden_size]
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
    let seq_len = input.shape[0];
    let batch_size = input.shape[1];
    let input_size = input.shape[2];

    let num_directions = w.shape[0];
    let hidden_size = w.shape[1] / 3;
    if num_directions != 1 {
        panic!("GRU: Only num_directions=1 supported");
    }
    if batch_size != 1 {
        panic!("GRU: Only batch_size=1 supported");
    }
    let w_data = &w.data;
    let r_data = &r.data;
    let default_bias = vec![0.0; 6 * hidden_size];
    let bias_data: &[f32] = if let Some(b) = bias {
        b.data.as_ref()
    } else {
        default_bias.as_slice()
    };
    let (bias_w_slice, bias_r_slice) = bias_data.split_at(3 * hidden_size);

    out_h.resize(hidden_size, 0.0);
    if let Some(h) = initial_h {
        out_h.copy_from_slice(&h.data);
    } else {
        out_h.fill(0.0);
    }
    out_y.resize(seq_len * hidden_size, 0.0);

    let mut gates = vec![0.0f32; 3 * hidden_size];
    let mut w_contribution = vec![0.0f32; 3 * hidden_size];
    let mut r_contribution = vec![0.0f32; 3 * hidden_size];

    let m = 3 * hidden_size;

    for t in 0..seq_len {
        let input_offset = t * input_size;
        let x_t = &input.data[input_offset..input_offset + input_size];

        unsafe {
            let a = MatRef::<f32>::from_raw_parts(w_data.as_ptr(), m, input_size, input_size as isize, 1);
            let b = MatRef::<f32>::from_raw_parts(x_t.as_ptr(), input_size, 1, 1, 1);
            let c = MatMut::<f32>::from_raw_parts_mut(w_contribution.as_mut_ptr(), m, 1, 1, 1);
            matmul(c, Accum::Replace, a, b, 1.0, Par::Seq);

            let a = MatRef::<f32>::from_raw_parts(r_data.as_ptr(), m, hidden_size, hidden_size as isize, 1);
            let b = MatRef::<f32>::from_raw_parts(out_h.as_ptr(), hidden_size, 1, 1, 1);
            let c = MatMut::<f32>::from_raw_parts_mut(r_contribution.as_mut_ptr(), m, 1, 1, 1);
            matmul(c, Accum::Replace, a, b, 1.0, Par::Seq);
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe { gru_gate_fusion_avx2(
                    &w_contribution, &r_contribution, bias_w_slice, bias_r_slice,
                    out_h, &mut out_y[t * hidden_size..], hidden_size, linear_before_reset,
                ); }
                continue;
            }
        }

        for g in 0..m {
            gates[g] = w_contribution[g] + r_contribution[g] + bias_w_slice[g] + bias_r_slice[g];
        }

        let (z_wx, r_wx_hx) = gates.split_at(hidden_size);
        let (r_wx, h_wx) = r_wx_hx.split_at(hidden_size);

        if linear_before_reset {
            for k in 0..hidden_size {
                let z_gate = sigmoid(z_wx[k]);
                let r_gate = sigmoid(r_wx[k]);
                let r_bias_r = r_contribution[hidden_size + k] + bias_r_slice[hidden_size + k];
                let h_pre = h_wx[k] + r_gate * r_bias_r;
                let h_gate = tanh(h_pre);
                let ht = (1.0 - z_gate) * h_gate + z_gate * out_h[k];
                out_h[k] = ht;
                out_y[t * hidden_size + k] = ht;
            }
        } else {
            for k in 0..hidden_size {
                let z_gate = sigmoid(z_wx[k]);
                let r_gate = sigmoid(r_wx[k]);
                let wh_x_bh = w_contribution[2 * hidden_size + k] + bias_w_slice[2 * hidden_size + k];
                let r_rh = r_gate * (r_contribution[2 * hidden_size + k] + bias_r_slice[2 * hidden_size + k]);
                let h_pre = wh_x_bh + r_rh;
                let h_gate = tanh(h_pre);
                let ht = (1.0 - z_gate) * h_gate + z_gate * out_h[k];
                out_h[k] = ht;
                out_y[t * hidden_size + k] = ht;
            }
        }
    }

    let output_y_shape = vec![seq_len, 1, 1, hidden_size];
    let output_y = TensorView::from_slice(out_y, output_y_shape);
    let output_h_shape = vec![1, 1, hidden_size];
    let output_h = TensorView::from_slice(out_h, output_h_shape);
    (output_y, output_h)
}

#[cfg(target_arch = "x86_64")]
unsafe fn gru_gate_fusion_avx2(
    w_cont: &[f32],
    r_cont: &[f32],
    bias_w: &[f32],
    bias_r: &[f32],
    h: &mut [f32],
    y_out: &mut [f32],
    hidden_size: usize,
    _linear_before_reset: bool,
) {
    use std::arch::x86_64::*;

    let one = _mm256_set1_ps(1.0);

    let mut k = 0usize;
    while k + 8 <= hidden_size {
        // z gate: sigmoid(w_cont[k] + r_cont[k] + bias_w[k] + bias_r[k])
        let wz = _mm256_loadu_ps(w_cont.as_ptr().add(k));
        let wr = _mm256_loadu_ps(r_cont.as_ptr().add(k));
        let bwz = _mm256_loadu_ps(bias_w.as_ptr().add(k));
        let bwr = _mm256_loadu_ps(bias_r.as_ptr().add(k));
        let z_pre = _mm256_add_ps(_mm256_add_ps(wz, wr), _mm256_add_ps(bwz, bwr));
        let z_gate = crate::kernels::avx::math::avx2_sigmoid_ps(z_pre);

        // r gate
        let k2 = k + hidden_size;
        let wr2 = _mm256_loadu_ps(w_cont.as_ptr().add(k2));
        let rr2 = _mm256_loadu_ps(r_cont.as_ptr().add(k2));
        let bwr2 = _mm256_loadu_ps(bias_w.as_ptr().add(k2));
        let brr2 = _mm256_loadu_ps(bias_r.as_ptr().add(k2));
        let r_pre = _mm256_add_ps(_mm256_add_ps(wr2, rr2), _mm256_add_ps(bwr2, brr2));
        let r_gate = crate::kernels::avx::math::avx2_sigmoid_ps(r_pre);

        // h gate
        let k3 = k + 2 * hidden_size;
        let wh_x = _mm256_add_ps(
            _mm256_loadu_ps(w_cont.as_ptr().add(k3)),
            _mm256_loadu_ps(bias_w.as_ptr().add(k3)),
        );
        let r_rh = _mm256_mul_ps(
            r_gate,
            _mm256_add_ps(
                _mm256_loadu_ps(r_cont.as_ptr().add(k3)),
                _mm256_loadu_ps(bias_r.as_ptr().add(k3)),
            ),
        );
        let h_pre = _mm256_add_ps(wh_x, r_rh);
        let h_gate = crate::kernels::avx::math::avx2_tanh_ps(h_pre);

        // ht = (1-z)*h_gate + z*h_prev
        let h_prev = _mm256_loadu_ps(h.as_ptr().add(k));
        let ht = _mm256_fmadd_ps(_mm256_sub_ps(one, z_gate), h_gate, _mm256_mul_ps(z_gate, h_prev));
        _mm256_storeu_ps(h.as_mut_ptr().add(k), ht);
        _mm256_storeu_ps(y_out.as_mut_ptr().add(k), ht);

        k += 8;
    }
    while k < hidden_size {
        let z_pre = w_cont[k] + r_cont[k] + bias_w[k] + bias_r[k];
        let z_gate = sigmoid(z_pre);
        let r_pre = w_cont[k + hidden_size] + r_cont[k + hidden_size]
            + bias_w[k + hidden_size] + bias_r[k + hidden_size];
        let r_gate = sigmoid(r_pre);
        let wh_x_bh = w_cont[k + 2 * hidden_size] + bias_w[k + 2 * hidden_size];
        let r_rh = r_gate * (r_cont[k + 2 * hidden_size] + bias_r[k + 2 * hidden_size]);
        let h_pre = wh_x_bh + r_rh;
        let h_gate = tanh(h_pre);
        let ht = (1.0 - z_gate) * h_gate + z_gate * h[k];
        h[k] = ht;
        y_out[k] = ht;
        k += 1;
    }
}
