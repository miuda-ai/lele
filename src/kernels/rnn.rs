use crate::kernels::activations::{sigmoid, tanh};
use crate::tensor::TensorView;
#[cfg(not(target_arch = "wasm32"))]
use faer::linalg::matmul::matmul;
#[cfg(not(target_arch = "wasm32"))]
use faer::mat::{MatMut, MatRef};
#[cfg(not(target_arch = "wasm32"))]
use faer::{Accum, Par};
#[cfg(target_arch = "wasm32")]
use crate::kernels::wasm_matmul::{matmul, MatMut, MatRef, Accum, Par};

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

        #[cfg(target_arch = "aarch64")]
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

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
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
