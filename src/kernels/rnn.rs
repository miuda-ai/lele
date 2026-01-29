use crate::kernels::activations::{sigmoid, tanh};
use crate::tensor::TensorView;
use faer::linalg::matmul::matmul;
use faer::mat::{from_raw_parts, from_raw_parts_mut};
use faer::Parallelism;

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
            let a = from_raw_parts::<f32>(w_data.as_ptr(), m, k, k as isize, 1);
            let b = from_raw_parts::<f32>(x_t.as_ptr(), k, 1, 1, 1);
            let c = from_raw_parts_mut::<f32>(w_contribution.as_mut_ptr(), m, 1, 1, 1);

            matmul(c, a, b, None, 1.0, Parallelism::None);

            // R * h_{t-1}
            // R: [4*H, H]
            // out_h: [H]
            let a = from_raw_parts::<f32>(r_data.as_ptr(), m, hidden_size, hidden_size as isize, 1);
            let b = from_raw_parts::<f32>(out_h.as_ptr(), hidden_size, 1, 1, 1);
            let c = from_raw_parts_mut::<f32>(r_contribution.as_mut_ptr(), m, 1, 1, 1);

            matmul(c, a, b, None, 1.0, Parallelism::None);
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

        #[cfg(not(target_arch = "aarch64"))]
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
