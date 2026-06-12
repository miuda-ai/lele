use lele::tensor::TensorView;
use std::fs;

fn main() {
    let read_f32 = |path: &str| -> Vec<f32> {
        fs::read(path).unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };

    let x = read_f32("/tmp/test_ct_x.bin");
    let y_ref = read_f32("/tmp/test_ct_y.bin");
    let w = read_f32("/tmp/test_ct_w.bin");

    let x_tv = TensorView::new(&x, &[1, 32, 50]);
    let w_tv = TensorView::new(&w, &[32, 1, 12]);

    let mut out = Vec::new();
    let result = lele::kernels::conv_transpose(
        &x_tv, &w_tv, None, &[1], 32, &[0, 0], &[2], &mut out,
    );

    println!("lele shape: [{}, {}, {}]", result.shape[0], result.shape[1], result.shape[2]);

    let mut max_err = 0.0f32;
    for i in 0..y_ref.len().min(result.data.len()) {
        max_err = max_err.max((result.data[i] - y_ref[i]).abs());
    }
    println!("Max error: {:.6e}", max_err);
    println!("{}", if max_err < 1e-4 { "PASS" } else { "FAIL" });

    if max_err >= 1e-4 {
        let l_out = result.shape[2];
        for i in 0..y_ref.len().min(result.data.len()) {
            if (result.data[i] - y_ref[i]).abs() > 1e-4 {
                println!("  [ch={}, pos={}] lele={:.6} ref={:.6}",
                    i / l_out, i % l_out, result.data[i], y_ref[i]);
                break;
            }
        }
    }
}
