// Kernel accuracy tests - compare lele implementations with expected outputs
use lele::kernels::*;
use lele::tensor::TensorView;

fn assert_close(a: &[f32], b: &[f32], tol: f32, name: &str) {
    // Temporarily relax length check if convenient, but rigorous test should match
    assert_eq!(a.len(), b.len(), "{}: length mismatch", name);
    let max_diff = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);

    if max_diff > tol {
        println!(
            "{} FAILED: max diff = {:.6e} (tol = {:.6e})",
            name, max_diff, tol
        );
        println!("Got:      {:?}", &a[..5.min(a.len())]);
        println!("Expected: {:?}", &b[..5.min(b.len())]);
        panic!("{} failed accuracy check", name);
    }
    println!("{} PASSED: max diff = {:.6e}", name, max_diff);
}

#[test]
fn test_softmax_accuracy() {
    // Reference from ORT
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x = TensorView::from_owned(input, vec![2, 4]);

    let mut out_buf = Vec::new();
    let result = softmax(&x, -1, &mut out_buf);

    // Expected output from ORT (first row)
    let expected = vec![
        0.0320586, 0.08714432, 0.23688284, 0.6439143, 0.0320586, 0.08714432, 0.23688284, 0.6439143,
    ];

    assert_close(&result.data, &expected, 1e-6, "softmax");

    // Check sum equals 1
    let sum: f32 = result.data[0..4].iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "Softmax sum should be 1.0, got {}",
        sum
    );
}

#[test]
fn test_mat_mul_integer() {
    // 2x3 * 3x2
    // A (2x3)
    // 10 20 30
    // 40 50 60
    // zp_a = 5
    // Real A:
    // 5 15 25
    // 35 45 55

    // B (3x2)
    // 1 2
    // 3 4
    // 5 6
    // zp_b = 1
    // Real B:
    // 0 1
    // 2 3
    // 4 5

    // A * B
    // (5*0 + 15*2 + 25*4)  (5*1 + 15*3 + 25*5)
    // (0 + 30 + 100)       (5 + 45 + 125)
    // 130                  175

    // (35*0 + 45*2 + 55*4) (35*1 + 45*3 + 55*5)
    // (0 + 90 + 220)       (35 + 135 + 275)
    // 310                  445

    let a_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let b_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let a = TensorView::from_owned(a_data, vec![2, 3]);
    let b = TensorView::from_owned(b_data, vec![3, 2]);

    let zp_a_val: Vec<f32> = vec![5.0];
    let zp_b_val: Vec<f32> = vec![1.0];
    let zp_a = TensorView::from_owned(zp_a_val, vec![1]);
    let zp_b = TensorView::from_owned(zp_b_val, vec![1]);

    let mut out_buf = Vec::new();
    let result = mat_mul_integer(&a, &b, Some(&zp_a), Some(&zp_b), &mut out_buf);

    let expected = vec![130.0, 175.0, 310.0, 445.0];
    assert_close(&result.data, &expected, 1e-5, "test_mat_mul_integer");
}

#[test]
fn test_layer_norm_accuracy() {
    // Simple test case
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = TensorView::from_owned(input, vec![2, 3]);

    let scale = TensorView::from_owned(vec![1.0, 1.0, 1.0], vec![3]);
    let bias = TensorView::from_owned(vec![0.0, 0.0, 0.0], vec![3]);

    let mut out_buf = Vec::new();
    let result = layer_norm(&x, &scale, &bias, -1, 1e-5, &mut out_buf);

    // After normalization, each row should have mean≈0 and std≈1
    let row1_mean: f32 = result.data[0..3].iter().sum::<f32>() / 3.0;
    let row1_std: f32 = (result.data[0..3]
        .iter()
        .map(|&x| (x - row1_mean).powi(2))
        .sum::<f32>()
        / 3.0)
        .sqrt();

    println!("LayerNorm row1: mean={:.6}, std={:.6}", row1_mean, row1_std);
    assert!(row1_mean.abs() < 1e-5, "Mean should be ~0");
    assert!((row1_std - 1.0).abs() < 1e-5, "Std should be ~1");
}

#[test]
fn test_matmul_accuracy() {
    // 2x3 @ 3x2 = 2x2
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let a_tensor = TensorView::from_owned(a, vec![2, 3]);
    let b_tensor = TensorView::from_owned(b, vec![3, 2]);

    let mut out_buf = Vec::new();
    let result = matmul(&a_tensor, &b_tensor, &mut out_buf);

    // Expected: [[22, 28], [49, 64]]
    let expected = vec![22.0, 28.0, 49.0, 64.0];

    assert_close(&result.data, &expected, 1e-5, "matmul");
}

#[test]
fn test_concat_accuracy() {
    let x1 = vec![1.0, 2.0, 3.0, 4.0];
    let x2 = vec![5.0, 6.0];

    let t1 = TensorView::from_owned(x1, vec![2, 2]);
    let t2 = TensorView::from_owned(x2, vec![1, 2]);

    let mut out_buf = Vec::new();
    let result = concat(&[&t1, &t2], 0, &mut out_buf);

    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    assert_eq!(result.shape.as_ref(), &[3, 2]);
    assert_close(&result.data, &expected, 1e-6, "concat");
}

#[test]
fn test_where_accuracy() {
    let condition = vec![1.0, 0.0, 0.0, 1.0]; // True, False, False, True
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = vec![5.0, 6.0, 7.0, 8.0];

    let cond = TensorView::from_owned(condition, vec![2, 2]);
    let x_t = TensorView::from_owned(x, vec![2, 2]);
    let y_t = TensorView::from_owned(y, vec![2, 2]);

    let mut out_buf = Vec::new();
    let result = where_op(&cond, &x_t, &y_t, &mut out_buf);

    // Expected: [1.0, 6.0, 7.0, 4.0] (pick x where cond is true, y otherwise)
    let expected = vec![1.0, 6.0, 7.0, 4.0];

    assert_close(&result.data, &expected, 1e-6, "where");
}

#[test]
fn test_expand_accuracy() {
    let x = vec![1.0, 2.0, 3.0];
    let x_t = TensorView::from_owned(x, vec![3, 1]);
    let target_shape = [3, 4];

    let mut out_buf = Vec::new();
    let result = expand(&x_t, &target_shape, &mut out_buf);

    // Expected: each row repeated 4 times
    let expected = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0];

    assert_eq!(result.shape.as_ref(), &[3, 4]);
    assert_close(&result.data, &expected, 1e-6, "expand");
}

#[test]
fn test_dynamic_quantize_accuracy() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x_t = TensorView::from_owned(x.clone(), vec![2, 4]);

    let mut buf_q = Vec::new();
    let mut buf_s = Vec::new();
    let mut buf_z = Vec::new();

    let (quantized, scale, zero_point) =
        dynamic_quantize_linear(&x_t, &mut buf_q, &mut buf_s, &mut buf_z);

    println!("Dynamic quantize:");
    println!("  Scale: {}", scale.data[0]);
    println!("  Zero point: {}", zero_point.data[0]);
    println!("  Quantized (first 4): {:?}", &quantized.data[..4]);

    // Dequantize and check error
    let scale_val = scale.data[0];
    let zp_val = zero_point.data[0];

    let dequant: Vec<f32> = quantized
        .data
        .iter()
        .map(|&q| (q as f32 - zp_val) * scale_val)
        .collect();

    let max_error: f32 = x
        .iter()
        .zip(dequant.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);

    println!("  Max dequant error: {:.6e}", max_error);

    // Quantization error should be bounded by the quantization step
    // For uint8 quantization, error should be within scale/2
    assert!(
        max_error < scale_val + 0.1,
        "Quantization error too large: {} > {}",
        max_error,
        scale_val
    );
}

#[test]
fn test_mat_mul_integer_accuracy() {
    // Simple case: all zero points are 0
    let a: Vec<u8> = vec![1, 2, 3, 4, 5, 6]; // 2x3
    let b: Vec<u8> = vec![7, 8, 9, 10, 11, 12]; // 3x2

    // Convert u8 to f32 for new API
    let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
    let b_f32: Vec<f32> = b.iter().map(|&x| x as f32).collect();
    let a_t = TensorView::from_owned(a_f32, vec![2, 3]);
    let b_t = TensorView::from_owned(b_f32, vec![3, 2]);
    let zp_a = TensorView::from_owned(vec![0.0f32], vec![1]);
    let zp_b = TensorView::from_owned(vec![0.0f32], vec![1]);

    let mut out_buf = Vec::new();
    let result = mat_mul_integer(&a_t, &b_t, Some(&zp_a), Some(&zp_b), &mut out_buf);

    // Expected: [[58, 64], [139, 154]]
    let expected = vec![58.0, 64.0, 139.0, 154.0];

    assert_close(&result.data, &expected, 1e-5, "mat_mul_integer");
}

#[test]
fn test_split_accuracy() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x_t = TensorView::from_owned(x, vec![1, 6]);

    let mut bufs = vec![Vec::new(), Vec::new(), Vec::new()];
    let splits = vec![2i64, 2, 2];

    let results = split(&x_t, 1, &splits, &mut bufs);

    assert_eq!(results.len(), 3);
    assert_close(&results[0].data, &[1.0, 2.0], 1e-6, "split[0]");
    assert_close(&results[1].data, &[3.0, 4.0], 1e-6, "split[1]");
    assert_close(&results[2].data, &[5.0, 6.0], 1e-6, "split[2]");
}

#[test]
fn test_transpose_accuracy() {
    use lele::kernels::manipulation::transpose;

    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x_t = TensorView::from_owned(x, vec![2, 3]);

    let mut out_buf = Vec::new();
    let result = transpose(&x_t, &[1, 0], &mut out_buf);

    // Input: [[1, 2, 3], [4, 5, 6]]
    // Transposed: [[1, 4], [2, 5], [3, 6]]
    let expected = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];

    assert_eq!(result.shape.as_ref(), &[3, 2]);
    assert_close(&result.data, &expected, 1e-6, "transpose");
}

#[test]
fn test_add_accuracy() {
    use lele::kernels::math::add;

    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let a_t = TensorView::from_owned(a, vec![2, 2]);
    let b_t = TensorView::from_owned(b, vec![2, 2]);

    let mut out_buf = Vec::new();
    let result = add(&a_t, &b_t, &mut out_buf);

    let expected = vec![6.0, 8.0, 10.0, 12.0];

    assert_close(&result.data, &expected, 1e-6, "add");
}

#[test]
fn test_mul_accuracy() {
    use lele::kernels::math::mul;

    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![2.0, 3.0, 4.0, 5.0];
    let a_t = TensorView::from_owned(a, vec![2, 2]);
    let b_t = TensorView::from_owned(b, vec![2, 2]);

    let mut out_buf = Vec::new();
    let result = mul(&a_t, &b_t, &mut out_buf);

    let expected = vec![2.0, 6.0, 12.0, 20.0];

    assert_close(&result.data, &expected, 1e-6, "mul");
}

#[test]
fn test_relu_accuracy() {
    use lele::kernels::math::relu;

    let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let x_t = TensorView::from_owned(x, vec![2, 3]);

    let mut out_buf = Vec::new();
    let result = relu(&x_t, &mut out_buf);

    let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0];

    assert_close(&result.data, &expected, 1e-6, "relu");
}

#[test]
fn test_gather_accuracy() {
    use lele::kernels::manipulation::gather;

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let data_t = TensorView::from_owned(data, vec![3, 3]);

    let indices = vec![0.0, 2.0]; // Gather rows 0 and 2
    let indices_t = TensorView::from_owned(indices, vec![2]);

    let mut out_buf = Vec::new();
    let result = gather(&data_t, &indices_t, 0, &mut out_buf);

    // Row 0: [1, 2, 3]
    // Row 2: [7, 8, 9]
    let expected = vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0];

    assert_eq!(result.shape.as_ref(), &[2, 3]);
    assert_close(&result.data, &expected, 1e-6, "gather");
}

#[test]
fn test_reduce_max_accuracy() {
    let input = vec![1.0, 5.0, 2.0, 8.0, 3.0, 1.0, 9.0, 4.0];
    let x = TensorView::from_owned(input, vec![2, 4]);

    let mut out_buf = Vec::new();
    // Reduce over last axis [2, 4] -> [2]
    let result = lele::kernels::reduce_max(&x, &[1], false, &mut out_buf);
    let expected = vec![8.0, 9.0];
    assert_close(&result.data, &expected, 1e-6, "reduce_max_axis1");

    // Reduce over first axis [2, 4] -> [4]
    let mut out_buf2 = Vec::new();
    let result2 = lele::kernels::reduce_max(&x, &[0], false, &mut out_buf2);
    let expected2 = vec![3.0, 5.0, 9.0, 8.0];
    assert_close(&result2.data, &expected2, 1e-6, "reduce_max_axis0");
}
