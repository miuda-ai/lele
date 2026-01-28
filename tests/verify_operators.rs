use approx::assert_abs_diff_eq;
use lele::kernels::{layer_norm, matmul, softmax};
use lele::tensor::TensorView;

#[test]
fn test_matmul_simple() {
    // A: 2x3, B: 3x2 -> C: 2x2
    // A = [[1, 2, 3], [4, 5, 6]]
    // B = [[7, 8], [9, 10], [11, 12]]
    // C00 = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // C01 = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    // C10 = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    // C11 = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154

    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a_shape = vec![2, 3];
    let a = TensorView::from_owned(a_data, a_shape);

    let b_data = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let b_shape = vec![3, 2];
    let b = TensorView::from_owned(b_data, b_shape);

    let mut out_buf = Vec::new();
    let c = matmul(&a, &b, &mut out_buf);

    assert_eq!(c.shape.as_ref(), &[2, 2]);
    let expected = vec![58.0, 64.0, 139.0, 154.0];
    for (val, exp) in c.data.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(val, exp, epsilon = 1e-5);
    }
}

#[test]
fn test_layernorm_simple() {
    // Input: [1.0, 2.0, 3.0]
    // Mean = 2.0
    // Var = ((1-2)^2 + (2-2)^2 + (3-2)^2)/3 = (1+0+1)/3 = 0.6666
    // Std = sqrt(0.6666) = 0.8165
    // Norm = [-1.2247, 0.0, 1.2247]
    // Scale=1, Bias=0

    let input = TensorView::from_owned(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let scale = TensorView::from_owned(vec![1.0, 1.0, 1.0], vec![3]);
    let bias = TensorView::from_owned(vec![0.0, 0.0, 0.0], vec![3]);

    let mut out_buf = Vec::new();
    let out = layer_norm(&input, &scale, &bias, -1, 1e-5, &mut out_buf);

    let expected = vec![-1.2247356, 0.0, 1.2247356];
    for (val, exp) in out.data.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(val, exp, epsilon = 1e-4);
    }
}

#[test]
fn test_softmax_simple() {
    // Input: [1.0, 2.0, 3.0]
    // Exp: [2.718, 7.389, 20.085]
    // Sum: 30.192
    // Prob: [0.090, 0.244, 0.665]

    let input = TensorView::from_owned(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let mut out_buf = Vec::new();
    let out = softmax(&input, -1, &mut out_buf);

    let expected = vec![0.09003057, 0.24472847, 0.66524096];
    for (val, exp) in out.data.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(val, exp, epsilon = 1e-5);
    }
}
