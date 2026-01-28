use std::f32::consts::PI;
pub fn hann_window(size: usize) -> Vec<f32> {
    if size == 0 {
        return Vec::new();
    }
    if size == 1 {
        return vec![1.0];
    }
    (0..size)
        .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / (size - 1) as f32).cos()))
        .collect()
}
pub fn apply_window(input: &mut [f32], window: &[f32]) {
    assert_eq!(input.len(), window.len());
    for (x, w) in input.iter_mut().zip(window.iter()) {
        *x *= w;
    }
}
