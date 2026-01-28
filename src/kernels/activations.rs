pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
pub fn tanh(x: f32) -> f32 {
    x.tanh()
}
pub fn hard_sigmoid(x: f32, alpha: f32, beta: f32) -> f32 {
    (alpha * x + beta).clamp(0.0, 1.0)
}
pub fn relu_scalar(x: f32) -> f32 {
    x.max(0.0)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_sigmoid() {
        let y = sigmoid(0.0);
        assert!((y - 0.5).abs() < 1e-6);
    }
    #[test]
    fn test_hard_sigmoid() {
        assert_eq!(hard_sigmoid(0.0, 0.2, 0.5), 0.5);
        assert_eq!(hard_sigmoid(3.0, 0.2, 0.5), 1.0);
        assert_eq!(hard_sigmoid(-3.0, 0.2, 0.5), 0.0);
    }
}
