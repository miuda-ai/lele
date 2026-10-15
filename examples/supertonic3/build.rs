fn main() -> lele_build::Result {
    lele_build::build_models(
        "model.toml",
        "src",
        &[
            ("text_encoder.onnx", "TextEncoder"),
            ("duration_predictor.onnx", "DurationPredictor"),
            ("vector_estimator.onnx", "VectorEstimator"),
            ("vocoder.onnx", "Vocoder"),
        ],
    )
}
