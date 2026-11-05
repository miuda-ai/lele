fn main() -> lele_build::Result {
    lele_build::build_models(
        "../../../../examples/supertonic/model.toml",
        "src/gen",
        &[
            ("text_encoder.onnx", "TextEncoder"),
            ("duration_predictor.onnx", "DurationPredictor"),
            ("vector_estimator.onnx", "VectorEstimator"),
            ("vocoder.onnx", "Vocoder"),
        ],
    )
}
