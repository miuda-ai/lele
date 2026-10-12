use lele_build::*;

const SUPERTONIC_MODELS: &[(&str, &str)] = &[
    ("text_encoder.onnx", "TextEncoder"),
    ("duration_predictor.onnx", "DurationPredictor"),
    ("vector_estimator.onnx", "VectorEstimator"),
    ("vocoder.onnx", "Vocoder"),
];

fn main() -> Result {
    let output_dir = "src/gen";
    // SenseVoice (ASR), Silero VAD, YOLO26 (object detection), Supertonic (TTS)
    build_model("../../examples/sensevoice/model.toml", output_dir)?;
    build_model("../../examples/silero/model.toml", output_dir)?;
    build_model("../../examples/yolo26/model.toml", output_dir)?;
    build_models(
        "../../examples/supertonic/model.toml",
        output_dir,
        SUPERTONIC_MODELS,
    )
}
