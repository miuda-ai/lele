// Hojo-TTS-Light is a Token-LM TTS with three ONNX sub-models.
fn main() -> lele_build::Result {
    lele_build::build_models(
        "model.toml",
        "src",
        &[
            ("onnx/Hojo-TTS-Light-encoder.onnx", "Encoder"),
            ("onnx/Hojo-TTS-Light-llm.onnx", "Llm"),
            ("onnx/Hojo-TTS-Light-decoder.onnx", "Decoder"),
        ],
    )
}
