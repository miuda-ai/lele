use lele_build::*;
use std::path::{Path, PathBuf};

const TTS_REPO: &str = "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX";
const TTS_FILES: &[&str] = &[
    "moss_tts_prefill.onnx",
    "moss_tts_global_shared.data",
    "moss_tts_decode_step.onnx",
    "moss_tts_local_fixed_sampled_frame.onnx",
    "moss_tts_local_shared.data",
    "tokenizer.model",
    "tts_browser_onnx_meta.json",
    "browser_poc_manifest.json",
];

const CODEC_REPO: &str = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX";
const CODEC_FILES: &[&str] = &[
    "moss_audio_tokenizer_decode_full.onnx",
    "moss_audio_tokenizer_decode_shared.data",
    "codec_browser_onnx_meta.json",
];

fn main() -> Result {
    println!("cargo:rerun-if-changed=model.toml");

    if should_skip_codegen() {
        println!("cargo:warning=Skipping model code generation (LELE_SKIP_MODEL_GEN=1)");
        return Ok(());
    }

    let tts_dir = Path::new("models/tts");
    let codec_dir = Path::new("models/codec");
    let output_dir = Path::new("src");

    let models: [(PathBuf, &str); 4] = [
        (tts_dir.join("moss_tts_prefill.onnx"), "Prefill"),
        (tts_dir.join("moss_tts_decode_step.onnx"), "DecodeStep"),
        (
            tts_dir.join("moss_tts_local_fixed_sampled_frame.onnx"),
            "LocalFixedSampledFrame",
        ),
        (
            codec_dir.join("moss_audio_tokenizer_decode_full.onnx"),
            "CodecDecodeFull",
        ),
    ];
    let stale: Vec<_> = models
        .iter()
        .filter(|(onnx, class_name)| {
            need_regenerate_with_model(class_name, output_dir, "model.toml", Some(onnx))
        })
        .collect();
    if stale.is_empty() {
        println!("cargo:warning=Generated files are up to date, skipping regeneration");
        return Ok(());
    }

    for file in TTS_FILES {
        fetch_from_hf_hub(TTS_REPO, file, Some("main"), &tts_dir.join(file))?;
    }
    for file in CODEC_FILES {
        fetch_from_hf_hub(CODEC_REPO, file, Some("main"), &codec_dir.join(file))?;
    }

    for (onnx, class_name) in stale {
        generate_model_code(onnx, class_name, output_dir, &[])?;
    }

    Ok(())
}
