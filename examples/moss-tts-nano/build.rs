use lele_build::*;
use std::path::Path;

fn generate_custom_stub(
    class_name: &str,
    output_dir: &Path,
    error_msg: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;
    fs::create_dir_all(output_dir)?;
    let rs_path = output_dir.join(format!("{}.rs", class_name.to_lowercase()));
    let weights_path = output_dir.join(format!("{}_weights.bin", class_name.to_lowercase()));

    let code = format!(
        r#"
use lele::tensor::TensorView;
use std::marker::PhantomData;

#[allow(dead_code)]
pub struct {name}<'a> {{
    _marker: PhantomData<&'a ()>,
}}

#[allow(dead_code)]
impl<'a> {name}<'a> {{
    pub fn new(_weights: &'a [u8]) -> Self {{
        panic!("Model is not available: {err}");
    }}
}}
"#,
        name = class_name,
        err = error_msg
    );

    fs::write(rs_path, code)?;
    fs::write(weights_path, &[])?;
    Ok(())
}

fn download_model_files(
    repo: &str,
    revision: &str,
    files: &[(&str, &str)],
    model_dir: &Path,
) {
    for (file, dest_rel) in files {
        let dest_path = model_dir.join(dest_rel);
        if dest_path.exists() {
            continue;
        }
        if let Some(parent) = dest_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        match download_from_hf_hub(repo, file, Some(revision), Some(&get_model_cache_dir())) {
            Ok(cached_path) => {
                std::fs::copy(&cached_path, &dest_path).ok();
                println!("cargo:warning=Downloaded: {}/{} -> {}", repo, file, dest_path.display());
            }
            Err(e) => {
                println!("cargo:warning=Failed to download {}/{}: {}", repo, file, e);
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=model.toml");

    if should_skip_codegen() {
        println!("cargo:warning=Skipping model code generation (LELE_SKIP_MODEL_GEN=1)");
        return Ok(());
    }

    let model_dir = Path::new("models");
    let tts_dir = model_dir.join("tts");
    let codec_dir = model_dir.join("codec");

    // Download TTS model files
    let tts_files = [
        ("moss_tts_prefill.onnx", "moss_tts_prefill.onnx"),
        ("moss_tts_global_shared.data", "moss_tts_global_shared.data"),
        ("moss_tts_decode_step.onnx", "moss_tts_decode_step.onnx"),
        ("moss_tts_local_fixed_sampled_frame.onnx", "moss_tts_local_fixed_sampled_frame.onnx"),
        ("moss_tts_local_shared.data", "moss_tts_local_shared.data"),
        ("tokenizer.model", "tokenizer.model"),
        ("tts_browser_onnx_meta.json", "tts_browser_onnx_meta.json"),
        ("browser_poc_manifest.json", "browser_poc_manifest.json"),
    ];
    download_model_files(
        "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX",
        "main",
        &tts_files,
        &tts_dir,
    );

    // Download codec model files
    let codec_files = [
        ("moss_audio_tokenizer_decode_full.onnx", "moss_audio_tokenizer_decode_full.onnx"),
        ("moss_audio_tokenizer_decode_shared.data", "moss_audio_tokenizer_decode_shared.data"),
        ("codec_browser_onnx_meta.json", "codec_browser_onnx_meta.json"),
    ];
    download_model_files(
        "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX",
        "main",
        &codec_files,
        &codec_dir,
    );

    let output_dir = Path::new("src");

    // Models to generate code for
    let models = [
        ("moss_tts_prefill.onnx", &tts_dir, "Prefill"),
        ("moss_tts_decode_step.onnx", &tts_dir, "DecodeStep"),
        ("moss_tts_local_fixed_sampled_frame.onnx", &tts_dir, "LocalFixedSampledFrame"),
        ("moss_audio_tokenizer_decode_full.onnx", &codec_dir, "CodecDecodeFull"),
    ];

    for (onnx_name, dir, class_name) in models {
        let onnx_path = dir.join(onnx_name);
        if !onnx_path.exists() {
            println!("cargo:warning=ONNX file not found: {}", onnx_path.display());
            generate_custom_stub(class_name, output_dir, &format!("{} not found", onnx_path.display()))?;
            continue;
        }

        println!("cargo:warning=Generating code for {}...", class_name);
        if let Err(e) = generate_model_code(&onnx_path, class_name, output_dir, &[]) {
            println!("cargo:warning=Failed to generate code for {}: {}", class_name, e);
            generate_custom_stub(class_name, output_dir, &e.to_string())?;
        }
    }

    Ok(())
}
