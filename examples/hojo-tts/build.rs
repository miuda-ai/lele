use lele_build::*;
use std::path::Path;

// Hojo-TTS-Light is a Token-LM TTS with three ONNX sub-models.
// Each sub-model is compiled independently; if compilation fails (e.g. an
// operator lele does not yet support) we emit a stub with the exact same
// `forward` signature so the crate still builds and the failure is reported
// clearly at runtime.
fn generate_custom_stub(
    class_name: &str,
    output_dir: &Path,
    error_msg: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;
    fs::create_dir_all(output_dir)?;
    let rs_path = output_dir.join(format!("{}.rs", class_name.to_lowercase()));
    let weights_path = output_dir.join(format!("{}_weights.bin", class_name.to_lowercase()));

    // The forward signature is chosen to match the real generated code for the
    // ONNX graph of each sub-model (input/output element types differ).
    let forward = match class_name {
        // Encoder: fp16/f32 wav [1,1,T] -> int64 vq_codes
        "Encoder" => {
            "    pub fn forward<'a>(&self, _wav: lele::tensor::TensorView<'a>) -> lele::tensor::TensorView<'static, i64> { panic!(\"Model is not available: ERROR_MSG\") }"
        }
        // LLM: int64 input_ids [1,T] -> f32 logits [1,T,V]
        "Llm" => {
            "    pub fn forward<'a>(&self, _input_ids: lele::tensor::TensorView<'a, i64>) -> lele::tensor::TensorView<'static> { panic!(\"Model is not available: ERROR_MSG\") }"
        }
        // Decoder: int64 vq_codes [1,1,T] -> f32 wav [1,1,S]
        "Decoder" => {
            "    pub fn forward<'a>(&self, _vq_codes: lele::tensor::TensorView<'a, i64>) -> lele::tensor::TensorView<'static> { panic!(\"Model is not available: ERROR_MSG\") }"
        }
        _ => "",
    };

    let code = format!(
        r#"// Auto-generated stub (model generation failed)
// Error: {error_msg}
use std::marker::PhantomData;

#[allow(dead_code)]
pub struct {class_name}<'a> {{
    _marker: PhantomData<&'a ()>,
}}

#[allow(dead_code)]
impl<'a> {class_name}<'a> {{
    pub fn new(_weights: &'a [u8]) -> Self {{
        panic!("Model is not available: {error_msg}")
    }}
{forward}
}}
"#,
        error_msg = error_msg,
        class_name = class_name,
        forward = forward.replace("ERROR_MSG", error_msg),
    );

    fs::write(rs_path, code)?;
    fs::write(weights_path, &[])?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=model.toml");

    if should_skip_codegen() {
        println!("cargo:warning=Skipping model code generation (LELE_SKIP_MODEL_GEN=1)");
        return Ok(());
    }

    let config = match config::ModelConfig::load("model.toml") {
        Ok(cfg) => cfg,
        Err(e) => {
            println!("cargo:warning=Failed to load model.toml: {}", e);
            return Ok(());
        }
    };

    // 1. Download/get model + tokenizer files.
    match &config.model {
        config::ModelSource::Local { files } => {
            for file_spec in files {
                if !Path::new(&file_spec.path).exists() {
                    println!("cargo:warning=Local model file not found: {}", file_spec.path);
                }
                if let Some(dest) = &file_spec.dest {
                    let dest_path = Path::new(dest);
                    if dest_path != Path::new(&file_spec.path) && !dest_path.exists() {
                        if let Some(parent) = dest_path.parent() {
                            std::fs::create_dir_all(parent)?;
                        }
                        std::fs::copy(&file_spec.path, dest_path)?;
                    }
                }
            }
        }
        config::ModelSource::HuggingFaceHub {
            repo,
            revision,
            files,
        } => {
            println!(
                "cargo:warning=Downloading {} files from Hugging Face Hub: {}",
                files.len(),
                repo
            );
            for file_spec in files {
                let dest_name = file_spec
                    .dest
                    .as_ref()
                    .map(|s| s.as_str())
                    .unwrap_or(&file_spec.file);
                let dest_path = Path::new(dest_name);
                if dest_path.exists() && !should_force_regenerate() {
                    println!("cargo:warning=File already exists, skipping: {}", dest_name);
                    continue;
                }
                if let Some(parent) = dest_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                match download_from_hf_hub(
                    repo,
                    &file_spec.file,
                    revision.as_deref(),
                    Some(&get_model_cache_dir()),
                ) {
                    Ok(cached_path) => {
                        if cached_path != dest_path {
                            std::fs::copy(&cached_path, dest_path)?;
                        }
                        println!("cargo:warning=Downloaded: {} -> {}", file_spec.file, dest_name);
                    }
                    Err(e) => {
                        println!("cargo:warning=Failed to download {}: {}", file_spec.file, e);
                    }
                }
            }
        }
        config::ModelSource::Url { files } => {
            for file_spec in files {
                let dest_path = Path::new(&file_spec.dest);
                if dest_path.exists() && !should_force_regenerate() {
                    continue;
                }
                if let Some(parent) = dest_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                if let Err(e) = download_from_url(&file_spec.url, dest_path) {
                    println!("cargo:warning=Failed to download from {}: {}", file_spec.url, e);
                }
            }
        }
    }

    let output_dir = Path::new("src");

    // 2. Generate code for each ONNX sub-model (or a signature-matched stub).
    let models = [
        ("onnx/Hojo-TTS-Light-encoder.onnx", "Encoder"),
        ("onnx/Hojo-TTS-Light-llm.onnx", "Llm"),
        ("onnx/Hojo-TTS-Light-decoder.onnx", "Decoder"),
    ];

    for (onnx_name, class_name) in models {
        let onnx_path = Path::new(onnx_name);
        if !onnx_path.exists() {
            println!("cargo:warning=ONNX file not found: {}", onnx_name);
            generate_custom_stub(class_name, output_dir, &format!("{} not found", onnx_name))?;
            continue;
        }
        let rs_path = output_dir.join(format!("{}.rs", class_name.to_lowercase()));
        if rs_path.exists() && std::env::var("LELE_SKIP_GEN").is_ok() {
            println!("cargo:warning=Skipping code generation for {} (LELE_SKIP_GEN)", class_name);
            continue;
        }
        println!("cargo:warning=Generating code for {}...", class_name);
        match generate_model_code(onnx_path, class_name, output_dir, &[]) {
            Ok(()) => println!("cargo:warning=Code generation successful for {}", class_name),
            Err(e) => {
                println!(
                    "cargo:warning=Failed to generate code for {}: {}",
                    class_name, e
                );
                generate_custom_stub(class_name, output_dir, &e.to_string())?;
            }
        }
    }

    Ok(())
}
