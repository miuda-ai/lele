use lele_build::*;
use std::path::Path;

fn generate_model_stub(
    class_name: &str,
    output_dir: &Path,
    error_msg: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(output_dir)?;
    let rs_filename = format!("{}.rs", class_name.to_lowercase());
    let rs_path = output_dir.join(&rs_filename);

    // Generate API-compatible stubs with forward() methods matching real codegen output
    let stub_code = match class_name {
        "SenseVoice" => format!(
            r#"// Auto-generated stub (model not available)
// Error: {error}
use lele::tensor::TensorView;

#[allow(dead_code)]
pub struct SenseVoiceWorkspace;
#[allow(dead_code)]
impl SenseVoiceWorkspace {{
    pub fn new() -> Self {{ SenseVoiceWorkspace }}
}}
impl Default for SenseVoiceWorkspace {{
    fn default() -> Self {{ Self::new() }}
}}

#[allow(dead_code)]
pub struct SenseVoice<'a> {{
    _marker: std::marker::PhantomData<&'a ()>,
}}

#[allow(dead_code)]
impl<'a> SenseVoice<'a> {{
    pub fn new(_weights: &'a [u8]) -> Self {{
        Self {{ _marker: std::marker::PhantomData }}
    }}

    pub fn forward(
        &self,
        _x: TensorView<'a>,
        _x_length: TensorView<'a, i64>,
        _language: TensorView<'a, i64>,
        _text_norm: TensorView<'a, i64>,
    ) -> TensorView<'static> {{
        panic!("Model not available: {error}")
    }}

    pub fn forward_with_workspace<'w>(
        &self,
        _ws: &'w mut SenseVoiceWorkspace,
        _x: TensorView<'w>,
        _x_length: TensorView<'w, i64>,
        _language: TensorView<'w, i64>,
        _text_norm: TensorView<'w, i64>,
    ) -> TensorView<'w> {{
        panic!("Model not available: {error}")
    }}
}}
"#,
            error = error_msg
        ),
        "Yolo26" => format!(
            r#"// Auto-generated stub (model not available)
// Error: {error}
use lele::tensor::TensorView;

#[allow(dead_code)]
pub struct Yolo26Workspace;
#[allow(dead_code)]
impl Yolo26Workspace {{
    pub fn new() -> Self {{ Yolo26Workspace }}
}}
impl Default for Yolo26Workspace {{
    fn default() -> Self {{ Self::new() }}
}}

#[allow(dead_code)]
pub struct Yolo26<'a> {{
    _marker: std::marker::PhantomData<&'a ()>,
}}

#[allow(dead_code)]
impl<'a> Yolo26<'a> {{
    pub fn new(_weights: &'a [u8]) -> Self {{
        Self {{ _marker: std::marker::PhantomData }}
    }}

    pub fn forward(
        &self,
        _pixel_values: TensorView<'a>,
    ) -> (TensorView<'static>, TensorView<'static>) {{
        panic!("Model not available: {error}")
    }}

    pub fn forward_with_workspace<'w>(
        &self,
        _ws: &'w mut Yolo26Workspace,
        _pixel_values: TensorView<'w>,
    ) -> (TensorView<'w>, TensorView<'w>) {{
        panic!("Model not available: {error}")
    }}
}}
"#,
            error = error_msg
        ),
        "TextEncoder" => format!(
            r#"// Auto-generated stub (model not available)
// Error: {error}
use lele::tensor::TensorView;

#[allow(dead_code)]
pub struct TextEncoderWorkspace;
#[allow(dead_code)]
impl TextEncoderWorkspace {{
    pub fn new() -> Self {{ TextEncoderWorkspace }}
}}

#[allow(dead_code)]
pub struct TextEncoder<'a> {{
    _marker: std::marker::PhantomData<&'a ()>,
}}

#[allow(dead_code)]
impl<'a> TextEncoder<'a> {{
    pub fn new(_weights: &'a [u8]) -> Self {{
        Self {{ _marker: std::marker::PhantomData }}
    }}

    pub fn forward(
        &self,
        _text_ids: TensorView<'a, i64>,
        _style_ttl: TensorView<'a>,
        _text_mask: TensorView<'a>,
    ) -> TensorView<'static> {{
        panic!("Model not available: {error}")
    }}
}}
"#,
            error = error_msg
        ),
        "DurationPredictor" => format!(
            r#"// Auto-generated stub (model not available)
// Error: {error}
use lele::tensor::TensorView;

#[allow(dead_code)]
pub struct DurationPredictorWorkspace;
#[allow(dead_code)]
impl DurationPredictorWorkspace {{
    pub fn new() -> Self {{ DurationPredictorWorkspace }}
}}

#[allow(dead_code)]
pub struct DurationPredictor<'a> {{
    _marker: std::marker::PhantomData<&'a ()>,
}}

#[allow(dead_code)]
impl<'a> DurationPredictor<'a> {{
    pub fn new(_weights: &'a [u8]) -> Self {{
        Self {{ _marker: std::marker::PhantomData }}
    }}

    pub fn forward(
        &self,
        _text_ids: TensorView<'a, i64>,
        _style_dp: TensorView<'a>,
        _text_mask: TensorView<'a>,
    ) -> TensorView<'static> {{
        panic!("Model not available: {error}")
    }}
}}
"#,
            error = error_msg
        ),
        "VectorEstimator" => format!(
            r#"// Auto-generated stub (model not available)
// Error: {error}
use lele::tensor::TensorView;

#[allow(dead_code)]
pub struct VectorEstimatorWorkspace;
#[allow(dead_code)]
impl VectorEstimatorWorkspace {{
    pub fn new() -> Self {{ VectorEstimatorWorkspace }}
}}

#[allow(dead_code)]
pub struct VectorEstimator<'a> {{
    _marker: std::marker::PhantomData<&'a ()>,
}}

#[allow(dead_code)]
impl<'a> VectorEstimator<'a> {{
    pub fn new(_weights: &'a [u8]) -> Self {{
        Self {{ _marker: std::marker::PhantomData }}
    }}

    pub fn forward(
        &self,
        _noisy_latent: TensorView<'a>,
        _text_emb: TensorView<'a>,
        _style_ttl: TensorView<'a>,
        _latent_mask: TensorView<'a>,
        _text_mask: TensorView<'a>,
        _current_step: TensorView<'a>,
        _total_step: TensorView<'a>,
    ) -> TensorView<'static> {{
        panic!("Model not available: {error}")
    }}
}}
"#,
            error = error_msg
        ),
        "Vocoder" => format!(
            r#"// Auto-generated stub (model not available)
// Error: {error}
use lele::tensor::TensorView;

#[allow(dead_code)]
pub struct VocoderWorkspace;
#[allow(dead_code)]
impl VocoderWorkspace {{
    pub fn new() -> Self {{ VocoderWorkspace }}
}}

#[allow(dead_code)]
pub struct Vocoder<'a> {{
    _marker: std::marker::PhantomData<&'a ()>,
}}

#[allow(dead_code)]
impl<'a> Vocoder<'a> {{
    pub fn new(_weights: &'a [u8]) -> Self {{
        Self {{ _marker: std::marker::PhantomData }}
    }}

    pub fn forward(
        &self,
        _latent: TensorView<'a>,
    ) -> TensorView<'static> {{
        panic!("Model not available: {error}")
    }}
}}
"#,
            error = error_msg
        ),
        _ => {
            // Fallback to lele-build's default stub
            return generate_stub(class_name, output_dir, error_msg);
        }
    };

    std::fs::write(&rs_path, stub_code)?;

    // Create empty weights file
    let weights_filename = format!("{}_weights.bin", class_name.to_lowercase());
    let bin_path = output_dir.join(&weights_filename);
    if !bin_path.exists() {
        std::fs::write(bin_path, &[])?;
    }

    Ok(())
}

/// Generate code for a single model, downloading if needed.
fn process_model(
    model_toml_path: &str,
    model_file: &str,
    class_name: &str,
    output_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = match config::ModelConfig::load(model_toml_path) {
        Ok(cfg) => cfg,
        Err(e) => {
            println!("cargo:warning=Failed to load {}: {}", model_toml_path, e);
            generate_model_stub(
                class_name,
                output_dir,
                &format!("Config load failed: {}", e),
            )?;
            return Ok(());
        }
    };

    // Download model files
    match &config.model {
        config::ModelSource::Local { files } => {
            for file_spec in files {
                let path = Path::new(&file_spec.path);
                if !path.exists() {
                    println!("cargo:warning=Local file not found: {}", file_spec.path);
                }
                if let Some(dest) = &file_spec.dest {
                    let dest_path = Path::new(dest);
                    if dest_path != path {
                        if let Some(parent) = dest_path.parent() {
                            std::fs::create_dir_all(parent)?;
                        }
                        if path.exists() {
                            std::fs::copy(path, dest_path)?;
                        }
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
                "cargo:warning=Downloading {} files from HF: {}",
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
                    println!("cargo:warning=File exists, skipping: {}", dest_name);
                    continue;
                }
                match download_from_hf_hub(
                    repo,
                    &file_spec.file,
                    revision.as_deref(),
                    Some(&get_model_cache_dir()),
                ) {
                    Ok(cached_path) => {
                        println!(
                            "cargo:warning=Downloaded: {} -> {}",
                            file_spec.file, dest_name
                        );
                        if cached_path != dest_path {
                            if let Some(parent) = dest_path.parent() {
                                std::fs::create_dir_all(parent)?;
                            }
                            std::fs::copy(&cached_path, dest_path)?;
                        }
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
                let _ = download_from_url(&file_spec.url, dest_path);
            }
        }
    }

    let model_path = Path::new(model_file);
    if !model_path.exists() {
        println!(
            "cargo:warning=Model file not found: {}, generating stub",
            model_file
        );
        generate_model_stub(
            class_name,
            output_dir,
            &format!("Model file not found: {}", model_file),
        )?;
        return Ok(());
    }

    println!("cargo:warning=Generating code for: {}", class_name);
    match generate_model_code(model_path, class_name, output_dir, &[]) {
        Ok(()) => println!(
            "cargo:warning=Code generation successful for {}",
            class_name
        ),
        Err(e) => {
            println!(
                "cargo:warning=Code generation failed for {}: {}",
                class_name, e
            );
            generate_model_stub(class_name, output_dir, &e.to_string())?;
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");

    if should_skip_codegen() {
        println!("cargo:warning=Skipping model code generation (LELE_SKIP_MODEL_GEN=1)");
        return Ok(());
    }

    let output_dir = Path::new("src/gen");
    std::fs::create_dir_all(output_dir)?;

    // 1. SenseVoice (ASR)
    process_model(
        "../../examples/sensevoice/model.toml",
        "sensevoice.int8.onnx",
        "SenseVoice",
        output_dir,
    )?;

    // 2. Silero VAD
    process_model(
        "../../examples/silero/model.toml",
        "model.onnx",
        "SileroVad",
        output_dir,
    )?;

    // 3. YOLO26 (Object Detection)
    process_model(
        "../../examples/yolo26/model.toml",
        "yolo26.onnx",
        "Yolo26",
        output_dir,
    )?;

    // 4. Supertonic TTS (4 models)
    // First download all supertonic files
    if let Ok(config) = config::ModelConfig::load("../../examples/supertonic/model.toml") {
        match &config.model {
            config::ModelSource::HuggingFaceHub {
                repo,
                revision,
                files,
            } => {
                for file_spec in files {
                    let dest_name = file_spec
                        .dest
                        .as_ref()
                        .map(|s| s.as_str())
                        .unwrap_or(&file_spec.file);
                    let dest_path = Path::new(dest_name);
                    if dest_path.exists() && !should_force_regenerate() {
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
                        }
                        Err(e) => {
                            println!("cargo:warning=Failed to download {}: {}", file_spec.file, e);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let supertonic_models = [
        ("text_encoder.onnx", "TextEncoder"),
        ("duration_predictor.onnx", "DurationPredictor"),
        ("vector_estimator.onnx", "VectorEstimator"),
        ("vocoder.onnx", "Vocoder"),
    ];

    for (onnx_name, class_name) in &supertonic_models {
        let onnx_path = Path::new(onnx_name);
        if !onnx_path.exists() {
            println!("cargo:warning=Supertonic ONNX not found: {}", onnx_name);
            generate_model_stub(class_name, output_dir, &format!("{} not found", onnx_name))?;
            continue;
        }

        println!("cargo:warning=Generating code for {}...", class_name);
        if let Err(e) = generate_model_code(onnx_path, class_name, output_dir, &[]) {
            println!(
                "cargo:warning=Failed to generate code for {}: {}",
                class_name, e
            );
            generate_model_stub(class_name, output_dir, &e.to_string())?;
        }
    }

    Ok(())
}
