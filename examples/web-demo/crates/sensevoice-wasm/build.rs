use lele_build::*;
use std::path::Path;

fn generate_sensevoice_stub(
    output_dir: &Path,
    error_msg: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(output_dir)?;
    let stub_code = format!(
        r#"// Auto-generated stub (model not available)
// Error: {error}
use lele::tensor::TensorView;

#[allow(dead_code)]
pub struct SenseVoiceWorkspace;
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
    );
    std::fs::write(output_dir.join("sensevoice.rs"), stub_code)?;
    let bin_path = output_dir.join("sensevoice_weights.bin");
    if !bin_path.exists() {
        std::fs::write(bin_path, &[])?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");

    let output_dir = Path::new("src/gen");
    std::fs::create_dir_all(output_dir)?;

    if should_skip_codegen() {
        println!("cargo:warning=Skipping model code generation (LELE_SKIP_MODEL_GEN=1)");
        return Ok(());
    }

    let model_toml = "../../../../examples/sensevoice/model.toml";
    let config = match config::ModelConfig::load(model_toml) {
        Ok(cfg) => cfg,
        Err(e) => {
            println!("cargo:warning=Failed to load {}: {}", model_toml, e);
            generate_sensevoice_stub(output_dir, &format!("Config load failed: {}", e))?;
            return Ok(());
        }
    };

    // Download model files
    if let config::ModelSource::HuggingFaceHub {
        repo,
        revision,
        files,
    } = &config.model
    {
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
                continue;
            }
            match download_from_hf_hub(
                repo,
                &file_spec.file,
                revision.as_deref(),
                Some(&get_model_cache_dir()),
            ) {
                Ok(cached_path) => {
                    if cached_path != dest_path {
                        if let Some(parent) = dest_path.parent() {
                            std::fs::create_dir_all(parent)?;
                        }
                        std::fs::copy(&cached_path, dest_path)?;
                    }
                }
                Err(e) => println!("cargo:warning=Failed to download {}: {}", file_spec.file, e),
            }
        }
    }

    let model_file = "sensevoice.int8.onnx";
    // Search multiple candidate locations for the model file
    let candidates = [
        model_file.to_string(),
        format!("../../{}", model_file), // web-demo root
        format!("../../../../examples/sensevoice/{}", model_file), // sensevoice example
    ];
    let model_path = candidates
        .iter()
        .map(|c| Path::new(c).to_path_buf())
        .find(|p| p.exists())
        .unwrap_or_else(|| Path::new(model_file).to_path_buf());
    if !model_path.exists() {
        println!(
            "cargo:warning=Model file not found: {}, generating stub",
            model_file
        );
        generate_sensevoice_stub(output_dir, &format!("Model file not found: {}", model_file))?;
        return Ok(());
    }

    println!("cargo:warning=Generating code for SenseVoice...");
    match generate_model_code(model_path, "SenseVoice", output_dir, &[]) {
        Ok(()) => println!("cargo:warning=Code generation successful for SenseVoice"),
        Err(e) => {
            println!("cargo:warning=Code generation failed: {}", e);
            generate_sensevoice_stub(output_dir, &e.to_string())?;
        }
    }

    Ok(())
}
