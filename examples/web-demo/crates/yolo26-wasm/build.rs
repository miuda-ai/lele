use lele_build::*;
use std::path::Path;

fn generate_yolo26_stub(
    output_dir: &Path,
    error_msg: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(output_dir)?;
    let stub_code = format!(
        r#"// Auto-generated stub (model not available)
// Error: {error}
use lele::tensor::TensorView;

#[allow(dead_code)]
pub struct Yolo26Workspace;
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
    );
    std::fs::write(output_dir.join("yolo26.rs"), stub_code)?;
    let bin_path = output_dir.join("yolo26_weights.bin");
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

    let model_toml = "../../../../examples/yolo26/model.toml";
    let config = match config::ModelConfig::load(model_toml) {
        Ok(cfg) => cfg,
        Err(e) => {
            println!("cargo:warning=Failed to load {}: {}", model_toml, e);
            generate_yolo26_stub(output_dir, &format!("Config load failed: {}", e))?;
            return Ok(());
        }
    };

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

    let model_file = "yolo26.onnx";
    let candidates = [
        model_file.to_string(),
        format!("../../{}", model_file),
        format!("../../../../examples/yolo26/{}", model_file),
    ];
    let model_path = candidates
        .iter()
        .map(|c| std::path::PathBuf::from(c))
        .find(|p| p.exists())
        .unwrap_or_else(|| std::path::PathBuf::from(model_file));
    if !model_path.exists() {
        println!(
            "cargo:warning=Model file not found: {}, generating stub",
            model_file
        );
        generate_yolo26_stub(output_dir, &format!("Model file not found: {}", model_file))?;
        return Ok(());
    }

    println!("cargo:warning=Generating code for Yolo26...");
    match generate_model_code(model_path, "Yolo26", output_dir, &[]) {
        Ok(()) => println!("cargo:warning=Code generation successful for Yolo26"),
        Err(e) => {
            println!("cargo:warning=Code generation failed: {}", e);
            generate_yolo26_stub(output_dir, &e.to_string())?;
        }
    }

    Ok(())
}
