use lele_build::*;
use std::path::Path;

// Supertonic has multiple models, so we generate code for each one
fn generate_custom_stub(
    class_name: &str,
    output_dir: &Path,
    error_msg: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;
    fs::create_dir_all(output_dir)?;
    let rs_path = output_dir.join(format!("{}.rs", class_name.to_lowercase()));
    let weights_path = output_dir.join(format!("{}_weights.bin", class_name.to_lowercase()));

    // Common imports
    let mut code = String::from(
        r#"
use lele::tensor::TensorView;
use std::marker::PhantomData;

#[allow(dead_code)]
pub struct CLASSNAME<'a> {
    _marker: PhantomData<&'a ()>,
}

#[allow(dead_code)]
impl<'a> CLASSNAME<'a> {
    pub fn new(_weights: &'a [u8]) -> Self {
        panic!("Model is not available: ERROR_MSG");
    }
"#,
    );

    // Customize forward method based on main.rs usage
    let forward_sig = match class_name {
        "DurationPredictor" | "TextEncoder" => {
            "    pub fn forward<'w>(&self, _a: TensorView<'w, i64>, _b: TensorView<'w>, _c: TensorView<'w>) -> TensorView<'static> { panic!(\"Model not available\") }"
        }
        "VectorEstimator" => {
            "    pub fn forward<'w>(&self, _a: TensorView<'w>, _b: TensorView<'w>, _c: TensorView<'w>, _d: TensorView<'w>, _e: TensorView<'w>, _f: TensorView<'w>, _g: TensorView<'w>) -> TensorView<'static> { panic!(\"Model not available\") }"
        }
        "Vocoder" => {
            "    pub fn forward<'w>(&self, _a: TensorView<'w>) -> TensorView<'static> { panic!(\"Model not available\") }"
        }
        _ => "",
    };

    code = code
        .replace("CLASSNAME", class_name)
        .replace("ERROR_MSG", error_msg);
    code.push_str(forward_sig);
    code.push_str("\n}\n");

    fs::write(rs_path, code)?;
    fs::write(weights_path, &[])?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=model.toml");
    println!("cargo:warning=CWD: {:?}", std::env::current_dir()?);
    println!("cargo:warning=Checking files check...");

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

    // Download/get model files and metadata
    match &config.model {
        config::ModelSource::Local { files } => {
            for file_spec in files {
                let path = Path::new(&file_spec.path);
                if !path.exists() {
                    println!(
                        "cargo:warning=Local model file not found: {}",
                        file_spec.path
                    );
                    return Ok(());
                }

                if let Some(dest) = &file_spec.dest {
                    let dest_path = Path::new(dest);
                    if dest_path != path {
                        if let Some(parent) = dest_path.parent() {
                            std::fs::create_dir_all(parent)?;
                        }
                        std::fs::copy(path, dest_path)?;
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
                        // Continue to let stub generation handle missing files
                    }
                }
            }
        }
        config::ModelSource::Url { files } => {
            println!("cargo:warning=Downloading {} files from URLs", files.len());

            for file_spec in files {
                let dest_path = Path::new(&file_spec.dest);

                if dest_path.exists() && !should_force_regenerate() {
                    println!(
                        "cargo:warning=File already exists, skipping: {}",
                        file_spec.dest
                    );
                    continue;
                }

                if let Some(parent) = dest_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }

                if let Err(e) = download_from_url(&file_spec.url, dest_path) {
                    println!(
                        "cargo:warning=Failed to download from {}: {}",
                        file_spec.url, e
                    );
                    // Continue to let stub generation handle missing files
                }
            }
        }
    }

    let output_dir = Path::new("src");

    // List of models to generate
    let models = vec![
        ("text_encoder.onnx", "TextEncoder"),
        ("duration_predictor.onnx", "DurationPredictor"),
        ("vector_estimator.onnx", "VectorEstimator"),
        ("vocoder.onnx", "Vocoder"),
    ];

    for (onnx_name, class_name) in models {
        let onnx_path = Path::new(onnx_name);
        if !onnx_path.exists() {
            println!("cargo:warning=ONNX file not found: {}", onnx_name);
            generate_custom_stub(class_name, output_dir, &format!("{} not found", onnx_name))?;
            continue;
        }

        println!("cargo:warning=Generating code for {}...", class_name);
        if let Err(e) = generate_model_code(onnx_path, class_name, output_dir, &[]) {
            println!(
                "cargo:warning=Failed to generate code for {}: {}",
                class_name, e
            );
            generate_custom_stub(class_name, output_dir, &e.to_string())?;
        }
    }

    Ok(())
}
