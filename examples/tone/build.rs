use lele_build::*;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=model.toml");

    // Check if code generation should be skipped
    if should_skip_codegen() {
        println!("cargo:warning=Skipping model code generation (LELE_SKIP_MODEL_GEN=1)");
        return Ok(());
    }

    // Load model configuration
    let config = match config::ModelConfig::load("model.toml") {
        Ok(cfg) => cfg,
        Err(e) => {
            println!("cargo:warning=Failed to load model.toml: {}", e);
            return Ok(());
        }
    };

    let class_name = &config.codegen.class_name;
    let output_dir = Path::new("src");

    // Determine which file to use for code generation
    let model_file = config
        .codegen
        .model_file
        .as_ref()
        .map(|s| s.as_str())
        .unwrap_or("model.onnx");

    // Check if regeneration is needed
    if !need_regenerate(class_name, output_dir, "model.toml") && !should_force_regenerate() {
        println!("cargo:warning=Generated files are up to date, skipping regeneration");
        return Ok(());
    }

    // Download/get model files
    match &config.model {
        config::ModelSource::Local { files } => {
            // Check if local files exist
            for file_spec in files {
                let path = Path::new(&file_spec.path);
                if !path.exists() {
                    println!(
                        "cargo:warning=Local model file not found: {}",
                        file_spec.path
                    );
                    println!("cargo:warning=Generating stub implementation");
                    generate_stub(
                        class_name,
                        output_dir,
                        &format!("Model file not found: {}", file_spec.path),
                    )?;
                    return Ok(());
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

                // Skip if file already exists
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
                        // Copy to local directory
                        if cached_path != dest_path {
                            std::fs::copy(&cached_path, dest_path)?;
                        }
                    }
                    Err(e) => {
                        println!("cargo:warning=Failed to download {}: {}", file_spec.file, e);
                        println!("cargo:warning=Generating stub implementation");
                        generate_stub(class_name, output_dir, &format!("Download failed: {}", e))?;
                        return Ok(());
                    }
                }
            }
        }
        config::ModelSource::Url { files } => {
            println!("cargo:warning=Downloading {} files from URLs", files.len());

            for file_spec in files {
                let dest_path = Path::new(&file_spec.dest);

                // Skip if file already exists
                if dest_path.exists() && !should_force_regenerate() {
                    println!(
                        "cargo:warning=File already exists, skipping: {}",
                        file_spec.dest
                    );
                    continue;
                }

                match download_from_url(&file_spec.url, dest_path) {
                    Ok(_) => {
                        println!("cargo:warning=Downloaded: {}", file_spec.dest);
                    }
                    Err(e) => {
                        println!(
                            "cargo:warning=Failed to download from {}: {}",
                            file_spec.url, e
                        );
                        println!("cargo:warning=Generating stub implementation");
                        generate_stub(class_name, output_dir, &format!("Download failed: {}", e))?;
                        return Ok(());
                    }
                }
            }
        }
    }

    // Generate code from the model file
    let model_path = Path::new(model_file);
    if !model_path.exists() {
        println!("cargo:warning=Model file not found: {}", model_file);
        println!("cargo:warning=Generating stub implementation");
        generate_stub(
            class_name,
            output_dir,
            &format!("Model file not found: {}", model_file),
        )?;
        return Ok(());
    }

    println!("cargo:warning=Generating code for model: {}", class_name);
    match generate_model_code(
        model_path,
        class_name,
        output_dir,
        &config.codegen.custom_methods,
    ) {
        Ok(()) => {
            println!("cargo:warning=Code generation successful");
        }
        Err(e) => {
            println!("cargo:warning=Code generation failed: {}", e);
            println!("cargo:warning=Generating stub implementation");
            generate_stub(
                class_name,
                output_dir,
                &format!("Code generation failed: {}", e),
            )?;
        }
    }

    Ok(())
}
