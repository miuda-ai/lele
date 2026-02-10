use lele_build::*;
use std::path::Path;

fn generate_stub(class_name: &str, output_dir: &Path, error_msg: &str) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(output_dir)?;
    let rs_filename = format!("{}.rs", class_name.to_lowercase());

    let (forward_params, _ws_name) = match class_name {
        "TextEncoder" => (
            "        _text_ids: TensorView<'a, i64>,\n        _style_ttl: TensorView<'a>,\n        _text_mask: TensorView<'a>,",
            "TextEncoderWorkspace"
        ),
        "DurationPredictor" => (
            "        _text_ids: TensorView<'a, i64>,\n        _style_dp: TensorView<'a>,\n        _text_mask: TensorView<'a>,",
            "DurationPredictorWorkspace"
        ),
        "VectorEstimator" => (
            "        _noisy_latent: TensorView<'a>,\n        _text_emb: TensorView<'a>,\n        _style_ttl: TensorView<'a>,\n        _latent_mask: TensorView<'a>,\n        _text_mask: TensorView<'a>,\n        _current_step: TensorView<'a>,\n        _total_step: TensorView<'a>,",
            "VectorEstimatorWorkspace"
        ),
        "Vocoder" => (
            "        _latent: TensorView<'a>,",
            "VocoderWorkspace"
        ),
        _ => return lele_build::generate_stub(class_name, output_dir, error_msg),
    };

    let ws_name = _ws_name;
    let stub_code = format!(
        r#"// Auto-generated stub (model not available)
// Error: {error}
use lele::tensor::TensorView;

#[allow(dead_code)]
pub struct {ws_name};
impl {ws_name} {{
    pub fn new() -> Self {{ {ws_name} }}
}}

#[allow(dead_code)]
pub struct {cls}<'a> {{
    _marker: std::marker::PhantomData<&'a ()>,
}}

impl<'a> {cls}<'a> {{
    pub fn new(_weights: &'a [u8]) -> Self {{
        Self {{ _marker: std::marker::PhantomData }}
    }}

    pub fn forward(
        &self,
{params}
    ) -> TensorView<'static> {{
        panic!("Model not available: {error}")
    }}
}}
"#,
        error = error_msg,
        cls = class_name,
        ws_name = ws_name,
        params = forward_params
    );
    std::fs::write(output_dir.join(&rs_filename), stub_code)?;

    let weights_filename = format!("{}_weights.bin", class_name.to_lowercase());
    let bin_path = output_dir.join(&weights_filename);
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

    // Download all supertonic files
    let model_toml = "../../../../examples/supertonic/model.toml";
    if let Ok(config) = config::ModelConfig::load(model_toml) {
        if let config::ModelSource::HuggingFaceHub { repo, revision, files } = &config.model {
            for file_spec in files {
                let dest_name = file_spec.dest.as_ref().map(|s| s.as_str()).unwrap_or(&file_spec.file);
                let dest_path = Path::new(dest_name);
                if dest_path.exists() && !should_force_regenerate() {
                    continue;
                }
                if let Some(parent) = dest_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                match download_from_hf_hub(repo, &file_spec.file, revision.as_deref(), Some(&get_model_cache_dir())) {
                    Ok(cached_path) => {
                        if cached_path != dest_path {
                            std::fs::copy(&cached_path, dest_path)?;
                        }
                    }
                    Err(e) => println!("cargo:warning=Failed to download {}: {}", file_spec.file, e),
                }
            }
        }
    }

    let models = [
        ("text_encoder.onnx", "TextEncoder"),
        ("duration_predictor.onnx", "DurationPredictor"),
        ("vector_estimator.onnx", "VectorEstimator"),
        ("vocoder.onnx", "Vocoder"),
    ];

    for (onnx_name, class_name) in &models {
        let candidates = [
            onnx_name.to_string(),
            format!("../../{}", onnx_name),
            format!("../../../../examples/supertonic/{}", onnx_name),
        ];
        let onnx_path = candidates
            .iter()
            .map(|c| std::path::PathBuf::from(c))
            .find(|p| p.exists())
            .unwrap_or_else(|| std::path::PathBuf::from(*onnx_name));
        if !onnx_path.exists() {
            println!("cargo:warning=Supertonic ONNX not found: {}", onnx_name);
            generate_stub(class_name, output_dir, &format!("{} not found", onnx_name))?;
            continue;
        }

        println!("cargo:warning=Generating code for {}...", class_name);
        if let Err(e) = generate_model_code(onnx_path, class_name, output_dir, &[]) {
            println!("cargo:warning=Failed to generate code for {}: {}", class_name, e);
            generate_stub(class_name, output_dir, &e.to_string())?;
        }
    }

    Ok(())
}
