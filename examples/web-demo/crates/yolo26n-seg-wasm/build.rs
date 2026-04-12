use lele_build::*;
use std::path::Path;

fn generate_yolo26seg_stub(
    output_dir: &Path,
    error_msg: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(output_dir)?;
    let stub_code = format!(
        r#"// Auto-generated stub (model not available)
// Error: {error}
use lele::tensor::TensorView;

#[allow(dead_code)]
pub struct Yolo26SegWorkspace;
impl Yolo26SegWorkspace {{
    pub fn new() -> Self {{ Yolo26SegWorkspace }}
}}
impl Default for Yolo26SegWorkspace {{
    fn default() -> Self {{ Self::new() }}
}}

#[allow(dead_code)]
pub struct Yolo26Seg<'a> {{
    _marker: std::marker::PhantomData<&'a ()>,
}}

impl<'a> Yolo26Seg<'a> {{
    pub fn new(_weights: &'a [u8]) -> Self {{
        Self {{ _marker: std::marker::PhantomData }}
    }}

    pub fn forward(
        &self,
        _images: TensorView<'a>,
    ) -> (TensorView<'static>, TensorView<'static>) {{
        panic!("Model not available: {error}")
    }}

    pub fn forward_with_workspace<'w>(
        &self,
        _ws: &'w mut Yolo26SegWorkspace,
        _images: TensorView<'w>,
    ) -> (TensorView<'w>, TensorView<'w>) {{
        panic!("Model not available: {error}")
    }}
}}
"#,
        error = error_msg
    );
    std::fs::write(output_dir.join("yolo26seg.rs"), stub_code)?;
    let bin_path = output_dir.join("yolo26seg_weights.bin");
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

    let model_toml = "../../../../examples/yolo26n-seg/model.toml";
    let config = match config::ModelConfig::load(model_toml) {
        Ok(cfg) => cfg,
        Err(e) => {
            println!("cargo:warning=Failed to load {}: {}", model_toml, e);
            generate_yolo26seg_stub(output_dir, &format!("Config load failed: {}", e))?;
            return Ok(());
        }
    };

    if let config::ModelSource::Local { files } = &config.model {
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

    let model_file = "yolo26n-seg.onnx";
    let candidates = [
        model_file.to_string(),
        format!("../../{}", model_file),
        format!("../../../../examples/yolo26n-seg/{}", model_file),
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
        generate_yolo26seg_stub(output_dir, &format!("Model file not found: {}", model_file))?;
        return Ok(());
    }

    println!("cargo:warning=Generating code for Yolo26Seg...");
    match generate_model_code(model_path, "Yolo26Seg", output_dir, &[]) {
        Ok(()) => println!("cargo:warning=Code generation successful for Yolo26Seg"),
        Err(e) => {
            println!("cargo:warning=Code generation failed: {}", e);
            generate_yolo26seg_stub(output_dir, &e.to_string())?;
        }
    }

    Ok(())
}
