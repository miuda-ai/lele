use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

pub mod config {
    use serde::{Deserialize, Serialize};
    use std::path::Path;

    #[derive(Debug, Deserialize, Serialize)]
    pub struct ModelConfig {
        pub model: ModelSource,
        pub codegen: CodeGenConfig,
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(tag = "source")]
    pub enum ModelSource {
        #[serde(rename = "hf-hub")]
        HuggingFaceHub {
            repo: String,
            #[serde(default)]
            revision: Option<String>,
            files: Vec<HfFileSpec>,
        },
        #[serde(rename = "url")]
        Url { files: Vec<UrlFileSpec> },
        #[serde(rename = "local")]
        Local {
            #[serde(default)]
            files: Vec<FileSpec>,
        },
    }

    #[derive(Debug, Deserialize, Serialize, Clone)]
    pub struct FileSpec {
        pub path: String,
        #[serde(default)]
        pub dest: Option<String>,
    }

    #[derive(Debug, Deserialize, Serialize, Clone)]
    pub struct HfFileSpec {
        pub file: String,
        #[serde(default)]
        pub dest: Option<String>,
    }

    #[derive(Debug, Deserialize, Serialize, Clone)]
    pub struct UrlFileSpec {
        pub url: String,
        pub dest: String,
    }

    #[derive(Debug, Deserialize, Serialize)]
    pub struct CodeGenConfig {
        pub class_name: String,
        #[serde(default)]
        pub model_file: Option<String>,
        #[serde(default)]
        pub custom_methods: Vec<String>,
    }

    impl ModelConfig {
        pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
            let content = std::fs::read_to_string(path)?;
            Ok(toml::from_str(&content)?)
        }
    }
}

/// Download model from Hugging Face Hub using HF_ENDPOINT (mirror) and caching.
pub fn download_from_hf_hub(
    repo: &str,
    file: &str,
    revision: Option<&str>,
    cache_dir: Option<&Path>,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let base =
        std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string());
    if std::env::var("HF_ENDPOINT").is_ok() {
        println!("cargo:warning=Using HF mirror: {}", base);
    }

    let base = base.trim_end_matches('/');
    let rev = revision.unwrap_or("main");
    let url = format!("{}/{}/resolve/{}/{}", base, repo, rev, file);

    let cache_root = cache_dir
        .map(|p| p.to_path_buf())
        .unwrap_or_else(get_model_cache_dir);
    let cache_path = cache_root.join(repo).join(rev).join(file);

    if cache_path.exists() {
        return Ok(cache_path);
    }

    if let Some(parent) = cache_path.parent() {
        fs::create_dir_all(parent)?;
    }

    download_from_url(&url, &cache_path)
}

/// Download file from URL
pub fn download_from_url(url: &str, dest: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    println!("cargo:warning=Downloading from URL: {}", url);

    let config = ureq::config::Config::builder()
        .timeout_global(Some(std::time::Duration::from_secs(1200)))
        .build();
    let agent = ureq::Agent::new_with_config(config);
    let response = agent.get(url).call()?;

    println!("cargo:warning=Response status: {}", response.status());
    if !response.status().is_success() {
        return Err(format!("Download failed with status: {}", response.status()).into());
    }

    let content_len = response
        .headers()
        .get("content-length")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());

    if let Some(len) = content_len {
        println!("cargo:warning=Downloading {} bytes...", len);
    }

    let mut file = File::create(dest)?;
    let mut reader = response.into_body().into_reader();
    std::io::copy(&mut reader, &mut file)?;

    println!("cargo:warning=Downloaded to: {}", dest.display());
    Ok(dest.to_path_buf())
}

/// Get model cache directory
pub fn get_model_cache_dir() -> PathBuf {
    if let Ok(cache_dir) = std::env::var("LELE_MODEL_CACHE") {
        PathBuf::from(cache_dir)
    } else {
        let target_dir = std::env::var("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
                PathBuf::from(manifest_dir).join("../../target")
            });
        target_dir.join("lele_cache/models")
    }
}

/// Generate Rust code and binary weights from ONNX model
pub fn generate_model_code(
    model_path: impl AsRef<Path>,
    class_name: &str,
    output_dir: impl AsRef<Path>,
    _custom_methods: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    use lele::compiler::Compiler;
    use lele::model::OnnxModel;

    let model = OnnxModel::load(model_path.as_ref())?;
    let graph = model.graph().ok_or("Missing graph in ONNX model")?;

    let compiler = Compiler::new()
        .with_name(class_name)
        .with_default_optimizations()
        .with_constant_folding(true);

    let result = compiler.compile(graph)?;

    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir)?;

    // Write weights binary
    let weights_filename = format!("{}_weights.bin", class_name.to_lowercase());
    let bin_path = output_dir.join(&weights_filename);
    let mut bin_file = std::io::BufWriter::new(File::create(&bin_path)?);
    bin_file.write_all(&result.weights)?;
    bin_file.flush()?;

    // Write generated Rust code
    let rs_filename = format!("{}.rs", class_name.to_lowercase());
    let rs_path = output_dir.join(&rs_filename);
    let mut rs_file = std::io::BufWriter::new(File::create(&rs_path)?);
    rs_file.write_all(result.code.as_bytes())?;
    rs_file.flush()?;

    Ok(())
}

/// Generate a stub implementation when model download or generation fails
pub fn generate_stub(
    class_name: &str,
    output_dir: impl AsRef<Path>,
    error_msg: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir)?;

    let rs_filename = format!("{}.rs", class_name.to_lowercase());
    let rs_path = output_dir.join(&rs_filename);

    let stub_code = format!(
        r#"// Auto-generated stub (model generation failed)
// Error: {}

#[allow(dead_code)]
pub struct {}<'a> {{
    _marker: std::marker::PhantomData<&'a ()>,
}}

#[allow(dead_code)]
impl<'a> {}<'a> {{
    pub fn new(_weights: &'a [u8]) -> Self {{
        panic!("Model is not available: {}")
    }}
}}
"#,
        error_msg, class_name, class_name, error_msg
    );

    fs::write(rs_path, stub_code)?;

    // Create empty weights file
    let weights_filename = format!("{}_weights.bin", class_name.to_lowercase());
    let bin_path = output_dir.join(&weights_filename);
    fs::write(bin_path, &[])?;

    Ok(())
}

/// Check if code generation should be skipped
pub fn should_skip_codegen() -> bool {
    println!("cargo:rerun-if-env-changed=LELE_SKIP_MODEL_GEN");
    std::env::var("LELE_SKIP_MODEL_GEN")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false)
}

/// Check if code should be forcefully regenerated
pub fn should_force_regenerate() -> bool {
    println!("cargo:rerun-if-env-changed=LELE_FORCE_REGENERATE");
    println!("cargo:rerun-if-env-changed=LELE_FORCE_REGEN");
    std::env::var("LELE_FORCE_REGENERATE")
        .or_else(|_| std::env::var("LELE_FORCE_REGEN"))
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false)
}

/// Check if generated files already exist and are up to date
pub fn need_regenerate(
    class_name: &str,
    output_dir: impl AsRef<Path>,
    model_config_path: impl AsRef<Path>,
) -> bool {
    need_regenerate_with_model(class_name, output_dir, model_config_path, None)
}

/// Check if generated files already exist and are up to date, with optional model file path
pub fn need_regenerate_with_model(
    class_name: &str,
    output_dir: impl AsRef<Path>,
    model_config_path: impl AsRef<Path>,
    model_file_path: Option<&Path>,
) -> bool {
    if should_force_regenerate() {
        return true;
    }

    let output_dir = output_dir.as_ref();
    let rs_filename = format!("{}.rs", class_name.to_lowercase());
    let weights_filename = format!("{}_weights.bin", class_name.to_lowercase());

    let rs_path = output_dir.join(&rs_filename);
    let bin_path = output_dir.join(&weights_filename);

    println!(
        "cargo:warning=Checking if regeneration needed for {}",
        class_name
    );
    println!(
        "cargo:warning=rs_path: {:?} (exists: {}) abs: {:?}",
        rs_path,
        rs_path.exists(),
        fs::canonicalize(&rs_path).ok()
    );
    println!(
        "cargo:warning=bin_path: {:?} (exists: {}) abs: {:?}",
        bin_path,
        bin_path.exists(),
        fs::canonicalize(&bin_path).ok()
    );

    if !rs_path.exists() || !bin_path.exists() {
        println!("cargo:warning=File(s) missing, need regenerate");
        return true;
    }

    let rs_meta = match fs::metadata(&rs_path).and_then(|m| m.modified()) {
        Ok(t) => t,
        Err(_) => return true,
    };

    // Emit cargo:rerun-if-changed and check modification time for all source files
    let sources: Vec<&Path> = if let Some(mfp) = model_file_path {
        vec![model_config_path.as_ref(), mfp]
    } else {
        vec![model_config_path.as_ref()]
    };
    for src in &sources {
        println!("cargo:rerun-if-changed={}", src.display());
        if let Ok(src_time) = fs::metadata(src).and_then(|m| m.modified()) {
            if src_time > rs_meta {
                println!("cargo:warning=Source file {:?} is newer than generated code, need regenerate", src);
                return true;
            }
        }
    }

    false
}
