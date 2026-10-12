use std::fmt;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

/// What a build script fails with.
///
/// `Debug` prints the message alone, so a `main` that returns this reports
/// `Error: <message>` rather than a quoted string or a struct dump.
pub struct Error(String);

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl<E: std::error::Error> From<E> for Error {
    fn from(e: E) -> Self {
        Error(e.to_string())
    }
}

pub type Result<T = (), E = Error> = std::result::Result<T, E>;

/// Says what was being done when an error happened.
pub trait Context<T> {
    fn context(self, what: impl FnOnce() -> String) -> Result<T>;
}

impl<T, E: fmt::Display> Context<T> for std::result::Result<T, E> {
    fn context(self, what: impl FnOnce() -> String) -> Result<T> {
        self.map_err(|e| Error(format!("{}: {}", what(), e)))
    }
}

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
        pub fn load(path: impl AsRef<Path>) -> crate::Result<Self> {
            use crate::Context;
            let path = path.as_ref();
            let content =
                std::fs::read_to_string(path).context(|| format!("reading {}", path.display()))?;
            toml::from_str(&content).context(|| format!("parsing {}", path.display()))
        }
    }
}

/// Download model from Hugging Face Hub using HF_ENDPOINT (mirror) and caching.
pub fn download_from_hf_hub(
    repo: &str,
    file: &str,
    revision: Option<&str>,
    cache_dir: Option<&Path>,
) -> Result<PathBuf> {
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
///
/// The body goes to `<dest>.part` first and is renamed once complete: an
/// interrupted download would otherwise leave a truncated `dest`, which every
/// later build would take for the finished file.
pub fn download_from_url(url: &str, dest: &Path) -> Result<PathBuf> {
    println!("cargo:warning=Downloading from URL: {}", url);

    let config = ureq::config::Config::builder()
        .timeout_global(Some(std::time::Duration::from_secs(1200)))
        .build();
    let agent = ureq::Agent::new_with_config(config);
    let response = agent.get(url).call().context(|| format!("GET {url}"))?;

    let content_len = response
        .headers()
        .get("content-length")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());

    if let Some(len) = content_len {
        println!("cargo:warning=Downloading {} bytes...", len);
    }

    let part = PathBuf::from(format!("{}.part", dest.display()));
    let mut file = File::create(&part).context(|| format!("creating {}", part.display()))?;
    let mut reader = response.into_body().into_reader();
    std::io::copy(&mut reader, &mut file).context(|| format!("downloading {url}"))?;
    drop(file);
    fs::rename(&part, dest).context(|| format!("moving {} into place", part.display()))?;

    println!("cargo:warning=Downloaded to: {}", dest.display());
    Ok(dest.to_path_buf())
}

/// Puts `file` of the Hugging Face repo `repo` at `dest`, unless it is already
/// there and no regeneration is forced.
pub fn fetch_from_hf_hub(repo: &str, file: &str, revision: Option<&str>, dest: &Path) -> Result {
    if dest.exists() && !should_force_regenerate() {
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    let what = || format!("fetching {file} from Hugging Face repo {repo}");
    let cached =
        download_from_hf_hub(repo, file, revision, Some(&get_model_cache_dir())).context(what)?;
    if cached != dest {
        fs::copy(&cached, dest).context(|| format!("copying to {}", dest.display()))?;
    }
    Ok(())
}

/// Puts the file at `url` at `dest`, unless it is already there and no
/// regeneration is forced.
pub fn fetch_from_url(url: &str, dest: &Path) -> Result {
    if dest.exists() && !should_force_regenerate() {
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    download_from_url(url, dest)?;
    Ok(())
}

/// Puts every file `source` lists at its destination in the current crate:
/// downloads the remote ones that are missing, and copies the local ones,
/// whose paths are relative to `config_dir`, the directory of the model.toml.
pub fn fetch_model_files(source: &config::ModelSource, config_dir: &Path) -> Result {
    use config::ModelSource;
    match source {
        ModelSource::Local { files } => {
            for spec in files {
                let path = config_dir.join(&spec.path);
                if !path.exists() {
                    return Err(Error(format!(
                        "model file {} not found: its model.toml lists it as a local file, \
                         so it has to be put there by hand",
                        path.display()
                    )));
                }
                let dest = Path::new(spec.dest.as_deref().unwrap_or(&spec.path));
                if dest != path && (!dest.exists() || should_force_regenerate()) {
                    if let Some(parent) = dest.parent() {
                        fs::create_dir_all(parent)?;
                    }
                    fs::copy(&path, dest)
                        .context(|| format!("copying {} to {}", path.display(), dest.display()))?;
                }
            }
        }
        ModelSource::HuggingFaceHub {
            repo,
            revision,
            files,
        } => {
            for spec in files {
                let dest = spec.dest.as_deref().unwrap_or(&spec.file);
                fetch_from_hf_hub(repo, &spec.file, revision.as_deref(), Path::new(dest))?;
            }
        }
        ModelSource::Url { files } => {
            for spec in files {
                fetch_from_url(&spec.url, Path::new(&spec.dest))?;
            }
        }
    }
    Ok(())
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
) -> Result {
    let model_path = model_path.as_ref();
    println!("cargo:warning=Generating code for {}...", class_name);
    compile_model(model_path, class_name, output_dir.as_ref()).context(|| {
        format!(
            "generating code for {} from {}",
            class_name,
            model_path.display()
        )
    })
}

fn compile_model(
    model_path: &Path,
    class_name: &str,
    output_dir: &Path,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    use lele::compiler::Compiler;
    use lele::model::OnnxModel;

    let model = OnnxModel::load(model_path)?;
    let graph = model.graph().ok_or("Missing graph in ONNX model")?;

    let compiler = Compiler::new()
        .with_name(class_name)
        .with_default_optimizations()
        .with_constant_folding(true);

    let result = compiler.compile(graph)?;

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

/// Builds the model `model_toml` describes: fetches its files into the
/// current crate and generates `<output_dir>/<class>.rs` and its weights,
/// unless they are newer than both the config and the model.
pub fn build_model(model_toml: &str, output_dir: impl AsRef<Path>) -> Result {
    println!("cargo:rerun-if-changed={}", model_toml);
    if should_skip_codegen() {
        println!("cargo:warning=Skipping model code generation (LELE_SKIP_MODEL_GEN=1)");
        return Ok(());
    }

    let config = config::ModelConfig::load(model_toml)?;
    let codegen = &config.codegen;
    let model_file = Path::new(codegen.model_file.as_deref().unwrap_or("model.onnx"));
    let output_dir = output_dir.as_ref();
    if !need_regenerate_with_model(
        &codegen.class_name,
        output_dir,
        model_toml,
        Some(model_file),
    ) {
        println!("cargo:warning=Generated files are up to date, skipping regeneration");
        return Ok(());
    }

    fetch_model_files(&config.model, config_dir(model_toml))?;
    generate_model_code(
        model_file,
        &codegen.class_name,
        output_dir,
        &codegen.custom_methods,
    )
}

/// Builds several models from the files `model_toml` lists: fetches them into
/// the current crate and generates `<output_dir>/<class>.rs` and its weights
/// for each `(onnx file, class name)` of `models` whose generated files are
/// older than the config or the model.
pub fn build_models(
    model_toml: &str,
    output_dir: impl AsRef<Path>,
    models: &[(&str, &str)],
) -> Result {
    println!("cargo:rerun-if-changed={}", model_toml);
    if should_skip_codegen() {
        println!("cargo:warning=Skipping model code generation (LELE_SKIP_MODEL_GEN=1)");
        return Ok(());
    }

    let output_dir = output_dir.as_ref();
    let stale: Vec<_> = models
        .iter()
        .filter(|(onnx, class_name)| {
            need_regenerate_with_model(class_name, output_dir, model_toml, Some(Path::new(onnx)))
        })
        .collect();
    if stale.is_empty() {
        println!("cargo:warning=Generated files are up to date, skipping regeneration");
        return Ok(());
    }

    let config = config::ModelConfig::load(model_toml)?;
    fetch_model_files(&config.model, config_dir(model_toml))?;
    for (onnx, class_name) in stale {
        generate_model_code(onnx, class_name, output_dir, &[])?;
    }
    Ok(())
}

fn config_dir(model_toml: &str) -> &Path {
    Path::new(model_toml).parent().unwrap_or(Path::new(""))
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
        if let Ok(src_time) = fs::metadata(src).and_then(|m| m.modified())
            && src_time > rs_meta
        {
            println!(
                "cargo:warning=Source file {:?} is newer than generated code, need regenerate",
                src
            );
            return true;
        }
    }

    false
}
