use lele_build::*;
use std::path::Path;

struct ModelSpec {
    repo: &'static str,
    file: &'static str,
    dest: &'static str,
    class_name: &'static str,
}

const DET_SPEC: ModelSpec = ModelSpec {
    repo: "PaddlePaddle/PP-OCRv6_tiny_det_onnx",
    file: "inference.onnx",
    dest: "det.onnx",
    class_name: "PpocrDet",
};

const REC_SPEC: ModelSpec = ModelSpec {
    repo: "PaddlePaddle/PP-OCRv6_tiny_rec_onnx",
    file: "inference.onnx",
    dest: "rec.onnx",
    class_name: "PpocrRec",
};

fn download_and_generate(spec: &ModelSpec) -> Result<(), Box<dyn std::error::Error>> {
    let dest_path = Path::new(spec.dest);

    if !dest_path.exists() || should_force_regenerate() {
        match download_from_hf_hub(spec.repo, spec.file, Some("main"), Some(&get_model_cache_dir())) {
            Ok(cached_path) => {
                if cached_path != dest_path {
                    std::fs::copy(&cached_path, dest_path)?;
                }
            }
            Err(e) => {
                println!("cargo:warning=Failed to download {}: {}", spec.file, e);
                generate_stub(spec.class_name, "src", &format!("Download failed: {}", e))?;
                return Ok(());
            }
        }
    }

    let rs_path = format!("src/{}.rs", spec.class_name.to_lowercase());
    let bin_path = format!("src/{}_weights.bin", spec.class_name.to_lowercase());

    if (!Path::new(&rs_path).exists() || !Path::new(&bin_path).exists() || should_force_regenerate())
        && dest_path.exists()
    {
        println!("cargo:warning=Generating code for {}...", spec.class_name);
        match generate_model_code(dest_path, spec.class_name, "src", &[]) {
            Ok(()) => {
                println!("cargo:warning=Code generation successful for {}", spec.class_name);
            }
            Err(e) => {
                println!("cargo:warning=Code generation failed for {}: {}", spec.class_name, e);
                generate_stub(spec.class_name, "src", &format!("Code generation failed: {}", e))?;
            }
        }
    } else if Path::new(&rs_path).exists() {
        println!("cargo:warning={} already generated, skipping", spec.class_name);
    }

    Ok(())
}

fn download_rec_yaml_and_gen_dict() -> Result<(), Box<dyn std::error::Error>> {
    let dest = Path::new("rec_inference.yml");
    if !dest.exists() || should_force_regenerate() {
        match download_from_hf_hub(
            "PaddlePaddle/PP-OCRv6_tiny_rec_onnx",
            "inference.yml",
            Some("main"),
            Some(&get_model_cache_dir()),
        ) {
            Ok(cached_path) => {
                if cached_path != dest {
                    std::fs::copy(&cached_path, dest)?;
                }
            }
            Err(e) => {
                println!("cargo:warning=Failed to download inference.yml: {}", e);
                generate_dict_stub();
                return Ok(());
            }
        }
    }

    let yml = std::fs::read_to_string(dest)?;
    let chars = parse_character_dict(&yml);

    let out_path = Path::new("src/dict.rs");
    let mut content = String::new();
    content.push_str("// Auto-extracted from PP-OCRv6_tiny_rec inference.yml\n");
    content.push_str(&format!("// {} characters\n", chars.len()));
    content.push_str("pub const CHAR_DICT: &[&str] = &[\n");
    for c in &chars {
        let escaped = c.replace('\\', "\\\\").replace('"', "\\\"");
        content.push_str(&format!("    \"{}\",\n", escaped));
    }
    content.push_str("];\n");
    std::fs::write(out_path, content)?;
    println!("cargo:warning=Generated dict.rs with {} characters", chars.len());

    Ok(())
}

fn parse_character_dict(yml: &str) -> Vec<String> {
    let mut chars = Vec::new();
    let mut in_dict = false;
    for line in yml.lines() {
        let trimmed = line.trim();
        if trimmed == "character_dict:" {
            in_dict = true;
            continue;
        }
        if in_dict {
            if let Some(rest) = trimmed.strip_prefix("- ") {
                let val = if rest.starts_with('\'') && rest.ends_with('\'') {
                    &rest[1..rest.len() - 1]
                } else if rest.starts_with('"') && rest.ends_with('"') {
                    &rest[1..rest.len() - 1]
                } else {
                    rest
                };
                let unescaped = val
                    .replace("\\\"", "\"")
                    .replace("\\\\", "\\");
                chars.push(unescaped);
            } else if !trimmed.is_empty() && !trimmed.starts_with('-') {
                break;
            }
        }
    }
    chars
}

fn generate_dict_stub() {
    let stub = "pub const CHAR_DICT: &[&str] = &[];\n";
    let _ = std::fs::write("src/dict.rs", stub);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=model.toml");
    println!("cargo:rerun-if-env-changed=LELE_FORCE_REGEN");
    println!("cargo:rerun-if-env-changed=LELE_FORCE_REGENERATE");

    if should_skip_codegen() {
        println!("cargo:warning=Skipping model code generation (LELE_SKIP_MODEL_GEN=1)");
        return Ok(());
    }

    download_and_generate(&DET_SPEC)?;
    download_and_generate(&REC_SPEC)?;
    download_rec_yaml_and_gen_dict()?;

    Ok(())
}
