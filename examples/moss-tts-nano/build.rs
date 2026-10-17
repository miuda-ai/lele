//! Build script for the MOSS-TTS-Nano example.
//!
//! Two things happen on top of a plain code generation:
//!
//! 1. **Shared prefill/decode weights.** `moss_tts_prefill.onnx` and
//!    `moss_tts_decode_step.onnx` are two graphs over the *same* transformer
//!    (both reference `moss_tts_global_shared.data`). Generating them
//!    independently stores those ~440 MB twice. `DecodeStep` is compiled on top
//!    of the `Prefill` blob instead, so identical tensors keep their offsets and
//!    only the handful of graph-specific scalars are appended: one
//!    `tts_shared_weights.bin` feeds both modules.
//!
//! 2. **Weight-only int8 quantization** (via `quantize.py`), which shrinks the
//!    MatMul weights ~4x. See the caveat below.
//!
//! `MOSS_QUANTIZE=0` forces the f32 weights; the quantized models are cached as
//! `*.int8.onnx` next to the downloads and only rebuilt when missing.
//!
//! Caveat: on macOS-aarch64 the int8 GEMM path (`fused_dq_gemm_prepared_arm`)
//! dequantizes the weights back to f32 for Apple Accelerate/AMX and caches that
//! copy, so at runtime int8 costs extra memory instead of saving it (measured:
//! +1.4 GB peak RSS for this example, with no speed change). It is still worth
//! it when the goal is the size of the shipped artifacts. On aarch64 Linux the
//! NEON UDOT path is used instead and keeps the u8 weights as-is.

use lele_build::*;
use std::path::{Path, PathBuf};
use std::process::Command;

fn generate_custom_stub(
    class_name: &str,
    output_dir: &Path,
    error_msg: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;
    fs::create_dir_all(output_dir)?;
    let rs_path = output_dir.join(format!("{}.rs", class_name.to_lowercase()));
    let weights_path = output_dir.join(format!("{}_weights.bin", class_name.to_lowercase()));

    let code = format!(
        r#"
use lele::tensor::TensorView;
use std::marker::PhantomData;

#[allow(dead_code)]
pub struct {name}<'a> {{
    _marker: PhantomData<&'a ()>,
}}

#[allow(dead_code)]
impl<'a> {name}<'a> {{
    pub fn new(_weights: &'a [u8]) -> Self {{
        panic!("Model is not available: {err}");
    }}
}}
"#,
        name = class_name,
        err = error_msg
    );

    fs::write(rs_path, code)?;
    fs::write(weights_path, &[])?;
    Ok(())
}

fn download_model_files(
    repo: &str,
    revision: &str,
    files: &[(&str, &str)],
    model_dir: &Path,
) {
    for (file, dest_rel) in files {
        let dest_path = model_dir.join(dest_rel);
        if dest_path.exists() {
            continue;
        }
        if let Some(parent) = dest_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        match download_from_hf_hub(repo, file, Some(revision), Some(&get_model_cache_dir())) {
            Ok(cached_path) => {
                std::fs::copy(&cached_path, &dest_path).ok();
                println!("cargo:warning=Downloaded: {}/{} -> {}", repo, file, dest_path.display());
            }
            Err(e) => {
                println!("cargo:warning=Failed to download {}/{}: {}", repo, file, e);
            }
        }
    }
}

/// Path of the int8 variant of an ONNX file: `foo.onnx` -> `foo.int8.onnx`.
fn quantized_path(onnx: &Path) -> PathBuf {
    let stem = onnx.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    onnx.with_file_name(format!("{stem}.int8.onnx"))
}

/// Weight-only int8 quantization via `quantize.py`.
///
/// Returns the int8 model path when it already exists or can be produced, or
/// `None` to fall back to the f32 model (no python/onnxruntime, or disabled by
/// `MOSS_QUANTIZE=0`). The result is cached next to the source model.
fn quantized_or_none(onnx: &Path) -> Option<PathBuf> {
    if std::env::var("MOSS_QUANTIZE").is_ok_and(|v| v == "0") {
        return None;
    }
    let out = quantized_path(onnx);
    if out.exists() {
        return Some(out);
    }

    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("quantize.py");
    if !script.exists() {
        println!("cargo:warning=quantize.py not found; using f32 weights");
        return None;
    }

    let status = Command::new("python3")
        .arg(&script)
        .arg(onnx)
        .arg(&out)
        .status();
    match status {
        Ok(s) if s.success() && out.exists() => Some(out),
        Ok(_) => {
            println!(
                "cargo:warning=quantize.py failed for {}; using f32 weights",
                onnx.display()
            );
            None
        }
        Err(e) => {
            println!(
                "cargo:warning=could not run python3 ({}); using f32 weights",
                e
            );
            None
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=model.toml");
    println!("cargo:rerun-if-changed=quantize.py");
    println!("cargo:rerun-if-env-changed=MOSS_QUANTIZE");

    if should_skip_codegen() {
        println!("cargo:warning=Skipping model code generation (LELE_SKIP_MODEL_GEN=1)");
        return Ok(());
    }

    let model_dir = Path::new("models");
    let tts_dir = model_dir.join("tts");
    let codec_dir = model_dir.join("codec");

    // Download TTS model files
    let tts_files = [
        ("moss_tts_prefill.onnx", "moss_tts_prefill.onnx"),
        ("moss_tts_global_shared.data", "moss_tts_global_shared.data"),
        ("moss_tts_decode_step.onnx", "moss_tts_decode_step.onnx"),
        ("moss_tts_local_fixed_sampled_frame.onnx", "moss_tts_local_fixed_sampled_frame.onnx"),
        ("moss_tts_local_shared.data", "moss_tts_local_shared.data"),
        ("tokenizer.model", "tokenizer.model"),
        ("tts_browser_onnx_meta.json", "tts_browser_onnx_meta.json"),
        ("browser_poc_manifest.json", "browser_poc_manifest.json"),
    ];
    download_model_files(
        "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX",
        "main",
        &tts_files,
        &tts_dir,
    );

    // Download codec model files
    let codec_files = [
        ("moss_audio_tokenizer_decode_full.onnx", "moss_audio_tokenizer_decode_full.onnx"),
        ("moss_audio_tokenizer_decode_shared.data", "moss_audio_tokenizer_decode_shared.data"),
        ("codec_browser_onnx_meta.json", "codec_browser_onnx_meta.json"),
    ];
    download_model_files(
        "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX",
        "main",
        &codec_files,
        &codec_dir,
    );

    let output_dir = Path::new("src");

    // Pick the int8 weight file when we can produce it, otherwise the f32 one.
    let prefill_onnx = tts_dir.join("moss_tts_prefill.onnx");
    let decode_onnx = tts_dir.join("moss_tts_decode_step.onnx");
    let local_onnx = tts_dir.join("moss_tts_local_fixed_sampled_frame.onnx");
    let codec_onnx = codec_dir.join("moss_audio_tokenizer_decode_full.onnx");

    let prefill_src = quantized_or_none(&prefill_onnx).unwrap_or(prefill_onnx.clone());
    let decode_src = quantized_or_none(&decode_onnx).unwrap_or(decode_onnx.clone());
    let local_src = quantized_or_none(&local_onnx).unwrap_or(local_onnx.clone());
    let codec_src = quantized_or_none(&codec_onnx).unwrap_or(codec_onnx.clone());
    let quantized = prefill_src != prefill_onnx;

    println!("cargo:warning=Weights: {}", if quantized { "int8 (weight-only)" } else { "f32" });

    // --- TTS: local frame + prefill + decode step -------------------------
    //
    // `moss_tts_prefill` and `moss_tts_decode_step` are two graphs over the *same*
    // transformer weights. Compiling them independently stores those weights
    // twice; building the decode step on top of the prefill blob makes it reuse
    // every identical tensor and append only its own, so both modules load one
    // file.
    if !prefill_src.exists() || !decode_src.exists() {
        generate_custom_stub("Prefill", output_dir, "prefill onnx not found")?;
        generate_custom_stub("DecodeStep", output_dir, "decode_step onnx not found")?;
    } else {
        println!("cargo:warning=Generating code for Prefill...");
        let prefill = compile_model(&prefill_src, "Prefill", None)?;
        println!("cargo:warning=Generating code for DecodeStep...");
        let decode = compile_model(
            &decode_src,
            "DecodeStep",
            Some(SharedWeights::from_result(&prefill)),
        )?;

        write_model_code(output_dir, "Prefill", &prefill.code)?;
        write_model_code(output_dir, "DecodeStep", &decode.code)?;

        let shared = write_shared_weights(output_dir.join("tts_shared_weights.bin"), &decode.weights)?;
        println!(
            "cargo:warning=TTS weights: {} ({:.1} MB) serving Prefill ({:.1} MB) + DecodeStep",
            shared.display(),
            decode.weights.len() as f64 / 1e6,
            prefill.weights.len() as f64 / 1e6,
        );
    }

    // --- Local frame + codec: independent models --------------------------
    for (onnx, class_name) in [
        (local_src.as_path(), "LocalFixedSampledFrame"),
        (codec_src.as_path(), "CodecDecodeFull"),
    ] {
        if !onnx.exists() {
            println!("cargo:warning=ONNX file not found: {}", onnx.display());
            generate_custom_stub(class_name, output_dir, &format!("{} not found", onnx.display()))?;
            continue;
        }
        println!("cargo:warning=Generating code for {}...", class_name);
        if let Err(e) = generate_model_code(onnx, class_name, output_dir, &[]) {
            println!("cargo:warning=Failed to generate code for {}: {}", class_name, e);
            generate_custom_stub(class_name, output_dir, &e.to_string())?;
        }
    }

    Ok(())
}
