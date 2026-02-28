fn main() -> std::io::Result<()> {
    prost_build::compile_protos(&["src/onnx.proto"], &["src/"])?;

    // Link Apple Accelerate framework on macOS aarch64 for AMX-accelerated GEMM
    // Note: #[cfg] in build scripts checks the HOST, not TARGET.
    // Use CARGO_CFG_TARGET_* env vars to check the compilation target.
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_arch == "aarch64" && target_os == "macos" {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    Ok(())
}
