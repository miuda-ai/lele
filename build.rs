fn main() {
    println!("cargo:rustc-check-cfg=cfg(nightly_build)");

    // Link Apple Accelerate framework on macOS aarch64 for AMX-accelerated GEMM
    // Note: #[cfg] in build scripts checks the HOST, not TARGET.
    // Use CARGO_CFG_TARGET_* env vars to check the compilation target.
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_arch == "aarch64" && target_os == "macos" {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    // Detect nightly Rust compiler and expose cfg(nightly_build) for portable_simd.
    // Downstream crates compiled with stable Rust will skip all std::simd code
    // automatically without needing to set any feature flag.
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    if let Ok(out) = std::process::Command::new(&rustc).arg("--version").output() {
        if String::from_utf8_lossy(&out.stdout).contains("nightly") {
            println!("cargo:rustc-cfg=nightly_build");
        }
    }
}
