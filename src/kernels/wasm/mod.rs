/// WASM SIMD128 kernel implementations.
/// All WASM-specific code lives here; dispatched from the top-level kernel modules.
#[cfg(target_arch = "wasm32")]
pub mod math;
#[cfg(target_arch = "wasm32")]
pub mod norm;
