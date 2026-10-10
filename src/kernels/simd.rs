//! Shared plumbing for kernels written with `fearless_simd`.

/// Calls the `#[simd]` function `$f` with the token of `$level` and the given
/// arguments, like `fearless_simd::dispatch!(level, simd => f(simd, ...))`.
///
/// `dispatch!` runs its expression in a closure, which takes the arguments by
/// reference, so the kernel starts by loading them back through pointers.
/// Calling `$f` with a concrete token instead lets `#[simd]`'s own entry pass
/// them in registers; on short rows that is a few nanoseconds a call. Levels
/// not listed here still go through `dispatch!`.
///
/// `simd_call!(level, max = Avx2, f(...))` runs AVX-512 machines at the AVX2
/// level instead. Use it for kernels that stream memory with little compute
/// per element: each 512-bit access that is not 64-byte aligned is split
/// across two cache lines, against at most every other 256-bit one, and rows
/// of activations rarely start 64-byte aligned. On an Ice Lake Xeon this
/// turned LayerNorm and RMSNorm from up to 16% slower than AVX2 into parity.
macro_rules! simd_call {
    ($level:expr, max = Avx2, $f:ident($($arg:expr),* $(,)?)) => {{
        #[allow(unused_mut)]
        let mut level: fearless_simd::Level = $level;
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let fearless_simd::Level::Avx512(_) = level {
            level = fearless_simd::Level::Avx2(level.as_avx2().expect("AVX-512 level implies AVX2"));
        }
        $crate::kernels::simd::simd_call!(level, $f($($arg),*))
    }};
    ($level:expr, $f:ident($($arg:expr),* $(,)?)) => {{
        use fearless_simd::Level;
        let level: Level = $level;
        match level {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Level::Avx512(token) => $f(token, $($arg),*),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Level::Avx2(token) => $f(token, $($arg),*),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Level::Sse4_2(token) => $f(token, $($arg),*),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Level::Sse2(token) => $f(token, $($arg),*),
            #[cfg(target_arch = "aarch64")]
            Level::Neon(token) => $f(token, $($arg),*),
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Level::WasmSimd128(token) => $f(token, $($arg),*),
            #[allow(unreachable_patterns)]
            _ => fearless_simd::dispatch!(level, simd => $f(simd, $($arg),*)),
        }
    }};
}

pub(crate) use simd_call;
