//! Helpers for checking kernels against a scalar f64 reference: deterministic
//! inputs, lengths that reach every loop tail, and every SIMD level the
//! machine running the tests can execute.

use fearless_simd::Level;

/// Lengths on both sides of each vector width (4, 8, 16) and of the unrolled
/// steps built from them, so main loops, vector tails and scalar tails all run.
pub const AWKWARD_LENS: &[usize] = &[
    1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257,
    1000, 1027,
];

/// xorshift64*: deterministic, so a failure reproduces.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    /// Uniform in [lo, hi).
    pub fn f32(&mut self, lo: f32, hi: f32) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40;
        lo + (hi - lo) * (bits as f32 / (1u64 << 24) as f32)
    }

    pub fn vec(&mut self, len: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..len).map(|_| self.f32(lo, hi)).collect()
    }
}

/// Every SIMD level this CPU supports, best first, ending with the scalar
/// fallback.
pub fn levels() -> Vec<Level> {
    let best = Level::new();
    #[allow(unused_mut)]
    let mut levels: Vec<Level> = Vec::new();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    levels.extend(
        [
            best.as_avx512().map(Level::Avx512),
            best.as_avx2().map(Level::Avx2),
            best.as_sse4_2().map(Level::Sse4_2),
            best.as_sse2().map(Level::Sse2),
        ]
        .into_iter()
        .flatten(),
    );
    #[cfg(target_arch = "aarch64")]
    levels.extend(best.as_neon().map(Level::Neon));
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    levels.extend(best.as_wasm_simd128().map(Level::WasmSimd128));
    levels.push(Level::fallback());
    levels
}

/// Fails unless every element is within `tol` of the reference, relative to
/// the reference's magnitude once that exceeds 1; returns the worst error.
pub fn assert_close(got: &[f32], want: &[f64], tol: f64, what: &str) -> f64 {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = (0.0f64, 0usize);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let err = (g as f64 - w).abs() / w.abs().max(1.0);
        if !(err <= worst.0) {
            worst = (err, i);
        }
    }
    let (err, i) = worst;
    assert!(
        err <= tol,
        "{what}: error {err:.3e} > {tol:.0e} at {i}: got {}, want {}",
        got[i],
        want[i]
    );
    err
}
