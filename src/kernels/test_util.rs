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

/// The four binary operations, by name, as scalar functions.
pub const BINARY_OPS: [(&str, fn(f32, f32) -> f32); 4] = [
    ("add", |x, y| x + y),
    ("sub", |x, y| x - y),
    ("mul", |x, y| x * y),
    ("div", |x, y| x / y),
];

/// `op` over `a` and `b` under numpy broadcasting, element by element.
pub fn broadcast_reference(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
    op: impl Fn(f32, f32) -> f32,
) -> (Vec<usize>, Vec<f32>) {
    let shape = crate::kernels::utils::broadcast_shapes(a_shape, b_shape).unwrap();
    let rank = shape.len();
    let index = |operand: &[usize], coords: &[usize]| {
        let skipped = rank - operand.len();
        operand.iter().enumerate().fold(0, |idx, (d, &n)| {
            idx * n + if n == 1 { 0 } else { coords[skipped + d] }
        })
    };
    let out = (0..shape.iter().product::<usize>())
        .map(|flat| {
            let mut coords = vec![0; rank];
            let mut rest = flat;
            for d in (0..rank).rev() {
                coords[d] = rest % shape[d];
                rest /= shape[d];
            }
            op(a[index(a_shape, &coords)], b[index(b_shape, &coords)])
        })
        .collect();
    (shape, out)
}

/// Pairs of operand shapes: equal ones at awkward lengths, a scalar, one value
/// per channel or per row, a bias or mask repeated, a block in the middle, and
/// pairs where both operands broadcast.
pub fn binary_shapes() -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut shapes = Vec::new();
    for &n in AWKWARD_LENS {
        shapes.push((vec![n], vec![n]));
        shapes.push((vec![n], vec![1]));
        shapes.push((vec![1], vec![n]));
        shapes.push((vec![2, n], vec![n]));
        shapes.push((vec![n], vec![3, 1]));
    }
    for (a, b) in [
        // a scalar on either side
        (vec![2, 3, 5], vec![1, 1, 1]),
        (vec![], vec![4, 5]),
        // one value per channel, both ways round
        (vec![1, 8, 7, 7], vec![8, 1, 1]),
        (vec![1, 8, 7, 7], vec![1, 8, 1, 1]),
        (vec![2, 8, 7, 7], vec![2, 8, 1, 1]),
        (vec![2, 8, 7, 7], vec![1, 8, 1, 1]),
        (vec![8, 1, 1], vec![1, 8, 7, 7]),
        (vec![1, 64, 56, 56], vec![64, 1, 1]),
        (vec![1, 3, 100, 100], vec![3, 1, 1]),
        // one value per row
        (vec![10, 35], vec![10, 1]),
        (vec![10, 1], vec![10, 35]),
        (vec![2, 40, 33], vec![2, 40, 1]),
        // a trailing block repeated: a bias, a mask
        (vec![2, 3, 5, 7], vec![7]),
        (vec![2, 3, 5, 7], vec![5, 7]),
        (vec![2, 3, 5, 7], vec![1, 1, 5, 7]),
        (vec![7], vec![2, 3, 5, 7]),
        // as many values as channels, but along the last dimension
        (vec![1, 3, 4, 3], vec![3]),
        (vec![3], vec![1, 3, 4, 3]),
        (vec![1, 3, 4, 3], vec![1, 1, 1, 3]),
        (vec![40, 1920], vec![1920]),
        (vec![1, 8, 50, 50], vec![1, 1, 50, 50]),
        (vec![1, 1, 50, 50], vec![1, 8, 50, 50]),
        // a block in the middle
        (vec![2, 3, 4, 5], vec![3, 4, 1]),
        (vec![2, 3, 4, 5], vec![1, 3, 4, 1]),
        // not one block: both sides broadcast
        (vec![4, 1, 5], vec![1, 6, 1]),
        (vec![2, 3, 1, 5], vec![2, 1, 4, 1]),
        (vec![2, 1, 1, 5], vec![2, 3, 4, 5]),
        (vec![17, 1], vec![1, 19]),
    ] {
        shapes.push((a, b));
    }
    shapes
}

/// Operand values for the given shapes, with exact zeros among them, so that
/// division produces infinities and NaNs.
pub fn binary_inputs(a_shape: &[usize], b_shape: &[usize]) -> (Vec<f32>, Vec<f32>) {
    let values = |shape: &[usize], seed: u64| {
        let mut v = Rng::new(seed).vec(shape.iter().product(), -4.0, 4.0);
        for x in v.iter_mut().step_by(7) {
            *x = 0.0;
        }
        v
    };
    (values(a_shape, 1), values(b_shape, 2))
}

/// Fails unless the two are equal bit for bit, NaNs aside: the right check for
/// operations that are exactly rounded.
pub fn assert_same_bits(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()),
            "{what}: element {i}: got {g}, want {w}"
        );
    }
}
