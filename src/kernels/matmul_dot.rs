//! `C (+)= alpha * A * B` as dot products, for row-major `f32` matrices with few columns
//! of C, on `fearless_simd`.
//!
//! `B` is given transposed (`bt[j]` is column `j` of B, contiguous), so every element of C
//! is a dot product of two contiguous rows: nothing needs packing, A is read once, and no
//! lane of the tile is wasted on a column that does not exist. The price is a horizontal
//! sum per element of C, which is why this is for narrow C, or for a B that is already
//! transposed.

use crate::kernels::simd::simd_call;
use fearless_simd::{Level, Simd, f32x8};
use fearless_simd_macros::simd;

/// Rows of A per tile.
const MR: usize = 4;
/// Columns of C per tile.
const NB: usize = 2;

/// `A` is `m x k` with `lda` between rows, `bt` is `n x k` with `ldbt` between rows,
/// `C` is `m x n` with `ldc` between rows.
#[derive(Clone, Copy)]
pub(crate) struct DotDims {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lda: usize,
    pub ldbt: usize,
    pub ldc: usize,
    pub alpha: f32,
    pub add: bool,
}

pub(crate) fn gemm_dot(level: Level, d: &DotDims, a: &[f32], bt: &[f32], c: &mut [f32]) {
    simd_call!(level, gemm_dot_simd(d, a, bt, c))
}

#[simd]
fn gemm_dot_simd<S: Simd>(simd: S, d: &DotDims, a: &[f32], bt: &[f32], c: &mut [f32]) {
    let mut i = 0;
    while i < d.m {
        let rows = MR.min(d.m - i);
        let mut j = 0;
        while j < d.n {
            let cols = NB.min(d.n - j);
            macro_rules! tile {
                ($r:literal, $c:literal) => {
                    dot_tile::<S, $r, $c>(simd, d, &a[i * d.lda..], &bt[j * d.ldbt..], &mut c[i * d.ldc + j..])
                };
            }
            match (rows, cols) {
                (4, 2) => tile!(4, 2),
                (3, 2) => tile!(3, 2),
                (2, 2) => tile!(2, 2),
                (1, 2) => tile!(1, 2),
                (4, 1) => tile!(4, 1),
                (3, 1) => tile!(3, 1),
                (2, 1) => tile!(2, 1),
                _ => tile!(1, 1),
            }
            j += NB;
        }
        i += MR;
    }
}

/// `R` rows of A (from `a`) against `C` columns of B (from `bt`), into `c`.
#[inline(always)]
fn dot_tile<S: Simd, const R: usize, const C: usize>(
    simd: S,
    d: &DotDims,
    a: &[f32],
    bt: &[f32],
    c: &mut [f32],
) {
    use fearless_simd::prelude::*;
    let k = d.k;
    let a_rows: [&[f32]; R] = core::array::from_fn(|r| &a[r * d.lda..][..k]);
    let b_rows: [&[f32]; C] = core::array::from_fn(|j| &bt[j * d.ldbt..][..k]);
    let a8 = a_rows.map(|row| row.as_chunks::<8>().0);
    let b8 = b_rows.map(|row| row.as_chunks::<8>().0);
    let steps = a8[0].len();
    // One check each here instead of one per access in the loop.
    for rows in a8.iter().chain(&b8) {
        assert_eq!(rows.len(), steps);
    }
    let zero = f32x8::splat(simd, 0.0);
    let mut acc = [[zero; C]; R];
    for q in 0..steps {
        let bv: [f32x8<S>; C] = core::array::from_fn(|j| f32x8::load_array_ref(simd, &b8[j][q]));
        for r in 0..R {
            let av = f32x8::load_array_ref(simd, &a8[r][q]);
            for j in 0..C {
                acc[r][j] = av.mul_add(bv[j], acc[r][j]);
            }
        }
    }
    for r in 0..R {
        for j in 0..C {
            let mut sum = acc[r][j].reduce_sum();
            for t in steps * 8..k {
                sum += a_rows[r][t] * b_rows[j][t];
            }
            let out = &mut c[r * d.ldc + j];
            *out = d.alpha * sum + if d.add { *out } else { 0.0 };
        }
    }
}
