//! `f32` matrix multiplication on `fearless_simd`.
//!
//! `matmul` keeps the interface of `faer`'s (and of `wasm_matmul`), and looks at the
//! strides to pick a kernel written for that layout:
//!
//! - `gemm_nn`: A, B and C row-major, which is nearly every call. Blocked as in BLIS: a
//!   block of rows of A stays in L2 while each `KC x 16` panel of B stays in L1 under it,
//!   and a `6 x 16` tile of C is kept in registers across the depth of the panel. Rows of A
//!   are read in place, one broadcast per element, and so are the rows of B; only the last,
//!   narrower panel of B is packed.
//! - Any other layout is copied into row-major first.
//!
//! The only `unsafe` is turning the caller's pointers and strides into slices, once.

use crate::kernels::matmul_dot::{DotDims, gemm_dot};
use crate::kernels::matmul_nn::{Dims, SMALL_M, gemm_nn, gemm_small_m};
use fearless_simd::Level;

/// Accumulation mode for matmul output.
pub enum Accum {
    /// Overwrite the destination: C = alpha * A * B
    Replace,
    /// Accumulate into destination: C += alpha * A * B
    Add,
}

/// Parallelism control (always sequential).
pub struct Par;

impl Par {
    #[allow(non_upper_case_globals)]
    pub const Seq: Par = Par;
}

/// Read-only matrix with row and column strides (in elements).
pub struct MatRef<T> {
    ptr: *const T,
    nrows: usize,
    ncols: usize,
    rs: isize,
    cs: isize,
}

unsafe impl<T> Send for MatRef<T> {}
unsafe impl<T> Sync for MatRef<T> {}

impl MatRef<f32> {
    /// # Safety
    /// `ptr` must be valid for reads of every element the strides address.
    pub unsafe fn from_raw_parts(
        ptr: *const f32,
        nrows: usize,
        ncols: usize,
        rs: isize,
        cs: isize,
    ) -> Self {
        Self { ptr, nrows, ncols, rs, cs }
    }

    /// The elements the strides address, from the first to the last.
    fn data(&self) -> &[f32] {
        let len = extent(self.nrows, self.ncols, self.rs, self.cs);
        // SAFETY: `from_raw_parts` promised every addressed element is readable.
        unsafe { core::slice::from_raw_parts(self.ptr, len) }
    }
}

/// Mutable matrix with row and column strides (in elements).
pub struct MatMut<T> {
    ptr: *mut T,
    nrows: usize,
    ncols: usize,
    rs: isize,
    cs: isize,
}

unsafe impl<T> Send for MatMut<T> {}
unsafe impl<T> Sync for MatMut<T> {}

impl MatMut<f32> {
    /// # Safety
    /// `ptr` must be valid for reads and writes of every element the strides address.
    pub unsafe fn from_raw_parts_mut(
        ptr: *mut f32,
        nrows: usize,
        ncols: usize,
        rs: isize,
        cs: isize,
    ) -> Self {
        Self { ptr, nrows, ncols, rs, cs }
    }

    fn data(&mut self) -> &mut [f32] {
        let len = extent(self.nrows, self.ncols, self.rs, self.cs);
        // SAFETY: `from_raw_parts_mut` promised every addressed element is readable and writable.
        unsafe { core::slice::from_raw_parts_mut(self.ptr, len) }
    }
}

/// Number of elements from the first to the last one a matrix addresses.
fn extent(nrows: usize, ncols: usize, rs: isize, cs: isize) -> usize {
    assert!(rs >= 0 && cs >= 0, "matmul: negative strides are not supported");
    if nrows == 0 || ncols == 0 {
        return 0;
    }
    (nrows - 1) * rs as usize + (ncols - 1) * cs as usize + 1
}

/// Elements between rows, if the matrix is row-major.
fn row_major(nrows: usize, ncols: usize, rs: isize, cs: isize) -> Option<usize> {
    (cs == 1 && (rs >= ncols as isize || nrows <= 1)).then_some((rs as usize).max(ncols))
}

/// `C = alpha * A * B` (`Accum::Replace`) or `C += alpha * A * B` (`Accum::Add`).
pub fn matmul(
    dst: MatMut<f32>,
    accum: Accum,
    a: MatRef<f32>,
    b: MatRef<f32>,
    alpha: f32,
    _par: Par,
) {
    matmul_at(Level::new(), dst, accum, a, b, alpha)
}

pub(crate) fn matmul_at(
    level: Level,
    mut dst: MatMut<f32>,
    accum: Accum,
    a: MatRef<f32>,
    b: MatRef<f32>,
    alpha: f32,
) {
    let (m, k, n) = (a.nrows, a.ncols, b.ncols);
    assert_eq!(b.nrows, k, "matmul: inner dimensions differ");
    assert_eq!((dst.nrows, dst.ncols), (m, n), "matmul: output shape");
    if m == 0 || n == 0 {
        return;
    }
    let add = matches!(accum, Accum::Add);
    let (rsc, csc) = (dst.rs, dst.cs);
    let c = dst.data();
    if k == 0 {
        if !add {
            for i in 0..m {
                for j in 0..n {
                    c[i * rsc as usize + j * csc as usize] = 0.0;
                }
            }
        }
        return;
    }
    // Copy what is not row-major into row-major.
    let (a_data, b_data) = (a.data(), b.data());
    let a_rm;
    let (a_data, lda) = match row_major(m, k, a.rs, a.cs) {
        Some(lda) => (a_data, lda),
        None => {
            a_rm = gather(a_data, m, k, a.rs as usize, a.cs as usize);
            (&a_rm[..], k)
        }
    };
    // A narrow C is computed as dot products against the columns of B, a wider one
    // from panels of its rows.
    let b_rm;
    let operand = if n <= DOT_MAX_N {
        match row_major(n, k, b.cs, b.rs) {
            Some(ldbt) => BOperand::Columns(b_data, ldbt),
            None => {
                b_rm = gather(b_data, n, k, b.cs as usize, b.rs as usize);
                BOperand::Columns(&b_rm, k)
            }
        }
    } else if let Some(ldb) = row_major(k, n, b.rs, b.cs) {
        BOperand::Rows(b_data, ldb)
    } else if let Some(ldbt) = row_major(n, k, b.cs, b.rs) {
        BOperand::Columns(b_data, ldbt)
    } else {
        b_rm = gather(b_data, k, n, b.rs as usize, b.cs as usize);
        BOperand::Rows(&b_rm, n)
    };
    let run = |c: &mut [f32], ldc: usize| match operand {
        BOperand::Rows(b, ldb) => {
            let dims = Dims { m, n, k, lda, ldb, ldc, alpha, add, b_cols: false };
            if m <= SMALL_M {
                gemm_small_m(level, &dims, a_data, b, c)
            } else {
                gemm_nn(level, &dims, a_data, b, c)
            }
        }
        // Few rows or columns of C: dot products. Otherwise the columns are packed.
        BOperand::Columns(bt, ldbt) if n <= DOT_MAX_N || m <= SMALL_M => {
            gemm_dot(level, &DotDims { m, n, k, lda, ldbt, ldc, alpha, add }, a_data, bt, c)
        }
        BOperand::Columns(bt, ldb) => gemm_nn(
            level,
            &Dims { m, n, k, lda, ldb, ldc, alpha, add, b_cols: true },
            a_data,
            bt,
            c,
        ),
    };
    match row_major(m, n, rsc, csc) {
        Some(ldc) => run(c, ldc),
        None => {
            // Column-major (or otherwise strided) C: compute row-major, then scatter.
            let mut tmp = vec![0.0f32; m * n];
            if add {
                for i in 0..m {
                    for j in 0..n {
                        tmp[i * n + j] = c[i * rsc as usize + j * csc as usize];
                    }
                }
            }
            run(&mut tmp, n);
            for i in 0..m {
                for j in 0..n {
                    c[i * rsc as usize + j * csc as usize] = tmp[i * n + j];
                }
            }
        }
    }
}

/// At most this many columns of C are computed as dot products.
const DOT_MAX_N: usize = 12;

/// B as the kernels take it: row-major, or its transpose row-major (so its columns are
/// contiguous), each with the elements between rows.
#[derive(Clone, Copy)]
enum BOperand<'a> {
    Rows(&'a [f32], usize),
    Columns(&'a [f32], usize),
}

/// A row-major copy of a strided `rows x cols` matrix.
fn gather(data: &[f32], rows: usize, cols: usize, rs: usize, cs: usize) -> Vec<f32> {
    (0..rows)
        .flat_map(|i| (0..cols).map(move |j| (i, j)))
        .map(|(i, j)| data[i * rs + j * cs])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::test_util::{assert_close, levels, Rng};

    /// `C (+)= alpha * A B` in f64 over strided operands.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        (m, n, k): (usize, usize, usize),
        a: &[f32], (rsa, csa): (isize, isize),
        b: &[f32], (rsb, csb): (isize, isize),
        c: &[f32], (rsc, csc): (isize, isize),
        alpha: f32, add: bool,
    ) -> Vec<f64> {
        let mut out = Vec::new();
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f64;
                for p in 0..k {
                    s += a[(i as isize * rsa + p as isize * csa) as usize] as f64
                        * b[(p as isize * rsb + j as isize * csb) as usize] as f64;
                }
                let old = c[(i as isize * rsc + j as isize * csc) as usize] as f64;
                out.push(alpha as f64 * s + if add { old } else { 0.0 });
            }
        }
        out
    }

    #[test]
    fn test_matmul_matches_reference_at_every_level() {
        let mut rng = Rng::new(3);
        let sizes = [
            (1, 1, 1), (1, 16, 8), (6, 16, 4), (7, 17, 5), (12, 33, 9), (5, 10, 384),
            (13, 5, 40), (100, 31, 257), (97, 130, 300), (2, 200, 513), (64, 64, 64),
            (1, 77, 130), (3, 40, 50), (4, 37, 70), (4, 100, 9), (9, 1, 33), (20, 12, 40), (20, 13, 40),
            // Several depth blocks, column blocks and row groups, at and past the packing threshold.
            (33, 200, 1100), (40, 30, 700), (35, 220, 1030),
        ];
        for level in levels() {
            for &(m, n, k) in &sizes {
                // Row-major, then each operand transposed, then C column-major.
                for (ta, tb, tc) in [(false, false, false), (true, false, false), (false, true, false), (false, false, true), (true, true, true)] {
                    for (alpha, add) in [(1.0, false), (0.5, false), (1.0, true), (-1.5, true)] {
                        let a = rng.vec(m * k, -1.0, 1.0);
                        let b = rng.vec(k * n, -1.0, 1.0);
                        let c0 = rng.vec(m * n, -1.0, 1.0);
                        let sa = if ta { (1, m as isize) } else { (k as isize, 1) };
                        let sb = if tb { (1, k as isize) } else { (n as isize, 1) };
                        let sc = if tc { (1, m as isize) } else { (n as isize, 1) };
                        let want = reference((m, n, k), &a, sa, &b, sb, &c0, sc, alpha, add);
                        let mut c = c0.clone();
                        unsafe {
                            matmul_at(
                                level,
                                MatMut::from_raw_parts_mut(c.as_mut_ptr(), m, n, sc.0, sc.1),
                                if add { Accum::Add } else { Accum::Replace },
                                MatRef::from_raw_parts(a.as_ptr(), m, k, sa.0, sa.1),
                                MatRef::from_raw_parts(b.as_ptr(), k, n, sb.0, sb.1),
                                alpha,
                            );
                        }
                        let got: Vec<f32> = (0..m)
                            .flat_map(|i| (0..n).map(move |j| (i, j)))
                            .map(|(i, j)| c[(i as isize * sc.0 + j as isize * sc.1) as usize])
                            .collect();
                        assert_close(&got, &want, 1e-4, &format!("{level:?} {m}x{n}x{k} t={ta}{tb}{tc} alpha {alpha} add {add}"));
                    }
                }
            }
        }
    }

    #[test]
    fn test_matmul_with_padded_rows() {
        // Row-major operands whose rows are longer than their columns.
        let mut rng = Rng::new(5);
        let (m, n, k, lda, ldb, ldc) = (9, 20, 13, 17, 25, 31);
        let a = rng.vec(m * lda, -1.0, 1.0);
        let b = rng.vec(k * ldb, -1.0, 1.0);
        let c0 = rng.vec(m * ldc, -1.0, 1.0);
        for level in levels() {
            let want = reference((m, n, k), &a, (lda as isize, 1), &b, (ldb as isize, 1), &c0, (ldc as isize, 1), 1.0, true);
            let mut c = c0.clone();
            unsafe {
                matmul_at(
                    level,
                    MatMut::from_raw_parts_mut(c.as_mut_ptr(), m, n, ldc as isize, 1),
                    Accum::Add,
                    MatRef::from_raw_parts(a.as_ptr(), m, k, lda as isize, 1),
                    MatRef::from_raw_parts(b.as_ptr(), k, n, ldb as isize, 1),
                    1.0,
                );
            }
            let got: Vec<f32> = (0..m).flat_map(|i| (0..n).map(move |j| (i, j))).map(|(i, j)| c[i * ldc + j]).collect();
            assert_close(&got, &want, 1e-4, &format!("{level:?} padded"));
            // Columns past `n` are left alone.
            for i in 0..m {
                assert_eq!(c[i * ldc + n..(i + 1) * ldc], c0[i * ldc + n..(i + 1) * ldc], "{level:?} row {i}");
            }
        }
    }
}
