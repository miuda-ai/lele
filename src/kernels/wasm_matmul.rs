//! Optimized matrix multiplication for wasm32 targets using SIMD128.
//!
//! Provides API-compatible shims for faer types (MatRef, MatMut, Accum, Par)
//! so that kernel code can use the same call sites on both native and wasm targets.
//!
//! Uses wasm_simd128 intrinsics with tiled micro-kernel for high throughput.

#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

/// Accumulation mode for matmul output.
pub enum Accum {
    /// Overwrite the destination: C = alpha * A * B
    Replace,
    /// Accumulate into destination: C += alpha * A * B
    Add,
}

/// Parallelism control (always sequential on wasm).
pub struct Par;

impl Par {
    #[allow(non_upper_case_globals)]
    pub const Seq: Par = Par;
}

/// Read-only matrix reference with stride information.
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
    /// Create a MatRef from raw parts (mirrors faer API).
    ///
    /// # Safety
    /// The pointer must be valid for `nrows * ncols` elements with the given strides.
    pub unsafe fn from_raw_parts(
        ptr: *const f32,
        nrows: usize,
        ncols: usize,
        rs: isize,
        cs: isize,
    ) -> Self {
        Self {
            ptr,
            nrows,
            ncols,
            rs,
            cs,
        }
    }

    #[inline(always)]
    unsafe fn get(&self, i: usize, j: usize) -> f32 {
        unsafe { *self.ptr.offset(i as isize * self.rs + j as isize * self.cs) }
    }

    #[inline(always)]
    unsafe fn row_ptr(&self, i: usize) -> *const f32 {
        unsafe { self.ptr.offset(i as isize * self.rs) }
    }

    pub fn as_ref(&self) -> &Self {
        self
    }

    /// Returns true if matrix is stored in row-major contiguous layout (cs=1).
    #[inline(always)]
    fn is_row_major(&self) -> bool {
        self.cs == 1
    }
}

/// Mutable matrix reference with stride information.
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
    /// Create a MatMut from raw parts (mirrors faer API).
    ///
    /// # Safety
    /// The pointer must be valid for `nrows * ncols` elements with the given strides.
    pub unsafe fn from_raw_parts_mut(
        ptr: *mut f32,
        nrows: usize,
        ncols: usize,
        rs: isize,
        cs: isize,
    ) -> Self {
        Self {
            ptr,
            nrows,
            ncols,
            rs,
            cs,
        }
    }

    #[inline(always)]
    unsafe fn get_mut(&mut self, i: usize, j: usize) -> *mut f32 {
        unsafe { self.ptr.offset(i as isize * self.rs + j as isize * self.cs) }
    }

    #[inline(always)]
    unsafe fn row_ptr_mut(&mut self, i: usize) -> *mut f32 {
        unsafe { self.ptr.offset(i as isize * self.rs) }
    }

    pub fn as_mut(&mut self) -> &mut Self {
        self
    }

    /// Returns true if matrix is stored in row-major contiguous layout (cs=1).
    #[inline(always)]
    fn is_row_major(&self) -> bool {
        self.cs == 1
    }
}

// Tile sizes for cache-friendly blocking
const MC: usize = 64; // Panel height for A
const NC: usize = 256; // Panel width for B
const KC: usize = 64; // Panel depth (K dimension)

/// Matrix multiplication: C = alpha * A * B (Replace) or C += alpha * A * B (Add).
///
/// Uses WASM SIMD128 with tiled micro-kernel when matrices are row-major contiguous.
/// Falls back to scalar for non-contiguous strides.
pub fn matmul(
    mut dst: MatMut<f32>,
    accum: Accum,
    a: MatRef<f32>,
    b: MatRef<f32>,
    alpha: f32,
    _par: Par,
) {
    let m = a.nrows;
    let k = a.ncols;
    let n = b.ncols;

    // Use optimized SIMD path when all matrices are row-major contiguous
    if a.is_row_major() && b.is_row_major() && dst.is_row_major() {
        unsafe {
            matmul_simd_tiled(&mut dst, &accum, &a, &b, alpha, m, k, n);
        }
        return;
    }

    // Fallback: strided access (rare in practice)
    unsafe {
        match accum {
            Accum::Replace => {
                for i in 0..m {
                    for j in 0..n {
                        *dst.get_mut(i, j) = 0.0;
                    }
                }
            }
            Accum::Add => {}
        }

        for i in 0..m {
            for p in 0..k {
                let a_val = a.get(i, p) * alpha;
                if a_val == 0.0 {
                    continue;
                }
                for j in 0..n {
                    let b_val = b.get(p, j);
                    let dst_ptr = dst.get_mut(i, j);
                    *dst_ptr += a_val * b_val;
                }
            }
        }
    }
}

/// SIMD128 tiled matmul for row-major contiguous matrices.
#[cfg(target_arch = "wasm32")]
#[inline(never)]
unsafe fn matmul_simd_tiled(
    dst: &mut MatMut<f32>,
    accum: &Accum,
    a: &MatRef<f32>,
    b: &MatRef<f32>,
    alpha: f32,
    m: usize,
    k: usize,
    n: usize,
) {
    let dst_ptr = dst.ptr;
    let a_ptr = a.ptr;
    let b_ptr = b.ptr;
    let a_rs = a.rs as usize;
    let b_rs = b.rs as usize;
    let d_rs = dst.rs as usize;

    // Initialize C based on accumulation mode
    match accum {
        Accum::Replace => {
            for i in 0..m {
                let row = dst_ptr.add(i * d_rs);
                let mut j = 0;
                let zero = f32x4_splat(0.0);
                while j + 4 <= n {
                    v128_store(row.add(j) as *mut v128, zero);
                    j += 4;
                }
                while j < n {
                    *row.add(j) = 0.0;
                    j += 1;
                }
            }
        }
        Accum::Add => {}
    }

    let alpha_v = f32x4_splat(alpha);

    // Tiled loop: iterate over K in blocks for cache locality
    let mut kk = 0;
    while kk < k {
        let kb = KC.min(k - kk);

        let mut ii = 0;
        while ii < m {
            let ib = MC.min(m - ii);

            // Process rows of A-panel × B-panel
            for i in ii..ii + ib {
                let c_row = dst_ptr.add(i * d_rs);
                let a_row = a_ptr.add(i * a_rs + kk);

                // For each k in this tile, broadcast a[i,p] and do SIMD FMA across n
                for p in 0..kb {
                    let a_val = *a_row.add(p) * alpha;
                    if a_val == 0.0 {
                        continue;
                    }
                    let a_splat = f32x4_splat(a_val);
                    let b_row = b_ptr.add((kk + p) * b_rs);

                    let mut j = 0;
                    // Process 16 elements per iteration (4x unroll)
                    while j + 16 <= n {
                        let c0 = v128_load(c_row.add(j) as *const v128);
                        let c1 = v128_load(c_row.add(j + 4) as *const v128);
                        let c2 = v128_load(c_row.add(j + 8) as *const v128);
                        let c3 = v128_load(c_row.add(j + 12) as *const v128);

                        let b0 = v128_load(b_row.add(j) as *const v128);
                        let b1 = v128_load(b_row.add(j + 4) as *const v128);
                        let b2 = v128_load(b_row.add(j + 8) as *const v128);
                        let b3 = v128_load(b_row.add(j + 12) as *const v128);

                        v128_store(c_row.add(j) as *mut v128, f32x4_add(c0, f32x4_mul(a_splat, b0)));
                        v128_store(c_row.add(j + 4) as *mut v128, f32x4_add(c1, f32x4_mul(a_splat, b1)));
                        v128_store(c_row.add(j + 8) as *mut v128, f32x4_add(c2, f32x4_mul(a_splat, b2)));
                        v128_store(c_row.add(j + 12) as *mut v128, f32x4_add(c3, f32x4_mul(a_splat, b3)));

                        j += 16;
                    }
                    // Process 4 elements per iteration
                    while j + 4 <= n {
                        let c0 = v128_load(c_row.add(j) as *const v128);
                        let b0 = v128_load(b_row.add(j) as *const v128);
                        v128_store(c_row.add(j) as *mut v128, f32x4_add(c0, f32x4_mul(a_splat, b0)));
                        j += 4;
                    }
                    // Scalar tail
                    while j < n {
                        *c_row.add(j) += a_val * *b_row.add(j);
                        j += 1;
                    }
                }
            }

            ii += ib;
        }

        kk += kb;
    }
}

/// Fallback for non-wasm targets (should not be reached).
#[cfg(not(target_arch = "wasm32"))]
unsafe fn matmul_simd_tiled(
    dst: &mut MatMut<f32>,
    accum: &Accum,
    a: &MatRef<f32>,
    b: &MatRef<f32>,
    alpha: f32,
    m: usize,
    k: usize,
    n: usize,
) {
    let dst_ptr = dst.ptr;
    let a_ptr = a.ptr;
    let b_ptr = b.ptr;
    let a_rs = a.rs as usize;
    let b_rs = b.rs as usize;
    let d_rs = dst.rs as usize;

    match accum {
        Accum::Replace => {
            for i in 0..m {
                let row = dst_ptr.add(i * d_rs);
                for j in 0..n {
                    *row.add(j) = 0.0;
                }
            }
        }
        Accum::Add => {}
    }

    for i in 0..m {
        let c_row = dst_ptr.add(i * d_rs);
        let a_row = a_ptr.add(i * a_rs);
        for p in 0..k {
            let a_val = *a_row.add(p) * alpha;
            if a_val == 0.0 {
                continue;
            }
            let b_row = b_ptr.add(p * b_rs);
            for j in 0..n {
                *c_row.add(j) += a_val * *b_row.add(j);
            }
        }
    }
}
