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

// Tile sizes for cache-friendly 3-level blocking
/// M-dimension panel (number of C-rows per macro-kernel call).
const MC: usize = 64;
/// N-dimension panel (number of C-cols per jj-tile).  Sized so that the
/// B sub-panel (KC × NC × 4 bytes) fits in L2 cache (~256 KB).
const NC: usize = 512;
/// K-dimension panel depth.
const KC: usize = 128;
/// Micro-kernel row count: process MR rows of C simultaneously to amortise B loads.
const MR: usize = 4;
/// Micro-kernel col count: 4 SIMD f32x4 = 16 scalars per unrolled step.
const NR: usize = 16;

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

    // Pre-transpose non-row-major matrices into contiguous buffers,
    // then use the fast SIMD tiled kernel. This is critical for Gemm
    // ops with transB=1 (attention QKV, FFN layers).
    unsafe {
        let a_buf;
        let a_ref = if !a.is_row_major() {
            a_buf = transpose_to_row_major(&a, m, k);
            MatRef::<f32>::from_raw_parts(a_buf.as_ptr(), m, k, k as isize, 1)
        } else {
            a_buf = Vec::new();
            MatRef::<f32>::from_raw_parts(a.ptr, m, k, a.rs, a.cs)
        };

        let b_buf;
        let b_ref = if !b.is_row_major() {
            b_buf = transpose_to_row_major(&b, k, n);
            MatRef::<f32>::from_raw_parts(b_buf.as_ptr(), k, n, n as isize, 1)
        } else {
            b_buf = Vec::new();
            MatRef::<f32>::from_raw_parts(b.ptr, k, n, b.rs, b.cs)
        };

        matmul_simd_tiled(&mut dst, &accum, &a_ref, &b_ref, alpha, m, k, n);
    }
}

/// Transpose a strided matrix into a contiguous row-major buffer.
#[inline]
unsafe fn transpose_to_row_major(mat: &MatRef<f32>, rows: usize, cols: usize) -> Vec<f32> {
    let mut buf = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            *buf.get_unchecked_mut(i * cols + j) = mat.get(i, j);
        }
    }
    buf
}

/// SIMD128 tiled matmul for row-major contiguous matrices.
///
/// Loop order: jj (NC) → kk (KC) → ii (MC) → MR micro-kernel
/// B-panel packed:  B[kk:kk+KC, jj:jj+NC] → contiguous KC×NC buffer (reused across all ii).
/// Accumulator buffer: MR×NC per thread-local, eliminates repeated C loads across kk panels.
///   - kk==0: zero acc, accumulate into acc
///   - kk>0:  accumulate into acc (already has partial sums)
///   - kk is last: acc → C (replace) or acc+C → C (add)
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
    let a_rs = a.rs as usize; // weight matrix row stride (== k, small)
    let b_rs = b.rs as usize; // im2col matrix row stride (can be huge: out_h*out_w)
    let d_rs = dst.rs as usize;

    let do_replace = matches!(accum, Accum::Replace);

    // Thread-local buffers.
    // B-pack: KC × NC floats (256KB)
    // Acc:    MC × NC floats (128KB) — accumulator buffer, avoids C loads across kk panels
    thread_local! {
        static B_PACK_BUF: std::cell::RefCell<Vec<f32>> =
            std::cell::RefCell::new(Vec::with_capacity(KC * NC));
        static ACC_BUF: std::cell::RefCell<Vec<f32>> =
            std::cell::RefCell::new(Vec::with_capacity(MC * NC));
    }

    let b_pack_size = KC * NC;
    let acc_buf_size = MC * NC;
    let mut b_pack_raw: *mut f32 = std::ptr::null_mut();
    let mut acc_raw: *mut f32 = std::ptr::null_mut();
    B_PACK_BUF.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < b_pack_size { buf.resize(b_pack_size, 0.0); }
        b_pack_raw = buf.as_mut_ptr();
    });
    ACC_BUF.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < acc_buf_size { buf.resize(acc_buf_size, 0.0); }
        acc_raw = buf.as_mut_ptr();
    });
    let b_pack_ptr: *mut f32 = b_pack_raw;
    let acc_ptr: *mut f32 = acc_raw;

    let num_kk = (k + KC - 1) / KC; // total kk panels

    // ── outer jj loop: B-column panels ────────────────────────────────────────
    let mut jj = 0;
    while jj < n {
        let nb = NC.min(n - jj);

        // ── kk loop: K panels (B repacked here, reused across all ii) ─────────
        let mut kk = 0;
        let mut kk_panel = 0usize; // which panel index (0-based) for first/last detection
        while kk < k {
            let kb = KC.min(k - kk);
            let is_first_kk = kk_panel == 0;
            let is_last_kk  = kk_panel == num_kk - 1;

            // Pack B sub-panel: B[kk:kk+kb, jj:jj+nb] → b_pack[0..kb*NC]
            for p in 0..kb {
                let src = b_ptr.add((kk + p) * b_rs + jj);
                let dst_p = b_pack_ptr.add(p * NC);
                core::ptr::copy_nonoverlapping(src, dst_p, nb);
            }
            let b_packed = b_pack_ptr as *const f32;

            // ── ii loop: M panels ─────────────────────────────────────────────
            let mut ii = 0;
            while ii < m {
                let ib = MC.min(m - ii);
                let a_row_base = a_ptr.add(ii * a_rs + kk); // A[ii, kk]

                // ── MR stripe loop ─────────────────────────────────────────────
                let mut ir = 0;
                while ir < ib {
                    let rows_left = ib - ir;
                    let i = ii + ir;

                    if rows_left >= MR {
                        let acc_row = acc_ptr.add(ir * NC); // acc row for this stripe

                        if is_first_kk {
                            // Zero acc for this (ii, jj) from the very first kk pass
                            let zero = f32x4_splat(0.0);
                            let mut ji = 0;
                            while ji + 16 <= nb {
                                v128_store(acc_row.add(ji     ) as *mut v128, zero);
                                v128_store(acc_row.add(ji +  4) as *mut v128, zero);
                                v128_store(acc_row.add(ji +  8) as *mut v128, zero);
                                v128_store(acc_row.add(ji + 12) as *mut v128, zero);
                                v128_store(acc_row.add(ji      + NC) as *mut v128, zero);
                                v128_store(acc_row.add(ji +  4 + NC) as *mut v128, zero);
                                v128_store(acc_row.add(ji +  8 + NC) as *mut v128, zero);
                                v128_store(acc_row.add(ji + 12 + NC) as *mut v128, zero);
                                v128_store(acc_row.add(ji      + 2*NC) as *mut v128, zero);
                                v128_store(acc_row.add(ji +  4 + 2*NC) as *mut v128, zero);
                                v128_store(acc_row.add(ji +  8 + 2*NC) as *mut v128, zero);
                                v128_store(acc_row.add(ji + 12 + 2*NC) as *mut v128, zero);
                                v128_store(acc_row.add(ji      + 3*NC) as *mut v128, zero);
                                v128_store(acc_row.add(ji +  4 + 3*NC) as *mut v128, zero);
                                v128_store(acc_row.add(ji +  8 + 3*NC) as *mut v128, zero);
                                v128_store(acc_row.add(ji + 12 + 3*NC) as *mut v128, zero);
                                ji += 16;
                            }
                            while ji + 4 <= nb {
                                for r in 0..MR {
                                    v128_store(acc_row.add(ji + r * NC) as *mut v128, zero);
                                }
                                ji += 4;
                            }
                            while ji < nb {
                                for r in 0..MR { *acc_row.add(ji + r * NC) = 0.0; }
                                ji += 1;
                            }
                        }

                        // Accumulate into acc (loaded from acc, add A*B, store to acc)
                        micro_kernel_4r_packed(
                            nb, kb,
                            a_row_base.add(ir * a_rs), // A[ir, kk]
                            b_packed,
                            acc_row,   // acc row stride = NC
                            a_rs, NC, NC, alpha,
                        );

                        if is_last_kk {
                            // Write acc → C (replace or add)
                            let c_row = dst_ptr.add(i * d_rs + jj);
                            if do_replace {
                                for r in 0..MR {
                                    core::ptr::copy_nonoverlapping(
                                        acc_row.add(r * NC),
                                        c_row.add(r * d_rs),
                                        nb,
                                    );
                                }
                            } else {
                                for r in 0..MR {
                                    let src = acc_row.add(r * NC);
                                    let dst_r = c_row.add(r * d_rs);
                                    let mut ji = 0;
                                    while ji + 4 <= nb {
                                        let cv = v128_load(dst_r.add(ji) as *const v128);
                                        let av = v128_load(src.add(ji) as *const v128);
                                        v128_store(dst_r.add(ji) as *mut v128, f32x4_add(cv, av));
                                        ji += 4;
                                    }
                                    while ji < nb {
                                        *dst_r.add(ji) += *src.add(ji);
                                        ji += 1;
                                    }
                                }
                            }
                        }

                        ir += MR;
                    } else {
                        // Tail rows: handle first/last kk for C directly
                        let c_base = dst_ptr.add(i * d_rs + jj);
                        if is_first_kk && do_replace {
                            for r in 0..rows_left {
                                let row = c_base.add(r * d_rs);
                                let mut ji = 0;
                                while ji + 4 <= nb {
                                    v128_store(row.add(ji) as *mut v128, f32x4_splat(0.0));
                                    ji += 4;
                                }
                                while ji < nb { *row.add(ji) = 0.0; ji += 1; }
                            }
                        }
                        for r in 0..rows_left {
                            let c_r = c_base.add(r * d_rs);
                            let a_row_r = a_row_base.add((ir + r) * a_rs);
                            for p in 0..kb {
                                let av = *a_row_r.add(p) * alpha;
                                if av == 0.0 { continue; }
                                let b_row = b_packed.add(p * NC);
                                let av_v = f32x4_splat(av);
                                let mut ji = 0;
                                while ji + 4 <= nb {
                                    let cv = v128_load(c_r.add(ji) as *const v128);
                                    let bv = v128_load(b_row.add(ji) as *const v128);
                                    v128_store(c_r.add(ji) as *mut v128,
                                        f32x4_add(cv, f32x4_mul(av_v, bv)));
                                    ji += 4;
                                }
                                while ji < nb {
                                    *c_r.add(ji) += av * *b_row.add(ji);
                                    ji += 1;
                                }
                            }
                        }
                        ir += rows_left;
                    }
                }

                ii += ib;
            }
            kk += kb;
            kk_panel += 1;
        }
        jj += nb;
    }
}

/// 4-row × nb-col register-blocked micro-kernel.
///
/// `a` points to A row data (direct, stride = a_rs).
/// `b` points to packed B panel: kb rows × NC cols (b_rs_packed = NC, contiguous).
/// `c` is the output sub-matrix with stride `c_rs`.
/// Computes: C[0..4, 0..nb] += alpha * A[0..4, 0..kb] × B_packed[0..kb, 0..nb]
#[cfg(target_arch = "wasm32")]
#[inline(always)]
unsafe fn micro_kernel_4r_packed(
    nb: usize,
    kb: usize,
    a: *const f32,  // A rows (direct, stride = a_rs)
    b: *const f32,  // packed B: kb rows × NC cols (b_rs_packed = NC)
    c: *mut f32,    // output C: 4 rows × nb, stride = c_rs
    a_rs: usize,    // A row stride (== k, weight matrix width)
    b_rs: usize,    // = NC (packed)
    c_rs: usize,    // output stride
    alpha: f32,
) {
    let a0 = a;
    let a1 = a.add(a_rs);
    let a2 = a.add(2 * a_rs);
    let a3 = a.add(3 * a_rs);

    let c0 = c;
    let c1 = c.add(c_rs);
    let c2 = c.add(2 * c_rs);
    let c3 = c.add(3 * c_rs);

    let mut j = 0;

    // ── 16-col unrolled loop (4 SIMD f32x4 per row) ──────────────────────────
    while j + NR <= nb {
        // Load 4×4 accumulator (current C values)
        let mut r00 = v128_load(c0.add(j     ) as *const v128);
        let mut r01 = v128_load(c0.add(j +  4) as *const v128);
        let mut r02 = v128_load(c0.add(j +  8) as *const v128);
        let mut r03 = v128_load(c0.add(j + 12) as *const v128);

        let mut r10 = v128_load(c1.add(j     ) as *const v128);
        let mut r11 = v128_load(c1.add(j +  4) as *const v128);
        let mut r12 = v128_load(c1.add(j +  8) as *const v128);
        let mut r13 = v128_load(c1.add(j + 12) as *const v128);

        let mut r20 = v128_load(c2.add(j     ) as *const v128);
        let mut r21 = v128_load(c2.add(j +  4) as *const v128);
        let mut r22 = v128_load(c2.add(j +  8) as *const v128);
        let mut r23 = v128_load(c2.add(j + 12) as *const v128);

        let mut r30 = v128_load(c3.add(j     ) as *const v128);
        let mut r31 = v128_load(c3.add(j +  4) as *const v128);
        let mut r32 = v128_load(c3.add(j +  8) as *const v128);
        let mut r33 = v128_load(c3.add(j + 12) as *const v128);

        // K accumulation – B is contiguous (packed), b.add(p*NC+j) is sequential
        // Fast path: skip alpha multiply when alpha == 1.0 (common case)
        let mut p = 0;
        if alpha == 1.0 {
            while p < kb {
                let b_row = b.add(p * b_rs + j);
                let bv0 = v128_load(b_row        as *const v128);
                let bv1 = v128_load(b_row.add( 4) as *const v128);
                let bv2 = v128_load(b_row.add( 8) as *const v128);
                let bv3 = v128_load(b_row.add(12) as *const v128);

                let s0 = f32x4_splat(*a0.add(p));
                let s1 = f32x4_splat(*a1.add(p));
                let s2 = f32x4_splat(*a2.add(p));
                let s3 = f32x4_splat(*a3.add(p));

                r00 = f32x4_add(r00, f32x4_mul(s0, bv0));
                r01 = f32x4_add(r01, f32x4_mul(s0, bv1));
                r02 = f32x4_add(r02, f32x4_mul(s0, bv2));
                r03 = f32x4_add(r03, f32x4_mul(s0, bv3));

                r10 = f32x4_add(r10, f32x4_mul(s1, bv0));
                r11 = f32x4_add(r11, f32x4_mul(s1, bv1));
                r12 = f32x4_add(r12, f32x4_mul(s1, bv2));
                r13 = f32x4_add(r13, f32x4_mul(s1, bv3));

                r20 = f32x4_add(r20, f32x4_mul(s2, bv0));
                r21 = f32x4_add(r21, f32x4_mul(s2, bv1));
                r22 = f32x4_add(r22, f32x4_mul(s2, bv2));
                r23 = f32x4_add(r23, f32x4_mul(s2, bv3));

                r30 = f32x4_add(r30, f32x4_mul(s3, bv0));
                r31 = f32x4_add(r31, f32x4_mul(s3, bv1));
                r32 = f32x4_add(r32, f32x4_mul(s3, bv2));
                r33 = f32x4_add(r33, f32x4_mul(s3, bv3));

                p += 1;
            }
        } else {
            while p < kb {
                let b_row = b.add(p * b_rs + j);
                let bv0 = v128_load(b_row        as *const v128);
                let bv1 = v128_load(b_row.add( 4) as *const v128);
                let bv2 = v128_load(b_row.add( 8) as *const v128);
                let bv3 = v128_load(b_row.add(12) as *const v128);

                let s0 = f32x4_splat(*a0.add(p) * alpha);
                let s1 = f32x4_splat(*a1.add(p) * alpha);
                let s2 = f32x4_splat(*a2.add(p) * alpha);
                let s3 = f32x4_splat(*a3.add(p) * alpha);

                r00 = f32x4_add(r00, f32x4_mul(s0, bv0));
                r01 = f32x4_add(r01, f32x4_mul(s0, bv1));
                r02 = f32x4_add(r02, f32x4_mul(s0, bv2));
                r03 = f32x4_add(r03, f32x4_mul(s0, bv3));

                r10 = f32x4_add(r10, f32x4_mul(s1, bv0));
                r11 = f32x4_add(r11, f32x4_mul(s1, bv1));
                r12 = f32x4_add(r12, f32x4_mul(s1, bv2));
                r13 = f32x4_add(r13, f32x4_mul(s1, bv3));

                r20 = f32x4_add(r20, f32x4_mul(s2, bv0));
                r21 = f32x4_add(r21, f32x4_mul(s2, bv1));
                r22 = f32x4_add(r22, f32x4_mul(s2, bv2));
                r23 = f32x4_add(r23, f32x4_mul(s2, bv3));

                r30 = f32x4_add(r30, f32x4_mul(s3, bv0));
                r31 = f32x4_add(r31, f32x4_mul(s3, bv1));
                r32 = f32x4_add(r32, f32x4_mul(s3, bv2));
                r33 = f32x4_add(r33, f32x4_mul(s3, bv3));

                p += 1;
            }
        } // end if alpha == 1.0

        v128_store(c0.add(j     ) as *mut v128, r00);
        v128_store(c0.add(j +  4) as *mut v128, r01);
        v128_store(c0.add(j +  8) as *mut v128, r02);
        v128_store(c0.add(j + 12) as *mut v128, r03);

        v128_store(c1.add(j     ) as *mut v128, r10);
        v128_store(c1.add(j +  4) as *mut v128, r11);
        v128_store(c1.add(j +  8) as *mut v128, r12);
        v128_store(c1.add(j + 12) as *mut v128, r13);

        v128_store(c2.add(j     ) as *mut v128, r20);
        v128_store(c2.add(j +  4) as *mut v128, r21);
        v128_store(c2.add(j +  8) as *mut v128, r22);
        v128_store(c2.add(j + 12) as *mut v128, r23);

        v128_store(c3.add(j     ) as *mut v128, r30);
        v128_store(c3.add(j +  4) as *mut v128, r31);
        v128_store(c3.add(j +  8) as *mut v128, r32);
        v128_store(c3.add(j + 12) as *mut v128, r33);

        j += NR;
    }

    // ── 4-col tail ───────────────────────────────────────────────────────────
    while j + 4 <= nb {
        let mut r0 = v128_load(c0.add(j) as *const v128);
        let mut r1 = v128_load(c1.add(j) as *const v128);
        let mut r2 = v128_load(c2.add(j) as *const v128);
        let mut r3 = v128_load(c3.add(j) as *const v128);

        for p in 0..kb {
            let bv = v128_load(b.add(p * b_rs + j) as *const v128);
            r0 = f32x4_add(r0, f32x4_mul(f32x4_splat(*a0.add(p) * alpha), bv));
            r1 = f32x4_add(r1, f32x4_mul(f32x4_splat(*a1.add(p) * alpha), bv));
            r2 = f32x4_add(r2, f32x4_mul(f32x4_splat(*a2.add(p) * alpha), bv));
            r3 = f32x4_add(r3, f32x4_mul(f32x4_splat(*a3.add(p) * alpha), bv));
        }

        v128_store(c0.add(j) as *mut v128, r0);
        v128_store(c1.add(j) as *mut v128, r1);
        v128_store(c2.add(j) as *mut v128, r2);
        v128_store(c3.add(j) as *mut v128, r3);

        j += 4;
    }

    // ── scalar tail ──────────────────────────────────────────────────────────
    while j < nb {
        for p in 0..kb {
            let av0 = *a0.add(p) * alpha;
            let av1 = *a1.add(p) * alpha;
            let av2 = *a2.add(p) * alpha;
            let av3 = *a3.add(p) * alpha;
            let bv = *b.add(p * b_rs + j);
            *c0.add(j) += av0 * bv;
            *c1.add(j) += av1 * bv;
            *c2.add(j) += av2 * bv;
            *c3.add(j) += av3 * bv;
        }
        j += 1;
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
