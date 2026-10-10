//! Apple's Accelerate framework, on macOS (aarch64), whose `cblas_sgemm` runs
//! on the AMX units. `kernels::matmul::matmul` hands it every product it can
//! express; nothing else calls it.

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;

unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
    fn setenv(name: *const std::ffi::c_char, value: *const std::ffi::c_char, overwrite: i32) -> i32;
}

/// Keeps Accelerate on one thread, set before its first call: the products
/// here are small enough that spawning threads costs more than it saves.
fn single_threaded() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| unsafe {
        setenv(c"VECLIB_MAXIMUM_THREADS".as_ptr(), c"1".as_ptr(), 1);
    });
}

/// `C = alpha * op(A) * op(B) + beta * C` for an `m x n` row-major C, `ldc`
/// entries between its rows. `op(A)` is `m x k`: `a` holds it row-major with
/// `lda` entries between rows, or, with `trans_a`, its transpose that way.
/// The same for `op(B)`, `k x n`. With `beta == 0`, C is only written.
///
/// # Safety
///
/// The pointers must be valid for every element those sizes and strides
/// address, and C must not overlap A or B.
pub(crate) unsafe fn sgemm(
    (trans_a, trans_b): (bool, bool),
    (m, n, k): (usize, usize, usize),
    alpha: f32,
    (a, lda): (*const f32, usize),
    (b, ldb): (*const f32, usize),
    beta: f32,
    (c, ldc): (*mut f32, usize),
) {
    single_threaded();
    let op = |t: bool| if t { CBLAS_TRANS } else { CBLAS_NO_TRANS };
    let int = |v: usize| i32::try_from(v).expect("Accelerate: dimension exceeds i32");
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            op(trans_a),
            op(trans_b),
            int(m),
            int(n),
            int(k),
            alpha,
            a,
            int(lda),
            b,
            int(ldb),
            beta,
            c,
            int(ldc),
        );
    }
}
