//! C ABI wrappers exposing lele kernels for Python benchmarking.
//!
//! Build with:
//!   cargo build --release -p kernel-bench
//! Produces: target/release/libkernel_bench.so (Linux)

use lele::tensor::TensorView;
use std::borrow::Cow;

// ── helpers ─────────────────────────────────────────────────────────────────

unsafe fn tv<'a>(ptr: *const f32, shape: &'a [usize]) -> TensorView<'a> {
    let len: usize = shape.iter().product();
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    TensorView {
        data: Cow::Borrowed(slice),
        shape: Cow::Borrowed(shape),
    }
}

unsafe fn tv_u8<'a>(ptr: *const u8, shape: &'a [usize]) -> TensorView<'a, u8> {
    let len: usize = shape.iter().product();
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    TensorView {
        data: Cow::Borrowed(slice),
        shape: Cow::Borrowed(shape),
    }
}

fn copy_out(result: &TensorView<f32>, out: *mut f32) {
    let slice = unsafe { std::slice::from_raw_parts_mut(out, result.data.len()) };
    slice.copy_from_slice(&result.data);
}

// ── 1. MatMul ────────────────────────────────────────────────────────────────
// A[M,K] * B[K,N] -> C[M,N]

#[unsafe(no_mangle)]
pub extern "C" fn lele_matmul(
    a_ptr: *const f32,
    m: i64,
    k: i64,
    b_ptr: *const f32,
    n: i64,
    out_ptr: *mut f32,
) {
    let ash = [m as usize, k as usize];
    let bsh = [k as usize, n as usize];
    let a = unsafe { tv(a_ptr, &ash) };
    let b = unsafe { tv(b_ptr, &bsh) };
    let mut buf = vec![0.0f32; (m * n) as usize];
    let r = lele::kernels::matmul(&a, &b, &mut buf);
    copy_out(&r, out_ptr);
}

// ── 2. GEMM (transB=true, with optional bias) ────────────────────────────────
// A[M,K] * B_T[N,K]^T + bias[N] -> C[M,N]

#[unsafe(no_mangle)]
pub extern "C" fn lele_gemm(
    a_ptr: *const f32,
    m: i64,
    k: i64,
    b_ptr: *const f32,
    n: i64,
    bias_ptr: *const f32, // null = no bias
    trans_a: i32,
    trans_b: i32,
    out_ptr: *mut f32,
) {
    let ash = [m as usize, k as usize];
    let bsh = [n as usize, k as usize]; // stored as N×K, transposed during gemm
    let bsh2 = [n as usize];
    let a = unsafe { tv(a_ptr, &ash) };
    let b = unsafe { tv(b_ptr, &bsh) };
    let bias_opt = if bias_ptr.is_null() {
        None
    } else {
        Some(unsafe { tv(bias_ptr, &bsh2) })
    };
    let mut buf = vec![0.0f32; (m * n) as usize];
    let r = lele::kernels::gemm(
        &a,
        &b,
        bias_opt.as_ref(),
        1.0,
        1.0,
        trans_a != 0,
        trans_b != 0,
        &mut buf,
    );
    copy_out(&r, out_ptr);
}

// ── 3. Layer Norm ────────────────────────────────────────────────────────────
// x[batch, hidden], scale[hidden], bias[hidden] -> out[batch, hidden]

#[unsafe(no_mangle)]
pub extern "C" fn lele_layer_norm(
    x_ptr: *const f32,
    batch: i64,
    hidden: i64,
    scale_ptr: *const f32,
    bias_ptr: *const f32, // null = no bias
    axis: i64,
    epsilon: f32,
    out_ptr: *mut f32,
) {
    let xsh = [batch as usize, hidden as usize];
    let ssh = [hidden as usize];
    let x = unsafe { tv(x_ptr, &xsh) };
    let scale = unsafe { tv(scale_ptr, &ssh) };
    let bias_storage;
    let bias = if bias_ptr.is_null() {
        TensorView::empty()
    } else {
        bias_storage = unsafe { tv(bias_ptr, &ssh) };
        bias_storage
    };
    let mut buf = vec![0.0f32; (batch * hidden) as usize];
    let r = lele::kernels::layer_norm(&x, &scale, &bias, axis as i32, epsilon, &mut buf);
    copy_out(&r, out_ptr);
}

// ── 4. RMS Norm ──────────────────────────────────────────────────────────────
// x[batch, hidden], scale[hidden] -> out[batch, hidden]

#[unsafe(no_mangle)]
pub extern "C" fn lele_rms_norm(
    x_ptr: *const f32,
    batch: i64,
    hidden: i64,
    scale_ptr: *const f32,
    epsilon: f32,
    out_ptr: *mut f32,
) {
    let xsh = [batch as usize, hidden as usize];
    let ssh = [hidden as usize];
    let x = unsafe { tv(x_ptr, &xsh) };
    let scale = unsafe { tv(scale_ptr, &ssh) };
    let mut buf = vec![0.0f32; (batch * hidden) as usize];
    let r = lele::kernels::norm::rms_norm(&x, &scale, -1i32, epsilon, &mut buf);
    copy_out(&r, out_ptr);
}

// ── 5. Softmax ───────────────────────────────────────────────────────────────
// x[batch, seq] -> out[batch, seq]

#[unsafe(no_mangle)]
pub extern "C" fn lele_softmax(
    x_ptr: *const f32,
    batch: i64,
    seq: i64,
    axis: i64,
    out_ptr: *mut f32,
) {
    let xsh = [batch as usize, seq as usize];
    let x = unsafe { tv(x_ptr, &xsh) };
    let mut buf = vec![0.0f32; (batch * seq) as usize];
    let r = lele::kernels::norm::softmax(&x, axis as i32, &mut buf);
    copy_out(&r, out_ptr);
}

// ── 6. Elementwise activations ───────────────────────────────────────────────

macro_rules! activation_ffi {
    ($sym:ident, $fn:expr) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $sym(x_ptr: *const f32, n: i64, out_ptr: *mut f32) {
            let sh = [n as usize];
            let x = unsafe { tv(x_ptr, &sh) };
            let mut buf = vec![0.0f32; n as usize];
            let r = $fn(&x, &mut buf);
            copy_out(&r, out_ptr);
        }
    };
}

activation_ffi!(lele_relu, lele::kernels::math::relu);
activation_ffi!(lele_silu, lele::kernels::math::silu);
activation_ffi!(lele_gelu, lele::kernels::math::gelu);
activation_ffi!(lele_fast_gelu, lele::kernels::math::fast_gelu);
activation_ffi!(lele_tanh, lele::kernels::math::tanh_kernel);
activation_ffi!(lele_sigmoid, lele::kernels::math::sigmoid);
activation_ffi!(lele_erf, lele::kernels::math::erf);

// ── 7. Transpose ─────────────────────────────────────────────────────────────
// Generic N-D transpose. shape/perm passed as flat arrays.

#[unsafe(no_mangle)]
pub extern "C" fn lele_transpose(
    x_ptr: *const f32,
    shape_ptr: *const i64,
    ndim: i64,
    perm_ptr: *const i64,
    out_ptr: *mut f32,
) {
    let ndim = ndim as usize;
    let raw_shape = unsafe { std::slice::from_raw_parts(shape_ptr, ndim) };
    let perm = unsafe { std::slice::from_raw_parts(perm_ptr, ndim) };
    let shape: Vec<usize> = raw_shape.iter().map(|&d| d as usize).collect();
    let n: usize = shape.iter().product();
    let x = unsafe { tv(x_ptr, &shape) };
    let mut buf = vec![0.0f32; n];
    let r = lele::kernels::manipulation::transpose(&x, perm, &mut buf);
    copy_out(&r, out_ptr);
}

// ── 8. Conv2d ────────────────────────────────────────────────────────────────
// x[N,C,H,W], w[M,C,kH,kW], bias[M] or null -> out[N,M,oH,oW]

#[unsafe(no_mangle)]
pub extern "C" fn lele_conv2d(
    x_ptr: *const f32,
    n: i64,
    c: i64,
    h: i64,
    w: i64,
    w_ptr: *const f32,
    m: i64,
    kh: i64,
    kw: i64,
    bias_ptr: *const f32, // null = no bias
    pad_h: i64,
    pad_w: i64,
    stride_h: i64,
    stride_w: i64,
    dilation_h: i64,
    dilation_w: i64,
    group: i64,
    out_ptr: *mut f32,
) {
    let xsh = [n as usize, c as usize, h as usize, w as usize];
    let wsh = [m as usize, (c / group) as usize, kh as usize, kw as usize];
    let bsh = [m as usize];
    let x = unsafe { tv(x_ptr, &xsh) };
    let wt = unsafe { tv(w_ptr, &wsh) };
    let bias_val;
    let bias = if bias_ptr.is_null() {
        None
    } else {
        bias_val = unsafe { tv(bias_ptr, &bsh) };
        Some(&bias_val)
    };
    let dilations = [dilation_h, dilation_w];
    let pads = [pad_h, pad_w, pad_h, pad_w];
    let strides = [stride_h, stride_w];
    let oh = (h + 2 * pad_h - dilation_h * (kh - 1) - 1) / stride_h + 1;
    let ow = (w + 2 * pad_w - dilation_w * (kw - 1) - 1) / stride_w + 1;
    let mut buf = vec![0.0f32; (n * m * oh * ow) as usize];
    let r = lele::kernels::conv2d(&x, &wt, bias, &dilations, group, &pads, &strides, &mut buf);
    copy_out(&r, out_ptr);
}

// ── 9. Conv1d ─────────────────────────────────────────────────────────────────
// x[N,C,L], w[M,C,K] -> out[N,M,oL]

#[unsafe(no_mangle)]
pub extern "C" fn lele_conv1d(
    x_ptr: *const f32,
    n: i64,
    c: i64,
    l: i64,
    w_ptr: *const f32,
    m: i64,
    k: i64,
    bias_ptr: *const f32,
    pad: i64,
    stride: i64,
    dilation: i64,
    group: i64,
    out_ptr: *mut f32,
) {
    let xsh = [n as usize, c as usize, l as usize];
    let wsh = [m as usize, (c / group) as usize, k as usize];
    let bsh_1d = [m as usize];
    let x = unsafe { tv(x_ptr, &xsh) };
    let wt = unsafe { tv(w_ptr, &wsh) };
    let bias_val;
    let bias = if bias_ptr.is_null() {
        None
    } else {
        bias_val = unsafe { tv(bias_ptr, &bsh_1d) };
        Some(&bias_val)
    };
    let dilations = [dilation];
    let pads = [pad, pad];
    let strides = [stride];
    let ol = (l + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
    let mut buf = vec![0.0f32; (n * m * ol) as usize];
    let r = lele::kernels::conv1d(&x, &wt, bias, &dilations, group, &pads, &strides, &mut buf);
    copy_out(&r, out_ptr);
}

// ── 10. MatMulInteger (u8 quantized) ─────────────────────────────────────────
// A[M,K] u8, B[K,N] u8, zp_a scalar, zp_b scalar -> C[M,N] f32

#[unsafe(no_mangle)]
pub extern "C" fn lele_mat_mul_integer(
    a_ptr: *const u8,
    m: i64,
    k: i64,
    b_ptr: *const u8,
    n: i64,
    zp_a: u8,
    zp_b: u8,
    out_ptr: *mut f32,
) {
    let ash = [m as usize, k as usize];
    let bsh = [k as usize, n as usize];
    // kernel takes f32 tensors representing u8 values
    let a_raw = unsafe { tv_u8(a_ptr, &ash) };
    let b_raw = unsafe { tv_u8(b_ptr, &bsh) };
    let a_f: Vec<f32> = a_raw.data.iter().map(|&v| v as f32).collect();
    let b_f: Vec<f32> = b_raw.data.iter().map(|&v| v as f32).collect();
    let a = TensorView::from_owned(a_f, ash.to_vec());
    let b = TensorView::from_owned(b_f, bsh.to_vec());
    let zpa_data = vec![zp_a as f32];
    let zpb_data = vec![zp_b as f32];
    let zpa = TensorView::from_owned(zpa_data, vec![1]);
    let zpb = TensorView::from_owned(zpb_data, vec![1]);
    let mut buf = vec![0.0f32; (m * n) as usize];
    let r = lele::kernels::mat_mul_integer(&a, &b, Some(&zpa), Some(&zpb), &mut buf);
    copy_out(&r, out_ptr);
}
