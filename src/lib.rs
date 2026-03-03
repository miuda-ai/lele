#![feature(portable_simd)]
pub mod compiler;
pub mod features;
pub mod kernels;
pub mod model;
pub mod tensor;
pub use kernels::*;

/// Lightweight profiling: accumulates nanosecond timing for key kernel categories.
pub mod profiling {
    use std::sync::atomic::{AtomicU64, Ordering};

    static QUANT_GEMM_NS: AtomicU64 = AtomicU64::new(0);
    static F32_MATMUL_NS: AtomicU64 = AtomicU64::new(0);
    static SOFTMAX_NS: AtomicU64 = AtomicU64::new(0);
    static TRANSPOSE_NS: AtomicU64 = AtomicU64::new(0);
    static WHERE_OP_NS: AtomicU64 = AtomicU64::new(0);
    static CONV1D_NS: AtomicU64 = AtomicU64::new(0);
    static LAYERNORM_NS: AtomicU64 = AtomicU64::new(0);
    static ELEMENTWISE_NS: AtomicU64 = AtomicU64::new(0);
    static CONV2D_NS: AtomicU64 = AtomicU64::new(0);
    static CONV2D_IM2COL_NS: AtomicU64 = AtomicU64::new(0);
    static CONV2D_GEMM_NS: AtomicU64 = AtomicU64::new(0);
    static PAD_NS: AtomicU64 = AtomicU64::new(0);
    static CONCAT_NS: AtomicU64 = AtomicU64::new(0);
    static SPLIT_NS: AtomicU64 = AtomicU64::new(0);
    static ERF_NS: AtomicU64 = AtomicU64::new(0);
    static UNARY_NS: AtomicU64 = AtomicU64::new(0);

    macro_rules! timer_fns {
        ($add:ident, $atom:ident) => {
            #[inline]
            pub fn $add(ns: u64) { $atom.fetch_add(ns, Ordering::Relaxed); }
        };
    }
    timer_fns!(add_quant_gemm, QUANT_GEMM_NS);
    timer_fns!(add_f32_matmul, F32_MATMUL_NS);
    timer_fns!(add_softmax, SOFTMAX_NS);
    timer_fns!(add_transpose, TRANSPOSE_NS);
    timer_fns!(add_where_op, WHERE_OP_NS);
    timer_fns!(add_conv1d, CONV1D_NS);
    timer_fns!(add_layernorm, LAYERNORM_NS);
    timer_fns!(add_elementwise, ELEMENTWISE_NS);
    timer_fns!(add_conv2d, CONV2D_NS);
    timer_fns!(add_conv2d_im2col, CONV2D_IM2COL_NS);
    timer_fns!(add_conv2d_gemm, CONV2D_GEMM_NS);
    timer_fns!(add_pad, PAD_NS);
    timer_fns!(add_concat, CONCAT_NS);
    timer_fns!(add_split, SPLIT_NS);
    timer_fns!(add_erf, ERF_NS);
    timer_fns!(add_unary, UNARY_NS);

    pub fn reset() {
        QUANT_GEMM_NS.store(0, Ordering::Relaxed);
        F32_MATMUL_NS.store(0, Ordering::Relaxed);
        SOFTMAX_NS.store(0, Ordering::Relaxed);
        TRANSPOSE_NS.store(0, Ordering::Relaxed);
        WHERE_OP_NS.store(0, Ordering::Relaxed);
        CONV1D_NS.store(0, Ordering::Relaxed);
        LAYERNORM_NS.store(0, Ordering::Relaxed);
        ELEMENTWISE_NS.store(0, Ordering::Relaxed);
        CONV2D_NS.store(0, Ordering::Relaxed);
        CONV2D_IM2COL_NS.store(0, Ordering::Relaxed);
        CONV2D_GEMM_NS.store(0, Ordering::Relaxed);
        PAD_NS.store(0, Ordering::Relaxed);
        CONCAT_NS.store(0, Ordering::Relaxed);
        SPLIT_NS.store(0, Ordering::Relaxed);
        ERF_NS.store(0, Ordering::Relaxed);
        UNARY_NS.store(0, Ordering::Relaxed);
    }

    pub fn print_report() {
        let qg = QUANT_GEMM_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let fm = F32_MATMUL_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let sf = SOFTMAX_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let tr = TRANSPOSE_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let wo = WHERE_OP_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let c1 = CONV1D_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let ln = LAYERNORM_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let ew = ELEMENTWISE_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let c2 = CONV2D_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let c2_im2col = CONV2D_IM2COL_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let c2_gemm = CONV2D_GEMM_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let pd = PAD_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let ct = CONCAT_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let sp = SPLIT_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let ef = ERF_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let un = UNARY_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let total = qg + fm + sf + tr + wo + c1 + ln + ew + c2 + pd + ct + sp + ef + un;
        eprintln!("=== Kernel Profiling (ms) ===");
        eprintln!("  Quant GEMM:   {qg:8.2}");
        eprintln!("  f32 MatMul:   {fm:8.2}");
        eprintln!("  Softmax:      {sf:8.2}");
        eprintln!("  Transpose:    {tr:8.2}");
        eprintln!("  Where:        {wo:8.2}");
        eprintln!("  Conv1d:       {c1:8.2}");
        eprintln!("  LayerNorm:    {ln:8.2}");
        eprintln!("  Elementwise:  {ew:8.2}");
        eprintln!("  Conv2d total: {c2:8.2}");
        eprintln!("    im2col:     {c2_im2col:8.2}");
        eprintln!("    gemm:       {c2_gemm:8.2}");
        eprintln!("  Pad:          {pd:8.2}");
        eprintln!("  Concat:       {ct:8.2}");
        eprintln!("  Split:        {sp:8.2}");
        eprintln!("  Erf:          {ef:8.2}");
        eprintln!("  Unary/Other:  {un:8.2}");
        eprintln!("  Accounted:    {total:8.2}");
    }
}
