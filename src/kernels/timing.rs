/// Simple per-operation timing for profiling.
/// Enabled by `TIMING_ENABLED = true`. Set to false for production.
#[cfg(target_arch = "wasm32")]
pub const TIMING_ENABLED: bool = false;

#[cfg(not(target_arch = "wasm32"))]
pub const TIMING_ENABLED: bool = true;

use std::sync::atomic::{AtomicU64, Ordering};

pub static CONV1X1_NS: AtomicU64 = AtomicU64::new(0);
pub static CONV3X3_NS: AtomicU64 = AtomicU64::new(0);
pub static CONV_OTHER_NS: AtomicU64 = AtomicU64::new(0);
pub static SPLIT_NS: AtomicU64 = AtomicU64::new(0);
pub static CONCAT_NS: AtomicU64 = AtomicU64::new(0);
pub static ADD_NS: AtomicU64 = AtomicU64::new(0);
pub static SIGMOID_NS: AtomicU64 = AtomicU64::new(0);
pub static MUL_NS: AtomicU64 = AtomicU64::new(0);
pub static SILU_NS: AtomicU64 = AtomicU64::new(0);
pub static RESIZE_NS: AtomicU64 = AtomicU64::new(0);
pub static CONV_TRANS_NS: AtomicU64 = AtomicU64::new(0);
pub static OTHER_NS: AtomicU64 = AtomicU64::new(0);
pub static TOTAL_CONV_CALLS: AtomicU64 = AtomicU64::new(0);

pub fn reset() {
    CONV1X1_NS.store(0, Ordering::Relaxed);
    CONV3X3_NS.store(0, Ordering::Relaxed);
    CONV_OTHER_NS.store(0, Ordering::Relaxed);
    SPLIT_NS.store(0, Ordering::Relaxed);
    CONCAT_NS.store(0, Ordering::Relaxed);
    ADD_NS.store(0, Ordering::Relaxed);
    SIGMOID_NS.store(0, Ordering::Relaxed);
    MUL_NS.store(0, Ordering::Relaxed);
    SILU_NS.store(0, Ordering::Relaxed);
    RESIZE_NS.store(0, Ordering::Relaxed);
    CONV_TRANS_NS.store(0, Ordering::Relaxed);
    OTHER_NS.store(0, Ordering::Relaxed);
    TOTAL_CONV_CALLS.store(0, Ordering::Relaxed);
}

pub fn print() {
    let c1 = CONV1X1_NS.load(Ordering::Relaxed);
    let c3 = CONV3X3_NS.load(Ordering::Relaxed);
    let co = CONV_OTHER_NS.load(Ordering::Relaxed);
    let sp = SPLIT_NS.load(Ordering::Relaxed);
    let ca = CONCAT_NS.load(Ordering::Relaxed);
    let ad = ADD_NS.load(Ordering::Relaxed);
    let sig = SIGMOID_NS.load(Ordering::Relaxed);
    let mu = MUL_NS.load(Ordering::Relaxed);
    let si = SILU_NS.load(Ordering::Relaxed);
    let re = RESIZE_NS.load(Ordering::Relaxed);
    let ct = CONV_TRANS_NS.load(Ordering::Relaxed);
    let ot = OTHER_NS.load(Ordering::Relaxed);
    let total = c1 + c3 + co + sp + ca + ad + sig + mu + si + re + ct + ot;
    println!("\n[Timing breakdown]");
    println!(
        "  conv2d 1×1:    {:>8.2}ms  ({:.1}%)",
        c1 as f64 / 1e6,
        100.0 * c1 as f64 / total.max(1) as f64
    );
    println!(
        "  conv2d 3×3:    {:>8.2}ms  ({:.1}%)",
        c3 as f64 / 1e6,
        100.0 * c3 as f64 / total.max(1) as f64
    );
    println!(
        "  conv2d other:  {:>8.2}ms  ({:.1}%)",
        co as f64 / 1e6,
        100.0 * co as f64 / total.max(1) as f64
    );
    println!(
        "  split_owned:   {:>8.2}ms  ({:.1}%)",
        sp as f64 / 1e6,
        100.0 * sp as f64 / total.max(1) as f64
    );
    println!(
        "  concat:        {:>8.2}ms  ({:.1}%)",
        ca as f64 / 1e6,
        100.0 * ca as f64 / total.max(1) as f64
    );
    println!(
        "  add:           {:>8.2}ms  ({:.1}%)",
        ad as f64 / 1e6,
        100.0 * ad as f64 / total.max(1) as f64
    );
    println!(
        "  sigmoid:       {:>8.2}ms  ({:.1}%)",
        sig as f64 / 1e6,
        100.0 * sig as f64 / total.max(1) as f64
    );
    println!(
        "  mul:           {:>8.2}ms  ({:.1}%)",
        mu as f64 / 1e6,
        100.0 * mu as f64 / total.max(1) as f64
    );
    println!(
        "  silu_alone:    {:>8.2}ms  ({:.1}%)",
        si as f64 / 1e6,
        100.0 * si as f64 / total.max(1) as f64
    );
    println!(
        "  resize:        {:>8.2}ms  ({:.1}%)",
        re as f64 / 1e6,
        100.0 * re as f64 / total.max(1) as f64
    );
    println!(
        "  conv_transpose:{:>8.2}ms  ({:.1}%)",
        ct as f64 / 1e6,
        100.0 * ct as f64 / total.max(1) as f64
    );
    println!(
        "  other:         {:>8.2}ms  ({:.1}%)",
        ot as f64 / 1e6,
        100.0 * ot as f64 / total.max(1) as f64
    );
    println!("  [tracked total:{:>8.2}ms]", total as f64 / 1e6);
}
