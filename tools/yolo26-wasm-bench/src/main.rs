//! YOLO26 WASM performance benchmark.
//!
//! Compile and run with wasmtime:
//!   cargo build --release --target wasm32-wasip1 -p yolo26-wasm-bench
//!   wasmtime --wasm-features simd run target/wasm32-wasip1/release/yolo26-wasm-bench.wasm
#![allow(unused_imports)]

#[path = "gen/yolo26.rs"]
mod yolo26;

use lele::tensor::TensorView;
use std::time::{Duration, Instant};

fn main() {
    eprintln!("=== YOLO26 WASM Benchmark (wasmtime) ===\n");

    // Embed weights at compile time to avoid file I/O issues in WASM
    let bin: &[u8] = include_bytes!("gen/yolo26_weights.bin");
    eprintln!(
        "✓ Weights embedded: {:.1} MB",
        bin.len() as f64 / 1024.0 / 1024.0
    );

    let model = yolo26::Yolo26::new(bin);
    let mut ws = yolo26::Yolo26Workspace::new();

    // Dummy 640×640 RGB input (all 0.5)
    let input_data = vec![0.5f32; 1 * 3 * 640 * 640];

    let n_warmup = 1;
    let n_runs = 5;

    eprintln!("Warming up ({} run)...", n_warmup);
    for _ in 0..n_warmup {
        let inp = TensorView::from_owned(input_data.clone(), vec![1, 3, 640, 640]);
        let _ = model.forward_with_workspace(&mut ws, inp);
    }

    eprintln!("Benchmarking ({} runs)...\n", n_runs);
    let mut times_ms: Vec<f64> = Vec::with_capacity(n_runs);
    for i in 0..n_runs {
        let inp = TensorView::from_owned(input_data.clone(), vec![1, 3, 640, 640]);
        let start = Instant::now();
        let _ = model.forward_with_workspace(&mut ws, inp);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        times_ms.push(elapsed_ms);
        eprintln!("  run {:2}: {:.1} ms  ({:.3} FPS)", i + 1, elapsed_ms, 1000.0 / elapsed_ms);
    }

    times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = times_ms[0];
    let max = times_ms[times_ms.len() - 1];
    let avg = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    let median = times_ms[times_ms.len() / 2];

    eprintln!("\n=== Summary ===");
    eprintln!("  Min:    {:.1} ms  ({:.3} FPS)", min, 1000.0 / min);
    eprintln!("  Median: {:.1} ms  ({:.3} FPS)", median, 1000.0 / median);
    eprintln!("  Avg:    {:.1} ms  ({:.3} FPS)", avg, 1000.0 / avg);
    eprintln!("  Max:    {:.1} ms  ({:.3} FPS)", max, 1000.0 / max);

    // Print as JSON for easy parsing
    println!(
        "{{\"min_ms\":{:.1},\"median_ms\":{:.1},\"avg_ms\":{:.1},\"max_ms\":{:.1},\"target_fps\":1.0}}",
        min, median, avg, max
    );
}
