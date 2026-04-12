mod yolo26seg;

use lele::tensor::TensorView;
use std::time::Instant;

fn main() {
    println!("=== YOLO26n-Seg Benchmark ===\n");

    // Load weights
    let bin = std::fs::read("examples/yolo26n-seg/src/yolo26seg_weights.bin")
        .or_else(|_| std::fs::read("src/yolo26seg_weights.bin"))
        .or_else(|_| std::fs::read("examples/yolo26n-seg/yolo26seg_weights.bin"))
        .or_else(|_| std::fs::read("yolo26seg_weights.bin"))
        .expect("Failed to load yolo26seg_weights.bin. Run: cargo build -p yolo26n-seg-example");

    let model = yolo26seg::Yolo26Seg::new(&bin);
    println!(
        "✓ Model loaded ({:.1} MB)",
        bin.len() as f64 / 1024.0 / 1024.0
    );

    // Create dummy input [1, 3, 640, 640]
    let input_data = vec![0.5f32; 1 * 3 * 640 * 640];
    let input = TensorView::from_owned(input_data, vec![1, 3, 640, 640]);

    let mut ws = yolo26seg::Yolo26SegWorkspace::new();

    // Warmup
    let n_warmup = 3;
    let n_runs = 10;

    println!("Warming up ({} runs)...", n_warmup);
    for _ in 0..n_warmup {
        let _ = model.forward_with_workspace(&mut ws, input.clone());
    }

    // Benchmark
    println!("Benchmarking ({} runs)...", n_runs);
    lele::kernels::timing::reset();
    let mut times = Vec::with_capacity(n_runs);
    for _ in 0..n_runs {
        let start = Instant::now();
        let _ = model.forward_with_workspace(&mut ws, input.clone());
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = times[0];
    let max = times[times.len() - 1];
    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let median = times[times.len() / 2];

    // RTF: relative to 30fps (33.33ms per frame)
    let frame_ms = 1000.0 / 30.0;
    let rtf_avg = avg / frame_ms;
    let rtf_min = min / frame_ms;

    println!("\n=== Results (lele) ===");
    println!("  Min:    {:.2}ms  RTF={:.4}", min, rtf_min);
    println!("  Max:    {:.2}ms", max);
    println!("  Avg:    {:.2}ms  RTF={:.4}", avg, rtf_avg);
    println!("  Median: {:.2}ms", median);
    println!("  (RTF@30fps < 1.0 = real-time capable)");

    for (i, t) in times.iter().enumerate() {
        println!("  run {}: {:.2}ms", i + 1, t);
    }

    // Print per-operation timing breakdown
    lele::kernels::timing::print();

    eprintln!("\n(profiling totals for {} runs)", n_runs);
}
