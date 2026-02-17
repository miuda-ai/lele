mod yolo26;

use lele::tensor::TensorView;
use std::time::Instant;

fn main() {
    println!("=== YOLO26 Benchmark ===\n");

    // Load weights
    let bin = std::fs::read("examples/yolo26/src/yolo26_weights.bin")
        .or_else(|_| std::fs::read("yolo26_weights.bin"))
        .or_else(|_| std::fs::read("src/yolo26_weights.bin"))
        .expect("Failed to load yolo26_weights.bin");

    let model = yolo26::Yolo26::new(&bin);
    println!(
        "✓ Model loaded ({:.1} MB)",
        bin.len() as f64 / 1024.0 / 1024.0
    );

    // Create dummy input [1, 3, 640, 640]
    let input_data = vec![0.5f32; 1 * 3 * 640 * 640];
    let input = TensorView::from_owned(input_data, vec![1, 3, 640, 640]);

    let mut ws = yolo26::Yolo26Workspace::new();

    // Warmup
    let n_warmup = 3;
    let n_runs = 10;

    println!("Warming up ({} runs)...", n_warmup);
    for _ in 0..n_warmup {
        let _ = model.forward_with_workspace(&mut ws, input.clone());
    }

    // Benchmark
    println!("Benchmarking ({} runs)...", n_runs);
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

    println!("\n=== Results ===");
    println!("  Min:    {:.2}ms", min);
    println!("  Max:    {:.2}ms", max);
    println!("  Avg:    {:.2}ms", avg);
    println!("  Median: {:.2}ms", median);

    for (i, t) in times.iter().enumerate() {
        println!("  run {}: {:.2}ms", i + 1, t);
    }
}
