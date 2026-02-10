mod image;
mod yolo26;

use image::{Image, postprocess};
use lele::tensor::TensorView;
use std::env;
use std::time::Instant;

fn main() {
    println!("=== YOLO26 Pure Rust Inference ===\n");

    // 1. Load Model Weights
    println!("Loading YOLO26 weights...");
    let bin = std::fs::read("examples/yolo26/src/yolo26_weights.bin")
        .or_else(|_| std::fs::read("yolo26_weights.bin"))
        .or_else(|_| std::fs::read("src/yolo26_weights.bin"))
        .expect("Failed to load yolo26_weights.bin");
    
    let model = yolo26::Yolo26::new(&bin);
    println!(
        "✓ Model loaded ({:.1} MB)\n",
        bin.len() as f64 / 1024.0 / 1024.0
    );

    // 2. Load Image
    let img_path = env::args().nth(1).unwrap_or_else(|| "fixtures/bus.jpg".to_string());
    println!("Loading image: {}", img_path);
    
    let img = Image::load(&img_path).expect("Failed to load image");
    println!("✓ Image loaded: {}x{}\n", img.width, img.height);

    // 3. Preprocess
    println!("--- Preprocessing ---");
    let start = Instant::now();
    let input_data = img.preprocess();
    println!(
        "✓ Preprocessed to 1x3x640x640, took {:.2}ms\n",
        start.elapsed().as_secs_f64() * 1000.0
    );

    // 4. Inference
    println!("--- Inference ---");
    let start_total = Instant::now();
    
    let input = TensorView::from_owned(input_data, vec![1, 3, 640, 640]);
    
    // Warmup + timed runs
    let mut ws = yolo26::Yolo26Workspace::new();
    let n_warmup = 1;
    let n_runs = 3;
    
    // Warmup
    for _ in 0..n_warmup {
        let _ = model.forward_with_workspace(&mut ws, input.clone());
    }
    
    // Timed runs
    let mut best_time = f64::MAX;
    let mut times = Vec::new();
    for _ in 0..n_runs {
        lele::kernels::reset_conv_stats();
        let start_infer = Instant::now();
        let _ = model.forward_with_workspace(&mut ws, input.clone());
        let elapsed = start_infer.elapsed().as_secs_f64() * 1000.0;
        if elapsed < best_time {
            best_time = elapsed;
        }
        times.push(elapsed);
    }
    // Print stats from last run
    lele::kernels::print_conv_stats();
    // Final run for output
    let (logits, pred_boxes) = model.forward_with_workspace(&mut ws, input.clone());
    let logits: TensorView<'static> = logits.to_owned();
    let pred_boxes: TensorView<'static> = pred_boxes.to_owned();
    for (i, t) in times.iter().enumerate() {
        eprintln!("  run {}: {:.2}ms", i + 1, t);
    }
    
    println!(
        "✓ Inference completed: {:.2}ms\n",
        best_time
    );

    // Debug: check output tensor values
    {
        let logits_data = logits.data.as_ref();
        let boxes_data = pred_boxes.data.as_ref();
        eprintln!("DEBUG logits shape: {:?}, len={}", logits.shape.as_ref(), logits_data.len());
        eprintln!("DEBUG boxes shape: {:?}, len={}", pred_boxes.shape.as_ref(), boxes_data.len());
        // Show first few logit values
        let n = logits_data.len().min(20);
        eprintln!("DEBUG logits[0..{}]: {:?}", n, &logits_data[..n]);
        // Show max logit and its index
        let (max_idx, max_val) = logits_data.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        eprintln!("DEBUG max logit: {} at index {} (query={}, class={})", max_val, max_idx, max_idx / 80, max_idx % 80);
        let sigmoid_max = 1.0 / (1.0 + (-max_val).exp());
        eprintln!("DEBUG max sigmoid score: {}", sigmoid_max);
        // Show top-10 logits across all queries
        let mut all_logits: Vec<(usize, f32)> = logits_data.iter().enumerate()
            .map(|(i, &v)| (i, v)).collect();
        all_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("DEBUG top-10 logits:");
        for &(idx, val) in all_logits.iter().take(10) {
            let query = idx / 80;
            let class = idx % 80;
            let score = 1.0 / (1.0 + (-val).exp());
            eprintln!("  query={}, class={}, logit={:.4}, score={:.4}", query, class, val, score);
        }
        // Show first few box values
        let bn = boxes_data.len().min(20);
        eprintln!("DEBUG boxes[0..{}]: {:?}", bn, &boxes_data[..bn]);
    }

    // 5. Postprocess
    println!("--- Postprocessing ---");
    let start = Instant::now();
    let detections = postprocess(
        logits.data.as_ref(),
        pred_boxes.data.as_ref(),
        img.width,
        img.height,
        0.3, // threshold
    );
    println!(
        "✓ Found {} detections, took {:.2}ms\n",
        detections.len(),
        start.elapsed().as_secs_f64() * 1000.0
    );

    // 6. Display Results
    println!("=== Detections ===");
    for (i, det) in detections.iter().enumerate() {
        println!(
            "  {}: {:15} score={:.3} bbox=[{:.0}, {:.0}, {:.0}, {:.0}]",
            i + 1,
            det.class_name,
            det.score,
            det.bbox[0],
            det.bbox[1],
            det.bbox[2],
            det.bbox[3]
        );
    }

    let total_time = start_total.elapsed();
    println!("\n✓ Total time: {:.2}ms", total_time.as_secs_f64() * 1000.0);
}
