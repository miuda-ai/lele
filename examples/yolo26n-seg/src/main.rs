mod image;
mod yolo26seg;

use image::{Image, postprocess_segmentation};
use lele::tensor::TensorView;
use std::env;
use std::time::Instant;

fn main() {
    println!("=== YOLO26n Segmentation Pure Rust Inference ===\n");

    // 1. Load Model Weights
    println!("Loading YOLO26n-Seg weights...");
    let bin = std::fs::read("src/yolo26seg_weights.bin")
        .or_else(|_| std::fs::read("yolo26seg_weights.bin"))
        .or_else(|_| std::fs::read("../yolo26n-seg/src/yolo26seg_weights.bin"))
        .expect("Failed to load yolo26seg_weights.bin");

    let model = yolo26seg::Yolo26Seg::new(&bin);
    println!(
        "✓ Model loaded ({:.1} MB)\n",
        bin.len() as f64 / 1024.0 / 1024.0
    );

    // 2. Load Image
    let img_path = env::args()
        .nth(1)
        .unwrap_or_else(|| "../../fixtures/bus.jpg".to_string());
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
    let mut ws = yolo26seg::Yolo26SegWorkspace::new();
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
    let (logits, mask_features) = model.forward_with_workspace(&mut ws, input.clone());
    let logits: TensorView<'static, f32> = logits.to_owned();
    let mask_features: TensorView<'static, f32> = mask_features.to_owned();
    for (i, t) in times.iter().enumerate() {
        eprintln!("  run {}: {:.2}ms", i + 1, t);
    }

    println!("✓ Inference completed: {:.2}ms\n", best_time);


    // 5. Postprocess (segmentation)
    println!("--- Postprocessing ---");
    let start = Instant::now();

    let (detections, _mask_img) = postprocess_segmentation(
        logits.data.as_ref(),
        mask_features.data.as_ref(),
        img.width,
        img.height,
        0.3, // threshold
        None,
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
