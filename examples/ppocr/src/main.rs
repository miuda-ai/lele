mod image;
mod dict;
mod ppocrdet;
mod ppocrrec;

use image::{Image, det_preprocess, det_postprocess, crop_text_region, rec_preprocess, rec_postprocess};
use lele::tensor::TensorView;
use std::env;
use std::time::Instant;

fn main() {
    let img_path = env::args()
        .nth(1)
        .unwrap_or_else(|| "fixtures/ocr_test.png".to_string());

    println!("=== PP-OCRv6 Pure Rust Inference ===\n");
    println!("Loading image: {}", img_path);
    let img = Image::load(&img_path).expect("Failed to load image");
    println!("Image: {}x{}\n", img.width, img.height);

    // ==================== Detection ====================
    println!("--- Detection ---");
    let t0 = Instant::now();

    let det_bin = std::fs::read("examples/ppocr/src/ppocrdet_weights.bin")
        .or_else(|_| std::fs::read("src/ppocrdet_weights.bin"))
        .expect("Failed to load det weights");
    let det_model = ppocrdet::PpocrDet::new(&det_bin);
    println!("Det model: {:.2} MB", det_bin.len() as f64 / 1048576.0);

    let det_input = det_preprocess(&img, 960);
    println!(
        "Det input: 1x3x{}x{}",
        det_input.resized_h, det_input.resized_w
    );

    let det_tv = TensorView::from_owned(
        det_input.chw,
        vec![1, 3, det_input.resized_h, det_input.resized_w],
    );

    let mut det_ws = ppocrdet::PpocrDetWorkspace::new();
    // Warmup
    let _ = det_model.forward_with_workspace(&mut det_ws, det_tv.clone());

    let t1 = Instant::now();
    let det_out = det_model.forward_with_workspace(&mut det_ws, det_tv.clone());
    let det_time = t1.elapsed().as_secs_f64() * 1000.0;

    let det_shape = det_out.shape.as_ref();
    let out_h = det_shape[2];
    let out_w = det_shape[3];
    println!("Det output: {:?}", det_shape);

    let boxes = det_postprocess(
        det_out.data.as_ref(),
        out_h,
        out_w,
        det_input.ratio_h,
        det_input.ratio_w,
        img.height,
        img.width,
        0.2,   // thresh
        0.4,   // box_thresh
        1.4,   // unclip_ratio
        3000,  // max_candidates
        3,     // min_size
    );
    println!(
        "Detected {} text regions in {:.2}ms\n",
        boxes.len(),
        det_time
    );

    // ==================== Recognition ====================
    println!("--- Recognition ---");
    let rec_bin = std::fs::read("examples/ppocr/src/ppocrrec_weights.bin")
        .or_else(|_| std::fs::read("src/ppocrrec_weights.bin"))
        .expect("Failed to load rec weights");
    let rec_model = ppocrrec::PpocrRec::new(&rec_bin);
    println!("Rec model: {:.2} MB", rec_bin.len() as f64 / 1048576.0);

    let mut rec_ws = ppocrrec::PpocrRecWorkspace::new();
    let img_h = 48usize;
    let num_classes = 6906usize;

    let mut results = Vec::new();
    let mut total_rec_time = 0.0f64;

    for (i, box_pts) in boxes.iter().enumerate() {
        let (region, rw, rh) = crop_text_region(&img.data, img.width, box_pts);
        if rw == 0 || rh == 0 {
            continue;
        }

        let (chw, resized_w) = rec_preprocess(&region, rw, rh, img_h);
        let rec_tv =
            TensorView::from_owned(chw, vec![1, 3, img_h, resized_w]);

        // Warmup on first
        if i == 0 {
            let _ = rec_model.forward_with_workspace(&mut rec_ws, rec_tv.clone());
        }

        let t2 = Instant::now();
        let rec_out = rec_model.forward_with_workspace(&mut rec_ws, rec_tv.clone());
        let rec_time = t2.elapsed().as_secs_f64() * 1000.0;
        total_rec_time += rec_time;

        let rec_shape = rec_out.shape.as_ref();
        let seq_len = rec_shape[1];
        let (text, conf) =
            rec_postprocess(rec_out.data.as_ref(), seq_len, num_classes, dict::CHAR_DICT);

        let cy = (box_pts[0][1] + box_pts[1][1] + box_pts[2][1] + box_pts[3][1]) * 0.25;
        let cx = (box_pts[0][0] + box_pts[1][0] + box_pts[2][0] + box_pts[3][0]) * 0.25;

        results.push((text.clone(), conf, cx, cy, rec_time));
        println!(
            "  [{:2}] ({:5.1}ms) conf={:.3}  \"{}\"  @({:.0},{:.0})",
            i + 1,
            rec_time,
            conf,
            text,
            cx,
            cy
        );
    }

    println!("\n=== Results ===");
    for (text, conf, _, _, _) in &results {
        println!("  [conf={:.3}] {}", conf, text);
    }

    println!("\n--- Timing Summary ---");
    println!("  Detection:  {:.2}ms", det_time);
    println!("  Recognition: {:.2}ms total ({:.2}ms avg for {} regions)", total_rec_time, total_rec_time / results.len().max(1) as f64, results.len());
    println!("  Total:       {:.2}ms", t0.elapsed().as_secs_f64() * 1000.0);
}
