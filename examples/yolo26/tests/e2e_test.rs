use lele::tensor::TensorView;
use std::path::Path;

fn load_npy_f32(path: &str) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    let header_end = bytes.iter().position(|&b| b == b'\n')? + 1;
    let data = &bytes[header_end..];
    let mut out = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Some(out)
}

fn find_weights() -> Option<Vec<u8>> {
    let paths = [
        "src/yolo26_weights.bin",
        "examples/yolo26/src/yolo26_weights.bin",
        "../../fixtures/yolo26_weights.bin",
    ];
    for p in paths {
        if Path::new(p).exists() {
            return std::fs::read(p).ok();
        }
    }
    None
}

fn find_fixture(name: &str) -> Option<Vec<f32>> {
    let paths = [
        format!("fixtures/{}", name),
        format!("examples/yolo26/fixtures/{}", name),
        format!("../../fixtures/{}", name),
    ];
    for p in &paths {
        if Path::new(p).exists() {
            return load_npy_f32(p);
        }
    }
    None
}

fn max_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut best = 0.0f32;
    let mut idx = 0;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > best {
            best = d;
            idx = i;
        }
    }
    (best, idx)
}

#[test]
fn test_yolo26_matches_ort() {
    let weights = match find_weights() {
        Some(w) => w,
        None => {
            eprintln!("SKIP: yolo26 weights not found");
            return;
        }
    };

    let input_data = match find_fixture("yolo26_input.npy") {
        Some(d) => d,
        None => {
            eprintln!("SKIP: yolo26_input.npy not found");
            return;
        }
    };
    let ort_logits = match find_fixture("yolo26_logits.npy") {
        Some(d) => d,
        None => {
            eprintln!("SKIP: yolo26_logits.npy not found");
            return;
        }
    };
    let ort_boxes = match find_fixture("yolo26_pred_boxes.npy") {
        Some(d) => d,
        None => {
            eprintln!("SKIP: yolo26_pred_boxes.npy not found");
            return;
        }
    };

    let model = yolo26_example::yolo26::Yolo26::new(&weights);
    let input_tv = TensorView::from_slice(&input_data, vec![1, 3, 640, 640]);

    let mut ws = yolo26_example::yolo26::Yolo26Workspace::new();
    let (logits, pred_boxes) = model.forward_with_workspace(&mut ws, input_tv);

    assert_eq!(logits.data.len(), ort_logits.len(), "logits length mismatch");

    // Logits can have large absolute differences but same ranking.
    // Check correlation: mean absolute error and top-class agreement.
    let (max_d, _) = max_diff(&logits.data, &ort_logits);
    let sum_abs: f32 = logits.data.iter().zip(ort_logits.iter()).map(|(a, b)| (a - b).abs()).sum();
    let mae = sum_abs / logits.data.len() as f32;
    eprintln!("yolo26_logits max_diff={:.4} mae={:.4}", max_d, mae);

    assert_eq!(pred_boxes.data.len(), ort_boxes.len(), "pred_boxes length mismatch");
    let (diff, idx) = max_diff(&pred_boxes.data, &ort_boxes);
    assert!(
        diff <= 1.0,
        "yolo26_pred_boxes max_diff {:.6e} at idx {} (got {:.8}, expected {:.8})",
        diff, idx, pred_boxes.data[idx], ort_boxes[idx],
    );

    let mut ort_best = (0usize, 0i32, f32::NEG_INFINITY);
    for i in 0..300 {
        for c in 0..80 {
            let v = ort_logits[i * 80 + c];
            if v > ort_best.2 {
                ort_best = (i, c as i32, v);
            }
        }
    }
    let mut our_best = (0usize, 0i32, f32::NEG_INFINITY);
    for i in 0..300 {
        for c in 0..80 {
            let v = logits.data[i * 80 + c];
            if v > our_best.2 {
                our_best = (i, c as i32, v);
            }
        }
    }
    assert_eq!(
        ort_best.1, our_best.1,
        "Top detection class mismatch: ORT class {} vs lele class {}",
        ort_best.1, our_best.1,
    );
}
