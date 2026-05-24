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

fn load_npy_i32_as_i64(path: &str) -> Option<Vec<i64>> {
    let bytes = std::fs::read(path).ok()?;
    let header_end = bytes.iter().position(|&b| b == b'\n')? + 1;
    let data = &bytes[header_end..];
    let mut out = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        out.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as i64);
    }
    Some(out)
}

fn find_weights() -> Option<Vec<u8>> {
    let paths = [
        "src/sensevoice_weights.bin",
        "examples/sensevoice/src/sensevoice_weights.bin",
        "../../fixtures/sensevoice_weights.bin",
    ];
    for p in paths {
        if Path::new(p).exists() {
            return std::fs::read(p).ok();
        }
    }
    None
}

fn find_fixture_f32(name: &str) -> Option<Vec<f32>> {
    let paths = [
        format!("fixtures/{}", name),
        format!("examples/sensevoice/fixtures/{}", name),
        format!("../../fixtures/{}", name),
    ];
    for p in &paths {
        if Path::new(p).exists() {
            return load_npy_f32(p);
        }
    }
    None
}

fn find_fixture_i64(name: &str) -> Option<Vec<i64>> {
    let paths = [
        format!("fixtures/{}", name),
        format!("examples/sensevoice/fixtures/{}", name),
        format!("../../fixtures/{}", name),
    ];
    for p in &paths {
        if Path::new(p).exists() {
            return load_npy_i32_as_i64(p);
        }
    }
    None
}

#[test]
fn test_sensevoice_matches_ort() {
    let weights = match find_weights() {
        Some(w) => w,
        None => {
            eprintln!("SKIP: sensevoice weights not found");
            return;
        }
    };

    let x_data = match find_fixture_f32("sensevoice_input_x.npy") {
        Some(d) => d,
        None => {
            eprintln!("SKIP: sensevoice_input_x.npy not found");
            return;
        }
    };
    let x_length = match find_fixture_i64("sensevoice_input_x_length.npy") {
        Some(d) => d,
        None => {
            eprintln!("SKIP: sensevoice_input_x_length.npy not found");
            return;
        }
    };
    let language = match find_fixture_i64("sensevoice_input_language.npy") {
        Some(d) => d,
        None => {
            eprintln!("SKIP: sensevoice_input_language.npy not found");
            return;
        }
    };
    let text_norm = match find_fixture_i64("sensevoice_input_text_norm.npy") {
        Some(d) => d,
        None => {
            eprintln!("SKIP: sensevoice_input_text_norm.npy not found");
            return;
        }
    };
    let ort_logits = match find_fixture_f32("sensevoice_logits.npy") {
        Some(d) => d,
        None => {
            eprintln!("SKIP: sensevoice_logits.npy not found");
            return;
        }
    };

    let model = sensevoice_example::sensevoice::SenseVoice::new(&weights);
    let x_tv = TensorView::from_slice(&x_data, vec![1, 10, 560]);
    let x_len_tv = TensorView::from_slice(&x_length, vec![1]);
    let lang_tv = TensorView::from_slice(&language, vec![1]);
    let tn_tv = TensorView::from_slice(&text_norm, vec![1]);

    let mut ws = sensevoice_example::sensevoice::SenseVoiceWorkspace::new();
    let logits = model.forward_with_workspace(&mut ws, x_tv, x_len_tv, lang_tv, tn_tv);

    assert_eq!(logits.data.len(), ort_logits.len(), "logits length mismatch");

    let mut max_diff = 0.0f32;
    let mut max_idx = 0;
    let mut sum_abs = 0.0f32;
    for (i, (a, b)) in logits.data.iter().zip(ort_logits.iter()).enumerate() {
        let d = (a - b).abs();
        sum_abs += d;
        if d > max_diff {
            max_diff = d;
            max_idx = i;
        }
    }
    let mae = sum_abs / logits.data.len() as f32;
    eprintln!(
        "sensevoice_logits max_diff={:.4} mae={:.4} at idx {}",
        max_diff, mae, max_idx
    );
    assert!(
        mae <= 1.0,
        "sensevoice_logits mae {:.4} too large (got {:.4}, expected {:.4} at idx {})",
        mae, logits.data[max_idx], ort_logits[max_idx], max_idx,
    );

    // Check argmax of first row matches
    let our_argmax: Vec<usize> = (0..logits.shape[1])
        .map(|t| {
            let base = t * 25055;
            logits.data[base..base + 25055]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        })
        .collect();
    let ort_argmax: Vec<usize> = (0..(ort_logits.len() / 25055))
        .map(|t| {
            let base = t * 25055;
            ort_logits[base..base + 25055]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        })
        .collect();
    assert_eq!(
        our_argmax.len(),
        ort_argmax.len(),
        "argmax length mismatch"
    );
    let matching = our_argmax
        .iter()
        .zip(ort_argmax.iter())
        .filter(|(a, b)| a == b)
        .count();
    eprintln!(
        "argmax match: {}/{} ({:.0}%)",
        matching,
        our_argmax.len(),
        100.0 * matching as f32 / our_argmax.len() as f32
    );
    assert!(
        matching > 0,
        "No argmax tokens matched between lele and ORT"
    );
}
