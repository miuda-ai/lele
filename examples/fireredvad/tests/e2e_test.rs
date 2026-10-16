//! fixtures/zh.wav against the reference pipeline: kaldi-native-fbank
//! features and onnxruntime probabilities, and the segments of the official
//! `FireRedStreamVad.detect_full`. Skipped until build.rs has generated the
//! model.

use fireredvad_example::{Cmvn, Config, Detector, Event, Segmenter, load_weights, read_wav};

const WAV: &str = "../../fixtures/zh.wav";

/// Probabilities fed as `pieces`-sample pieces; `None` without the model.
fn probs(pieces: usize) -> Option<Vec<f32>> {
    let Ok(weights) = load_weights() else {
        eprintln!("skipping: no generated model, build the example first");
        return None;
    };
    let (audio, rate) = read_wav(WAV).unwrap();
    assert_eq!(rate, 16000);
    let mut detector = Detector::new(&weights, Cmvn::load().unwrap());
    let mut probs = Vec::new();
    for piece in audio.chunks(pieces) {
        detector.feed(piece, &mut probs);
    }
    Some(probs)
}

#[test]
fn matches_reference() {
    let Some(probs) = probs(1600) else { return };
    assert_eq!(probs.len(), 557);
    // onnxruntime, fed the reference features one frame at a time.
    let want = [
        (0, 0.020146698),
        (50, 0.0048108101),
        (66, 0.0058964193),
        (100, 0.98615575),
        (300, 0.99892008),
        (527, 0.0079532266),
        (556, 0.015062034),
    ];
    for (i, w) in want {
        assert!(
            (probs[i] - w).abs() < 1e-5,
            "frame {i}: {} vs {w}",
            probs[i]
        );
    }

    let mut segmenter = Segmenter::new(Config::default());
    let mut events: Vec<Event> = probs.iter().flat_map(|&p| segmenter.push(p)).collect();
    events.extend(segmenter.finish());
    // 0.67 s to 5.28 s.
    assert_eq!(events, [Event::Start(67), Event::End(67, 528)]);
}

#[test]
fn pieces_agree() {
    let Some(whole) = probs(usize::MAX) else {
        return;
    };
    // A frame per call, and pieces that leave a frame's audio half in.
    for pieces in [160, 401] {
        let probs = probs(pieces).unwrap();
        assert_eq!(probs.len(), whole.len());
        for (i, (a, b)) in probs.iter().zip(&whole).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "pieces {pieces}, frame {i}: {a} vs {b}"
            );
        }
    }
}
