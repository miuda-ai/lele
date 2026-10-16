use fireredvad_example::{
    Cmvn, Config, Detector, Event, FRAME_RATE, SAMPLE_RATE, Segmenter, load_weights, read_wav,
};
use std::time::Instant;

fn seconds(frame: usize) -> f64 {
    frame as f64 / FRAME_RATE as f64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "fixtures/zh.wav".into());
    let (audio, rate) = read_wav(&path)?;
    if rate != SAMPLE_RATE {
        return Err(format!(
            "FireRedVAD expects 16 kHz audio, got {rate} Hz; convert with `ffmpeg -i in -ar 16000 -ac 1 out.wav`"
        )
        .into());
    }

    let weights = load_weights()?;
    let mut detector = Detector::new(&weights, Cmvn::load()?);
    let mut segmenter = Segmenter::new(Config::default());

    // Fed as it would arrive live, 100 ms at a time.
    let mut probs = Vec::new();
    let mut segments = Vec::new();
    let mut compute = 0.0;
    for piece in audio.chunks(SAMPLE_RATE as usize / 10) {
        let first = probs.len();
        let start = Instant::now();
        detector.feed(piece, &mut probs);
        compute += start.elapsed().as_secs_f64();
        for &p in &probs[first..] {
            for event in segmenter.push(p) {
                match event {
                    Event::Start(s) => println!("{:7.2}s  speech starts", seconds(s)),
                    Event::End(s, e) => {
                        println!("{:7.2}s  speech ends", seconds(e));
                        segments.push((s, e));
                    }
                }
            }
        }
    }
    if let Some(Event::End(s, e)) = segmenter.finish() {
        println!("{:7.2}s  speech ends (end of audio)", seconds(e));
        segments.push((s, e));
    }

    println!("\nSegments:");
    for (s, e) in &segments {
        println!("  {:.2}s - {:.2}s", seconds(*s), seconds(*e));
    }
    let duration = audio.len() as f64 / SAMPLE_RATE as f64;
    eprintln!(
        "\n{:.2}s of audio, {} frames in {:.2} ms ({:.1} µs per 10 ms frame, RTF {:.5})",
        duration,
        probs.len(),
        compute * 1e3,
        compute * 1e6 / probs.len().max(1) as f64,
        compute / duration
    );
    Ok(())
}
