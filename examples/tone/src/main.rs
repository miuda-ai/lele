use std::time::Instant;
use tone_example::{CHUNK, GreedyCtc, Recognizer, SAMPLE_RATE, load_weights};

/// Reads a 16-bit PCM WAV, walking the RIFF chunks rather than assuming a
/// 44-byte header, and averages the channels down to mono.
fn read_wav(path: &str) -> Result<(Vec<i32>, u32), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }
    let (mut channels, mut rate, mut bits) = (0usize, 0u32, 0u16);
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into()?) as usize;
        let body = &bytes[pos + 8..(pos + 8 + len).min(bytes.len())];
        if id == b"fmt " {
            channels = u16::from_le_bytes([body[2], body[3]]) as usize;
            rate = u32::from_le_bytes(body[4..8].try_into()?);
            bits = u16::from_le_bytes([body[14], body[15]]);
        } else if id == b"data" {
            if bits != 16 || channels == 0 {
                return Err(format!("need 16-bit PCM, got {bits}-bit x {channels}").into());
            }
            let samples = body
                .chunks_exact(2 * channels)
                .map(|frame| {
                    let sum: i32 = frame
                        .chunks_exact(2)
                        .map(|s| i16::from_le_bytes([s[0], s[1]]) as i32)
                        .sum();
                    sum / channels as i32
                })
                .collect();
            return Ok((samples, rate));
        }
        pos += 8 + len + (len & 1);
    }
    Err("no data chunk".into())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: tone <8kHz 16-bit mono .wav>")?;
    let (audio, rate) = read_wav(&path)?;
    if rate != SAMPLE_RATE {
        return Err(format!(
            "T-one expects 8 kHz audio, got {rate} Hz; convert with `ffmpeg -i in -ar 8000 -ac 1 out.wav`"
        )
        .into());
    }

    let weights = load_weights()?;
    let mut recognizer = Recognizer::new(&weights);

    // Like T-one's `forward_offline`: one chunk of silence on each side, so the
    // first and last words are not cut by the model's lookahead.
    let mut signal = vec![0i32; CHUNK];
    signal.extend_from_slice(&audio);
    signal.resize(signal.len() + CHUNK, 0);
    signal.resize(signal.len().div_ceil(CHUNK) * CHUNK, 0);

    let mut ctc = GreedyCtc::default();
    let start = Instant::now();
    for chunk in signal.chunks_exact(CHUNK) {
        for token in recognizer.feed(chunk) {
            ctc.push(token);
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    let chunks = signal.len() / CHUNK;
    let duration = audio.len() as f64 / SAMPLE_RATE as f64;

    println!("{}", ctc.text());
    eprintln!(
        "{:.2}s of audio in {:.1} ms ({:.2} ms per 300 ms chunk, RTF {:.4})",
        duration,
        elapsed * 1e3,
        elapsed * 1e3 / chunks as f64,
        elapsed / duration
    );
    Ok(())
}
