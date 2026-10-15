//! `whisper in.wav [--lang ru]`: transcribes a 16 kHz recording; the language
//! is detected unless given.

use std::time::Instant;
use whisper_example::wav::read_wav;
use whisper_example::{Weights, Whisper};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let language = args
        .iter()
        .position(|a| a == "--lang")
        .map(|i| args.get(i + 1).ok_or("--lang needs a language code"))
        .transpose()?
        .map(String::as_str);
    let path = args
        .iter()
        .enumerate()
        .find(|(i, a)| !a.starts_with("--") && (*i == 0 || args[*i - 1] != "--lang"))
        .map(|(_, a)| a)
        .ok_or("usage: whisper <16 kHz .wav> [--lang <code>]")?;

    let (audio, rate) = read_wav(path)?;
    if rate != 16_000 {
        return Err(format!(
            "Whisper expects 16 kHz audio, got {rate} Hz; convert with `ffmpeg -i in -ar 16000 -ac 1 out.wav`"
        )
        .into());
    }

    let weights = Weights::load()?;
    let mut whisper = Whisper::new(&weights)?;
    let started = Instant::now();
    let transcript = whisper.transcribe(&audio, language)?;
    let elapsed = started.elapsed().as_secs_f64();
    let duration = audio.len() as f64 / 16_000.0;

    println!("{}", transcript.text);
    eprintln!(
        "[{}] {duration:.2}s of audio in {:.0} ms (RTF {:.3})",
        transcript.language,
        elapsed * 1e3,
        elapsed / duration
    );
    Ok(())
}
