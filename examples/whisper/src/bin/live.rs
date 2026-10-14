//! Whisper on the microphone: speech is cut into utterances at pauses
//! ([`whisper_example::endpoint`]), and each is transcribed when it ends.
//!
//! `whisper-live [--lang en] [part of an input device's name]`
//!
//! `whisper-live --wav in.wav [--lang en]` runs a recording through the same path, as
//! if it were being spoken.
//!
//! The bottom line shows the live state: the level of the microphone against the
//! threshold above which it counts as speech. Whisper is not a streaming model, so
//! a transcript comes about a second after a pause, whatever the utterance's length.

use std::io::Write;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use whisper_example::capture::Microphone;
use whisper_example::endpoint::{Config, Endpointer, Event, SAMPLE_RATE};
use whisper_example::resample::Resampler;
use whisper_example::wav::read_wav;
use whisper_example::{Weights, Whisper};

/// Audio needed to trust a detected language: Whisper takes a word or two for
/// another language, so shorter utterances reuse the last one detected.
const DETECTION_SECS: f64 = 2.0;

struct Session<'a> {
    whisper: Whisper<'a>,
    endpoint: Endpointer,
    /// The language to expect, if given; otherwise it is detected.
    language: Option<String>,
    detected: Option<String>,
    /// Utterances transcribed, and what it took.
    utterances: usize,
    audio_secs: f64,
    took: Duration,
}

impl Session<'_> {
    fn feed(&mut self, audio: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
        let mut events = Vec::new();
        self.endpoint.push(audio, &mut |e| events.push(e));
        for event in events {
            match event {
                Event::Start { at } => {
                    println!("\r\x1b[2K\x1b[2m[{at:6.2}s] speech starts\x1b[0m");
                }
                Event::End { at, audio, forced } => self.transcribe(at, &audio, forced)?,
            }
        }
        Ok(())
    }

    fn transcribe(
        &mut self,
        at: f64,
        audio: &[f32],
        forced: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let secs = audio.len() as f64 / SAMPLE_RATE as f64;
        print!("\r\x1b[2K\x1b[2m[{at:6.2}s] {secs:.1} s of speech; transcribing…\x1b[0m");
        std::io::stdout().flush()?;

        let expected = self.language.clone().or(self.detected.clone());
        let started = Instant::now();
        let transcript = self.whisper.transcribe(audio, expected.as_deref())?;
        let took = started.elapsed();
        self.utterances += 1;
        self.audio_secs += secs;
        self.took += took;
        if self.language.is_none() && secs >= DETECTION_SECS {
            self.detected = Some(transcript.language.clone());
        }

        print!("\r\x1b[2K");
        if transcript.text.is_empty() {
            println!("           \x1b[2m(nothing heard)\x1b[0m");
        } else {
            println!(
                "           \x1b[1m[{}] {}\x1b[0m \x1b[2m({secs:.1} s read in {:.0} ms{})\x1b[0m",
                transcript.language,
                transcript.text,
                took.as_secs_f64() * 1e3,
                if forced { ", cut for length" } else { "" }
            );
        }
        Ok(())
    }

    /// The bottom line: whether speech is heard, and the level against the threshold.
    fn status(&self, lag: f64) -> String {
        let e = &self.endpoint;
        // -60..0 dB over 20 cells; the threshold marked on it.
        let cell = |db: f32| ((db + 60.0) / 3.0).clamp(0.0, 19.0) as usize;
        let (level, threshold) = (cell(e.level_db()), cell(e.threshold_db()));
        let bar: String = (0..20)
            .map(|i| match (i <= level, i == threshold) {
                (_, true) => '|',
                (true, _) => '█',
                _ => '░',
            })
            .collect();
        let lag = if lag > 0.5 {
            format!("  [{lag:.1}s behind]")
        } else {
            String::new()
        };
        format!(
            "{} {bar} {:4.0} dB{lag}",
            if e.in_speech() {
                "speaking"
            } else {
                "listening"
            },
            e.level_db()
        )
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |flag: &str| -> Result<Option<String>, String> {
        args.iter()
            .position(|a| a == flag)
            .map(|i| {
                args.get(i + 1)
                    .cloned()
                    .ok_or(format!("{flag} needs a value"))
            })
            .transpose()
    };
    let language = value("--lang")?;
    let wav = value("--wav")?;
    let device = args
        .iter()
        .enumerate()
        .find(|(i, a)| {
            !a.starts_with("--")
                && (*i == 0 || !matches!(args[*i - 1].as_str(), "--lang" | "--wav"))
        })
        .map(|(_, a)| a.as_str());

    let weights = Weights::load()?;
    let mut session = Session {
        whisper: Whisper::new(&weights)?,
        endpoint: Endpointer::new(Config::default()),
        language,
        detected: None,
        utterances: 0,
        audio_secs: 0.0,
        took: Duration::ZERO,
    };
    // A bad language fails here rather than at the first utterance.
    if let Some(code) = &session.language
        && session.whisper.tokens().language(code).is_none()
    {
        return Err(format!("unknown language {code:?}").into());
    }

    if let Some(path) = wav {
        let (audio, rate) = read_wav(&path)?;
        let mut audio16k = Vec::new();
        if rate as usize == SAMPLE_RATE {
            audio16k = audio;
        } else {
            let mut resampler = Resampler::new(rate, SAMPLE_RATE as u32);
            resampler.process(&audio, &mut audio16k);
            resampler.process(&vec![0.0; rate as usize / 10], &mut audio16k);
        }
        let duration = audio16k.len() as f64 / SAMPLE_RATE as f64;
        // Long enough for the last utterance to end.
        audio16k.resize(audio16k.len() + 2 * SAMPLE_RATE, 0.0);
        for block in audio16k.chunks(SAMPLE_RATE / 10) {
            session.feed(block)?;
        }
        println!(
            "\n{duration:.1} s of audio, {} utterances ({:.1} s of speech) read in {:.0} ms",
            session.utterances,
            session.audio_secs,
            session.took.as_secs_f64() * 1e3
        );
        return Ok(());
    }

    let mic = Microphone::open(device, SAMPLE_RATE as u32)?;
    eprintln!(
        "Listening on {} ({} Hz, {} ch). Speak; a pause ends an utterance. Ctrl-C to stop.\n",
        mic.name, mic.sample_rate, mic.channels
    );
    let mut resampler = Resampler::new(mic.sample_rate, SAMPLE_RATE as u32);
    let mut audio = Vec::new();
    for block in &mic.blocks {
        audio.clear();
        if mic.sample_rate as usize == SAMPLE_RATE {
            audio.extend_from_slice(&block);
        } else {
            resampler.process(&block, &mut audio);
        }
        session.feed(&audio)?;
        // How far the transcription trails the microphone.
        let lag = mic.captured.load(Ordering::Relaxed) as f64 / mic.sample_rate as f64
            - session.endpoint.time();
        print!("\r\x1b[2K{}", session.status(lag));
        std::io::stdout().flush()?;
    }
    Ok(())
}
