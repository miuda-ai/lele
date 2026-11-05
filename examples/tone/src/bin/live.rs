//! Live transcription from the microphone.
//!
//! The current phrase is redrawn in place as each 300 ms chunk is decoded; a
//! phrase ends, and a new line starts, once nobody has spoken for a while.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, SizedSample, StreamConfig};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use tone_example::{CHUNK, GreedyCtc, Recognizer, SAMPLE_RATE, is_speech, load_weights};

/// Frames (30 ms each) without speech that close a phrase.
const PHRASE_END_FRAMES: usize = 20;
/// Characters of the current phrase kept on screen, so the redrawn line does
/// not wrap (a wrapped line cannot be redrawn with `\r`).
const MAX_LINE: usize = 100;

/// Windowed-sinc resampler for an arbitrary rate ratio, fed in blocks.
struct Resampler {
    /// Input samples per output sample.
    step: f64,
    /// Low-pass cutoff in cycles per input sample.
    cutoff: f64,
    /// Kernel half-width in input samples.
    half: f64,
    buf: Vec<f32>,
    /// Position of the next output sample, in input samples from `buf[0]`.
    pos: f64,
}

impl Resampler {
    fn new(from: u32, to: u32) -> Self {
        let step = from as f64 / to as f64;
        // Pass up to 90% of the output's Nyquist; above it would alias.
        let cutoff = 0.45 / step.max(1.0);
        Self {
            step,
            cutoff,
            half: 16.0 * step.max(1.0),
            buf: Vec::new(),
            pos: 0.0,
        }
    }

    fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        self.buf.extend_from_slice(input);
        while self.pos + self.half < self.buf.len() as f64 {
            let lo = (self.pos - self.half).ceil().max(0.0) as usize;
            let hi = (self.pos + self.half).floor() as usize;
            let mut acc = 0.0f64;
            for (i, &x) in self.buf[lo..=hi].iter().enumerate() {
                let t = (lo + i) as f64 - self.pos;
                let arg = 2.0 * self.cutoff * t;
                let sinc = if arg.abs() < 1e-9 {
                    1.0
                } else {
                    (std::f64::consts::PI * arg).sin() / (std::f64::consts::PI * arg)
                };
                // Hann window over [-half, half].
                let window = 0.5 + 0.5 * (std::f64::consts::PI * t / self.half).cos();
                acc += x as f64 * 2.0 * self.cutoff * sinc * window;
            }
            out.push(acc as f32);
            self.pos += self.step;
        }
        // Drop input no future output can reach.
        let keep_from = ((self.pos - self.half).floor().max(0.0) as usize).min(self.buf.len());
        self.buf.drain(..keep_from);
        self.pos -= keep_from as f64;
    }
}

#[cfg(test)]
mod tests {
    use super::Resampler;

    /// Resamples a 48 kHz tone to 8 kHz in uneven blocks; returns the output
    /// after the filter has settled.
    fn resample_tone(freq: f64) -> Vec<f32> {
        let input: Vec<f32> = (0..48000)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 48000.0).sin() as f32)
            .collect();
        let mut r = Resampler::new(48000, 8000);
        let mut out = Vec::new();
        for block in input.chunks(997) {
            r.process(block, &mut out);
        }
        out[100..out.len() - 100].to_vec()
    }

    #[test]
    fn keeps_speech_band_and_rate() {
        let out = resample_tone(1000.0);
        // One second in, one second (less the filter's tail) out.
        assert!((7900..=8000).contains(&(out.len() + 200)), "{}", out.len());
        let peak = out.iter().fold(0f32, |m, &v| m.max(v.abs()));
        assert!((0.97..1.03).contains(&peak), "1 kHz peak {peak}");
        // 8 output samples per cycle: sign changes every 4 samples.
        let crossings = out.windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count();
        let expected = out.len() / 4;
        assert!(crossings.abs_diff(expected) <= 2, "{crossings} vs {expected}");
    }

    #[test]
    fn removes_content_above_new_nyquist() {
        // 6 kHz would alias to 2 kHz at 8 kHz without the low-pass.
        let peak = resample_tone(6000.0).iter().fold(0f32, |m, &v| m.max(v.abs()));
        assert!(peak < 0.01, "6 kHz leaked through at {peak}");
    }
}

/// Picks a config that delivers 8 kHz directly if the device (or the sound
/// server behind it) offers one, preferring mono; otherwise its default.
fn pick_config(device: &cpal::Device) -> Result<(StreamConfig, SampleFormat), Box<dyn std::error::Error>> {
    let usable = |f: SampleFormat| matches!(f, SampleFormat::F32 | SampleFormat::I16 | SampleFormat::I32);
    let mut direct: Vec<_> = device
        .supported_input_configs()?
        .filter(|c| usable(c.sample_format()))
        .filter_map(|c| c.try_with_sample_rate(SAMPLE_RATE))
        .collect();
    direct.sort_by_key(|c| c.channels());
    let chosen = match direct.into_iter().next() {
        Some(c) => c,
        None => device.default_input_config()?,
    };
    Ok((chosen.config(), chosen.sample_format()))
}

/// Starts capture, sending mono f32 blocks in [-1, 1] at the device's rate
/// and counting the frames captured.
fn start_capture<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    tx: mpsc::Sender<Vec<f32>>,
    captured: Arc<AtomicUsize>,
) -> Result<cpal::Stream, Box<dyn std::error::Error>>
where
    T: SizedSample,
    f32: cpal::FromSample<T>,
{
    let channels = config.channels as usize;
    let stream = device.build_input_stream(
        config.clone(),
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            captured.fetch_add(data.len() / channels, Ordering::Relaxed);
            let mono = data
                .chunks_exact(channels)
                .map(|frame| {
                    frame
                        .iter()
                        .map(|&s| <f32 as cpal::FromSample<T>>::from_sample_(s))
                        .sum::<f32>()
                        / channels as f32
                })
                .collect();
            let _ = tx.send(mono);
        },
        |err| eprintln!("\ncapture error: {err}"),
        None,
    )?;
    stream.play()?;
    Ok(stream)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let host = cpal::default_host();
    // Optional argument: part of an input device's name.
    let device = match std::env::args().nth(1) {
        Some(want) => host
            .input_devices()?
            .find(|d| {
                d.description()
                    .map(|desc| desc.name().to_lowercase().contains(&want.to_lowercase()))
                    .unwrap_or(false)
            })
            .ok_or_else(|| format!("no input device matching {want:?}"))?,
        None => host.default_input_device().ok_or("no default input device")?,
    };
    let (config, format) = pick_config(&device)?;
    let name = device
        .description()
        .map(|d| d.name().to_string())
        .unwrap_or_else(|_| "?".into());
    eprintln!(
        "Listening on {name} ({} Hz, {} ch, {format:?}). Speak Russian; Ctrl-C to stop.\n",
        config.sample_rate, config.channels
    );

    let weights = load_weights()?;
    let mut recognizer = Recognizer::new(&weights);

    let (tx, rx) = mpsc::channel();
    let captured = Arc::new(AtomicUsize::new(0));
    let _stream = match format {
        SampleFormat::F32 => start_capture::<f32>(&device, &config, tx, captured.clone())?,
        SampleFormat::I16 => start_capture::<i16>(&device, &config, tx, captured.clone())?,
        SampleFormat::I32 => start_capture::<i32>(&device, &config, tx, captured.clone())?,
        other => return Err(format!("unsupported sample format {other:?}").into()),
    };

    let mut resampler = Resampler::new(config.sample_rate, SAMPLE_RATE);
    let mut resampled = Vec::new();
    let mut pending: Vec<i32> = Vec::new();
    // Samples at 8 kHz fed to the model so far.
    let mut decoded = 0usize;
    let mut ctc = GreedyCtc::default();
    let (mut silent_frames, mut heard_speech) = (0usize, false);
    let mut stdout = std::io::stdout();

    for block in rx {
        resampled.clear();
        resampler.process(&block, &mut resampled);
        pending.extend(
            resampled
                .iter()
                .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i32),
        );
        // Drain everything that has queued up before redrawing.
        while pending.len() >= CHUNK {
            let tokens = recognizer.feed(&pending[..CHUNK]);
            pending.drain(..CHUNK);
            for token in tokens {
                ctc.push(token);
                if is_speech(token) {
                    heard_speech = true;
                    silent_frames = 0;
                } else {
                    silent_frames += 1;
                }
            }

            let text = ctc.text();
            let chars: Vec<char> = text.chars().collect();
            let shown: String = if chars.len() > MAX_LINE {
                std::iter::once('…')
                    .chain(chars[chars.len() - MAX_LINE..].iter().copied())
                    .collect()
            } else {
                text.clone()
            };
            // How far decoding trails the microphone, beyond the chunk being
            // gathered; it grows without bound if the machine is too slow.
            decoded += CHUNK;
            let lag = captured.load(Ordering::Relaxed) as f64 / config.sample_rate as f64
                - decoded as f64 / SAMPLE_RATE as f64
                - CHUNK as f64 / SAMPLE_RATE as f64;
            let lag_note = if lag > 0.5 {
                format!("  [{lag:.1}s behind]")
            } else {
                String::new()
            };
            write!(stdout, "\r\x1b[2K{shown}{lag_note}")?;

            if heard_speech && silent_frames >= PHRASE_END_FRAMES {
                if !text.is_empty() {
                    write!(stdout, "\r\x1b[2K{text}\n")?;
                }
                ctc.clear();
                heard_speech = false;
            }
            stdout.flush()?;
        }
    }
    Ok(())
}
