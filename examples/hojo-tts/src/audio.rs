use anyhow::{bail, Result};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

pub struct WavWriter;

impl WavWriter {
    pub fn save(path: &str, samples: &[f32], sample_rate: u32) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        let num_channels = 1u16;
        let bits_per_sample = 16u16;
        let data_size = (samples.len() * 2) as u32;
        let file_size = 36 + data_size;

        writer.write_all(b"RIFF")?;
        writer.write_all(&file_size.to_le_bytes())?;
        writer.write_all(b"WAVE")?;
        writer.write_all(b"fmt ")?;
        writer.write_all(&(16u32).to_le_bytes())?;
        writer.write_all(&(1u16).to_le_bytes())?;
        writer.write_all(&num_channels.to_le_bytes())?;
        writer.write_all(&sample_rate.to_le_bytes())?;
        writer.write_all(&(sample_rate * num_channels as u32 * 2).to_le_bytes())?;
        writer.write_all(&(num_channels * 2).to_le_bytes())?;
        writer.write_all(&bits_per_sample.to_le_bytes())?;
        writer.write_all(b"data")?;
        writer.write_all(&data_size.to_le_bytes())?;
        for &sample in samples {
            let s = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
            writer.write_all(&s.to_le_bytes())?;
        }
        writer.flush()?;
        Ok(())
    }
}

/// Minimal PCM WAV reader. Returns mono f32 samples and the sample rate.
pub struct WavReader;

impl WavReader {
    pub fn load(path: &str) -> Result<(Vec<f32>, u32)> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut header = [0u8; 12];
        reader.read_exact(&mut header)?;
        if &header[0..4] != b"RIFF" || &header[8..12] != b"WAVE" {
            bail!("Not a RIFF/WAVE file: {}", path);
        }

        let mut sample_rate = 0u32;
        let mut bits_per_sample = 16u16;
        let mut num_channels = 1u16;
        let mut data: Vec<u8> = Vec::new();

        loop {
            let mut chunk_header = [0u8; 8];
            match reader.read_exact(&mut chunk_header) {
                Ok(()) => {}
                Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            let id = &chunk_header[0..4];
            let size = u32::from_le_bytes([
                chunk_header[4],
                chunk_header[5],
                chunk_header[6],
                chunk_header[7],
            ]);
            let mut payload = vec![0u8; size as usize];
            reader.read_exact(&mut payload)?;

            if id == b"fmt " {
                // PCM fmt chunk: audioFormat(2) channels(2) sampleRate(4) ...
                num_channels = u16::from_le_bytes([payload[2], payload[3]]);
                sample_rate = u32::from_le_bytes([
                    payload[4],
                    payload[5],
                    payload[6],
                    payload[7],
                ]);
                bits_per_sample = u16::from_le_bytes([payload[14], payload[15]]);
            } else if id == b"data" {
                data = payload;
            }
        }

        if data.is_empty() {
            bail!("No data chunk found in WAV: {}", path);
        }

        let samples = decode_pcm(&data, bits_per_sample, num_channels)?;
        Ok((samples, sample_rate))
    }
}

fn decode_pcm(data: &[u8], bits: u16, channels: u16) -> Result<Vec<f32>> {
    let mut mono = Vec::new();
    match bits {
        16 => {
            let frames = data.chunks_exact(channels as usize * 2);
            for frame in frames {
                let mut acc = 0.0f32;
                for ch in 0..channels as usize {
                    let off = ch * 2;
                    let v = i16::from_le_bytes([frame[off], frame[off + 1]]);
                    acc += v as f32 / 32768.0;
                }
                mono.push(acc / channels as f32);
            }
        }
        8 => {
            let frames = data.chunks_exact(channels as usize);
            for frame in frames {
                let mut acc = 0.0f32;
                for &b in frame {
                    acc += ((b as i32 - 128) as f32) / 128.0;
                }
                mono.push(acc / channels as f32);
            }
        }
        32 => {
            // assume f32 PCM
            let frames = data.chunks_exact(channels as usize * 4);
            for frame in frames {
                let mut acc = 0.0f32;
                for ch in 0..channels as usize {
                    let off = ch * 4;
                    let v = f32::from_le_bytes([
                        frame[off],
                        frame[off + 1],
                        frame[off + 2],
                        frame[off + 3],
                    ]);
                    acc += v;
                }
                mono.push(acc / channels as f32);
            }
        }
        other => bail!("Unsupported WAV bit depth: {}", other),
    }
    Ok(mono)
}

/// Sinc resampling matching torchaudio.functional.resample (sinc_interp_hann).
pub fn resample_sinc(samples: &[f32], from_sr: u32, to_sr: u32) -> Vec<f32> {
    if from_sr == to_sr || samples.is_empty() {
        return samples.to_vec();
    }

    let g = gcd(from_sr, to_sr);
    let orig_freq = from_sr / g;
    let new_freq = to_sr / g;

    let lowpass_filter_width: f32 = 6.0;
    let rolloff: f32 = 0.99;

    let base_freq = orig_freq.min(new_freq) as f32 * rolloff;
    let width = ((lowpass_filter_width * orig_freq as f32) / base_freq).ceil() as usize;
    let num_taps = 2 * width + orig_freq as usize;

    let scale = base_freq / orig_freq as f32;

    let mut kernel = vec![0.0f32; new_freq as usize * num_taps];

    for phase in 0..new_freq as usize {
        for k in 0..num_taps {
            let idx_val = (k as i64 - width as i64) as f32 / orig_freq as f32;
            let mut t = -(phase as f32) / new_freq as f32 + idx_val;
            t *= base_freq;
            t = t.clamp(-lowpass_filter_width, lowpass_filter_width);
            let window = (t * std::f32::consts::PI / lowpass_filter_width / 2.0).cos().powi(2);
            t *= std::f32::consts::PI;
            let sinc = if t == 0.0 { 1.0 } else { t.sin() / t };
            kernel[phase * num_taps + k] = sinc * window * scale;
        }
    }

    let length = samples.len();
    let target_length =
        ((new_freq as f64 * length as f64) / orig_freq as f64).ceil() as usize;

    let pad_left = width;
    let pad_right = width + orig_freq as usize;
    let padded_len = length + pad_left + pad_right;
    let mut padded = vec![0.0f32; padded_len];
    padded[pad_left..pad_left + length].copy_from_slice(samples);

    let stride = orig_freq as usize;
    let n_conv = (padded_len.saturating_sub(num_taps)) / stride + 1;

    let mut output = Vec::with_capacity(target_length);
    'outer: for i in 0..n_conv {
        let base = i * stride;
        for phase in 0..new_freq as usize {
            if output.len() >= target_length {
                break 'outer;
            }
            let kp = &kernel[phase * num_taps..phase * num_taps + num_taps];
            let mut sum = 0.0f32;
            for k in 0..num_taps {
                sum += padded[base + k] * kp[k];
            }
            output.push(sum);
        }
    }

    output.truncate(target_length);
    output
}

fn gcd(a: u32, b: u32) -> u32 {
    let mut a = a;
    let mut b = b;
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}
