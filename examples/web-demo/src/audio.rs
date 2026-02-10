/// Audio utilities for WASM (no filesystem access).

/// Decode WAV bytes into f32 samples.
pub fn decode_wav_bytes(data: &[u8]) -> Result<(Vec<f32>, u32), String> {
    if data.len() < 44 {
        return Err("WAV data too short".into());
    }

    if &data[0..4] != b"RIFF" {
        return Err("Not a valid WAV file (missing RIFF)".into());
    }
    if &data[8..12] != b"WAVE" {
        return Err("Not a valid WAV file (missing WAVE)".into());
    }

    let audio_format = u16::from_le_bytes([data[20], data[21]]);
    if audio_format != 1 {
        return Err(format!("Unsupported audio format: {} (only PCM)", audio_format));
    }

    let num_channels = u16::from_le_bytes([data[22], data[23]]);
    let sample_rate = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
    let bits_per_sample = u16::from_le_bytes([data[34], data[35]]);

    // Find data chunk
    let mut pos = 36;
    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]) as usize;

        if chunk_id == b"data" {
            let audio_data = &data[pos + 8..data.len().min(pos + 8 + chunk_size)];
            let samples: Vec<f32> = match bits_per_sample {
                16 => audio_data
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                    .collect(),
                8 => audio_data
                    .iter()
                    .map(|&b| (b as f32 - 128.0) / 128.0)
                    .collect(),
                32 => audio_data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                _ => return Err(format!("Unsupported bits per sample: {}", bits_per_sample)),
            };

            // Convert to mono if stereo
            let mono = if num_channels == 2 {
                samples
                    .chunks(2)
                    .map(|ch| {
                        if ch.len() == 2 {
                            (ch[0] + ch[1]) / 2.0
                        } else {
                            ch[0]
                        }
                    })
                    .collect()
            } else {
                samples
            };

            return Ok((mono, sample_rate));
        }

        pos += 8 + chunk_size;
    }

    Err("No data chunk found in WAV file".into())
}

/// Encode f32 samples to WAV bytes.
pub fn encode_wav_bytes(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let num_channels = 1u16;
    let bits_per_sample = 16u16;
    let data_size = (samples.len() * 2) as u32;
    let file_size = 36 + data_size;

    let mut buf = Vec::with_capacity(44 + data_size as usize);

    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // fmt chunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&num_channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&(sample_rate * num_channels as u32 * 2).to_le_bytes());
    buf.extend_from_slice(&(num_channels * 2).to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());

    for &sample in samples {
        let s = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        buf.extend_from_slice(&s.to_le_bytes());
    }

    buf
}

/// Simple linear resampling.
pub fn resample(samples: &[f32], from_rate: usize, to_rate: usize) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = (samples.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos as usize;
        let frac = src_pos - idx as f64;

        if idx + 1 < samples.len() {
            let val = samples[idx] as f64 * (1.0 - frac) + samples[idx + 1] as f64 * frac;
            output.push(val as f32);
        } else if idx < samples.len() {
            output.push(samples[idx]);
        }
    }

    output
}
