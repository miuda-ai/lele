use anyhow::Result;
use std::fs::File;
use std::io::{BufWriter, Write};

pub struct WavWriter;

impl WavWriter {
    pub fn save_stereo(path: &str, samples: &[f32], num_channels: usize, sample_rate: u32) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        let bits_per_sample = 16u16;
        let data_size = (samples.len() * 2) as u32;
        let file_size = 36 + data_size;

        writer.write_all(b"RIFF")?;
        writer.write_all(&file_size.to_le_bytes())?;
        writer.write_all(b"WAVE")?;

        writer.write_all(b"fmt ")?;
        writer.write_all(&(16u32).to_le_bytes())?;
        writer.write_all(&(1u16).to_le_bytes())?;
        writer.write_all(&(num_channels as u16).to_le_bytes())?;
        writer.write_all(&sample_rate.to_le_bytes())?;
        writer.write_all(&(sample_rate * num_channels as u32 * 2).to_le_bytes())?;
        writer.write_all(&((num_channels * 2) as u16).to_le_bytes())?;
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
