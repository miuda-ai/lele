use std::fs::File;
use std::io::Read;

/// Simple WAV file reader for 16kHz mono PCM audio
pub struct WavReader;

impl WavReader {
    /// Load a WAV file and return the audio samples as f32 in [-1, 1] range
    pub fn load(path: &str) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
        let mut file = File::open(path)?;
        let mut header = vec![0u8; 44]; // Standard WAV header is 44 bytes
        file.read_exact(&mut header)?;

        // Verify RIFF header
        if &header[0..4] != b"RIFF" {
            return Err("Not a valid WAV file (missing RIFF)".into());
        }

        if &header[8..12] != b"WAVE" {
            return Err("Not a valid WAV file (missing WAVE)".into());
        }

        // Parse audio format (bytes 20-21: 1=PCM)
        let audio_format = u16::from_le_bytes([header[20], header[21]]);
        if audio_format != 1 {
            return Err(format!(
                "Unsupported audio format: {} (only PCM supported)",
                audio_format
            )
            .into());
        }

        // Parse number of channels (bytes 22-23)
        let num_channels = u16::from_le_bytes([header[22], header[23]]);

        // Parse sample rate (bytes 24-27)
        let sample_rate = u32::from_le_bytes([header[24], header[25], header[26], header[27]]);

        // Parse bits per sample (bytes 34-35)
        let bits_per_sample = u16::from_le_bytes([header[34], header[35]]);

        println!(
            "WAV info: {} Hz, {} channels, {} bits",
            sample_rate, num_channels, bits_per_sample
        );

        // Read remaining audio data
        let mut audio_data = Vec::new();
        file.read_to_end(&mut audio_data)?;

        // Convert PCM data to f32
        let samples = match bits_per_sample {
            16 => {
                let mut samples = Vec::with_capacity(audio_data.len() / 2);
                for chunk in audio_data.chunks_exact(2) {
                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                    samples.push(sample as f32 / 32768.0);
                }
                samples
            }
            8 => audio_data
                .iter()
                .map(|&b| (b as f32 - 128.0) / 128.0)
                .collect(),
            _ => return Err(format!("Unsupported bits per sample: {}", bits_per_sample).into()),
        };

        // Convert to mono if stereo
        let mono_samples = if num_channels == 2 {
            samples.chunks(2).map(|ch| (ch[0] + ch[1]) / 2.0).collect()
        } else {
            samples
        };

        Ok((mono_samples, sample_rate))
    }
}
