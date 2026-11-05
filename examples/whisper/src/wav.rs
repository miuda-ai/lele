/// Reads a PCM WAV (8, 16, 24 or 32 bits, or 32-bit float), walking the RIFF
/// chunks rather than assuming a 44-byte header, and averages the channels down
/// to mono in [-1, 1]. Returns the samples and their rate.
pub fn read_wav(path: &str) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }
    let (mut format, mut channels, mut rate, mut bits) = (0u16, 0usize, 0u32, 0u16);
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into()?) as usize;
        let body = &bytes[pos + 8..(pos + 8).saturating_add(len).min(bytes.len())];
        if id == b"fmt " {
            if body.len() < 16 {
                return Err("truncated fmt chunk".into());
            }
            format = u16::from_le_bytes([body[0], body[1]]);
            channels = u16::from_le_bytes([body[2], body[3]]) as usize;
            rate = u32::from_le_bytes(body[4..8].try_into()?);
            bits = u16::from_le_bytes([body[14], body[15]]);
            // WAVE_FORMAT_EXTENSIBLE: the real format is the sub-format's first word.
            if format == 0xFFFE && body.len() >= 26 {
                format = u16::from_le_bytes([body[24], body[25]]);
            }
        } else if id == b"data" {
            if channels == 0 || rate == 0 {
                return Err("data chunk before fmt chunk".into());
            }
            let width = bits as usize / 8;
            let sample = |b: &[u8]| -> Result<f32, String> {
                Ok(match (format, bits) {
                    (1, 8) => (b[0] as f32 - 128.0) / 128.0,
                    (1, 16) => i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0,
                    (1, 24) => {
                        (i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8) as f32 / 8_388_608.0
                    }
                    (1, 32) => {
                        i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2_147_483_648.0
                    }
                    (3, 32) => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                    _ => return Err(format!("unsupported WAV format {format} with {bits} bits")),
                })
            };
            if width == 0 {
                return Err("zero bits per sample".into());
            }
            let mut samples = Vec::with_capacity(body.len() / (width * channels));
            for frame in body.chunks_exact(width * channels) {
                let mut sum = 0.0;
                for s in frame.chunks_exact(width) {
                    sum += sample(s)?;
                }
                samples.push(sum / channels as f32);
            }
            return Ok((samples, rate));
        }
        pos = pos.saturating_add(8 + len + (len & 1));
    }
    Err("no data chunk".into())
}
