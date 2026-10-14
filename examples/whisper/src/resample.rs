/// Windowed-sinc resampler for an arbitrary rate ratio, fed in blocks.
pub struct Resampler {
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
    pub fn new(from: u32, to: u32) -> Self {
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

    /// Appends the resampled `input` to `out`. The output trails the input by
    /// the kernel's half-width.
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
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
        let crossings = out
            .windows(2)
            .filter(|w| (w[0] < 0.0) != (w[1] < 0.0))
            .count();
        let expected = out.len() / 4;
        assert!(
            crossings.abs_diff(expected) <= 2,
            "{crossings} vs {expected}"
        );
    }

    #[test]
    fn removes_content_above_new_nyquist() {
        // 6 kHz would alias to 2 kHz at 8 kHz without the low-pass.
        let peak = resample_tone(6000.0)
            .iter()
            .fold(0f32, |m, &v| m.max(v.abs()));
        assert!(peak < 0.01, "6 kHz leaked through at {peak}");
    }
}
