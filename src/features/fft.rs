pub struct RealFft {
    fft: crate::kernels::fft::Rfft,
}

impl RealFft {
    pub fn new(length: usize) -> Self {
        Self { fft: crate::kernels::fft::Rfft::new(length) }
    }

    pub fn scratch_len(&self) -> usize {
        self.fft.len()
    }

    pub fn process_with_scratch(
        &self,
        input: &[f32],
        output: &mut [Complex<f32>],
        _scratch: &mut [Complex<f32>],
    ) {
        let n = self.fft.len();
        assert_eq!(input.len(), n);
        let half = n / 2 + 1;
        let mut buf = vec![0.0f32; self.fft.scratch_len() + 2 * half];
        let (scratch, freq) = buf.split_at_mut(self.fft.scratch_len());
        let (freq_re, freq_im) = freq.split_at_mut(half);

        self.fft.forward(input, scratch, freq_re, freq_im);

        for i in 0..half {
            output[i] = Complex { re: freq_re[i], im: freq_im[i] };
        }
        for i in half..n {
            let j = n - i;
            output[i] = Complex { re: freq_re[j], im: -freq_im[j] };
        }
    }

    pub fn process(&self, input: &[f32], output: &mut [Complex<f32>]) {
        let mut scratch = vec![Complex { re: 0.0, im: 0.0 }; self.scratch_len()];
        self.process_with_scratch(input, output, &mut scratch);
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Complex<T> {
    pub re: T,
    pub im: T,
}
