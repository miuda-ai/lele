pub struct RealFft {
    n: usize,
    tw_re: Vec<f32>,
    tw_im: Vec<f32>,
    bit_rev: Vec<usize>,
}

impl RealFft {
    pub fn new(length: usize) -> Self {
        let (tw_re, tw_im, bit_rev) = crate::kernels::fft::precompute_twiddles(length);
        Self { n: length, tw_re, tw_im, bit_rev }
    }

    pub fn scratch_len(&self) -> usize {
        self.n
    }

    pub fn process_with_scratch(
        &self,
        input: &[f32],
        output: &mut [Complex<f32>],
        _scratch: &mut [Complex<f32>],
    ) {
        assert_eq!(input.len(), self.n);
        let half = self.n / 2 + 1;
        let mut re_buf = vec![0.0f32; self.n];
        let mut im_buf = vec![0.0f32; self.n];
        let mut freq_re = vec![0.0f32; half];
        let mut freq_im = vec![0.0f32; half];

        crate::kernels::fft::rfft_forward_f32_precomputed(
            input, &self.tw_re, &self.tw_im, &self.bit_rev,
            &mut re_buf, &mut im_buf, &mut freq_re, &mut freq_im,
        );

        for i in 0..half {
            output[i] = Complex { re: freq_re[i], im: freq_im[i] };
        }
        for i in half..self.n {
            let j = self.n - i;
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
