//! Real-input FFT.
//!
//! A real signal of length `n` is packed into a complex one of length
//! `m = n / 2`, even samples as real parts and odd ones as imaginary, which a
//! Stockham radix-2 FFT transforms. One pass then unpacks the result into the
//! `n / 2 + 1` bins of the real transform. Stockham sorts its output as it
//! goes, so there is no bit reversal, and each stage reads the two contiguous
//! halves of its input:
//!
//! `y[q + 2sp] = a + b`, `y[q + 2sp + s] = (a - b)·w_p`, where
//! `a = x[q + sp]` and `b = x[q + sp + m/2]`,
//!
//! for stride `s` = 1, 2, 4, …, `m/2`, `p < m/(2s)` and `q < s`. From `s = 16`
//! on, a vector covers 16 values of `q` and shares one twiddle. Below that it
//! covers `16 / s` values of `p`, and the two results are interleaved back in
//! blocks of `s`.

use crate::kernels::simd::simd_call;
use fearless_simd::{Level, Simd, f32x16, f64x8, prelude::*};
use fearless_simd_macros::simd;

const LANES: usize = 16;

/// Floats between the scratch buffers; see `Rfft::split_scratch`.
const SCRATCH_GAP: usize = 64;

/// Twiddles of one Stockham stage.
struct Stage {
    /// Stride: the number of butterflies in a row that share a twiddle.
    s: usize,
    /// `w_p = exp(-2πi·p·s/m)`. Repeated `s` times each when `s < LANES`, so
    /// that butterfly `t = sp + q` reads its own at `t`; once per `p` otherwise.
    re: Vec<f32>,
    im: Vec<f32>,
}

/// A real FFT of one power-of-two length, with its twiddles.
pub struct Rfft {
    n: usize,
    /// Detected once here: on short lengths, detecting per call costs a
    /// noticeable part of the transform.
    level: Level,
    stages: Vec<Stage>,
    /// `exp(-2πi·k/n)` for unpacking bin `k`, `k < n/2`.
    post_re: Vec<f32>,
    post_im: Vec<f32>,
}

impl Rfft {
    pub fn new(n: usize) -> Self {
        assert!(n.is_power_of_two(), "FFT length {n} is not a power of two");
        let m = n / 2;
        let twiddle = |num: usize, den: usize| {
            let angle = -2.0 * std::f64::consts::PI * num as f64 / den as f64;
            (angle.cos() as f32, angle.sin() as f32)
        };
        let mut stages = Vec::new();
        let mut s = 1;
        while s < m {
            let per_p = if s < LANES { s } else { 1 };
            let (re, im) = (0..m / (2 * s) * per_p).map(|i| twiddle(i / per_p * s, m)).unzip();
            stages.push(Stage { s, re, im });
            s *= 2;
        }
        let (post_re, post_im) = (0..m).map(|k| twiddle(k, n)).unzip();
        Self { n, level: Level::new(), stages, post_re, post_im }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    /// Floats of scratch that `forward` needs.
    pub fn scratch_len(&self) -> usize {
        4 * (self.n / 2 + SCRATCH_GAP)
    }

    /// The real and imaginary parts of the two buffers each stage reads from
    /// and writes to, `n / 2` floats each.
    ///
    /// They sit `SCRATCH_GAP` floats apart rather than back to back: when
    /// `n / 2` is a multiple of 1024, buffers that touch would start a
    /// multiple of 4 KiB apart, and x86 cores then stall loads from one on
    /// stores to another (4K aliasing).
    fn split_scratch<'s>(&self, scratch: &'s mut [f32]) -> [&'s mut [f32]; 4] {
        let m = self.n / 2;
        let mut parts = scratch[..self.scratch_len()].chunks_exact_mut(m + SCRATCH_GAP).map(|c| &mut c[..m]);
        std::array::from_fn(|_| parts.next().expect("four buffers"))
    }

    /// Bins `0..=n/2` of the DFT of `input` into `out_re` and `out_im`.
    /// `scratch` needs `scratch_len()` floats, of any content.
    pub fn forward(&self, input: &[f32], scratch: &mut [f32], out_re: &mut [f32], out_im: &mut [f32]) {
        self.forward_at(self.level, input, scratch, out_re, out_im)
    }

    fn forward_at(
        &self,
        level: Level,
        input: &[f32],
        scratch: &mut [f32],
        out_re: &mut [f32],
        out_im: &mut [f32],
    ) {
        let n = self.n;
        assert_eq!(input.len(), n, "FFT input length");
        assert!(scratch.len() >= self.scratch_len(), "FFT scratch too short");
        let (out_re, out_im) = (&mut out_re[..n / 2 + 1], &mut out_im[..n / 2 + 1]);
        if n == 1 {
            out_re[0] = input[0];
            out_im[0] = 0.0;
        } else if n / 2 < 2 * LANES {
            rfft_scalar(self, input, scratch, out_re, out_im);
        } else {
            simd_call!(level, rfft_simd(self, input, scratch, out_re, out_im));
        }
    }
}

/// Bins `0..=n/2` of the DFT of `input`, which must have a power-of-two length.
pub fn rfft_forward_f32(input: &[f32], output_re: &mut [f32], output_im: &mut [f32]) {
    let fft = Rfft::new(input.len());
    let mut scratch = vec![0.0; fft.scratch_len()];
    fft.forward(input, &mut scratch, output_re, output_im);
}

/// Bins 0 and `m` of the real transform, from bin 0 of the packed one: the
/// sum and the alternating sum of the samples.
fn unpack_ends(zr: &[f32], zi: &[f32], out_re: &mut [f32], out_im: &mut [f32]) {
    let m = zr.len();
    out_re[0] = zr[0] + zi[0];
    out_im[0] = 0.0;
    out_re[m] = zr[0] - zi[0];
    out_im[m] = 0.0;
}

/// Lengths too short for `rfft_simd`, which needs `m / 2` to be a whole
/// number of vectors.
fn rfft_scalar(fft: &Rfft, input: &[f32], scratch: &mut [f32], out_re: &mut [f32], out_im: &mut [f32]) {
    let m = fft.n / 2;
    let [mut xr, mut xi, mut yr, mut yi] = fft.split_scratch(scratch);
    for (j, pair) in input.chunks_exact(2).enumerate() {
        (xr[j], xi[j]) = (pair[0], pair[1]);
    }
    for st in &fft.stages {
        let s = st.s;
        for t in 0..m / 2 {
            let (p, q) = (t / s, t % s);
            let w = if s < LANES { t } else { p };
            let (ar, ai, br, bi) = (xr[t], xi[t], xr[t + m / 2], xi[t + m / 2]);
            let (dr, di) = (ar - br, ai - bi);
            let at = q + 2 * s * p;
            (yr[at], yi[at]) = (ar + br, ai + bi);
            yr[at + s] = dr * st.re[w] - di * st.im[w];
            yi[at + s] = dr * st.im[w] + di * st.re[w];
        }
        std::mem::swap(&mut xr, &mut yr);
        std::mem::swap(&mut xi, &mut yi);
    }
    unpack_ends(xr, xi, out_re, out_im);
    for k in 1..m {
        let (ar, ai, cr, ci) = (xr[k], xi[k], xr[m - k], xi[m - k]);
        let (er, ei) = (0.5 * (ar + cr), 0.5 * (ai - ci));
        let (or, oi) = (0.5 * (ai + ci), 0.5 * (cr - ar));
        let (wr, wi) = (fft.post_re[k], fft.post_im[k]);
        out_re[k] = er + wr * or - wi * oi;
        out_im[k] = ei + wr * oi + wi * or;
    }
}

#[inline(always)]
fn load<S: Simd>(simd: S, x: &[f32], at: usize) -> f32x16<S> {
    f32x16::from_slice(simd, &x[at..at + LANES])
}

#[inline(always)]
fn store<S: Simd>(v: f32x16<S>, x: &mut [f32], at: usize) {
    v.store_slice(&mut x[at..at + LANES]);
}

/// `(a - b)·w` and `a + b`, on split complex vectors.
#[inline(always)]
fn butterfly<S: Simd>(
    (ar, ai): (f32x16<S>, f32x16<S>),
    (br, bi): (f32x16<S>, f32x16<S>),
    (wr, wi): (f32x16<S>, f32x16<S>),
) -> ((f32x16<S>, f32x16<S>), (f32x16<S>, f32x16<S>)) {
    let (dr, di) = (ar - br, ai - bi);
    ((ar + br, ai + bi), (dr.mul_sub(wr, di * wi), dr.mul_add(wi, di * wr)))
}

/// `a` and `b` interleaved in blocks of `B` lanes: `a`'s first block, `b`'s
/// first, `a`'s second, and so on.
#[inline(always)]
fn interleave_blocks<S: Simd, const B: usize>(a: f32x16<S>, b: f32x16<S>) -> (f32x16<S>, f32x16<S>) {
    match B {
        1 => a.interleave(b),
        2 => {
            let (lo, hi) = a.bitcast::<f64x8<S>>().interleave(b.bitcast::<f64x8<S>>());
            (lo.bitcast(), hi.bitcast())
        }
        4 => {
            let ((a01, a23), (b01, b23)) = (a.split(), b.split());
            let ((a0, a1), (a2, a3)) = (a01.split(), a23.split());
            let ((b0, b1), (b2, b3)) = (b01.split(), b23.split());
            (a0.combine(b0).combine(a1.combine(b1)), a2.combine(b2).combine(a3.combine(b3)))
        }
        8 => {
            let ((a0, a1), (b0, b1)) = (a.split(), b.split());
            (a0.combine(b0), a1.combine(b1))
        }
        _ => unreachable!("block of {B}"),
    }
}

/// A stage with stride `B < LANES`: each vector holds `LANES / B` runs of
/// butterflies, whose two outputs land `B` apart.
#[inline(always)]
fn narrow_stage<S: Simd, const B: usize>(
    simd: S,
    st: &Stage,
    (xr, xi): (&[f32], &[f32]),
    (yr, yi): (&mut [f32], &mut [f32]),
) {
    let half = xr.len() / 2;
    for t in (0..half).step_by(LANES) {
        let a = (load(simd, xr, t), load(simd, xi, t));
        let b = (load(simd, xr, t + half), load(simd, xi, t + half));
        let w = (load(simd, &st.re, t), load(simd, &st.im, t));
        let ((sr, si), (dr, di)) = butterfly(a, b, w);
        let (lo, hi) = interleave_blocks::<S, B>(sr, dr);
        store(lo, yr, 2 * t);
        store(hi, yr, 2 * t + LANES);
        let (lo, hi) = interleave_blocks::<S, B>(si, di);
        store(lo, yi, 2 * t);
        store(hi, yi, 2 * t + LANES);
    }
}

/// A stage with stride `s >= LANES`: runs of whole vectors share a twiddle.
#[inline(always)]
fn wide_stage<S: Simd>(simd: S, st: &Stage, (xr, xi): (&[f32], &[f32]), (yr, yi): (&mut [f32], &mut [f32])) {
    let (s, half) = (st.s, xr.len() / 2);
    for (p, (&wr, &wi)) in st.re.iter().zip(&st.im).enumerate() {
        let w = (f32x16::splat(simd, wr), f32x16::splat(simd, wi));
        for t in (s * p..s * (p + 1)).step_by(LANES) {
            let a = (load(simd, xr, t), load(simd, xi, t));
            let b = (load(simd, xr, t + half), load(simd, xi, t + half));
            let ((sr, si), (dr, di)) = butterfly(a, b, w);
            store(sr, yr, t + s * p);
            store(si, yi, t + s * p);
            store(dr, yr, t + s * p + s);
            store(di, yi, t + s * p + s);
        }
    }
}

/// `m / 2` a whole number of vectors, at least one.
#[simd]
fn rfft_simd<S: Simd>(simd: S, fft: &Rfft, input: &[f32], scratch: &mut [f32], out_re: &mut [f32], out_im: &mut [f32]) {
    let m = fft.n / 2;
    let [mut xr, mut xi, mut yr, mut yi] = fft.split_scratch(scratch);
    for j in (0..m).step_by(LANES) {
        let (even, odd) = simd.deinterleave_f32x16(load(simd, input, 2 * j), load(simd, input, 2 * j + LANES));
        store(even, xr, j);
        store(odd, xi, j);
    }
    for st in &fft.stages {
        let (x, y) = ((&*xr, &*xi), (&mut *yr, &mut *yi));
        match st.s {
            1 => narrow_stage::<S, 1>(simd, st, x, y),
            2 => narrow_stage::<S, 2>(simd, st, x, y),
            4 => narrow_stage::<S, 4>(simd, st, x, y),
            8 => narrow_stage::<S, 8>(simd, st, x, y),
            _ => wide_stage(simd, st, x, y),
        }
        std::mem::swap(&mut xr, &mut yr);
        std::mem::swap(&mut xi, &mut yi);
    }

    // Bin k pairs packed bins k and m - k, read as one reversed vector. The
    // k = 1..m run is one short of a whole number of vectors, so its last
    // vector overlaps the one before.
    unpack_ends(xr, xi, out_re, out_im);
    let half = f32x16::splat(simd, 0.5);
    let mut k = 1;
    while k < m {
        let k0 = k.min(m - LANES);
        let (ar, ai) = (load(simd, xr, k0), load(simd, xi, k0));
        let (cr, ci) = (load(simd, xr, m - k0 - (LANES - 1)).reverse(), load(simd, xi, m - k0 - (LANES - 1)).reverse());
        let (er, ei) = ((ar + cr) * half, (ai - ci) * half);
        let (or, oi) = ((ai + ci) * half, (cr - ar) * half);
        let (wr, wi) = (load(simd, &fft.post_re, k0), load(simd, &fft.post_im, k0));
        store(er + wr.mul_sub(or, wi * oi), out_re, k0);
        store(ei + wr.mul_add(oi, wi * or), out_im, k0);
        k += LANES;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::test_util::{Rng, levels};

    /// The DFT of `x` in f64, bins `0..=n/2`.
    fn dft(x: &[f32]) -> (Vec<f64>, Vec<f64>) {
        let n = x.len();
        (0..=n / 2)
            .map(|k| {
                x.iter().enumerate().fold((0.0, 0.0), |(re, im), (j, &v)| {
                    let angle = -2.0 * std::f64::consts::PI * ((j * k) % n) as f64 / n as f64;
                    (re + v as f64 * angle.cos(), im + v as f64 * angle.sin())
                })
            })
            .unzip()
    }

    /// Largest error over all bins, relative to the largest bin magnitude.
    fn relative_error(got: (&[f32], &[f32]), want: (&[f64], &[f64])) -> f64 {
        let scale = want.0.iter().zip(want.1).map(|(r, i)| r.hypot(*i)).fold(1e-30, f64::max);
        let err = (0..want.0.len())
            .map(|k| (got.0[k] as f64 - want.0[k]).hypot(got.1[k] as f64 - want.1[k]))
            .fold(0.0, f64::max);
        err / scale
    }

    #[test]
    fn test_rfft_matches_dft_at_every_length_and_level() {
        for level in levels() {
            let mut rng = Rng::new(3);
            for n in (0..=12).map(|b| 1usize << b) {
                let x = rng.vec(n, -1.0, 1.0);
                let fft = Rfft::new(n);
                // Scratch and outputs start as NaN, so any value read before it
                // is written shows up.
                let mut scratch = vec![f32::NAN; fft.scratch_len()];
                let (mut re, mut im) = (vec![f32::NAN; n / 2 + 1], vec![f32::NAN; n / 2 + 1]);
                fft.forward_at(level, &x, &mut scratch, &mut re, &mut im);
                let (want_re, want_im) = dft(&x);
                let err = relative_error((&re, &im), (&want_re, &want_im));
                assert!(err < 1e-6, "{level:?} n={n}: relative error {err:.3e}");
            }
        }
    }

    #[test]
    fn test_rfft_of_tone_has_one_bin() {
        let n = 512;
        let x: Vec<f32> = (0..n).map(|j| (2.0 * std::f32::consts::PI * 5.0 * j as f32 / n as f32).cos()).collect();
        let (mut re, mut im) = (vec![0.0; n / 2 + 1], vec![0.0; n / 2 + 1]);
        rfft_forward_f32(&x, &mut re, &mut im);
        for k in 0..=n / 2 {
            let want = if k == 5 { n as f32 / 2.0 } else { 0.0 };
            assert!((re[k] - want).abs() < 1e-3 && im[k].abs() < 1e-3, "bin {k}: {} {}", re[k], im[k]);
        }
    }

    #[test]
    fn test_rfft_reuses_scratch() {
        let n = 256;
        let fft = Rfft::new(n);
        let mut scratch = vec![0.0; fft.scratch_len()];
        let (mut re, mut im) = (vec![0.0; n / 2 + 1], vec![0.0; n / 2 + 1]);
        let mut rng = Rng::new(4);
        for _ in 0..3 {
            let x = rng.vec(n, -1.0, 1.0);
            fft.forward(&x, &mut scratch, &mut re, &mut im);
            let (want_re, want_im) = dft(&x);
            assert!(relative_error((&re, &im), (&want_re, &want_im)) < 1e-6);
        }
    }

    #[test]
    #[should_panic(expected = "not a power of two")]
    fn test_rfft_rejects_other_lengths() {
        Rfft::new(400);
    }
}
