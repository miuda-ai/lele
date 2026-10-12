//! Depthwise convolution and max pooling over the planes of an NCHW tensor:
//! every channel slides its own window over its own plane.

use crate::kernels::conv2d::Activation;
use crate::kernels::simd::simd_call;
use crate::kernels::simd_math as vm;
use fearless_simd::{Level, Simd, f32x16};
use fearless_simd_macros::simd;

/// Output `(oh, ow)` of a channel reads input
/// `(oh * stride_h + ki * dilation_h - pad_top, ow * stride_w + kj * dilation_w - pad_left)`
/// for every tap `(ki, kj)` of the window; taps outside the input read the padding.
/// The input is any number of `[channels, in_h, in_w]` blocks.
pub(crate) struct Window {
    pub channels: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub dilation_h: usize,
    pub dilation_w: usize,
    pub pad_top: usize,
    pub pad_left: usize,
    pub out_h: usize,
    pub out_w: usize,
}

const VECTOR: usize = 16;

thread_local! {
    static PADDED: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Where one channel is copied before the window slides over it: a plane with
/// the padding written out, so no tap needs a bounds test and no vector load
/// runs off the end. Under stride 2 the even and the odd columns go to two
/// separate planes, so 16 neighbouring outputs of one tap read 16 consecutive
/// entries of one of them.
struct Layout {
    /// 2 under stride 2, else 1.
    phases: usize,
    rows: usize,
    row_len: usize,
    /// Entries of a plane row between neighbouring outputs: 1, except under
    /// strides above 2, which are not vectorized.
    step: usize,
    /// Where each tap, window row by window row, reads relative to where
    /// output `(oh, 0)` starts: entry `oh * stride_h * row_len`.
    taps: Vec<usize>,
    max_tap: usize,
}

impl Layout {
    fn new(w: &Window) -> Self {
        let phases = if w.stride_w == 2 { 2 } else { 1 };
        let step = w.stride_w / phases;
        let rows = (w.pad_top + w.in_h)
            .max((w.out_h - 1) * w.stride_h + (w.kernel_h - 1) * w.dilation_h + 1);
        // A row shorter than a vector is computed as a whole vector.
        let reach = (w.out_w.max(VECTOR) - 1) * step + (w.kernel_w - 1) * w.dilation_w / phases + 1;
        let row_len = (w.pad_left + w.in_w).div_ceil(phases).max(reach);
        // Padded column `ow * stride_w + kj * dilation_w` is entry `ow * step + col / phases`
        // of phase `col % phases`, where `col = kj * dilation_w`.
        let taps = (0..w.kernel_h)
            .flat_map(|ki| {
                (0..w.kernel_w).map(move |kj| {
                    let col = kj * w.dilation_w;
                    (col % phases * rows + ki * w.dilation_h) * row_len + col / phases
                })
            })
            .collect::<Vec<_>>();
        let max_tap = *taps.iter().max().expect("a window has a tap");
        Self {
            phases,
            rows,
            row_len,
            step,
            taps,
            max_tap,
        }
    }
    fn len(&self) -> usize {
        self.phases * self.rows * self.row_len
    }

    /// Copies one channel into `padded`; the entries around it keep the padding.
    #[inline(always)]
    fn fill<S: Simd>(&self, simd: S, w: &Window, src: &[f32], padded: &mut [f32]) {
        use fearless_simd::prelude::*;
        let (even, odd) = padded.split_at_mut(self.rows * self.row_len);
        for (ih, src) in src.chunks_exact(w.in_w).enumerate() {
            let row = (w.pad_top + ih) * self.row_len;
            if self.phases == 1 {
                even[row + w.pad_left..][..w.in_w].copy_from_slice(src);
                continue;
            }
            // Input column `iw` is padded column `pad_left + iw`: even ones go
            // to the first phase, odd ones to the second.
            let (first, second) = if w.pad_left % 2 == 0 {
                (&mut *even, &mut *odd)
            } else {
                (&mut *odd, &mut *even)
            };
            let first = &mut first[row + w.pad_left / 2..][..w.in_w.div_ceil(2)];
            let second = &mut second[row + (w.pad_left + 1) / 2..][..w.in_w / 2];
            let (pairs, rest) = src.as_chunks::<{ 2 * VECTOR }>();
            for (j, pair) in pairs.iter().enumerate() {
                let (a, b) = pair.split_at(VECTOR);
                let (e, o) = simd
                    .deinterleave_f32x16(f32x16::from_slice(simd, a), f32x16::from_slice(simd, b));
                e.store_slice(&mut first[j * VECTOR..][..VECTOR]);
                o.store_slice(&mut second[j * VECTOR..][..VECTOR]);
            }
            let done = pairs.len() * VECTOR;
            for (i, &v) in rest.iter().enumerate() {
                let half = if i % 2 == 0 {
                    &mut *first
                } else {
                    &mut *second
                };
                half[done + i / 2] = v;
            }
        }
    }
}

/// `out[b, c] = act(bias[c] + sum_k input[b, c] at tap k * weights[c, k])`, with
/// the input zero outside its plane; `weights` is `[channels, kernel_h, kernel_w]`.
pub(crate) fn depthwise(
    level: Level,
    w: &Window,
    input: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    act: Activation,
    out: &mut [f32],
) {
    let layout = Layout::new(w);
    PADDED.with_borrow_mut(|padded| {
        padded.clear();
        padded.resize(layout.len(), 0.0);
        simd_call!(
            level,
            depthwise_simd(w, &layout, input, weights, bias, act, out, padded)
        )
    })
}

/// `out[b, c]` is the largest input of `[b, c]` under the window; padding never wins.
pub(crate) fn max_pool(level: Level, w: &Window, input: &[f32], out: &mut [f32]) {
    let layout = Layout::new(w);
    PADDED.with_borrow_mut(|padded| {
        padded.clear();
        padded.resize(layout.len(), f32::NEG_INFINITY);
        simd_call!(level, max_pool_simd(w, &layout, input, out, padded))
    })
}

/// Stores `$at(t)`, an array with outputs `t..t + 16` of each row of `$ys`,
/// over those whole rows. The last vector overlaps its predecessor; rows
/// shorter than a vector are computed whole and the part that exists copied.
///
/// A macro rather than a function taking a closure: the closure would be
/// compiled without the target features of the `#[simd]` caller, and every
/// vector operation in it would become a call.
macro_rules! sweep {
    ($ys:expr, |$t:ident| $at:expr) => {{
        let mut ys = $ys;
        let n = ys[0].len();
        if n < VECTOR {
            let $t = 0;
            for (v, y) in $at.into_iter().zip(ys.iter_mut()) {
                let mut tile = [0.0f32; VECTOR];
                v.store_slice(&mut tile);
                y.copy_from_slice(&tile[..n]);
            }
        } else {
            let mut start = 0;
            while start + VECTOR <= n {
                let $t = start;
                for (v, y) in $at.into_iter().zip(ys.iter_mut()) {
                    v.store_slice(&mut y[start..][..VECTOR]);
                }
                start += VECTOR;
            }
            if start < n {
                let $t = n - VECTOR;
                for (v, y) in $at.into_iter().zip(ys.iter_mut()) {
                    v.store_slice(&mut y[n - VECTOR..]);
                }
            }
        }
    }};
}

#[inline(always)]
fn activate<S: Simd>(x: f32x16<S>, act: Activation) -> f32x16<S> {
    match act {
        Activation::None => x,
        Activation::Relu => vm::relu(x),
        Activation::SiLU => vm::silu(x),
    }
}

fn activate_scalar(x: f32, act: Activation) -> f32 {
    match act {
        Activation::None => x,
        Activation::Relu => x.max(0.0),
        Activation::SiLU => x / (1.0 + (-x).exp()),
    }
}

#[simd]
fn depthwise_simd<S: Simd>(
    simd: S,
    w: &Window,
    l: &Layout,
    input: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    act: Activation,
    out: &mut [f32],
    padded: &mut [f32],
) {
    use fearless_simd::prelude::*;
    let taps = &l.taps[..];
    let planes = input
        .chunks_exact(w.in_h * w.in_w)
        .zip(out.chunks_exact_mut(w.out_h * w.out_w));
    for (c, (src, dst)) in planes.enumerate() {
        let ch = c % w.channels;
        let k = &weights[ch * taps.len()..][..taps.len()];
        let b = bias.map_or(0.0, |b| b[ch]);
        l.fill(simd, w, src, padded);
        if l.step != 1 {
            for (oh, y) in dst.chunks_exact_mut(w.out_w).enumerate() {
                let x = &padded[oh * w.stride_h * l.row_len..];
                for (ow, y) in y.iter_mut().enumerate() {
                    let at = &x[ow * l.step..];
                    let s = taps.iter().zip(k).fold(b, |s, (&o, &kv)| s + kv * at[o]);
                    *y = activate_scalar(s, act);
                }
            }
            continue;
        }
        // Two output rows at a time, so each weight is broadcast once for both.
        let row_step = w.stride_h * l.row_len;
        // The common windows get their size at compile time, so their loop
        // over the taps unrolls and the broadcast weights stay in registers.
        macro_rules! rows {
            ($taps:expr, $k:expr) => {{
                let (taps, k): (&[usize], &[f32]) = ($taps, $k);
                let mut rows = dst.chunks_exact_mut(w.out_w).enumerate();
                while let Some((oh, y0)) = rows.next() {
                    let x = &padded[oh * row_step..];
                    // What the loads of one row need; `t` stays below `max(out_w, 16) - 16`.
                    let need = l.max_tap + w.out_w.max(VECTOR);
                    match rows.next() {
                        Some((_, y1)) => {
                            assert!(x.len() >= row_step + need);
                            sweep!([y0, y1], |t| unsafe {
                                depthwise_rows::<S, 2>(simd, &x[t..], row_step, taps, k, b, act)
                            })
                        }
                        None => {
                            assert!(x.len() >= need);
                            sweep!([y0], |t| unsafe {
                                depthwise_rows::<S, 1>(simd, &x[t..], row_step, taps, k, b, act)
                            })
                        }
                    }
                }
            }};
        }
        match taps.len() {
            9 => rows!(
                &<[usize; 9]>::try_from(taps).unwrap()[..],
                &<[f32; 9]>::try_from(k).unwrap()[..]
            ),
            25 => rows!(
                &<[usize; 25]>::try_from(taps).unwrap()[..],
                &<[f32; 25]>::try_from(k).unwrap()[..]
            ),
            _ => rows!(taps, k),
        }
    }
}

/// 16 neighbouring outputs of each of `R` output rows, `row_step` entries of
/// `x` apart; tap 0 of the first row reads `x[taps[0]]`.
///
/// # Safety
///
/// `(R - 1) * row_step + max(taps) + 16 <= x.len()`.
#[inline(always)]
unsafe fn depthwise_rows<S: Simd, const R: usize>(
    simd: S,
    x: &[f32],
    row_step: usize,
    taps: &[usize],
    k: &[f32],
    bias: f32,
    act: Activation,
) -> [f32x16<S>; R] {
    use fearless_simd::prelude::*;
    let mut acc = [f32x16::splat(simd, bias); R];
    for (&o, &kv) in taps.iter().zip(k) {
        let kv = f32x16::splat(simd, kv);
        for (r, acc) in acc.iter_mut().enumerate() {
            *acc = kv.mul_add(unsafe { load(simd, x, r * row_step + o) }, *acc);
        }
    }
    for acc in &mut acc {
        *acc = activate(*acc, act);
    }
    acc
}

/// # Safety
///
/// `at + 16 <= x.len()`.
#[inline(always)]
unsafe fn load<S: Simd>(simd: S, x: &[f32], at: usize) -> f32x16<S> {
    use fearless_simd::prelude::*;
    debug_assert!(at + VECTOR <= x.len());
    f32x16::from_slice(simd, unsafe { x.get_unchecked(at..at + VECTOR) })
}

#[simd]
fn max_pool_simd<S: Simd>(
    simd: S,
    w: &Window,
    l: &Layout,
    input: &[f32],
    out: &mut [f32],
    padded: &mut [f32],
) {
    use fearless_simd::prelude::*;
    let taps = &l.taps[..];
    let planes = input
        .chunks_exact(w.in_h * w.in_w)
        .zip(out.chunks_exact_mut(w.out_h * w.out_w));
    for (src, dst) in planes {
        l.fill(simd, w, src, padded);
        for (oh, y) in dst.chunks_exact_mut(w.out_w).enumerate() {
            let x = &padded[oh * w.stride_h * l.row_len..];
            if l.step != 1 {
                for (ow, y) in y.iter_mut().enumerate() {
                    let at = &x[ow * l.step..];
                    *y = taps.iter().fold(f32::NEG_INFINITY, |m, &o| m.max(at[o]));
                }
                continue;
            }
            // `t` stays below `max(out_w, 16) - 16`.
            assert!(x.len() >= l.max_tap + w.out_w.max(VECTOR));
            sweep!([y], |t| [unsafe { max_pool_at(simd, &x[t..], taps) }]);
        }
    }
}

/// 16 neighbouring outputs of a max-pool row, whose first tap 0 reads `x[taps[0]]`.
///
/// # Safety
///
/// `max(taps) + 16 <= x.len()`.
#[inline(always)]
unsafe fn max_pool_at<S: Simd>(simd: S, x: &[f32], taps: &[usize]) -> f32x16<S> {
    use fearless_simd::prelude::*;
    let (first, rest) = taps.split_first().expect("a window has a tap");
    let mut m = unsafe { load(simd, x, *first) };
    let mut m2 = f32x16::splat(simd, f32::NEG_INFINITY);
    let (pairs, last) = rest.as_chunks::<2>();
    for &[o0, o1] in pairs {
        m = m.max(unsafe { load(simd, x, o0) });
        m2 = m2.max(unsafe { load(simd, x, o1) });
    }
    if let &[o] = last {
        m = m.max(unsafe { load(simd, x, o) });
    }
    m.max(m2)
}
