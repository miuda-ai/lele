//! Direct 1D convolutions: dot products straight from the input, with no
//! im2col copy and no matrix multiply.

use crate::kernels::simd::simd_call;
use fearless_simd::{Level, Simd, f32x8, f32x16};
use fearless_simd_macros::simd;

/// Shape of a depthwise convolution: every channel has its own `kernel` taps
/// (`weights` is `[channels, kernel]`), and `pad_left` zeros precede the input.
pub(crate) struct Depthwise<'a> {
    pub batch: usize,
    pub channels: usize,
    pub len: usize,
    pub kernel: usize,
    pub stride: usize,
    pub pad_left: usize,
    pub out_len: usize,
    pub bias: Option<&'a [f32]>,
    pub relu: bool,
}

/// `out[b, c, t] = act(bias[c] + sum_k input[b, c, t * stride + k - pad_left] * weights[c, k])`,
/// with the input zero outside `0..len`.
pub(crate) fn depthwise(
    level: Level,
    shape: &Depthwise,
    input: &[f32],
    weights: &[f32],
    out: &mut [f32],
) {
    // Each channel is copied into `padded` between zeros, so no tap needs a
    // bounds test and a vector load never runs off the end.
    let padded_len = (shape.pad_left + shape.len).max((shape.out_len - 1) * shape.stride + shape.kernel)
        + 2 * VECTOR;
    DEPTHWISE_PADDED.with_borrow_mut(|padded| {
        padded.clear();
        padded.resize(padded_len, 0.0);
        simd_call!(level, depthwise_simd(shape, input, weights, out, padded))
    })
}

const VECTOR: usize = 16;

thread_local! {
    static DEPTHWISE_PADDED: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[simd]
fn depthwise_simd<S: Simd>(
    simd: S,
    s: &Depthwise,
    input: &[f32],
    weights: &[f32],
    out: &mut [f32],
    padded: &mut [f32],
) {
    use fearless_simd::prelude::*;
    let (k, len, out_len, stride) = (s.kernel, s.len, s.out_len, s.stride);
    for c in 0..s.batch * s.channels {
        // Only the row's own span changes between channels; the zeros around it stay.
        padded[s.pad_left..][..len].copy_from_slice(&input[c * len..][..len]);
        let x = &*padded;
        let y = &mut out[c * out_len..][..out_len];
        let w = &weights[(c % s.channels) * k..][..k];
        let bias = s.bias.map_or(0.0, |b| b[c % s.channels]);
        if stride != 1 {
            for (t, y) in y.iter_mut().enumerate() {
                let mut sum = bias;
                for (&xv, &w) in x[t * stride..][..k].iter().zip(w) {
                    sum += xv * w;
                }
                *y = if s.relu { sum.max(0.0) } else { sum };
            }
            continue;
        }
        // `$n` outputs at a time with vectors of type `$v`.
        macro_rules! sweep {
            ($v:ident, $n:literal) => {{
                let vbias = $v::splat(simd, bias);
                let zero = $v::splat(simd, 0.0);
                let at = |t: usize| {
                    let mut acc = vbias;
                    for (kk, &w) in w.iter().enumerate() {
                        let xv = $v::from_slice(simd, &x[t + kk..][..$n]);
                        acc = $v::splat(simd, w).mul_add(xv, acc);
                    }
                    if s.relu { acc.max(zero) } else { acc }
                };
                if out_len >= $n {
                    let mut t = 0;
                    while t + $n <= out_len {
                        at(t).store_slice(&mut y[t..][..$n]);
                        t += $n;
                    }
                    if t < out_len {
                        // The last vector overlaps its predecessor.
                        at(out_len - $n).store_slice(&mut y[out_len - $n..]);
                    }
                } else {
                    let mut tile = [0.0f32; $n];
                    at(0).store_slice(&mut tile);
                    y.copy_from_slice(&tile[..out_len]);
                }
            }};
        }
        if out_len >= 16 {
            sweep!(f32x16, 16)
        } else {
            sweep!(f32x8, 8)
        }
    }
}

/// Shape of a convolution of a single input channel (`weights` is
/// `[out_channels, kernel]`) with no padding or dilation.
pub(crate) struct SingleChannel<'a> {
    pub batch: usize,
    pub input_len: usize,
    pub out_channels: usize,
    pub kernel: usize,
    pub stride: usize,
    pub out_len: usize,
    pub bias: Option<&'a [f32]>,
    pub relu: bool,
}

/// `out[b, oc, t] = act(bias[oc] + sum_k input[b, t * stride + k] * weights[oc, k])`.
pub(crate) fn single_channel(
    level: Level,
    shape: &SingleChannel,
    input: &[f32],
    weights: &[f32],
    out: &mut [f32],
) {
    // Dot products pay a horizontal sum per output, which dominates short
    // kernels; with many outputs per channel it is cheaper to repack the weights
    // once and run the lanes across output channels instead.
    if shape.out_len >= LANES_MIN_OUT_LEN && shape.kernel <= LANES_MAX_KERNEL {
        PACKED.with_borrow_mut(|packed| {
            pack_weights(weights, shape.out_channels, shape.kernel, packed);
            simd_call!(level, single_channel_lanes(shape, input, packed, out))
        });
    } else {
        simd_call!(level, single_channel_simd(shape, input, weights, out))
    }
}

/// Output channels per vector.
const LANES: usize = 8;
/// Below this many outputs per channel the repack costs more than it saves.
const LANES_MIN_OUT_LEN: usize = 32;
/// Above this the dot products have enough work per horizontal sum.
const LANES_MAX_KERNEL: usize = 32;

thread_local! {
    static PACKED: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// `packed[(block * kernel + k) * LANES + lane] = weights[block * LANES + lane, k]`,
/// zero for channels past `out_channels`.
fn pack_weights(weights: &[f32], out_channels: usize, kernel: usize, packed: &mut Vec<f32>) {
    packed.clear();
    packed.resize(out_channels.div_ceil(LANES) * kernel * LANES, 0.0);
    for (oc, row) in weights.chunks_exact(kernel).take(out_channels).enumerate() {
        let (block, lane) = (oc / LANES, oc % LANES);
        for (k, &w) in row.iter().enumerate() {
            packed[(block * kernel + k) * LANES + lane] = w;
        }
    }
}

#[simd]
fn single_channel_lanes<S: Simd>(simd: S, s: &SingleChannel, input: &[f32], packed: &[f32], out: &mut [f32]) {
    let blocks = s.out_channels.div_ceil(LANES);
    for b in 0..s.batch {
        let x = &input[b * s.input_len..][..s.input_len];
        let out = &mut out[b * s.out_channels * s.out_len..][..s.out_channels * s.out_len];
        let mut block = 0;
        // Two blocks at a time share the broadcast input values.
        while block + 2 <= blocks {
            lanes_blocks::<S, 2>(simd, s, x, packed, out, block);
            block += 2;
        }
        if block < blocks {
            lanes_blocks::<S, 1>(simd, s, x, packed, out, block);
        }
    }
}

/// Output channels `block * LANES ..` (`U` blocks) of one batch item.
#[inline(always)]
fn lanes_blocks<S: Simd, const U: usize>(
    simd: S,
    s: &SingleChannel,
    x: &[f32],
    packed: &[f32],
    out: &mut [f32],
    block: usize,
) {
    use fearless_simd::prelude::*;
    let (k, out_len) = (s.kernel, s.out_len);
    let zero = f32x8::splat(simd, 0.0);
    let wb: [&[[f32; LANES]]; U] = core::array::from_fn(|u| {
        let w = &packed[(block + u) * k * LANES..][..k * LANES];
        w.as_chunks::<LANES>().0
    });
    let oc = block * LANES;
    let mut bias = [[0.0f32; LANES]; U];
    if let Some(src) = s.bias {
        for (u, bias) in bias.iter_mut().enumerate() {
            let lo = oc + u * LANES;
            let hi = (lo + LANES).min(s.out_channels);
            bias[..hi - lo].copy_from_slice(&src[lo..hi]);
        }
    }
    let bias = bias.map(|b| f32x8::load_array(simd, b));
    // Four output positions per pass share each weight vector.
    let mut t = 0;
    while t < out_len {
        let n = (out_len - t).min(4);
        let xs: [&[f32]; 4] = core::array::from_fn(|i| {
            let t = (t + i).min(out_len - 1);
            &x[t * s.stride..][..k]
        });
        for xk in &xs {
            assert_eq!(xk.len(), k);
        }
        let mut acc = [[zero; 4]; U];
        for kk in 0..k {
            let xv = [0, 1, 2, 3].map(|i| f32x8::splat(simd, xs[i][kk]));
            for u in 0..U {
                let w = f32x8::load_array_ref(simd, &wb[u][kk]);
                for i in 0..4 {
                    acc[u][i] = w.mul_add(xv[i], acc[u][i]);
                }
            }
        }
        for u in 0..U {
            let y = acc[u].map(|a| {
                let y = a + bias[u];
                *(if s.relu { y.max(zero) } else { y }).as_array()
            });
            let first = oc + u * LANES;
            let lanes = (s.out_channels - first).min(LANES);
            // The positions of one channel are contiguous in the output.
            for l in 0..lanes {
                let row = [y[0][l], y[1][l], y[2][l], y[3][l]];
                out[(first + l) * out_len + t..][..n].copy_from_slice(&row[..n]);
            }
        }
        t += n;
    }
}

/// Dot products of one input window with `N` weight rows, which share each
/// input vector. The sums go 8 lanes at a time, then one by one.
#[inline(always)]
fn dots<S: Simd, const N: usize>(simd: S, x: &[f32], ws: [&[f32]; N]) -> [f32; N] {
    use fearless_simd::prelude::*;
    let (x8, x_rest) = x.as_chunks::<8>();
    let w8 = ws.map(|w| {
        let chunks = w.as_chunks::<8>().0;
        // One check here instead of one per access in the loop.
        assert_eq!(chunks.len(), x8.len());
        chunks
    });
    let mut acc = [f32x8::splat(simd, 0.0); N];
    for (i, xv) in x8.iter().enumerate() {
        let xv = f32x8::load_array_ref(simd, xv);
        for j in 0..N {
            acc[j] = f32x8::load_array_ref(simd, &w8[j][i]).mul_add(xv, acc[j]);
        }
    }
    let mut sums = acc.map(|a| a.reduce_sum());
    let done = x8.len() * 8;
    for (r, &xv) in x_rest.iter().enumerate() {
        for j in 0..N {
            sums[j] += xv * ws[j][done + r];
        }
    }
    sums
}

#[simd]
fn single_channel_simd<S: Simd>(
    simd: S,
    s: &SingleChannel,
    input: &[f32],
    weights: &[f32],
    out: &mut [f32],
) {
    let (k, out_len) = (s.kernel, s.out_len);
    let finish = |sum: f32, bias: f32| {
        let v = sum + bias;
        if s.relu { v.max(0.0) } else { v }
    };
    for b in 0..s.batch {
        let x = &input[b * s.input_len..][..s.input_len];
        let out = &mut out[b * s.out_channels * out_len..][..s.out_channels * out_len];
        // Four output channels at a time, so each input window is loaded once for four.
        let first = s.out_channels / 4 * 4;
        let mut rows = out.chunks_exact_mut(4 * out_len);
        for (g, rows) in rows.by_ref().enumerate() {
            let oc = g * 4;
            let ws: [&[f32]; 4] = core::array::from_fn(|j| &weights[(oc + j) * k..][..k]);
            let bias: [f32; 4] = core::array::from_fn(|j| s.bias.map_or(0.0, |b| b[oc + j]));
            let (r0, rest) = rows.split_at_mut(out_len);
            let (r1, rest) = rest.split_at_mut(out_len);
            let (r2, r3) = rest.split_at_mut(out_len);
            for t in 0..out_len {
                let sums = dots(simd, &x[t * s.stride..][..k], ws);
                r0[t] = finish(sums[0], bias[0]);
                r1[t] = finish(sums[1], bias[1]);
                r2[t] = finish(sums[2], bias[2]);
                r3[t] = finish(sums[3], bias[3]);
            }
        }
        for (i, row) in rows.into_remainder().chunks_exact_mut(out_len).enumerate() {
            let oc = first + i;
            let ws = [&weights[oc * k..][..k]];
            let bias = s.bias.map_or(0.0, |b| b[oc]);
            for (t, y) in row.iter_mut().enumerate() {
                let [sum] = dots(simd, &x[t * s.stride..][..k], ws);
                *y = finish(sum, bias);
            }
        }
    }
}
