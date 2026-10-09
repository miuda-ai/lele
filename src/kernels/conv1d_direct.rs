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
    /// Only 1 with a stride other than 1.
    pub dilation: usize,
    pub pad_left: usize,
    pub out_len: usize,
    pub bias: Option<&'a [f32]>,
    pub relu: bool,
}

/// `out[b, c, t] = act(bias[c] + sum_k input[b, c, t * stride + k * dilation - pad_left] * weights[c, k])`,
/// with the input zero outside `0..len`.
pub(crate) fn depthwise(
    level: Level,
    shape: &Depthwise,
    input: &[f32],
    weights: &[f32],
    out: &mut [f32],
) {
    // Each channel is copied into `padded` between zeros, so no tap needs a
    // bounds test and a vector load never runs off the end. Stride 2 also
    // keeps the even and the odd entries of the row apart, after the copy.
    let phases = if shape.stride == 2 { padded_row_len(shape) } else { 0 };
    DEPTHWISE_PADDED.with_borrow_mut(|padded| {
        padded.clear();
        padded.resize(padded_row_len(shape) + phases, 0.0);
        simd_call!(level, depthwise_simd(shape, input, weights, out, padded))
    })
}

/// Room for a row between zeros, as an even number of entries.
fn padded_row_len(shape: &Depthwise) -> usize {
    let reach = (shape.out_len - 1) * shape.stride + (shape.kernel - 1) * shape.dilation + 1;
    let used = (shape.pad_left + shape.len).max(reach);
    (used + 2 * VECTOR).next_multiple_of(2)
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
    let (padded, phases) = padded.split_at_mut(padded_row_len(s));
    let (even, odd) = phases.split_at_mut(phases.len() / 2);
    // With no padding on either side and a whole vector of outputs, every tap
    // of every vector lies inside the row, which can be read in place.
    let in_place = s.pad_left == 0
        && stride == 1
        && out_len >= 8
        && out_len + (k - 1) * s.dilation <= len;
    for c in 0..s.batch * s.channels {
        if !in_place {
            // Only the row's own span changes between channels; the zeros around it stay.
            let (dst, src) = (&mut padded[s.pad_left..][..len], &input[c * len..][..len]);
            if len < 32 {
                // A short copy is cheaper written out than as a call to `memcpy`.
                for (d, &v) in dst.iter_mut().zip(src) {
                    *d = v;
                }
            } else {
                dst.copy_from_slice(src);
            }
        }
        let x = if in_place { &input[c * len..][..len] } else { &*padded };
        if stride == 2 {
            for ((e, o), pair) in even.iter_mut().zip(odd.iter_mut()).zip(x.chunks_exact(2)) {
                (*e, *o) = (pair[0], pair[1]);
            }
        }
        let phase: [&[f32]; 2] = [&*even, &*odd];
        let y = &mut out[c * out_len..][..out_len];
        let w = &weights[(c % s.channels) * k..][..k];
        let bias = s.bias.map_or(0.0, |b| b[c % s.channels]);
        if stride > 2 {
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
            ($at:ident, $n:literal, $m:literal, $srcs:expr) => {{
                let srcs: [&[f32]; 2] = $srcs;
                if out_len >= $n {
                    let mut t = 0;
                    while t + $n <= out_len {
                        $at::<S, $m>(simd, srcs, w, bias, s.relu, s.dilation, t)
                            .store_slice(&mut y[t..][..$n]);
                        t += $n;
                    }
                    if t < out_len {
                        // The last vector overlaps its predecessor.
                        $at::<S, $m>(simd, srcs, w, bias, s.relu, s.dilation, out_len - $n)
                            .store_slice(&mut y[out_len - $n..]);
                    }
                } else {
                    let mut tile = [0.0f32; $n];
                    $at::<S, $m>(simd, srcs, w, bias, s.relu, s.dilation, 0).store_slice(&mut tile);
                    y.copy_from_slice(&tile[..out_len]);
                }
            }};
        }
        // Stride 2 reads input `2t + kk`, which is entry `t + kk / 2` of phase `kk % 2`.
        match (stride, out_len >= 16) {
            (1, true) => sweep!(depthwise_at16, 16, 1, [x, x]),
            (1, false) => sweep!(depthwise_at8, 8, 1, [x, x]),
            (_, true) => sweep!(depthwise_at16, 16, 2, phase),
            (_, false) => sweep!(depthwise_at8, 8, 2, phase),
        }
    }
}

/// Outputs `t..t + $n` of one depthwise channel as a vector of type `$v`. Tap
/// `kk` reads entry `t + kk / M * dilation` of `srcs[kk % M]`: `M` is 1, or 2
/// for stride 2, which has no dilation.
macro_rules! depthwise_at {
    ($name:ident, $v:ident, $n:literal) => {
        #[inline(always)]
        fn $name<S: Simd, const M: usize>(
            simd: S,
            srcs: [&[f32]; 2],
            w: &[f32],
            bias: f32,
            relu: bool,
            dilation: usize,
            t: usize,
        ) -> $v<S> {
            use fearless_simd::prelude::*;
            // Two chains of multiply-adds, so a short kernel is not bound by
            // the latency of one.
            let (mut acc, mut acc2) = ($v::splat(simd, bias), $v::splat(simd, 0.0));
            let (pairs, odd) = w.as_chunks::<2>();
            for (i, &[w0, w1]) in pairs.iter().enumerate() {
                let (k0, k1) = (2 * i, 2 * i + 1);
                let x0 = $v::from_slice(simd, &srcs[k0 % M][t + k0 / M * dilation..][..$n]);
                let x1 = $v::from_slice(simd, &srcs[k1 % M][t + k1 / M * dilation..][..$n]);
                acc = $v::splat(simd, w0).mul_add(x0, acc);
                acc2 = $v::splat(simd, w1).mul_add(x1, acc2);
            }
            if let &[w0] = odd {
                let k0 = 2 * pairs.len();
                let x0 = $v::from_slice(simd, &srcs[k0 % M][t + k0 / M * dilation..][..$n]);
                acc = $v::splat(simd, w0).mul_add(x0, acc);
            }
            acc = acc + acc2;
            if relu { acc.max($v::splat(simd, 0.0)) } else { acc }
        }
    };
}

depthwise_at!(depthwise_at16, f32x16, 16);
depthwise_at!(depthwise_at8, f32x8, 8);

/// Shape of a convolution with kernel 3, stride 1, padding 1 and one group
/// (`weights` is `[out_channels, in_channels, 3]`); the output is as long as the input.
pub(crate) struct Kernel3<'a> {
    pub batch: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub len: usize,
    pub bias: Option<&'a [f32]>,
    pub relu: bool,
}

pub(crate) fn kernel3(
    level: Level,
    shape: &Kernel3,
    input: &[f32],
    weights: &[f32],
    out: &mut [f32],
) {
    // Rows with a zero on each side, so no tap needs a bounds test.
    // A row too short for a vector is padded out to one, and its outputs are
    // computed into a vector-wide tile per channel and copied.
    let row = shape.len.max(8) + 2;
    let tile = if shape.len < 8 { shape.out_channels * 8 } else { 0 };
    // The shortest rows lay out their taps instead.
    let size = (shape.in_channels * row + tile).max(shape.len * shape.in_channels * 3);
    PADDED.with_borrow_mut(|padded| {
        padded.clear();
        padded.resize(size, 0.0);
        if shape.len < DOT_MAX_LEN {
            simd_call!(level, kernel3_cols_simd(shape, input, weights, out, padded))
        } else {
            simd_call!(level, kernel3_simd(shape, input, weights, out, padded))
        }
    })
}

/// Rows shorter than this are computed as dot products.
const DOT_MAX_LEN: usize = 4;

thread_local! {
    static PADDED: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Computes `OC` output channels from `oc` over positions `t..t + $n`, with
/// vectors of type `$v`.
macro_rules! kernel3_tile {
    ($name:ident, $v:ident, $n:literal) => {
        #[inline(always)]
        fn $name<S: Simd, const OC: usize>(
            simd: S,
            s: &Kernel3,
            padded: &[f32],
            weights: &[f32],
            out: &mut [f32],
            oc: usize,
            t: usize,
        ) {
            use fearless_simd::prelude::*;
            let ic = s.in_channels;
            let w: [&[[f32; 3]]; OC] = core::array::from_fn(|j| {
                weights[(oc + j) * ic * 3..][..ic * 3].as_chunks::<3>().0
            });
            let mut acc: [$v<S>; OC] = core::array::from_fn(|j| {
                $v::splat(simd, s.bias.map_or(0.0, |b| b[oc + j]))
            });
            for (c, row) in padded.chunks_exact(s.len.max(8) + 2).take(ic).enumerate() {
                // Taps 0, 1 and 2 of outputs `t..` read the input at `t - 1 + k`,
                // which sits at `t + k` in the padded row.
                let x0 = $v::from_slice(simd, &row[t..][..$n]);
                let x1 = $v::from_slice(simd, &row[t + 1..][..$n]);
                let x2 = $v::from_slice(simd, &row[t + 2..][..$n]);
                for j in 0..OC {
                    let [w0, w1, w2] = w[j][c];
                    acc[j] = $v::splat(simd, w0).mul_add(x0, acc[j]);
                    acc[j] = $v::splat(simd, w1).mul_add(x1, acc[j]);
                    acc[j] = $v::splat(simd, w2).mul_add(x2, acc[j]);
                }
            }
            let zero = $v::splat(simd, 0.0);
            for j in 0..OC {
                let y = if s.relu { acc[j].max(zero) } else { acc[j] };
                y.store_slice(&mut out[(oc + j) * s.len.max(8) + t..][..$n]);
            }
        }
    };
}

/// Runs `$tile` for `$rows` output channels from `$oc` over every position;
/// the last vector overlaps its predecessor.
macro_rules! kernel3_positions {
    ($tile:ident, $n:literal, $rows:literal, $simd:expr, $s:expr, $padded:expr, $weights:expr, $out:expr, $oc:expr) => {{
        let len = $s.len;
        let mut t = 0;
        while t + $n <= len {
            $tile::<S, $rows>($simd, $s, $padded, $weights, $out, $oc, t);
            t += $n;
        }
        if t < len {
            $tile::<S, $rows>($simd, $s, $padded, $weights, $out, $oc, len - $n);
        }
    }};
}

kernel3_tile!(kernel3_tile16, f32x16, 16);
kernel3_tile!(kernel3_tile8, f32x8, 8);

/// `kernel3` for rows shorter than `DOT_MAX_LEN`.
#[simd]
fn kernel3_cols_simd<S: Simd>(
    simd: S,
    s: &Kernel3,
    input: &[f32],
    weights: &[f32],
    out: &mut [f32],
    padded: &mut [f32],
) {
    let len = s.len;
    {
        // A vector of positions would be mostly empty. Lay out the taps of
        // each position as one row, and take dot products with the weights.
        let (ic, ic3) = (s.in_channels, s.in_channels * 3);
        let cols = &mut padded[..len * ic3];
        for b in 0..s.batch {
            let x = &input[b * ic * len..][..ic * len];
            let out = &mut out[b * s.out_channels * len..][..s.out_channels * len];
            for (t, col) in cols.chunks_exact_mut(ic3).enumerate() {
                for (c, taps) in col.chunks_exact_mut(3).enumerate() {
                    for (k, tap) in taps.iter_mut().enumerate() {
                        // Input `t + k - 1`, zero outside the row.
                        *tap = (t + k).checked_sub(1).filter(|&i| i < len).map_or(0.0, |i| x[c * len + i]);
                    }
                }
            }
            let cols = &*cols;
            let finish = |sum: f32, bias: f32| if s.relu { (sum + bias).max(0.0) } else { sum + bias };
            let mut oc = 0;
            while oc + 4 <= s.out_channels {
                let ws: [&[f32]; 4] = core::array::from_fn(|j| &weights[(oc + j) * ic3..][..ic3]);
                for t in 0..len {
                    let [sums] = dots(simd, [&cols[t * ic3..][..ic3]], ws);
                    for j in 0..4 {
                        let bias = s.bias.map_or(0.0, |b| b[oc + j]);
                        out[(oc + j) * len + t] = finish(sums[j], bias);
                    }
                }
                oc += 4;
            }
            while oc < s.out_channels {
                let ws = [&weights[oc * ic3..][..ic3]];
                let bias = s.bias.map_or(0.0, |b| b[oc]);
                for t in 0..len {
                    let [[sum]] = dots(simd, [&cols[t * ic3..][..ic3]], ws);
                    out[oc * len + t] = finish(sum, bias);
                }
                oc += 1;
            }
        }
    }
}

#[simd]
fn kernel3_simd<S: Simd>(
    simd: S,
    s: &Kernel3,
    input: &[f32],
    weights: &[f32],
    out: &mut [f32],
    padded: &mut [f32],
) {
    let len = s.len;
    let (padded, tile) = padded.split_at_mut(s.in_channels * (len.max(8) + 2));
    for b in 0..s.batch {
        let x = &input[b * s.in_channels * len..][..s.in_channels * len];
        let out = &mut out[b * s.out_channels * len..][..s.out_channels * len];
        for (row, src) in padded.chunks_exact_mut(len.max(8) + 2).zip(x.chunks_exact(len)) {
            row[1..=len].copy_from_slice(src);
        }
        let padded = &*padded;
        // Four output channels at a time share the three input vectors.
        // Macros rather than closures, which would not be compiled with the target features.
        macro_rules! sweep {
            ($tile:ident, $n:literal) => {{
                let mut oc = 0;
                while oc + 4 <= s.out_channels {
                    kernel3_positions!($tile, $n, 4, simd, s, padded, weights, out, oc);
                    oc += 4;
                }
                while oc < s.out_channels {
                    kernel3_positions!($tile, $n, 1, simd, s, padded, weights, out, oc);
                    oc += 1;
                }
            }};
        }
        if len >= 16 {
            sweep!(kernel3_tile16, 16)
        } else if len >= 8 {
            sweep!(kernel3_tile8, 8)
        } else {
            // Too short for a vector: one vector-wide tile per channel.
            let mut oc = 0;
            while oc + 4 <= s.out_channels {
                kernel3_tile8::<S, 4>(simd, s, padded, weights, tile, oc, 0);
                oc += 4;
            }
            while oc < s.out_channels {
                kernel3_tile8::<S, 1>(simd, s, padded, weights, tile, oc, 0);
                oc += 1;
            }
            for (y, t) in out.chunks_exact_mut(len).zip(tile.chunks_exact(8)) {
                y.copy_from_slice(&t[..len]);
            }
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
                let dst = &mut out[(first + l) * out_len + t..][..n];
                if let Some(dst) = dst.first_chunk_mut::<4>() {
                    *dst = row;
                } else {
                    // A copy of unknown length would be a call to `memcpy`.
                    for (d, &v) in dst.iter_mut().zip(&row) {
                        *d = v;
                    }
                }
            }
        }
        t += n;
    }
}

/// Dot products of `M` input windows with `N` weight rows: `sums[m][n]`. The
/// windows share each weight vector and the rows share each input vector. The
/// sums go 8 lanes at a time, then one by one.
#[inline(always)]
fn dots<S: Simd, const N: usize, const M: usize>(
    simd: S,
    xs: [&[f32]; M],
    ws: [&[f32]; N],
) -> [[f32; N]; M] {
    use fearless_simd::prelude::*;
    let k = xs[0].len();
    // One check each here instead of one per access in the loop.
    let x8 = xs.map(|x| {
        assert_eq!(x.len(), k);
        x.as_chunks::<8>().0
    });
    let w8 = ws.map(|w| {
        let chunks = w.as_chunks::<8>().0;
        assert_eq!(chunks.len(), x8[0].len());
        chunks
    });
    let mut acc = [[f32x8::splat(simd, 0.0); N]; M];
    for i in 0..x8[0].len() {
        let xv = x8.map(|x| f32x8::load_array_ref(simd, &x[i]));
        for j in 0..N {
            let wv = f32x8::load_array_ref(simd, &w8[j][i]);
            for m in 0..M {
                acc[m][j] = wv.mul_add(xv[m], acc[m][j]);
            }
        }
    }
    let mut sums = acc.map(|a| a.map(|a| a.reduce_sum()));
    for r in x8[0].len() * 8..k {
        for m in 0..M {
            for j in 0..N {
                sums[m][j] += xs[m][r] * ws[j][r];
            }
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
            // Two output positions at a time share each weight vector.
            let mut t = 0;
            while t + 2 <= out_len {
                let xs = [&x[t * s.stride..][..k], &x[(t + 1) * s.stride..][..k]];
                let sums = dots(simd, xs, ws);
                for (m, sums) in sums.iter().enumerate() {
                    r0[t + m] = finish(sums[0], bias[0]);
                    r1[t + m] = finish(sums[1], bias[1]);
                    r2[t + m] = finish(sums[2], bias[2]);
                    r3[t + m] = finish(sums[3], bias[3]);
                }
                t += 2;
            }
            if t < out_len {
                let [sums] = dots(simd, [&x[t * s.stride..][..k]], ws);
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
                let [[sum]] = dots(simd, [&x[t * s.stride..][..k]], ws);
                *y = finish(sum, bias);
            }
        }
    }
}
