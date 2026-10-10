//! ONNX `ConvInteger`.
//!
//! When the depth of a group (`c * kh * kw`) is at most [`EXACT_F32_DEPTH`], every partial
//! sum of centered codes fits in an `f32` mantissa, so the f32 convolution of the centered
//! input and weights is exact and runs as it is; it has the paths for the shallow cases
//! (a first layer over 3 channels, depthwise convolutions) that a GEMM handles badly.
//!
//! Deeper groups run on the int8 GEMM of [`crate::kernels::qgemm`]. Each output position's
//! receptive field becomes a row of A, so a group's convolution is
//! `patches [positions, kh * kw * c] x weights [kh * kw * c, out channels]`. The input is
//! made channels-last first and the weights are packed with their depth ordered
//! `(kh, kw, c)` to match, so a patch row is a contiguous copy of `c` channels per tap (of
//! `kw * c` per kernel row, when the taps of a row are adjacent). A pointwise convolution
//! with no padding or stride uses the channels-last input as A as it is.
//!
//! Padding takes the input's zero point, which contributes nothing to the sums, as
//! onnxruntime pads.

use crate::kernels::qgemm::{Epilogue, Layout, QWeights, run_at};
use crate::kernels::utils;
use crate::tensor::TensorView;
use fearless_simd::Level;

/// The deepest group whose sums are exact in `f32`: `258 * 255 * 255 < 2^24`.
const EXACT_F32_DEPTH: usize = 258;

/// The weights of a `ConvInteger`, `[out channels, channels / groups, kh, kw]`, prepared
/// for whichever path its depth takes.
pub struct ConvWeights {
    out_channels: usize,
    /// Input channels per group.
    channels: usize,
    kernel: [usize; 2],
    groups: usize,
    kind: Kind,
}

enum Kind {
    /// The weights less their zero points, in their own layout, for the f32 convolution.
    Float(Vec<f32>),
    /// One block per group, `kh * kw * channels x out_channels / groups`.
    Int(Vec<QWeights>),
}

impl ConvWeights {
    /// Packs `codes`, `i8` if `signed` and `u8` otherwise, of the given `[oc, c, kh, kw]`
    /// shape, with one zero point or one per output channel.
    pub fn new(codes: &[u8], shape: &[usize], groups: usize, signed: bool, zero_point: &[i32]) -> Self {
        let code = |i: usize| if signed { codes[i] as i8 as f32 } else { codes[i] as f32 };
        Self::pack(Layout::for_level(Level::new()), EXACT_F32_DEPTH, shape, groups, zero_point, code)
    }

    /// As [`ConvWeights::new`], from codes held in `f32`.
    pub fn from_f32_codes(codes: &[f32], shape: &[usize], groups: usize, zero_point: &[i32]) -> Self {
        Self::pack(Layout::for_level(Level::new()), EXACT_F32_DEPTH, shape, groups, zero_point, |i| codes[i])
    }

    /// Groups no deeper than `float_depth` take the f32 path.
    fn pack(
        layout: Layout,
        float_depth: usize,
        shape: &[usize],
        groups: usize,
        zero_point: &[i32],
        code: impl Fn(usize) -> f32,
    ) -> Self {
        let &[out_channels, channels, kh, kw] = shape else {
            panic!("ConvInteger weights of shape {shape:?}");
        };
        assert!(groups > 0 && out_channels % groups == 0, "{out_channels} output channels in {groups} groups");
        assert!(zero_point.len() == 1 || zero_point.len() == out_channels);
        let per_group = out_channels / groups;
        let depth = kh * kw * channels;
        let zp = |o: usize| zero_point[if zero_point.len() == 1 { 0 } else { o }] as f32;
        let kind = if depth <= float_depth {
            Kind::Float((0..out_channels * depth).map(|i| code(i) - zp(i / depth)).collect())
        } else {
            let mut t = vec![0.0; depth * per_group];
            Kind::Int(
                (0..groups)
                    .map(|g| {
                        for j in 0..per_group {
                            let o = g * per_group + j;
                            for c in 0..channels {
                                for y in 0..kh {
                                    for x in 0..kw {
                                        t[((y * kw + x) * channels + c) * per_group + j] =
                                            code(((o * channels + c) * kh + y) * kw + x);
                                    }
                                }
                            }
                        }
                        let zp = if zero_point.len() == 1 { zero_point } else { &zero_point[g * per_group..][..per_group] };
                        QWeights::from_f32_codes_with_layout(layout, &t, depth, per_group, zp, &[1.0])
                    })
                    .collect(),
            )
        };
        Self { out_channels, channels, kernel: [kh, kw], groups, kind }
    }
}

/// ONNX `ConvInteger` with weights given at run time: prepares them, then runs
/// [`conv_integer_packed`]. Inputs and outputs are codes and sums held in `f32`.
pub fn conv_integer<'a>(
    input: &TensorView<'_>,
    weights: &TensorView<'_>,
    x_zero_point: Option<&TensorView<'_>>,
    w_zero_point: Option<&TensorView<'_>>,
    dilations: &[i64],
    group: i64,
    pads: &[i64],
    strides: &[i64],
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let x_zp = x_zero_point.and_then(|z| z.data.first().copied()).unwrap_or(0.0) as i32;
    let w_zp: Vec<i32> = match w_zero_point {
        Some(z) if !z.data.is_empty() => z.data.iter().map(|&v| v as i32).collect(),
        _ => vec![0],
    };
    let w = ConvWeights::from_f32_codes(&weights.data, &weights.shape, group as usize, &w_zp);
    conv_integer_packed(input, &w, x_zp, dilations, pads, strides, out)
}

/// ONNX `ConvInteger` against prepared weights: `input` is `[n, c, h, w]` codes (`u8`, or
/// `i8` if any is negative) with zero point `x_zero_point`; the output is
/// `[n, oc, out_h, out_w]` exact integer sums.
pub fn conv_integer_packed<'a>(
    input: &TensorView<'_>,
    w: &ConvWeights,
    x_zero_point: i32,
    dilations: &[i64],
    pads: &[i64],
    strides: &[i64],
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    conv_integer_at(Level::new(), input, w, x_zero_point, dilations, pads, strides, out)
}

/// The two spatial values of an attribute given as one for both, two, or none.
fn pair(v: &[i64], default: usize) -> [usize; 2] {
    match v {
        [] => [default; 2],
        [a] => [*a as usize; 2],
        [a, b, ..] => [*a as usize, *b as usize],
    }
}

/// `dst[j * rows + i] = src[i * cols + j]` for a row-major `rows x cols` `src`, a block at
/// a time so both sides stay in cache.
fn transpose(src: &[f32], rows: usize, cols: usize, dst: &mut [f32]) {
    const B: usize = 16;
    let (src, dst) = (&src[..rows * cols], &mut dst[..rows * cols]);
    for i0 in (0..rows).step_by(B) {
        for j0 in (0..cols).step_by(B) {
            for i in i0..(i0 + B).min(rows) {
                for j in j0..(j0 + B).min(cols) {
                    dst[j * rows + i] = src[i * cols + j];
                }
            }
        }
    }
}

thread_local! {
    static CENTERED: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    static CHANNELS_LAST: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    static PATCHES: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    static SUMS: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Elements of patches gathered at a time: as many rows as fit, at least enough for
/// the GEMM's tiles to stream the weights only a few times.
const PATCH_ELEMS: usize = 1 << 20;

pub(crate) fn conv_integer_at<'a>(
    level: Level,
    input: &TensorView<'_>,
    w: &ConvWeights,
    x_zero_point: i32,
    dilations: &[i64],
    pads: &[i64],
    strides: &[i64],
    out: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let &[batch, in_channels, in_h, in_w] = &input.shape[..] else {
        panic!("ConvInteger input of shape {:?}", input.shape);
    };
    let groups = w.groups;
    assert_eq!(in_channels, groups * w.channels, "ConvInteger of {:?} in {groups} groups", input.shape);
    let [kh, kw] = w.kernel;
    let pad = x_zero_point as f32;

    let packed = match &w.kind {
        Kind::Int(packed) => packed,
        Kind::Float(centered) => {
            return CENTERED.with_borrow_mut(|x| {
                utils::ensure_capacity(x, input.data.len());
                for (x, &v) in x.iter_mut().zip(input.data.iter()) {
                    *x = v - pad;
                }
                let x = TensorView::from_slice(&x[..input.data.len()], input.shape.to_vec());
                let wv = TensorView::from_slice(centered, vec![w.out_channels, w.channels, kh, kw]);
                let shape = {
                    let r = crate::kernels::conv2d(&x, &wv, None, dilations, groups as i64, pads, strides, out);
                    r.shape.to_vec()
                };
                TensorView::from_slice(out, shape)
            });
        }
    };

    let [dh, dw] = pair(dilations, 1);
    let [sh, sw] = pair(strides, 1);
    let [pt, pl, pb, pr] = match pads {
        [t, l, b, r, ..] => [*t, *l, *b, *r].map(|p| p as usize),
        [t, l] => [*t, *l, *t, *l].map(|p| p as usize),
        _ => [0; 4],
    };
    let span = |size: usize, pad: usize, d: usize, k: usize, s: usize| {
        let extent = d * (k - 1) + 1;
        assert!(size + pad >= extent, "ConvInteger kernel larger than its padded input");
        (size + pad - extent) / s + 1
    };
    let out_h = span(in_h, pt + pb, dh, kh, sh);
    let out_w = span(in_w, pl + pr, dw, kw, sw);
    let positions = out_h * out_w;
    let (oc, per_group, cg) = (w.out_channels, w.out_channels / groups, w.channels);
    let depth = kh * kw * cg;

    utils::ensure_capacity(out, batch * oc * positions);
    // `u8` codes go into the GEMM as they are; `i8` ones shifted into `u8` first.
    let signed = x_zero_point < 0 || input.data.iter().any(|&x| x < 0.0);
    let (shift, center) = if signed { (128, x_zero_point + 128) } else { (0, x_zero_point) };
    let e = Epilogue { scale: 1.0, bias: None, relu: false };
    let pointwise = kh == 1 && kw == 1 && sh == 1 && sw == 1 && pt + pl + pb + pr == 0;
    // A kernel row's taps are one run of `kw * in_channels` values when they are adjacent
    // and take every channel.
    let row_runs = dw == 1 && groups == 1;
    let spatial = in_h * in_w;
    let rows = (PATCH_ELEMS / depth).max(96).min(positions).max(1);

    CHANNELS_LAST.with_borrow_mut(|nhwc| {
        PATCHES.with_borrow_mut(|patches| {
            SUMS.with_borrow_mut(|sums| {
                utils::ensure_capacity(nhwc, spatial * in_channels);
                utils::ensure_capacity(sums, positions * per_group);
                for b in 0..batch {
                    let x = &input.data[b * in_channels * spatial..][..in_channels * spatial];
                    transpose(x, in_channels, spatial, nhwc);
                    for (g, wg) in packed.iter().enumerate() {
                        for r0 in (0..positions).step_by(rows) {
                            let m = rows.min(positions - r0);
                            let a: &[f32] = if pointwise && groups == 1 {
                                &nhwc[r0 * in_channels..][..m * in_channels]
                            } else {
                                utils::ensure_capacity(patches, m * depth);
                                for (i, row) in patches[..m * depth].chunks_exact_mut(depth).enumerate() {
                                    let (oh, ow) = ((r0 + i) / out_w, (r0 + i) % out_w);
                                    let iw0 = (ow * sw).wrapping_sub(pl);
                                    let whole_row = row_runs && iw0 < in_w && iw0 + kw <= in_w;
                                    for (y, row) in row.chunks_exact_mut(kw * cg).enumerate() {
                                        let ih = (oh * sh + y * dh).wrapping_sub(pt);
                                        if ih >= in_h {
                                            row.fill(pad);
                                        } else if whole_row {
                                            row.copy_from_slice(&nhwc[(ih * in_w + iw0) * in_channels..][..kw * cg]);
                                        } else {
                                            for (xk, dst) in row.chunks_exact_mut(cg).enumerate() {
                                                let iw = (ow * sw + xk * dw).wrapping_sub(pl);
                                                if iw < in_w {
                                                    dst.copy_from_slice(&nhwc[(ih * in_w + iw) * in_channels + g * cg..][..cg]);
                                                } else {
                                                    dst.fill(pad);
                                                }
                                            }
                                        }
                                    }
                                }
                                &patches[..m * depth]
                            };
                            run_at(level, wg, a, m, 1.0, shift, center, &e, &mut sums[r0 * per_group..][..m * per_group]);
                        }
                        let o = &mut out[(b * oc + g * per_group) * positions..][..per_group * positions];
                        transpose(sums, positions, per_group, o);
                    }
                }
            })
        })
    });
    TensorView::from_slice(out, vec![batch, oc, out_h, out_w])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::test_util::{Rng, levels};

    /// ConvInteger straight from its definition, padding with the input's zero point.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        x: &[f32],
        [n, c, h, wd]: [usize; 4],
        x_zp: i32,
        wt: &[f32],
        [oc, cg, kh, kw]: [usize; 4],
        w_zp: &[i32],
        groups: usize,
        [dh, dw]: [usize; 2],
        [pt, pl, pb, pr]: [usize; 4],
        [sh, sw]: [usize; 2],
    ) -> (Vec<f32>, [usize; 2]) {
        let oh = (h + pt + pb - dh * (kh - 1) - 1) / sh + 1;
        let ow = (wd + pl + pr - dw * (kw - 1) - 1) / sw + 1;
        let per_group = oc / groups;
        let mut out = vec![0.0; n * oc * oh * ow];
        for b in 0..n {
            for o in 0..oc {
                let g = o / per_group;
                let zw = w_zp[if w_zp.len() == 1 { 0 } else { o }];
                for y in 0..oh {
                    for z in 0..ow {
                        let mut acc = 0i64;
                        for ci in 0..cg {
                            let c_in = g * cg + ci;
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let iy = (y * sh + ky * dh) as isize - pt as isize;
                                    let ix = (z * sw + kx * dw) as isize - pl as isize;
                                    let v = if iy >= 0 && ix >= 0 && (iy as usize) < h && (ix as usize) < wd {
                                        x[((b * c + c_in) * h + iy as usize) * wd + ix as usize] as i64
                                    } else {
                                        x_zp as i64
                                    };
                                    let wv = wt[((o * cg + ci) * kh + ky) * kw + kx] as i64;
                                    acc += (v - x_zp as i64) * (wv - zw as i64);
                                }
                            }
                        }
                        out[((b * oc + o) * oh + y) * ow + z] = acc as f32;
                    }
                }
            }
        }
        (out, [oh, ow])
    }

    #[test]
    fn test_conv_integer_matches_its_definition_at_every_level() {
        let mut rng = Rng::new(11);
        // (n, c, h, w, oc, kh, kw, groups, dilations, pads, strides)
        type Case = (usize, usize, usize, usize, usize, usize, usize, usize, [usize; 2], [usize; 4], [usize; 2]);
        let cases: &[Case] = &[
            (1, 3, 9, 11, 8, 3, 3, 1, [1, 1], [1, 1, 1, 1], [1, 1]),
            (2, 16, 6, 6, 24, 1, 1, 1, [1, 1], [0; 4], [1, 1]),
            (1, 8, 10, 7, 5, 3, 3, 1, [1, 1], [0, 1, 2, 1], [2, 2]),
            (1, 6, 12, 12, 4, 3, 2, 2, [2, 1], [2, 0, 1, 1], [1, 2]),
            (1, 8, 7, 9, 8, 3, 3, 8, [1, 1], [1, 1, 1, 1], [1, 1]),
            (1, 40, 5, 5, 70, 3, 3, 1, [1, 1], [1, 1, 1, 1], [1, 1]),
            (1, 4, 3, 3, 3, 3, 3, 1, [1, 1], [0; 4], [1, 1]),
            (1, 4, 9, 8, 6, 3, 3, 1, [1, 2], [1, 2, 1, 0], [1, 1]),
        ];
        // The GEMM with both layouts at every level (the quads emulate VNNI off the AVX-512
        // level), then the f32 path wherever its sums are exact.
        let runs = levels()
            .into_iter()
            .flat_map(|l| [(l, Layout::Pairs, 0), (l, Layout::Quads, 0)])
            .chain([(Level::new(), Layout::for_level(Level::new()), EXACT_F32_DEPTH)]);
        for (level, layout, float_depth) in runs {
            for (i, &(n, c, h, wd, oc, kh, kw, groups, d, p, s)) in cases.iter().enumerate() {
                for case in 0..4 {
                    let x_signed = case % 2 == 1;
                    let w_signed = case >= 2;
                    let cg = c / groups;
                    let code = |rng: &mut Rng, signed: bool| {
                        if signed { rng.f32(-128.0, 128.0).floor() } else { rng.f32(0.0, 256.0).floor() }
                    };
                    let x: Vec<f32> = (0..n * c * h * wd).map(|_| code(&mut rng, x_signed)).collect();
                    let wt: Vec<f32> = (0..oc * cg * kh * kw).map(|_| code(&mut rng, w_signed)).collect();
                    let x_zp = code(&mut rng, x_signed) as i32;
                    let w_zp: Vec<i32> = (0..if case == 2 { oc } else { 1 })
                        .map(|_| if case == 0 { 0 } else { code(&mut rng, w_signed) as i32 })
                        .collect();
                    let (expected, [oh, ow]) =
                        reference(&x, [n, c, h, wd], x_zp, &wt, [oc, cg, kh, kw], &w_zp, groups, d, p, s);
                    let w = ConvWeights::pack(layout, float_depth, &[oc, cg, kh, kw], groups, &w_zp, |i| wt[i]);
                    let input = TensorView::from_slice(&x, vec![n, c, h, wd]);
                    let (d, p, s) = (d.map(|v| v as i64), p.map(|v| v as i64), s.map(|v| v as i64));
                    let mut out = Vec::new();
                    let got = conv_integer_at(level, &input, &w, x_zp, &d, &p, &s, &mut out);
                    let what = format!("{level:?} {layout:?} float depth {float_depth}, case {i}.{case}");
                    assert_eq!(&got.shape[..], &[n, oc, oh, ow], "{what}");
                    assert_eq!(&got.data[..], &expected[..], "{what}");
                }
            }
        }
    }
}
