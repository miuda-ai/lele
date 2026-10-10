//! `f32` add, sub, mul and div with numpy broadcasting, for the shapes models
//! use: equal shapes, a scalar, and one operand whose values each repeat
//! over a block of the other (a per-channel scale, a bias along the last
//! dimension, an attention mask). Other broadcasts are left to the generic
//! code in `math.rs`.

use crate::kernels::simd::simd_call;
use crate::kernels::utils;
use crate::tensor::TensorView;
use fearless_simd::{Level, Simd};
use fearless_simd_macros::simd;
use std::any::TypeId;

#[derive(Clone, Copy, Debug)]
pub(crate) enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

/// Computes `a op b` into `out` and returns the output shape, when `T` is
/// `f32` and the broadcast is one of the supported shapes; otherwise returns
/// `None` and leaves `out` alone.
pub(crate) fn try_f32<T: Copy + std::fmt::Debug + 'static>(
    op: Op,
    a: &TensorView<'_, T>,
    b: &TensorView<'_, T>,
    out: &mut Vec<T>,
) -> Option<Vec<usize>> {
    if TypeId::of::<T>() != TypeId::of::<f32>() {
        return None;
    }
    // SAFETY: `T` is `f32`, checked above.
    let (a_data, b_data, out) = unsafe {
        (
            std::slice::from_raw_parts(a.data.as_ptr() as *const f32, a.data.len()),
            std::slice::from_raw_parts(b.data.as_ptr() as *const f32, b.data.len()),
            &mut *(out as *mut Vec<T> as *mut Vec<f32>),
        )
    };
    binary_f32(Level::new(), op, a_data, &a.shape, b_data, &b.shape, out)
}

fn binary_f32(
    level: Level,
    op: Op,
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
    out: &mut Vec<f32>,
) -> Option<Vec<usize>> {
    let shape = if a_shape == b_shape {
        a_shape.to_vec()
    } else {
        utils::broadcast_shapes(a_shape, b_shape)?
    };
    let numel: usize = shape.iter().product();
    if numel == 0 {
        return None;
    }
    if a.len() == numel && b.len() == numel {
        utils::ensure_capacity(out, numel);
        tiled(level, op, false, a, b, out);
        return Some(shape);
    }
    // One operand has the output's shape; the other's values repeat over it.
    let (full, small, small_shape, small_first) = if a.len() == numel {
        (a, b, b_shape, false)
    } else if b.len() == numel {
        (b, a, a_shape, true)
    } else {
        return None;
    };
    let (outer, mid, inner) = blocks(small_shape, &shape)?;
    debug_assert_eq!(small.len(), mid);
    utils::ensure_capacity(out, numel);
    debug_assert_eq!(outer * mid * inner, numel);
    if inner == 1 {
        // The small operand is a tile that repeats `outer` times.
        tiled(level, op, small_first, full, small, out);
    } else {
        // Each value of the small operand covers `inner` elements.
        rows(level, op, small_first, full, small, inner, out);
    }
    Some(shape)
}

/// Splits the output into `outer × mid × inner`, if `small` (right-aligned to
/// `out`) takes each of its `mid` values over a block of `inner` elements and
/// repeats the whole run `outer` times; `None` if its dimensions are not one
/// run of the output's.
fn blocks(small: &[usize], out: &[usize]) -> Option<(usize, usize, usize)> {
    let rank = out.len();
    let skipped = rank - small.len();
    let dim = |d: usize| if d < skipped { 1 } else { small[d - skipped] };
    let Some(first) = (0..rank).find(|&d| dim(d) != 1) else {
        return Some((1, 1, out.iter().product())); // a single value
    };
    let last = (0..rank).rfind(|&d| dim(d) != 1)?;
    if !(first..=last).all(|d| dim(d) == out[d]) {
        return None;
    }
    Some((
        out[..first].iter().product(),
        out[first..=last].iter().product(),
        out[last + 1..].iter().product(),
    ))
}

#[inline(always)]
fn scalar(op: Op, x: f32, y: f32) -> f32 {
    match op {
        Op::Add => x + y,
        Op::Sub => x - y,
        Op::Mul => x * y,
        Op::Div => x / y,
    }
}

/// `out = full op small` (`small op full` if `small_first`), where `small`
/// repeats along `full`, whose length is a multiple of it. Tiles shorter than
/// a vector run as scalar loops: these four are exactly rounded, so that
/// gives what the vector kernels do.
///
/// The loop over tiles is inside the kernel: a call costs about as much as a
/// short tile, and a per-channel scale makes hundreds of them.
fn tiled(level: Level, op: Op, small_first: bool, full: &[f32], small: &[f32], out: &mut [f32]) {
    let tile = small.len();
    if tile < 16 {
        for (f, o) in full.chunks_exact(tile).zip(out.chunks_exact_mut(tile)) {
            for ((o, &x), &s) in o.iter_mut().zip(f).zip(small) {
                *o = if small_first { scalar(op, s, x) } else { scalar(op, x, s) };
            }
        }
        return;
    }
    match op {
        Op::Add => simd_call!(level, max = Avx2, add_tiled(small_first, full, small, out)),
        Op::Sub => simd_call!(level, max = Avx2, sub_tiled(small_first, full, small, out)),
        Op::Mul => simd_call!(level, max = Avx2, mul_tiled(small_first, full, small, out)),
        Op::Div => simd_call!(level, max = Avx2, div_tiled(small_first, full, small, out)),
    }
}

/// `out = full op small` with `small[i]` applied to the `i`-th run of `inner`
/// elements of `full` (`small` repeats from the start when it runs out), or
/// `small op full` if `small_first`.
fn rows(level: Level, op: Op, small_first: bool, full: &[f32], small: &[f32], inner: usize, out: &mut [f32]) {
    if inner < 16 {
        for (i, (f, o)) in full.chunks_exact(inner).zip(out.chunks_exact_mut(inner)).enumerate() {
            let s = small[i % small.len()];
            for (o, &x) in o.iter_mut().zip(f) {
                *o = if small_first { scalar(op, s, x) } else { scalar(op, x, s) };
            }
        }
        return;
    }
    match op {
        Op::Add => simd_call!(level, max = Avx2, add_rows(small_first, full, small, inner, out)),
        Op::Sub => simd_call!(level, max = Avx2, sub_rows(small_first, full, small, inner, out)),
        Op::Mul => simd_call!(level, max = Avx2, mul_rows(small_first, full, small, inner, out)),
        Op::Div => simd_call!(level, max = Avx2, div_rows(small_first, full, small, inner, out)),
    }
}

/// `out[i] = body(x = a[i], y = b[i])`, written `|x, y| body`. Needs
/// `a.len() >= 16`; `out` is a different buffer from the inputs, so a length
/// that is not a multiple of 16 ends with the last 16 elements overlapping the
/// previous vector, as in `simd_math::map!`. A macro for the same reason: the
/// body must be expanded in the `#[simd]` body to be compiled with its target
/// features.
macro_rules! zip_map {
    ($simd:expr, $a:expr, $b:expr, $out:expr, |$x:ident, $y:ident| $body:expr) => {{
        use fearless_simd::f32x16;
        use fearless_simd::prelude::*;
        let (simd, a, b, out): (_, &[f32], &[f32], &mut [f32]) = ($simd, $a, $b, $out);
        let (b, out) = (&b[..a.len()], &mut out[..a.len()]);
        let (chunks_a, tail) = a.as_chunks::<16>();
        let (chunks_b, _) = b.as_chunks::<16>();
        let (chunks_out, _) = out.as_chunks_mut::<16>();
        for ((va, vb), vo) in chunks_a.iter().zip(chunks_b).zip(chunks_out) {
            let ($x, $y) = (f32x16::load_array_ref(simd, va), f32x16::load_array_ref(simd, vb));
            ($body).store_array(vo);
        }
        if !tail.is_empty() {
            let $x = f32x16::load_array_ref(simd, a.last_chunk::<16>().unwrap());
            let $y = f32x16::load_array_ref(simd, b.last_chunk::<16>().unwrap());
            ($body).store_array(out.last_chunk_mut::<16>().unwrap());
        }
    }};
}

/// `out[i] = body(x = a[i], y = s)`, written `|x, y| body`, with the same
/// tail as `zip_map!`.
macro_rules! scalar_map {
    ($simd:expr, $a:expr, $s:expr, $out:expr, |$x:ident, $y:ident| $body:expr) => {{
        use fearless_simd::f32x16;
        use fearless_simd::prelude::*;
        let (simd, a, s, out): (_, &[f32], f32, &mut [f32]) = ($simd, $a, $s, $out);
        let out = &mut out[..a.len()];
        let $y = f32x16::splat(simd, s);
        let (chunks_a, tail) = a.as_chunks::<16>();
        let (chunks_out, _) = out.as_chunks_mut::<16>();
        for (va, vo) in chunks_a.iter().zip(chunks_out) {
            let $x = f32x16::load_array_ref(simd, va);
            ($body).store_array(vo);
        }
        if !tail.is_empty() {
            let $x = f32x16::load_array_ref(simd, a.last_chunk::<16>().unwrap());
            ($body).store_array(out.last_chunk_mut::<16>().unwrap());
        }
    }};
}

/// The kernels `$tiled` and `$rows` of one operation, `|x, y| body` with `x`
/// the left operand and `y` the right one. `small_first` says which is which;
/// both loops are written out for each order rather than choosing per element.
macro_rules! kernels {
    ($tiled:ident, $rows:ident, |$x:ident, $y:ident| $body:expr) => {
        #[simd]
        fn $tiled<S: Simd>(simd: S, small_first: bool, full: &[f32], small: &[f32], out: &mut [f32]) {
            let tile = small.len();
            if full.len() == tile && !small_first {
                // Equal shapes: no tiles to count (that is two integer divisions).
                zip_map!(simd, full, small, out, |$x, $y| $body);
                return;
            }
            for (f, o) in full.chunks_exact(tile).zip(out.chunks_exact_mut(tile)) {
                if small_first {
                    zip_map!(simd, small, f, o, |$x, $y| $body)
                } else {
                    zip_map!(simd, f, small, o, |$x, $y| $body)
                }
            }
        }

        #[simd]
        fn $rows<S: Simd>(
            simd: S,
            small_first: bool,
            full: &[f32],
            small: &[f32],
            inner: usize,
            out: &mut [f32],
        ) {
            let runs = full.chunks_exact(inner).zip(out.chunks_exact_mut(inner));
            let mut next = 0;
            for (f, o) in runs {
                let s = small[next];
                // Not `i % small.len()`: a division per row is a visible cost.
                next += 1;
                if next == small.len() {
                    next = 0;
                }
                if small_first {
                    scalar_map!(simd, f, s, o, |$y, $x| $body)
                } else {
                    scalar_map!(simd, f, s, o, |$x, $y| $body)
                }
            }
        }
    };
}

kernels!(add_tiled, add_rows, |x, y| x + y);
kernels!(sub_tiled, sub_rows, |x, y| x - y);
kernels!(mul_tiled, mul_rows, |x, y| x * y);
kernels!(div_tiled, div_rows, |x, y| x / y);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::test_util::{
        assert_same_bits, binary_inputs, binary_shapes, broadcast_reference, levels, BINARY_OPS,
    };

    fn op_of(name: &str) -> Op {
        match name {
            "add" => Op::Add,
            "sub" => Op::Sub,
            "mul" => Op::Mul,
            _ => Op::Div,
        }
    }

    #[test]
    fn test_matches_broadcast_reference_at_every_level() {
        for level in levels() {
            for (a_shape, b_shape) in &binary_shapes() {
                let (a, b) = binary_inputs(a_shape, b_shape);
                for (name, op) in BINARY_OPS {
                    let (want_shape, want) = broadcast_reference(&a, a_shape, &b, b_shape, op);
                    let mut out = Vec::new();
                    let what = format!("{level:?} {name} {a_shape:?} {b_shape:?}");
                    // Shapes that are not one block are left to the generic code.
                    if let Some(shape) = binary_f32(level, op_of(name), &a, a_shape, &b, b_shape, &mut out) {
                        assert_eq!(shape, want_shape, "{what}: shape");
                        assert_same_bits(&out, &want, &what);
                    }
                }
            }
        }
    }

    #[test]
    fn test_handles_the_shapes_models_use() {
        // Each of these must take the fast path, not fall through.
        let handled: &[(&[usize], &[usize])] = &[
            (&[1, 3, 100, 100], &[1, 3, 100, 100]),
            (&[1, 3, 100, 100], &[1]),
            (&[], &[4, 5]),
            (&[1, 64, 56, 56], &[64, 1, 1]),
            (&[2, 8, 7, 7], &[2, 8, 1, 1]),
            (&[10, 35], &[10, 1]),
            (&[40, 1920], &[1920]),
            (&[1, 8, 50, 50], &[1, 1, 50, 50]),
            (&[2, 3, 4, 5], &[1, 3, 4, 1]),
        ];
        for (a_shape, b_shape) in handled {
            for (a_shape, b_shape) in [(a_shape, b_shape), (b_shape, a_shape)] {
                let a = vec![1.0; a_shape.iter().product()];
                let b = vec![2.0; b_shape.iter().product()];
                let mut out = Vec::new();
                assert!(
                    binary_f32(Level::new(), Op::Add, &a, a_shape, &b, b_shape, &mut out).is_some(),
                    "{a_shape:?} {b_shape:?} not handled"
                );
            }
        }
        // Both operands broadcast, or a dimension of 1 inside the block.
        for (a_shape, b_shape) in [(&[4, 1, 5][..], &[1, 6, 1][..]), (&[2, 3, 1, 5], &[2, 1, 4, 1])] {
            let a = vec![1.0; a_shape.iter().product()];
            let b = vec![2.0; b_shape.iter().product()];
            let mut out = Vec::new();
            assert!(binary_f32(Level::new(), Op::Add, &a, a_shape, &b, b_shape, &mut out).is_none());
        }
    }

    #[test]
    fn test_other_types_are_not_handled() {
        let (a, b) = (vec![1i64, 2, 3], vec![4i64, 5, 6]);
        let (ta, tb) = (TensorView::from_slice(&a, vec![3]), TensorView::from_slice(&b, vec![3]));
        assert!(try_f32(Op::Add, &ta, &tb, &mut Vec::new()).is_none());
    }
}
