use crate::kernels::simd::simd_call;
use crate::kernels::utils;
use crate::tensor::TensorView;
use fearless_simd::{Level, Simd, f32x4, f32x8, f64x4, prelude::*};
use fearless_simd_macros::simd;

/// Side of the square blocks a matrix transpose works through, in elements:
/// a block of the source and one of the destination, 16 KiB each for f32,
/// stay in L1 together, so each cache line is read or filled completely
/// before it is evicted. 32 and 128 were up to 20% slower on model shapes.
const TRANSPOSE_BLOCK: usize = 64;

/// Most axes `transpose` plans for on the stack. Heap allocations were most
/// of the time of the small transposes models are full of, such as the
/// 2x2 and 3x2 ones in Silero VAD and SenseVoice; higher ranks are moved
/// element by element.
const MAX_RANK: usize = 8;

/// Up to `MAX_RANK` values, on the stack.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Small<T: Copy> {
    len: usize,
    items: [T; MAX_RANK],
}

impl<T: Copy + Default> Small<T> {
    fn new() -> Self {
        Self { len: 0, items: [T::default(); MAX_RANK] }
    }

    fn push(&mut self, value: T) {
        self.items[self.len] = value;
        self.len += 1;
    }
}

impl<T: Copy + Default> FromIterator<T> for Small<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut small = Self::new();
        for value in iter {
            small.push(value);
        }
        small
    }
}

impl<T: Copy> std::ops::Deref for Small<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.items[..self.len]
    }
}

impl<T: Copy> std::ops::DerefMut for Small<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.items[..self.len]
    }
}

impl<T: Copy + std::fmt::Debug> std::fmt::Debug for Small<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self[..].fmt(f)
    }
}

/// The input shape and permutation of a transpose equivalent to `perm` on
/// `shape`, with axes of length 1 dropped and every run of input axes that
/// stays adjacent and in order in the output merged into one axis.
///
/// After this, any permutation that moves no data is the identity of rank
/// 0 or 1, and one that keeps the innermost axis in place copies whole rows.
fn simplify_transpose(shape: &[usize], perm: &[usize]) -> (Small<usize>, Small<usize>) {
    // Runs of input axes, as (first, last, output position), in output order;
    // a run continues when the next axis is the next one of length > 1.
    let mut runs: Small<(usize, usize, usize)> = Small::new();
    for &p in perm.iter().filter(|&&p| shape[p] != 1) {
        match runs.last_mut() {
            Some(run) if run.1 < p && shape[run.1 + 1..p].iter().all(|&len| len == 1) => run.1 = p,
            _ => runs.push((p, p, runs.len)),
        }
    }
    runs.sort_unstable_by_key(|run| run.0);
    let mut new_shape = Small::new();
    let mut new_perm: Small<usize> = runs.iter().map(|_| 0).collect();
    for (axis, &(first, last, at)) in runs.iter().enumerate() {
        new_shape.push(shape[first..=last].iter().product());
        new_perm[at] = axis;
    }
    (new_shape, new_perm)
}

/// Calls `f(src_offset, dst_offset)` for every index of the axes `dims`,
/// each given as (length, source stride, destination stride), the last axis
/// fastest. With no axes, calls it once with zero offsets.
fn for_each_offset(dims: &[(usize, usize, usize)], mut f: impl FnMut(usize, usize)) {
    let Some((&(len, src_stride, dst_stride), dims)) = dims.split_last() else {
        return f(0, 0);
    };
    if len == 0 || dims.iter().any(|&(len, _, _)| len == 0) {
        return;
    }
    let mut index: Small<usize> = dims.iter().map(|_| 0).collect();
    let (mut src, mut dst) = (0, 0);
    loop {
        for i in 0..len {
            f(src + i * src_stride, dst + i * dst_stride);
        }
        let mut d = dims.len();
        loop {
            if d == 0 {
                return;
            }
            d -= 1;
            let (len, src_stride, dst_stride) = dims[d];
            index[d] += 1;
            if index[d] < len {
                src += src_stride;
                dst += dst_stride;
                break;
            }
            src -= (len - 1) * src_stride;
            dst -= (len - 1) * dst_stride;
            index[d] = 0;
        }
    }
}

/// `dst[c * dst_ld + r] = src[r * src_ld + c]` for `r < rows`, `c < cols`.
pub(crate) fn transpose_matrix<T: Copy + 'static>(
    src: &[T],
    src_ld: usize,
    dst: &mut [T],
    dst_ld: usize,
    rows: usize,
    cols: usize,
) {
    // Fewer elements than a tile or two, or too thin a side for even 4x4
    // tiles: not worth detecting and dispatching to a SIMD level.
    let small = rows * cols <= 64 || rows < 4 || cols < 4;
    if size_of::<T>() == 4 && align_of::<T>() == 4 && !small {
        // Only moved, never computed on, so any 4-byte type can go through
        // the vectors bit for bit.
        // SAFETY: `T` and `u32` have the same size and alignment, `T: Copy`
        // has no drop glue, and every bit pattern of `T` is a valid `u32`.
        let (src, dst) = unsafe {
            (
                std::slice::from_raw_parts(src.as_ptr().cast::<u32>(), src.len()),
                std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u32>(), dst.len()),
            )
        };
        transpose_matrix_u32(Level::new(), src, src_ld, dst, dst_ld, rows, cols);
    } else {
        assert_matrix_fits(src.len(), src_ld, dst.len(), dst_ld, rows, cols);
        // SAFETY: checked just above.
        unsafe { transpose_scalar(src.as_ptr(), src_ld, dst.as_mut_ptr(), dst_ld, 0..rows, 0..cols) };
    }
}

/// Copies `count` rows of `row` elements, `src_stride` apart in `src` and
/// `dst_stride` apart in `dst`.
fn copy_rows<T: Copy>(src: &[T], src_stride: usize, dst: &mut [T], dst_stride: usize, count: usize, row: usize) {
    if count == 0 || row == 0 {
        return;
    }
    assert!(
        (count - 1) * src_stride + row <= src.len() && (count - 1) * dst_stride + row <= dst.len(),
        "transpose: {count} rows of {row} overrun {} -> {} elements",
        src.len(),
        dst.len()
    );
    // Rows this short spent as long in the call to `memcpy` as in copying,
    // on Neoverse N1 most (BERT's rows of 26 heads were 6% slower).
    if size_of::<T>() == 4 && align_of::<T>() == 4 && (4..=MAX_INLINE_ROW).contains(&row) {
        // SAFETY: as in `transpose_matrix`.
        let (src, dst) = unsafe {
            (
                std::slice::from_raw_parts(src.as_ptr().cast::<f32>(), src.len()),
                std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<f32>(), dst.len()),
            )
        };
        simd_call!(Level::new(), copy_rows_simd(src, src_stride, dst, dst_stride, count, row));
    } else {
        for i in 0..count {
            dst[i * dst_stride..][..row].copy_from_slice(&src[i * src_stride..][..row]);
        }
    }
}

/// Longest row `copy_rows` copies with vectors rather than `memcpy`.
const MAX_INLINE_ROW: usize = 128;

/// `copy_rows` for rows of 4 to `MAX_INLINE_ROW` elements, whose extent has
/// been checked; the last vector of a row overlaps the one before.
#[simd]
fn copy_rows_simd<S: Simd>(
    simd: S,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
    count: usize,
    row: usize,
) {
    let (src, dst) = (src.as_ptr(), dst.as_mut_ptr());
    // SAFETY: `copy_rows` checked that every row is inside both slices, and
    // each access below stays inside its row.
    unsafe {
        for i in 0..count {
            let (s, d) = (src.add(i * src_stride), dst.add(i * dst_stride));
            if row >= 8 {
                let mut k = 0;
                while k + 8 < row {
                    f32x8::load_array_ref(simd, &*s.add(k).cast()).store_array(&mut *d.add(k).cast());
                    k += 8;
                }
                f32x8::load_array_ref(simd, &*s.add(row - 8).cast()).store_array(&mut *d.add(row - 8).cast());
            } else {
                f32x4::load_array_ref(simd, &*s.cast()).store_array(&mut *d.cast());
                f32x4::load_array_ref(simd, &*s.add(row - 4).cast()).store_array(&mut *d.add(row - 4).cast());
            }
        }
    }
}

/// Panics unless a `rows x cols` matrix with `src_ld` between rows fits in
/// `src_len` elements, and its transpose with `dst_ld` in `dst_len`.
fn assert_matrix_fits(src_len: usize, src_ld: usize, dst_len: usize, dst_ld: usize, rows: usize, cols: usize) {
    if rows == 0 || cols == 0 {
        return;
    }
    assert!(
        cols <= src_ld && rows <= dst_ld && (rows - 1) * src_ld + cols <= src_len && (cols - 1) * dst_ld + rows <= dst_len,
        "transpose: {rows}x{cols} matrix (leading dimensions {src_ld} -> {dst_ld}) overruns {src_len} -> {dst_len} elements"
    );
}

/// `transpose_matrix` for one range of rows and columns, a block at a time.
///
/// # Safety
/// `assert_matrix_fits` must hold for `src` and `dst` and matrices at least
/// `rows.end x cols.end`.
unsafe fn transpose_scalar<T: Copy>(
    src: *const T,
    src_ld: usize,
    dst: *mut T,
    dst_ld: usize,
    rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
) {
    for r0 in rows.clone().step_by(TRANSPOSE_BLOCK) {
        let r1 = (r0 + TRANSPOSE_BLOCK).min(rows.end);
        for c0 in cols.clone().step_by(TRANSPOSE_BLOCK) {
            let c1 = (c0 + TRANSPOSE_BLOCK).min(cols.end);
            // The inner loop runs along the longer side: the edges left by
            // the tiles are strips a few elements across.
            if r1 - r0 >= c1 - c0 {
                for c in c0..c1 {
                    for r in r0..r1 {
                        unsafe { *dst.add(c * dst_ld + r) = *src.add(r * src_ld + c) };
                    }
                }
            } else {
                for r in r0..r1 {
                    for c in c0..c1 {
                        unsafe { *dst.add(c * dst_ld + r) = *src.add(r * src_ld + c) };
                    }
                }
            }
        }
    }
}

/// `transpose_matrix` at `level`.
fn transpose_matrix_u32(
    level: Level,
    src: &[u32],
    src_ld: usize,
    dst: &mut [u32],
    dst_ld: usize,
    rows: usize,
    cols: usize,
) {
    simd_call!(level, transpose_matrix_simd(src, src_ld, dst, dst_ld, rows, cols))
}

#[simd]
fn transpose_matrix_simd<S: Simd>(
    simd: S,
    src: &[u32],
    src_ld: usize,
    dst: &mut [u32],
    dst_ld: usize,
    rows: usize,
    cols: usize,
) {
    assert_matrix_fits(src.len(), src_ld, dst.len(), dst_ld, rows, cols);
    let (src, dst) = (src.as_ptr(), dst.as_mut_ptr());
    // SAFETY: checked above, and every tile lies inside rows x cols.
    unsafe {
        if rows >= 8 && cols >= 8 {
            // 8x8 tiles over the bulk, then a strip of tiles along each edge
            // it leaves, 4x4 ones if they are wide enough, overlapping the
            // bulk: rewriting elements took less time than single ones.
            let (rows8, cols8) = (rows / 8 * 8, cols / 8 * 8);
            transpose_tiles::<S, 8>(simd, src, src_ld, dst, dst_ld, 0..rows8, 0..cols8);
            match cols - cols8 {
                0 => {}
                1..=4 => transpose_tiles::<S, 4>(simd, src, src_ld, dst, dst_ld, 0..rows, cols - 4..cols),
                _ => transpose_tiles::<S, 8>(simd, src, src_ld, dst, dst_ld, 0..rows, cols - 8..cols),
            }
            match rows - rows8 {
                0 => {}
                1..=4 => transpose_tiles::<S, 4>(simd, src, src_ld, dst, dst_ld, rows - 4..rows, 0..cols8),
                _ => transpose_tiles::<S, 8>(simd, src, src_ld, dst, dst_ld, rows - 8..rows, 0..cols8),
            }
        } else if rows >= 4 && cols >= 4 {
            transpose_tiles::<S, 4>(simd, src, src_ld, dst, dst_ld, 0..rows, 0..cols);
        } else {
            transpose_scalar(src, src_ld, dst, dst_ld, 0..rows, 0..cols);
        }
    }
}

/// `transpose_matrix` for ranges of at least `N` rows and columns, in
/// `N`x`N` tiles, a block of them at a time. Where a range is not a
/// multiple of `N` long, its last tiles overlap the ones before.
///
/// # Safety
/// As for `transpose_scalar`.
#[inline(always)]
unsafe fn transpose_tiles<S: Simd, const N: usize>(
    simd: S,
    src: *const u32,
    src_ld: usize,
    dst: *mut u32,
    dst_ld: usize,
    rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
) {
    // A function rather than a closure: closures don't take on the target
    // features `#[simd]` compiles this with, and the vector code would not
    // inline into them.
    #[inline(always)]
    unsafe fn tile<S: Simd, const N: usize>(
        simd: S,
        src: *const u32,
        src_ld: usize,
        dst: *mut u32,
        dst_ld: usize,
        (r, c): (usize, usize),
    ) {
        let (src, dst) = unsafe { (src.add(r * src_ld + c), dst.add(c * dst_ld + r)) };
        if N == 8 {
            unsafe { transpose_8x8(simd, src, src_ld, dst, dst_ld) };
        } else {
            unsafe { transpose_4x4(simd, src, src_ld, dst, dst_ld) };
        }
    }
    // Within a block, going across a tile row at a time ("rows first") keeps
    // a tile's worth of each destination row in cache until the next tile
    // row fills it in; going down a tile column at a time keeps the source
    // rows instead. Rows a multiple of 1 KiB apart fall into a few L1 sets,
    // too few for a block's 64, so the side with such a stride is the one
    // that must not wait in cache. Otherwise, measured on model shapes,
    // Neoverse N1 was up to 2x faster rows first and Zen 3 up to 2x faster
    // columns first.
    let (src_conflicts, dst_conflicts) = (src_ld.is_multiple_of(256), dst_ld.is_multiple_of(256));
    let rows_first = if src_conflicts != dst_conflicts {
        src_conflicts
    } else {
        cfg!(target_arch = "aarch64")
    };
    // The tile at row r, column c, moved back to overlap the one before where
    // it would run past the end.
    let at = |r: usize, c: usize| (r.min(rows.end - N), c.min(cols.end - N));
    for r0 in rows.clone().step_by(TRANSPOSE_BLOCK) {
        let r1 = (r0 + TRANSPOSE_BLOCK).min(rows.end);
        for c0 in cols.clone().step_by(TRANSPOSE_BLOCK) {
            let c1 = (c0 + TRANSPOSE_BLOCK).min(cols.end);
            if rows_first {
                for r in (r0..r1).step_by(N) {
                    for c in (c0..c1).step_by(N) {
                        unsafe { tile::<S, N>(simd, src, src_ld, dst, dst_ld, at(r, c)) };
                    }
                }
            } else {
                for c in (c0..c1).step_by(N) {
                    for r in (r0..r1).step_by(N) {
                        unsafe { tile::<S, N>(simd, src, src_ld, dst, dst_ld, at(r, c)) };
                    }
                }
            }
        }
    }
}

/// `a` and `b` interleaved within each half: the halves of `a.interleave(b)`
/// for the low halves, then for the high ones. The whole-vector `interleave`
/// moves data across the 128-bit halves; this compiles to one in-half
/// unpack per output on AVX2.
#[inline(always)]
fn interleave_halves<S: Simd>(a: f32x8<S>, b: f32x8<S>) -> (f32x8<S>, f32x8<S>) {
    let ((a0, a1), (b0, b1)) = (a.split(), b.split());
    let ((lo0, hi0), (lo1, hi1)) = (a0.interleave(b0), a1.interleave(b1));
    (lo0.combine(lo1), hi0.combine(hi1))
}

/// `interleave_halves` of pairs of elements.
#[inline(always)]
fn interleave_pair_halves<S: Simd>(a: f32x8<S>, b: f32x8<S>) -> (f32x8<S>, f32x8<S>) {
    let (a, b) = (a.bitcast::<f64x4<S>>(), b.bitcast::<f64x4<S>>());
    let ((a0, a1), (b0, b1)) = (a.split(), b.split());
    let ((lo0, hi0), (lo1, hi1)) = (a0.interleave(b0), a1.interleave(b1));
    (lo0.combine(lo1).bitcast(), hi0.combine(hi1).bitcast())
}

/// Transposes the 8x8 tile at `src` into `dst`: 4x4 transposes within the
/// halves of each pair of rows, by interleaving elements, then pairs of
/// them, and swapping the off-diagonal 4x4 blocks last.
///
/// The float vectors only move bits, so NaN payloads come through; on AVX2
/// they ran up to 10% faster than the same steps on integer vectors.
///
/// # Safety
/// 8 rows of 8 elements, `src_ld` and `dst_ld` apart, must be readable at
/// `src` and writable at `dst`.
#[inline(always)]
unsafe fn transpose_8x8<S: Simd>(simd: S, src: *const u32, src_ld: usize, dst: *mut u32, dst_ld: usize) {
    let (src, dst) = (src.cast::<f32>(), dst.cast::<f32>());
    // Loops rather than `array::from_fn`, for the reason `transpose_tiles`
    // gives for not using closures.
    let mut r = [f32x8::splat(simd, 0.0); 8];
    for (i, r) in r.iter_mut().enumerate() {
        *r = f32x8::load_array_ref(simd, unsafe { &*src.add(i * src_ld).cast() });
    }
    let (t0, t1) = interleave_halves(r[0], r[1]);
    let (t2, t3) = interleave_halves(r[2], r[3]);
    let (t4, t5) = interleave_halves(r[4], r[5]);
    let (t6, t7) = interleave_halves(r[6], r[7]);
    // Row i of the tile, in halves: columns 0-3 of rows 0-3 and 4-7 for
    // i < 4, columns 4-7 of them for i >= 4.
    let (s0, s1) = interleave_pair_halves(t0, t2);
    let (s2, s3) = interleave_pair_halves(t1, t3);
    let (s4, s5) = interleave_pair_halves(t4, t6);
    let (s6, s7) = interleave_pair_halves(t5, t7);
    for (i, (top, bottom)) in [(s0, s4), (s1, s5), (s2, s6), (s3, s7)].into_iter().enumerate() {
        let ((top0, top1), (bottom0, bottom1)) = (top.split(), bottom.split());
        top0.combine(bottom0).store_array(unsafe { &mut *dst.add(i * dst_ld).cast() });
        top1.combine(bottom1).store_array(unsafe { &mut *dst.add((i + 4) * dst_ld).cast() });
    }
}

/// `transpose_8x8` for 4x4 tiles, in two rounds.
#[inline(always)]
unsafe fn transpose_4x4<S: Simd>(simd: S, src: *const u32, src_ld: usize, dst: *mut u32, dst_ld: usize) {
    let (src, dst) = (src.cast::<f32>(), dst.cast::<f32>());
    let mut v = [f32x4::splat(simd, 0.0); 4];
    for (i, v) in v.iter_mut().enumerate() {
        *v = f32x4::load_array_ref(simd, unsafe { &*src.add(i * src_ld).cast() });
    }
    for _ in 0..2 {
        let mut next = v;
        for i in 0..2 {
            (next[2 * i], next[2 * i + 1]) = v[i].interleave(v[i + 2]);
        }
        v = next;
    }
    for (i, v) in v.into_iter().enumerate() {
        v.store_array(unsafe { &mut *dst.add(i * dst_ld).cast() });
    }
}

pub fn concat<'b, 'a, T: Clone + Copy + std::fmt::Debug>(
    inputs: &[&TensorView<'b, T>],
    axis: i64,
    out: &'a mut Vec<T>,
) -> TensorView<'a, T> {
    let _t0 = if crate::kernels::timing::TIMING_ENABLED {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let result = concat_impl(inputs, axis, out);
    if crate::kernels::timing::TIMING_ENABLED {
        let ns = _t0.unwrap().elapsed().as_nanos() as u64;
        crate::kernels::timing::CONCAT_NS.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
    }
    result
}

fn concat_impl<'b, 'a, T: Clone + Copy + std::fmt::Debug>(
    inputs: &[&TensorView<'b, T>],
    axis: i64,
    out: &'a mut Vec<T>,
) -> TensorView<'a, T> {
    if inputs.is_empty() {
        return TensorView::empty();
    }
    // Find first non-empty input to establish rank and base out_shape
    let mut first_non_empty = None;
    for (i, inp) in inputs.iter().enumerate() {
        if !inp.data.is_empty() {
            first_non_empty = Some((i, inp));
            break;
        }
    }

    let (_, first_inp) = match first_non_empty {
        Some(x) => x,
        None => return TensorView::empty(),
    };

    let ndim = first_inp.dim();
    let axis = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };

    // Calculate final shape and validate
    let mut out_shape = first_inp.shape.to_vec();
    out_shape[axis] = 0;

    for inp in inputs {
        if inp.data.is_empty() {
            continue;
        }
        assert_eq!(inp.dim(), ndim, "Concat: ranks mismatch");
        for d in 0..ndim {
            if d != axis {
                assert_eq!(inp.shape[d], out_shape[d], "Concat: inner dim mismatch");
            }
        }
        out_shape[axis] += inp.shape[axis];
    }

    let out_numel = out_shape.iter().product::<usize>();
    utils::ensure_capacity(out, out_numel);
    if out_numel == 0 {
        return TensorView::from_slice(out, out_shape);
    }

    let outer_dim: usize = out_shape.iter().take(axis).product();
    let inner_dim: usize = out_shape.iter().skip(axis + 1).product();
    let out_ptr = out.as_mut_ptr();
    let mut current_out_offset = 0;

    // Direct copy without allocating offset vectors
    for outer_i in 0..outer_dim {
        for inp in inputs {
            if inp.data.is_empty() {
                continue;
            }
            let axis_len = inp.shape[axis];
            let copy_len = axis_len * inner_dim;
            let src_offset = outer_i * copy_len;
            let inp_ptr = inp.data.as_ptr();

            unsafe {
                std::ptr::copy_nonoverlapping(
                    inp_ptr.add(src_offset),
                    out_ptr.add(current_out_offset),
                    copy_len,
                );
            }
            current_out_offset += copy_len;
        }
    }
    TensorView::from_slice(out, out_shape)
}
pub fn slice<'b, 'a, T: Clone + Copy + std::fmt::Debug>(
    input: &TensorView<'b, T>,
    starts: &[i64],
    ends: &[i64],
    axes: &[i64],
    steps: &[i64],
    out: &'a mut Vec<T>,
) -> TensorView<'a, T> {
    // Fast path: simple contiguous slice along first dimension
    // This handles the most common case: slicing rows of a 2D tensor
    if !axes.is_empty() && axes.len() == 1 && steps.is_empty() {
        let axis = if axes[0] < 0 {
            (input.dim() as i64 + axes[0]) as usize
        } else {
            axes[0] as usize
        };

        if axis == 0 && starts.len() == 1 && ends.len() == 1 {
            let dim_size = input.shape[0] as i64;
            let start = if starts[0] < 0 {
                (starts[0] + dim_size).max(0) as usize
            } else {
                starts[0].min(dim_size).max(0) as usize
            };
            let end = if ends[0] < 0 {
                (ends[0] + dim_size).max(0) as usize
            } else if ends[0] > 2_000_000_000 {
                dim_size as usize
            } else {
                ends[0].min(dim_size).max(0) as usize
            };

            if end > start {
                let stride: usize = input.shape[1..].iter().product();
                let start_offset = start * stride;
                let end_offset = end * stride;

                utils::ensure_capacity(out, end_offset - start_offset);
                out.copy_from_slice(&input.data[start_offset..end_offset]);

                let mut out_shape = input.shape.to_vec();
                out_shape[0] = end - start;
                return TensorView::from_slice(out, out_shape);
            }
        }
    }

    // Standard path: complex multi-dimensional slicing
    let ndim = input.dim();
    let num_ops = starts.len();
    let mut actual_starts = vec![0isize; ndim];
    let mut actual_ends = vec![0isize; ndim];
    let mut actual_steps = vec![1isize; ndim];
    for i in 0..ndim {
        actual_starts[i] = 0;
        actual_ends[i] = input.shape[i] as isize;
        actual_steps[i] = 1;
    }
    for i in 0..num_ops {
        let axis = if axes.is_empty() {
            i
        } else {
            let ax = axes[i];
            if ax < 0 {
                (ndim as i64 + ax) as usize
            } else {
                ax as usize
            }
        };
        let dim_size = input.shape[axis] as isize;
        let step = if i < steps.len() {
            steps[i] as isize
        } else {
            1
        };
        // Handle sentinel values (i64::MIN, i64::MAX) on the original i64
        // before casting to isize. On wasm32 (isize = 32-bit), casting
        // directly would truncate and break sentinel detection.
        let start_i64 = starts[i];
        let end_i64 = ends[i];

        // Detect sentinels BEFORE clamping. ONNX uses i64::MAX to mean
        // "to the end" (positive step) and i64::MIN to mean "to the
        // very beginning" (negative step).
        let end_is_max_sentinel = end_i64 > i64::MAX / 2;
        let end_is_min_sentinel = end_i64 < i64::MIN / 2;

        let start = if start_i64 > dim_size as i64 {
            dim_size
        } else if start_i64 < -(dim_size as i64) {
            -dim_size
        } else {
            start_i64 as isize
        };
        let end = if end_is_max_sentinel {
            dim_size
        } else if end_is_min_sentinel {
            -dim_size
        } else if end_i64 > dim_size as i64 {
            dim_size
        } else if end_i64 < -(dim_size as i64) {
            -dim_size
        } else {
            end_i64 as isize
        };
        let norm_start = if start < 0 { start + dim_size } else { start };
        let norm_end = if end_is_max_sentinel {
            if step > 0 { dim_size } else { -1 }
        } else if end_is_min_sentinel {
            if step > 0 { 0 } else { -1 }
        } else if end < 0 {
            end + dim_size
        } else {
            end
        };
        let (s, e) = if step > 0 {
            (
                norm_start.max(0).min(dim_size),
                norm_end.max(0).min(dim_size),
            )
        } else {
            (
                norm_start.max(0).min(dim_size - 1),
                norm_end.max(-1).min(dim_size - 1),
            )
        };
        actual_starts[axis] = s;
        actual_ends[axis] = e;
        actual_steps[axis] = step;
    }
    let mut out_shape = vec![0; ndim];
    for i in 0..ndim {
        let start = actual_starts[i];
        let end = actual_ends[i];
        let step = actual_steps[i];
        let count = if step > 0 {
            if start >= end {
                0
            } else {
                (end - start + step - 1) / step
            }
        } else {
            if start <= end {
                0
            } else {
                (start - end + (-step) - 1) / (-step)
            }
        };
        out_shape[i] = count as usize;
    }
    let out_numel = out_shape.iter().product::<usize>();
    utils::ensure_capacity(out, out_numel);

    let in_strides = utils::compute_strides(&input.shape);
    let mut coords = vec![0; ndim];
    let out_slice = out.as_mut_slice();
    for i in 0..out_numel {
        let mut in_off = 0isize;
        for d in 0..ndim {
            let in_idx = actual_starts[d] + (coords[d] as isize) * actual_steps[d];
            in_off += in_idx * (in_strides[d] as isize);
        }
        out_slice[i] = input.data[in_off as usize];
        for d in (0..ndim).rev() {
            coords[d] += 1;
            if coords[d] < out_shape[d] {
                break;
            }
            coords[d] = 0;
        }
    }
    TensorView::from_slice(out, out_shape)
}
pub fn pad<'b, 'a, T: Clone + Copy + std::fmt::Debug>(
    input: &TensorView<'b, T>,
    pads: &[i64],
    constant_value: Option<&TensorView<'b, T>>,
    mode: &str,
    out: &'a mut Vec<T>,
) -> TensorView<'a, T> {
    // Safely convert i64 → usize, clamping negatives to 0
    let raw_p: Vec<usize> = pads
        .iter()
        .map(|&x| if x < 0 { 0usize } else { x as usize })
        .collect();
    let rank = input.shape.len();
    // If pads is shorter than rank*2, it covers fewer dims.
    // ONNX pads layout: [begin_0..begin_n, end_0..end_n].
    // Zero-pad for the leading (batch/channel) dimensions.
    let p = if raw_p.len() < rank * 2 {
        let half = raw_p.len() / 2;
        let missing = rank - half;
        let mut full = vec![0usize; rank * 2];
        // Copy begins into positions [missing..rank]
        for i in 0..half {
            full[missing + i] = raw_p[i];
        }
        // Copy ends into positions [rank+missing..rank*2]
        for i in 0..half {
            full[rank + missing + i] = raw_p[half + i];
        }
        full
    } else {
        raw_p
    };
    let mut new_shape = input.shape.to_vec();
    for i in 0..rank {
        new_shape[i] += p[i] + p[i + rank];
    }
    let total = new_shape.iter().product::<usize>();
    utils::ensure_capacity(out, total);
    // For constant mode: use provided constant_value or default to 0 (per ONNX spec)
    let fill_val: T = if let Some(cv) = constant_value {
        if !cv.data.is_empty() {
            cv.data[0]
        } else {
            // ONNX default constant value is 0
            unsafe { std::mem::zeroed() }
        }
    } else {
        // ONNX default constant value is 0
        unsafe { std::mem::zeroed() }
    };

    // First copy input data into correct position, then handle padding
    out.fill(fill_val);
    // Copy input data into the center
    if rank == 1 {
        let start = p[0];
        out[start..start + input.shape[0]].copy_from_slice(&input.data);
    } else if rank == 2 {
        for i in 0..input.shape[0] {
            let dst_row = i + p[0];
            let dst_start = dst_row * new_shape[1] + p[1];
            let src_start = i * input.shape[1];
            out[dst_start..dst_start + input.shape[1]]
                .copy_from_slice(&input.data[src_start..src_start + input.shape[1]]);
        }
    } else if rank == 3 {
        for i in 0..input.shape[0] {
            for j in 0..input.shape[1] {
                let dst_row = i + p[0];
                let dst_col = j + p[1];
                let dst_start = (dst_row * new_shape[1] + dst_col) * new_shape[2] + p[2];
                let src_start = (i * input.shape[1] + j) * input.shape[2];
                let copy_len = input.shape[2];
                out[dst_start..dst_start + copy_len]
                    .copy_from_slice(&input.data[src_start..src_start + copy_len]);
            }
        }
    } else if rank == 4 {
        for n in 0..input.shape[0] {
            for c in 0..input.shape[1] {
                for h in 0..input.shape[2] {
                    let dst_n = n + p[0];
                    let dst_c = c + p[1];
                    let dst_h = h + p[2];
                    let dst_start = (((dst_n * new_shape[1] + dst_c) * new_shape[2] + dst_h)
                        * new_shape[3])
                        + p[3];
                    let src_start =
                        ((n * input.shape[1] + c) * input.shape[2] + h) * input.shape[3];
                    let copy_len = input.shape[3];
                    out[dst_start..dst_start + copy_len]
                        .copy_from_slice(&input.data[src_start..src_start + copy_len]);
                }
            }
        }
    } else {
        panic!("Pad: Rank {} not fully implemented", rank);
    }
    // For edge mode, replicate edge values into padding regions
    if mode == "edge" {
        pad_edge_inplace(out, &input.shape, &new_shape, &p, rank);
    } else if mode == "reflect" {
        pad_reflect_inplace(out, &input.shape, &new_shape, &p, rank);
    }
    TensorView::from_slice(out, new_shape)
}

/// Fill padding regions with edge (nearest) values for "edge" mode.
/// `out` already has the input data copied into the center region.
fn pad_edge_inplace<T: Clone + Copy>(
    out: &mut [T],
    input_shape: &[usize],
    new_shape: &[usize],
    p: &[usize],
    rank: usize,
) {
    // General n-dimensional edge padding via coordinate mapping
    let total: usize = new_shape.iter().product();
    let strides = utils::compute_strides(new_shape);

    // For each output element, clamp coordinates to the input range
    // to find the nearest edge value
    let mut coords = vec![0usize; rank];
    for idx in 0..total {
        // Check if this position is in the center (already filled)
        let mut in_center = true;
        for d in 0..rank {
            if coords[d] < p[d] || coords[d] >= p[d] + input_shape[d] {
                in_center = false;
                break;
            }
        }
        if !in_center {
            // Clamp coordinates to input region and read that value
            let mut src_idx = 0;
            for d in 0..rank {
                let clamped = if coords[d] < p[d] {
                    p[d]
                } else if coords[d] >= p[d] + input_shape[d] {
                    p[d] + input_shape[d] - 1
                } else {
                    coords[d]
                };
                src_idx += clamped * strides[d];
            }
            out[idx] = out[src_idx];
        }
        // Advance coordinates
        for d in (0..rank).rev() {
            coords[d] += 1;
            if coords[d] < new_shape[d] {
                break;
            }
            coords[d] = 0;
        }
    }
}

fn pad_reflect_inplace<T: Clone + Copy>(
    out: &mut [T],
    input_shape: &[usize],
    new_shape: &[usize],
    p: &[usize],
    rank: usize,
) {
    let total: usize = new_shape.iter().product();
    let strides = utils::compute_strides(new_shape);
    let mut coords = vec![0usize; rank];
    for idx in 0..total {
        let mut in_center = true;
        for d in 0..rank {
            if coords[d] < p[d] || coords[d] >= p[d] + input_shape[d] {
                in_center = false;
                break;
            }
        }
        if !in_center {
            let mut src_idx = 0;
            for d in 0..rank {
                let c = coords[d];
                let pad_begin = p[d];
                let pad_end = p[d] + input_shape[d];
                let reflected = if c < pad_begin {
                    pad_begin + (pad_begin - c)
                } else if c >= pad_end {
                    let past = c - pad_end;
                    pad_end - 1 - past
                } else {
                    c
                };
                src_idx += reflected * strides[d];
            }
            out[idx] = out[src_idx];
        }
        for d in (0..rank).rev() {
            coords[d] += 1;
            if coords[d] < new_shape[d] {
                break;
            }
            coords[d] = 0;
        }
    }
}

pub fn gather<'b, 'a, T, I>(
    data: &TensorView<'b, T>,
    indices: &TensorView<'b, I>,
    axis: i64,
    out: &'a mut Vec<T>,
) -> TensorView<'a, T>
where
    T: Copy + std::fmt::Debug,
    I: crate::kernels::utils::AsI64 + Copy + std::fmt::Debug,
{
    let axis = if axis < 0 {
        (data.dim() as i64 + axis) as usize
    } else {
        axis as usize
    };
    let mut out_shape = Vec::new();
    for i in 0..axis {
        out_shape.push(data.shape[i]);
    }
    out_shape.extend_from_slice(&indices.shape);
    for i in (axis + 1)..data.dim() {
        out_shape.push(data.shape[i]);
    }
    let out_numel = out_shape.iter().product::<usize>();
    utils::ensure_capacity(out, out_numel);
    let outer_dim: usize = data.shape[..axis].iter().product();
    let axis_dim = data.shape[axis];
    let inner_dim: usize = data.shape[axis + 1..].iter().product();
    let indices_len = indices.data.len();
    let mut out_idx = 0;

    for o in 0..outer_dim {
        for idx_i in 0..indices_len {
            let mut idx_val = indices.data[idx_i].as_i64();
            if idx_val < 0 {
                idx_val += axis_dim as i64;
            }
            let idx_val = idx_val as usize;
            let src_offset = o * axis_dim * inner_dim + idx_val * inner_dim;
            out[out_idx..out_idx + inner_dim]
                .copy_from_slice(&data.data[src_offset..src_offset + inner_dim]);
            out_idx += inner_dim;
        }
    }
    TensorView::from_slice(out, out_shape)
}
pub fn cast<'a>(input: &TensorView<'a>, _to: i64) -> TensorView<'a> {
    input.clone()
}
pub fn transpose<'b, 'a, T: Clone + Copy + std::fmt::Debug + 'static>(
    input: &TensorView<'b, T>,
    perm: &[i64],
    out: &'a mut Vec<T>,
) -> TensorView<'a, T> {
    let ndim = input.dim();
    // Missing trailing axes stay in place; no perm at all reverses the axes.
    let perm_at = |i: usize| match perm.get(i) {
        Some(&p) => p as usize,
        None if perm.is_empty() => ndim - 1 - i,
        None => i,
    };
    let input_shape = &input.shape;
    let input_data: &[T] = input.data.as_ref();
    assert!(
        (0..ndim).all(|i| perm_at(i) < ndim && (0..i).all(|j| perm_at(j) != perm_at(i))),
        "transpose: perm {perm:?} is not a permutation of the {ndim} axes of shape {input_shape:?}"
    );
    let out_shape: Vec<usize> = (0..ndim).map(|i| input_shape[perm_at(i)]).collect();
    let out_numel = input.data.len();
    utils::ensure_capacity(out, out_numel);
    let dst = &mut out[..out_numel];
    // Two axes longer than 1 that swap places, the rest of length 1, as in
    // most model transposes: a matrix transpose, without the planning that
    // was a good part of a small one's time.
    let mut long = [0; 2];
    let mut long_count = 0;
    for p in (0..ndim).map(perm_at).filter(|&p| input_shape[p] != 1) {
        if long_count < 2 {
            long[long_count] = p;
        }
        long_count += 1;
    }
    if long_count == 2 && long[0] > long[1] {
        let (rows, cols) = (input_shape[long[1]], input_shape[long[0]]);
        transpose_matrix(input_data, cols, dst, rows, rows, cols);
        return TensorView::from_slice(out, out_shape);
    }
    if ndim > MAX_RANK {
        let perm: Vec<usize> = (0..ndim).map(perm_at).collect();
        transpose_any_rank(input_data, input_shape, &perm, dst);
        return TensorView::from_slice(out, out_shape);
    }

    let perm_full: Small<usize> = (0..ndim).map(perm_at).collect();
    let (shape, perm) = simplify_transpose(input_shape, &perm_full);
    let n = shape.len();
    if n <= 1 {
        dst.copy_from_slice(input_data);
        return TensorView::from_slice(out, out_shape);
    }
    let in_stride = |axis: usize| shape[axis + 1..].iter().product::<usize>();
    // Each output axis as (length, input stride, output stride).
    let mut axes: Small<(usize, usize, usize)> = perm.iter().map(|&p| (shape[p], in_stride(p), 0)).collect();
    let mut out_stride = 1;
    for axis in axes.iter_mut().rev() {
        axis.2 = out_stride;
        out_stride *= axis.0;
    }

    if perm[n - 1] == n - 1 {
        // The innermost axis stays innermost: copy whole rows.
        let row = shape[n - 1];
        let (count, src_stride, dst_stride) = axes[n - 2];
        for_each_offset(&axes[..n - 2], |s, d| {
            copy_rows(&input_data[s..], src_stride, &mut dst[d..], dst_stride, count, row);
        });
    } else {
        // A matrix transpose for each index of the other axes: its rows run
        // along the input axis that becomes innermost, its columns along the
        // input's innermost axis.
        let rows_axis = perm[n - 1];
        let cols_at = perm.iter().position(|&p| p == n - 1).expect("perm is a permutation");
        let outer: Small<_> = (0..n - 1).filter(|&i| i != cols_at).map(|i| axes[i]).collect();
        for_each_offset(&outer, |s, d| {
            transpose_matrix(
                &input_data[s..],
                in_stride(rows_axis),
                &mut dst[d..],
                axes[cols_at].2,
                shape[rows_axis],
                shape[n - 1],
            );
        });
    }
    TensorView::from_slice(out, out_shape)
}

/// `transpose` for more than `MAX_RANK` axes, one element at a time.
fn transpose_any_rank<T: Copy>(src: &[T], shape: &[usize], perm: &[usize], dst: &mut [T]) {
    let strides = utils::compute_strides(shape);
    let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let mut index = vec![0; shape.len()];
    for value in dst.iter_mut() {
        *value = src[(0..shape.len()).map(|d| index[d] * strides[perm[d]]).sum::<usize>()];
        for d in (0..shape.len()).rev() {
            index[d] += 1;
            if index[d] < out_shape[d] {
                break;
            }
            index[d] = 0;
        }
    }
}
pub fn to_i64_vec<T: crate::kernels::utils::AsI64 + Copy + std::fmt::Debug>(
    input: &TensorView<T>,
) -> Vec<i64> {
    let mut out = Vec::with_capacity(input.data.len());
    for &val in input.data.iter() {
        out.push(val.as_i64());
    }
    out
}

/// The compiler emits all-zero sizes for a Split with no `split` input or
/// attribute, which ONNX defines as equal parts, the last one smaller when
/// the axis does not divide evenly.
fn resolve_split_sizes(splits: &[i64], dim: usize) -> std::borrow::Cow<'_, [i64]> {
    if splits.is_empty() || splits.iter().any(|&s| s != 0) {
        return std::borrow::Cow::Borrowed(splits);
    }
    let n = splits.len();
    let part = dim.div_ceil(n);
    std::borrow::Cow::Owned(
        (0..n)
            .map(|i| part.min(dim.saturating_sub(i * part)) as i64)
            .collect(),
    )
}

pub fn split<'a, T: Clone + Copy + std::fmt::Debug>(
    input: &TensorView<'_, T>,
    axis: i64,
    splits: &[i64],
    outputs: &'a mut [Vec<T>],
) -> Vec<TensorView<'a, T>> {
    let ndim = input.dim();
    let axis = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };
    assert!(axis < ndim, "Split: axis out of bounds (axis={}, ndim={}, shape={:?})", axis, ndim, &*input.shape);
    let splits = &*resolve_split_sizes(splits, input.shape[axis]);
    let num_splits = splits.len();
    assert_eq!(
        outputs.len(),
        num_splits,
        "Split: output buffers count mismatch"
    );
    let total: i64 = splits.iter().sum();
    assert_eq!(
        total, input.shape[axis] as i64,
        "Split: splits sum mismatch"
    );
    let outer_dim: usize = input.shape[..axis].iter().product();
    let inner_dim: usize = input.shape[axis + 1..].iter().product();
    let mut results = Vec::with_capacity(num_splits);
    let mut axis_offset = 0;
    for (i, &split_size) in splits.iter().enumerate() {
        let split_size = split_size as usize;
        let mut out_shape = input.shape.to_vec();
        out_shape[axis] = split_size;
        let out_numel = out_shape.iter().product::<usize>();
        utils::ensure_capacity(&mut outputs[i], out_numel);
        for outer_idx in 0..outer_dim {
            let src_offset = outer_idx * input.shape[axis] * inner_dim + axis_offset * inner_dim;
            let dst_offset = outer_idx * split_size * inner_dim;
            let copy_len = split_size * inner_dim;
            let out_slice = &mut outputs[i][dst_offset..dst_offset + copy_len];
            out_slice.copy_from_slice(&input.data[src_offset..src_offset + copy_len]);
        }
        axis_offset += split_size;
    }
    for (i, &split_size) in splits.iter().enumerate() {
        let mut out_shape = input.shape.to_vec();
        out_shape[axis] = split_size as usize;
        results.push(TensorView::from_slice(&outputs[i], out_shape));
    }
    results
}

/// Split a tensor into multiple owned tensors along an axis.
/// This is a convenience function that returns owned TensorViews directly,
/// avoiding the need for cloning when using the output buffers pattern.
///
/// # Arguments
/// * `input` - Input tensor to split
/// * `axis` - Axis along which to split (negative values index from the end)
/// * `splits` - Sizes of each split output
///
/// # Returns
/// Vector of owned TensorViews, one for each split
pub fn split_owned<T: Clone + Copy + std::fmt::Debug>(
    input: &TensorView<'_, T>,
    axis: i64,
    splits: &[i64],
) -> Vec<TensorView<'static, T>> {
    let _t0 = if crate::kernels::timing::TIMING_ENABLED {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let ndim = input.dim();
    let axis = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };
    assert!(axis < ndim, "Split: axis out of bounds (axis={}, ndim={}, shape={:?})", axis, ndim, &*input.shape);
    let splits = &*resolve_split_sizes(splits, input.shape[axis]);

    let num_splits = splits.len();
    let total: i64 = splits.iter().sum();
    assert_eq!(
        total, input.shape[axis] as i64,
        "Split: splits sum mismatch"
    );

    let outer_dim: usize = input.shape[..axis].iter().product();
    let inner_dim: usize = input.shape[axis + 1..].iter().product();

    let mut results = Vec::with_capacity(num_splits);
    let mut axis_offset = 0;

    for &split_size in splits {
        let split_size = split_size as usize;
        let mut out_shape = input.shape.to_vec();
        out_shape[axis] = split_size;
        let out_numel = out_shape.iter().product::<usize>();

        // Allocate owned buffer for this split
        let mut buffer = Vec::with_capacity(out_numel);
        unsafe {
            buffer.set_len(out_numel);
        }

        // Copy data from input to this split's buffer
        for outer_idx in 0..outer_dim {
            let src_offset = outer_idx * input.shape[axis] * inner_dim + axis_offset * inner_dim;
            let dst_offset = outer_idx * split_size * inner_dim;
            let copy_len = split_size * inner_dim;
            buffer[dst_offset..dst_offset + copy_len]
                .copy_from_slice(&input.data[src_offset..src_offset + copy_len]);
        }

        // Create owned TensorView
        results.push(TensorView::from_owned(buffer, out_shape));
        axis_offset += split_size;
    }
    if crate::kernels::timing::TIMING_ENABLED {
        let ns = _t0.unwrap().elapsed().as_nanos() as u64;
        crate::kernels::timing::SPLIT_NS.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
    }
    results
}
pub fn where_op<'b, 'a, T, C>(
    condition: &TensorView<'b, C>,
    x: &TensorView<'b, T>,
    y: &TensorView<'b, T>,
    out: &'a mut Vec<T>,
) -> TensorView<'a, T>
where
    T: Clone + Copy + std::fmt::Debug,
    C: Clone + Copy + std::fmt::Debug + crate::kernels::utils::AsI64,
{
    let result = where_op_inner(condition, x, y, out);
    result
}
fn where_op_inner<'b, 'a, T, C>(
    condition: &TensorView<'b, C>,
    x: &TensorView<'b, T>,
    y: &TensorView<'b, T>,
    out: &'a mut Vec<T>,
) -> TensorView<'a, T>
where
    T: Clone + Copy + std::fmt::Debug,
    C: Clone + Copy + std::fmt::Debug + crate::kernels::utils::AsI64,
{
    let cond_data = &condition.data;
    let x_data = &x.data;
    let y_data = &y.data;

    // Fast path: all same shape — direct elementwise, no coordinates
    if condition.shape == x.shape && x.shape == y.shape {
        let numel = cond_data.len();
        utils::ensure_capacity(out, numel);
        let o = out.as_mut_slice();
        for i in 0..numel {
            unsafe {
                *o.get_unchecked_mut(i) = if cond_data.get_unchecked(i).as_i64() != 0 {
                    *x_data.get_unchecked(i)
                } else {
                    *y_data.get_unchecked(i)
                };
            }
        }
        return TensorView::from_slice(out, condition.shape.to_vec());
    }

    let out_shape = utils::broadcast_shapes(&condition.shape, &x.shape)
        .and_then(|s| utils::broadcast_shapes(&s, &y.shape))
        .unwrap_or_else(|| condition.shape.to_vec());
    let out_numel: usize = out_shape.iter().product();
    let dims = out_shape.len();
    utils::ensure_capacity(out, out_numel);
    let o = out.as_mut_slice();

    // Fast path: both x and y are scalars — just fill based on condition
    // This is the attention mask pattern: where(mask, 0.0, -inf)
    if x.data.len() == 1 && y.data.len() == 1 {
        let x_val = x.data[0];
        let y_val = y.data[0];
        let cond_numel = cond_data.len();
        if cond_numel == out_numel {
            for i in 0..out_numel {
                unsafe {
                    *o.get_unchecked_mut(i) = if cond_data.get_unchecked(i).as_i64() != 0 {
                        x_val
                    } else {
                        y_val
                    };
                }
            }
        } else if cond_numel > 0 && out_numel % cond_numel == 0 {
            let repeat = out_numel / cond_numel;
            for r in 0..repeat {
                let base = r * cond_numel;
                for i in 0..cond_numel {
                    unsafe {
                        *o.get_unchecked_mut(base + i) =
                            if cond_data.get_unchecked(i).as_i64() != 0 {
                                x_val
                            } else {
                                y_val
                            };
                    }
                }
            }
        }
        return TensorView::from_slice(out, out_shape);
    }

    // Fast path: scalar x, condition broadcasts, y is full-sized
    // This is the attention mask pattern: where(mask, -inf, attention_scores)
    if x.data.len() == 1 && y.data.len() == out_numel {
        let x_val = x.data[0];
        let cond_numel = cond_data.len();

        if cond_numel == 1 {
            if cond_data[0].as_i64() != 0 {
                for v in o.iter_mut() {
                    *v = x_val;
                }
            } else {
                o.copy_from_slice(y_data);
            }
        } else {
            // Condition repeats over leading dims
            let repeat = out_numel / cond_numel;
            // Copy y first, then overwrite where condition is true
            o.copy_from_slice(y_data);
            for r in 0..repeat {
                let base = r * cond_numel;
                for i in 0..cond_numel {
                    unsafe {
                        if cond_data.get_unchecked(i).as_i64() != 0 {
                            *o.get_unchecked_mut(base + i) = x_val;
                        }
                    }
                }
            }
        }
        return TensorView::from_slice(out, out_shape);
    }

    if x.data.len() == out_numel && y.data.len() == out_numel {
        // Condition broadcasts, x and y are full — use stride-based cond index
        let cond_numel = cond_data.len();
        if cond_numel == 1 {
            // Scalar condition
            if cond_data[0].as_i64() != 0 {
                o.copy_from_slice(x_data);
            } else {
                o.copy_from_slice(y_data);
            }
        } else {
            // Condition is smaller, repeats over leading dims
            // E.g. cond=[T,T], out=[B,H,T,T]
            let repeat = out_numel / cond_numel;
            for r in 0..repeat {
                let base = r * cond_numel;
                for i in 0..cond_numel {
                    unsafe {
                        let idx = base + i;
                        *o.get_unchecked_mut(idx) = if cond_data.get_unchecked(i).as_i64() != 0 {
                            *x_data.get_unchecked(idx)
                        } else {
                            *y_data.get_unchecked(idx)
                        };
                    }
                }
            }
        }
        return TensorView::from_slice(out, out_shape);
    }

    // General broadcast: use coordinate-based indexing (slowest path)

    let mk_strides = |shape: &[usize]| -> Vec<usize> {
        let mut strides = vec![0; dims];
        let mut curr = 1;
        let offset = dims - shape.len();
        for i in (0..shape.len()).rev() {
            if shape[i] != 1 {
                strides[offset + i] = curr;
            }
            curr *= shape[i];
        }
        strides
    };
    let cs = mk_strides(&condition.shape);
    let xs = mk_strides(&x.shape);
    let ys = mk_strides(&y.shape);
    let mut coords = vec![0usize; dims];
    let mut off_c = 0usize;
    let mut off_x = 0usize;
    let mut off_y = 0usize;
    for j in 0..out_numel {
        unsafe {
            *o.get_unchecked_mut(j) = if cond_data.get_unchecked(off_c).as_i64() != 0 {
                *x_data.get_unchecked(off_x)
            } else {
                *y_data.get_unchecked(off_y)
            };
        }
        for d in (0..dims).rev() {
            coords[d] += 1;
            if coords[d] < out_shape[d] {
                off_c += cs[d];
                off_x += xs[d];
                off_y += ys[d];
                break;
            } else {
                off_c -= (coords[d] - 1) * cs[d];
                off_x -= (coords[d] - 1) * xs[d];
                off_y -= (coords[d] - 1) * ys[d];
                coords[d] = 0;
            }
        }
    }
    TensorView::from_slice(out, out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorView;
    #[test]
    fn test_concat() {
        let d1 = vec![1.0, 2.0, 3.0, 4.0];
        let t1 = TensorView::from_slice(&d1, vec![2, 2]);
        let d2 = vec![5.0, 6.0];
        let t2 = TensorView::from_slice(&d2, vec![2, 1]);
        let mut out = Vec::new();
        let res = concat(&[&t1, &t2], 1, &mut out);
        assert_eq!(res.shape, vec![2, 3]);
        assert_eq!(res.data, vec![1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
    }

    /// Transpose element by element: output index to input index.
    fn transpose_reference<T: Copy>(data: &[T], shape: &[usize], perm: &[usize]) -> Vec<T> {
        let strides = utils::compute_strides(shape);
        let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
        let mut index = vec![0; shape.len()];
        (0..data.len())
            .map(|mut i| {
                for d in (0..shape.len()).rev() {
                    index[d] = i % out_shape[d];
                    i /= out_shape[d];
                }
                data[(0..shape.len()).map(|d| index[d] * strides[perm[d]]).sum::<usize>()]
            })
            .collect()
    }

    fn permutations(n: usize) -> Vec<Vec<usize>> {
        if n == 0 {
            return vec![vec![]];
        }
        let mut all = Vec::new();
        for p in permutations(n - 1) {
            for at in 0..n {
                let mut q = p.clone();
                q.insert(at, n - 1);
                all.push(q);
            }
        }
        all
    }

    fn check_transpose<T: Copy + PartialEq + std::fmt::Debug + 'static>(
        shape: &[usize],
        perm: &[usize],
        value: impl Fn(usize) -> T,
    ) {
        let data: Vec<T> = (0..shape.iter().product()).map(value).collect();
        let input = TensorView::from_slice(&data, shape.to_vec());
        let perm_i64: Vec<i64> = perm.iter().map(|&p| p as i64).collect();
        let mut out = Vec::new();
        let got = transpose(&input, &perm_i64, &mut out);
        let want_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
        assert_eq!(&*got.shape, &want_shape[..], "shape {shape:?} perm {perm:?}");
        assert!(
            got.data == transpose_reference(&data, shape, perm),
            "shape {shape:?} perm {perm:?}"
        );
    }

    #[test]
    fn test_transpose_matches_reference_for_every_permutation() {
        // Lengths of 1 (dropped), and around the 4- and 8-wide tiles and the
        // 64-wide blocks.
        let lens = [1, 2, 3, 5, 8, 9, 65];
        for rank in 1..=4 {
            for perm in permutations(rank) {
                let mut shape = vec![0; rank];
                // Every length on every axis would be 7^4 shapes per perm; walk
                // each length over each axis in turn, the others cycling past it.
                for i in 0..lens.len() * rank {
                    for (d, len) in shape.iter_mut().enumerate() {
                        *len = lens[(i / rank + d * (i % rank + 1)) % lens.len()];
                    }
                    if shape.iter().product::<usize>() > 40_000 {
                        continue;
                    }
                    check_transpose(&shape, &perm, |i| i as f32);
                }
            }
        }
        for perm in permutations(5) {
            check_transpose(&[2, 3, 1, 4, 5], &perm, |i| i as f32);
        }
    }

    #[test]
    fn test_transpose_of_each_element_size() {
        for (shape, perm) in [
            (&[37, 41][..], &[1, 0][..]),
            (&[3, 17, 9], &[0, 2, 1]),
            (&[2, 9, 3, 10], &[0, 2, 3, 1]),
            (&[2, 9, 3, 10], &[2, 0, 1, 3]),
            (&[4, 3, 5], &[2, 1, 0]),
        ] {
            check_transpose(shape, perm, |i| i as u8);
            check_transpose(shape, perm, |i| i as i32 - 500);
            check_transpose(shape, perm, |i| i as i64 * 1_000_000_007);
            check_transpose(shape, perm, |i| i % 3 == 0);

            // NaN payloads, signalling ones included, and negative zero must
            // come through bit for bit.
            let bits: Vec<u32> = (0..shape.iter().product::<usize>())
                .map(|i| match i % 3 {
                    0 => 0x7f80_0001 + i as u32, // signalling NaN
                    1 => 0xffc0_0000 | i as u32, // quiet NaN, sign set
                    _ => 0x8000_0000,            // -0.0
                })
                .collect();
            let floats: Vec<f32> = bits.iter().map(|&b| f32::from_bits(b)).collect();
            let perm_i64: Vec<i64> = perm.iter().map(|&p| p as i64).collect();
            let mut out = Vec::new();
            let got = transpose(&TensorView::from_slice(&floats, shape.to_vec()), &perm_i64, &mut out);
            let got: Vec<u32> = got.data.iter().map(|v| v.to_bits()).collect();
            assert!(got == transpose_reference(&bits, shape, perm), "shape {shape:?} perm {perm:?}");
        }
    }

    #[test]
    fn test_transpose_model_shapes() {
        for (shape, perm) in [
            (&[1, 2, 400, 400][..], &[0, 1, 3, 2][..]), // YOLO26 head
            (&[1, 80, 8400], &[0, 2, 1]),                // YOLO26 output
            (&[1, 97, 4, 128], &[0, 2, 3, 1]),           // SenseVoice attention keys
            (&[1, 97, 4, 128], &[0, 2, 1, 3]),
            (&[1, 16, 384, 30], &[1, 0, 2, 3]),          // T-one, only moves an axis of 1
            (&[10, 1, 8, 48], &[1, 2, 3, 0]),
            (&[1, 1856, 3, 4, 64], &[2, 0, 3, 1, 4]),    // MOSS-TTS QKV split
        ] {
            check_transpose(shape, perm, |i| i as f32);
        }
    }

    #[test]
    fn test_transpose_matrix_at_every_level_and_stride() {
        for level in crate::kernels::test_util::levels() {
            // Every remainder by 8 on both sides, for the edge strips, and
            // sides under 8 and under 4.
            for rows in [1, 3, 4, 7, 8, 9, 12, 13, 31, 32, 33, 44, 70, 130] {
                for cols in [1, 2, 5, 8, 11, 12, 14, 16, 17, 40, 65, 129] {
                    // Padded leading dimensions, as for a matrix inside a larger
                    // tensor; multiples of 256 on either side, both or neither
                    // pick each order of walking the tiles.
                    let pad = |len: usize, wide: bool| if wide { len.next_multiple_of(256) } else { len + 3 };
                    for (wide_src, wide_dst) in [(false, false), (true, false), (false, true), (true, true)] {
                        let (src_ld, dst_ld) = (pad(cols, wide_src), pad(rows, wide_dst));
                        // Spread over all 32 bits, so many are NaN patterns
                        // once the tiles move them as floats.
                        let src: Vec<u32> = (0..rows * src_ld).map(|i| (i as u32).wrapping_mul(0x9e37_79b9)).collect();
                        let mut dst = vec![u32::MAX; cols * dst_ld];
                        transpose_matrix_u32(level, &src, src_ld, &mut dst, dst_ld, rows, cols);
                        for c in 0..cols {
                            for r in 0..dst_ld {
                                let want = if r < rows { src[r * src_ld + c] } else { u32::MAX };
                                assert_eq!(
                                    dst[c * dst_ld + r],
                                    want,
                                    "{level:?} {rows}x{cols}, ld {src_ld} -> {dst_ld}, at ({r}, {c})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_simplify_transpose() {
        let simplified = |shape: &[usize], perm: &[usize]| {
            let (shape, perm) = simplify_transpose(shape, perm);
            (shape.to_vec(), perm.to_vec())
        };
        // Axes of 1 go, and runs kept in order merge.
        assert_eq!(simplified(&[1, 16, 384, 30], &[1, 0, 2, 3]), (vec![184_320], vec![0]));
        assert_eq!(simplified(&[1, 97, 4, 128], &[0, 2, 3, 1]), (vec![97, 512], vec![1, 0]));
        assert_eq!(simplified(&[10, 1, 8, 48], &[1, 2, 3, 0]), (vec![10, 384], vec![1, 0]));
        assert_eq!(simplified(&[2, 62, 8, 64], &[2, 0, 1, 3]), (vec![124, 8, 64], vec![1, 0, 2]));
        assert_eq!(simplified(&[4, 3, 5], &[2, 1, 0]), (vec![4, 3, 5], vec![2, 1, 0]));
        assert_eq!(simplified(&[1, 1], &[1, 0]), (vec![], vec![]));
    }

    #[test]
    fn test_transpose_above_max_rank_and_default_perms() {
        let shape = [2, 1, 3, 2, 1, 2, 3, 1, 2];
        let perm = [8, 2, 0, 6, 1, 5, 3, 7, 4];
        check_transpose(&shape, &perm, |i| i as f32);

        // No perm reverses the axes; a short one leaves the rest in place.
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let input = TensorView::from_slice(&data, vec![2, 3, 4]);
        let mut out = Vec::new();
        assert!(transpose(&input, &[], &mut out).data == transpose_reference(&data, &[2, 3, 4], &[2, 1, 0]));
        assert!(transpose(&input, &[1, 0], &mut out).data == transpose_reference(&data, &[2, 3, 4], &[1, 0, 2]));
    }

    #[test]
    #[should_panic(expected = "not a permutation")]
    fn test_transpose_rejects_repeated_axis() {
        let data = [0.0f32; 6];
        transpose(&TensorView::from_slice(&data, vec![2, 3]), &[0, 0], &mut Vec::new());
    }
}
