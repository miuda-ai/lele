//! Shape inference, and folding of the shape computations it resolves.
//!
//! Exported models compute shapes at run time: `Shape(x)`, then `Gather`,
//! `Unsqueeze` and `Concat` to assemble a `Reshape` or `Expand` target or a
//! `ConstantOfShape` shape from dimensions of tensors and constants. Most of
//! those dimensions are fixed, and the others are dimensions of tensors
//! computed earlier. [`fold_shapes`] infers the shape of every value and the
//! elements of small integer tensors, each either a number or a dimension of
//! a value. It then replaces the integer tensors it knows completely by
//! initializers, and the operators whose target it knows up to dimensions of
//! values by [`Op::ReshapeTo`], [`Op::ExpandTo`] and
//! [`Op::ConstantOfShapeTo`], leaving the computation dead.
//!
//! Inference is conservative: a dimension is only derived where lele's
//! kernels agree with ONNX, and is [`Dim::Unknown`] otherwise.

use crate::compiler::ir::{
    Binary, ConvAttrs, DType, Dim, Graph, Node, Op, PoolAttrs, Reduction, TargetDim, ValueId,
};
use crate::model::onnx_proto::TensorProto;
use crate::model::tensor_to_array;
use std::collections::HashMap;

type Shape = Vec<Dim>;

/// Integer tensors with at most this many elements have them tracked.
const MAX_ELEMS: usize = 64;

/// What is known of an element of a small integer tensor.
#[derive(Clone, Debug, PartialEq)]
enum Elem {
    Fixed(i64),
    /// Dimension `axis` of a value of this graph, where it is not fixed.
    Dim(ValueId, usize),
    /// A symbolic dimension of a value of an enclosing graph.
    Sym(String),
    Unknown,
}

impl Elem {
    fn fixed(&self) -> Option<i64> {
        match self {
            Elem::Fixed(n) => Some(*n),
            _ => None,
        }
    }

    fn from_dim(d: &Dim) -> Self {
        match d {
            Dim::Fixed(n) => Elem::Fixed(*n),
            Dim::Sym(s) => Elem::Sym(s.clone()),
            Dim::Unknown => Elem::Unknown,
        }
    }

    /// This element as a dimension of a shape.
    fn to_dim(&self, g: &Graph) -> Dim {
        match self {
            Elem::Fixed(n) => Dim::Fixed(*n),
            Elem::Dim(v, axis) => g
                .value(*v)
                .shape
                .as_ref()
                .and_then(|s| s.get(*axis).cloned())
                .unwrap_or(Dim::Unknown),
            Elem::Sym(s) => Dim::Sym(s.clone()),
            Elem::Unknown => Dim::Unknown,
        }
    }
}

type Elems = HashMap<ValueId, Vec<Elem>>;

/// Shapes and elements of the values of enclosing graphs, by name.
#[derive(Clone, Default)]
struct Scope {
    shapes: HashMap<String, Shape>,
    elems: HashMap<String, Vec<Dim>>,
}

/// Infers shapes, then folds the integer tensors and targets they resolve,
/// in `g` and its subgraphs. Expects [`infer_dtypes`](super::infer_dtypes)
/// to have run.
pub(crate) fn fold_shapes(g: &mut Graph) {
    fold_graph(g, &Scope::default());
}

fn fold_graph(g: &mut Graph, outer: &Scope) {
    let elems = infer(g, outer);
    fold_ints(g, &elems);
    rewrite_targets(g, &elems);
}

/// Sets the shape of every value of `g` the model leaves open or only
/// partly states, recursing into subgraphs, and returns the elements known
/// of small integer tensors.
fn infer(g: &mut Graph, outer: &Scope) -> Elems {
    let mut elems = Elems::new();
    let captures: Vec<ValueId> = g.captures().collect();
    for v in captures {
        let name = g.value(v).name.clone();
        if let Some(shape) = outer.shapes.get(&name) {
            set_shape(g, v, shape.clone());
        }
        if let Some(e) = outer.elems.get(&name) {
            elems.insert(v, e.iter().map(Elem::from_dim).collect());
        }
    }
    // An initializer of an input's name is only its default value.
    for v in g.value_ids() {
        if !g.inputs.contains(&v)
            && let Some(e) = g.data(v).and_then(int_elems)
        {
            elems.insert(v, e);
        }
    }

    for id in g.node_ids() {
        if !g.node(id).op.subgraphs().is_empty() {
            let mut scope = outer.clone();
            for v in g.value_ids() {
                let value = g.value(v);
                if let Some(shape) = &value.shape {
                    scope.shapes.insert(value.name.clone(), shape.clone());
                }
                // Values of this graph mean nothing to the subgraph.
                if let Some(e) = elems.get(&v) {
                    let dims = e.iter().map(|e| e.to_dim(g)).collect();
                    scope.elems.insert(value.name.clone(), dims);
                }
            }
            for sub in g.node_mut(id).op.subgraphs_mut() {
                fold_graph(sub, &scope);
            }
        }
        let node = g.node(id);
        if let Op::Constant { value } = &node.op
            && let (Some(out), Some(e)) = (node.output(0), int_elems(value))
        {
            elems.insert(out, e);
        }
        let shapes = output_shapes(g, node, &elems);
        let out_elems = output_elems(g, node, &elems);
        let outputs = node.outputs().to_vec();
        for (v, shape) in outputs.iter().zip(shapes) {
            if let (Some(v), Some(shape)) = (v, shape) {
                set_shape(g, *v, shape);
            }
        }
        // Only scalars and vectors: the rules treat elements as a flat list.
        if let (Some(Some(v)), Some(e)) = (outputs.first(), out_elems)
            && e.len() <= MAX_ELEMS
            && g.value(*v).shape.as_ref().is_some_and(|s| s.len() <= 1)
        {
            elems.insert(*v, e);
        }
    }
    elems
}

/// Records `inferred` as the shape of `v`, keeping what the model states
/// where it is more specific.
fn set_shape(g: &mut Graph, v: ValueId, inferred: Shape) {
    let value = g.value_mut(v);
    value.shape = Some(match value.shape.take() {
        Some(stated) if stated.len() == inferred.len() => stated
            .into_iter()
            .zip(inferred)
            .map(|(s, i)| unify(&s, &i))
            .collect(),
        Some(stated) => stated,
        None => inferred,
    });
}

/// The more specific of two descriptions of one dimension.
fn unify(a: &Dim, b: &Dim) -> Dim {
    match (a, b) {
        (Dim::Fixed(_), _) => a.clone(),
        (_, Dim::Fixed(_)) => b.clone(),
        (Dim::Sym(_), _) => a.clone(),
        _ => b.clone(),
    }
}

/// The values of an integer scalar or vector small enough to track.
fn int_elems(t: &TensorProto) -> Option<Vec<Elem>> {
    if !matches!(DType::from_onnx(t.data_type), DType::I64 | DType::I32) || t.dims.len() > 1 {
        return None;
    }
    let (data, _) = tensor_to_array(t).ok()?;
    (data.len() <= MAX_ELEMS).then(|| data.iter().map(|&x| Elem::Fixed(x as i64)).collect())
}

fn fixed(d: &Dim) -> Option<i64> {
    match d {
        Dim::Fixed(n) => Some(*n),
        _ => None,
    }
}

/// Every dimension, if all are known.
fn all_fixed(dims: &[Dim]) -> Option<Vec<i64>> {
    dims.iter().map(fixed).collect()
}

/// Every element, if all are known.
fn all_fixed_elems(elems: &[Elem]) -> Option<Vec<i64>> {
    elems.iter().map(Elem::fixed).collect()
}

fn product(dims: &[Dim]) -> Option<i64> {
    all_fixed(dims).map(|d| d.iter().product())
}

/// `axis` counted from the front of a rank-`rank` tensor.
fn norm_axis(axis: i64, rank: usize) -> Option<usize> {
    let a = if axis < 0 { axis + rank as i64 } else { axis };
    (0..rank as i64).contains(&a).then_some(a as usize)
}

/// The dimension two broadcast dimensions produce.
fn broadcast_dim(a: &Dim, b: &Dim) -> Dim {
    match (a, b) {
        (Dim::Fixed(1), d) | (d, Dim::Fixed(1)) => d.clone(),
        (Dim::Fixed(x), Dim::Fixed(y)) if x != y => Dim::Unknown,
        // The other one is 1 or this.
        (Dim::Fixed(n), _) | (_, Dim::Fixed(n)) => Dim::Fixed(*n),
        (Dim::Sym(x), Dim::Sym(y)) if x == y => a.clone(),
        _ => Dim::Unknown,
    }
}

/// The shape numpy-style broadcasting of `shapes` produces.
fn broadcast(shapes: &[&[Dim]]) -> Shape {
    let rank = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
    (0..rank)
        .map(|i| {
            shapes
                .iter()
                .filter_map(|s| s.len().checked_sub(rank - i).map(|k| &s[k]))
                .fold(Dim::Fixed(1), |acc, d| broadcast_dim(&acc, d))
        })
        .collect()
}

/// `MatMul` of `a` and `b`.
fn matmul(a: &[Dim], b: &[Dim]) -> Option<Shape> {
    match (a.len(), b.len()) {
        (0, _) | (_, 0) => None,
        (1, 1) => Some(Vec::new()),
        (1, _) => {
            let mut out = b[..b.len() - 2].to_vec();
            out.push(b[b.len() - 1].clone());
            Some(out)
        }
        (_, 1) => Some(a[..a.len() - 1].to_vec()),
        _ => {
            let mut out = broadcast(&[&a[..a.len() - 2], &b[..b.len() - 2]]);
            out.push(a[a.len() - 2].clone());
            out.push(b[b.len() - 1].clone());
            Some(out)
        }
    }
}

/// Size of a sliding-window output dimension (convolution, pooling).
#[allow(clippy::too_many_arguments)]
fn window(
    input: &Dim,
    kernel: Option<i64>,
    stride: i64,
    dilation: i64,
    pads: (i64, i64),
    auto_pad: &str,
    ceil: bool,
) -> Dim {
    let Some(k) = kernel else {
        return Dim::Unknown;
    };
    let (begin, end) = match auto_pad {
        "" | "NOTSET" => pads,
        "VALID" => (0, 0),
        // lele pads for a stride of 1, where the output keeps the size.
        "SAME_UPPER" | "SAME_LOWER" if stride == 1 => return input.clone(),
        _ => return Dim::Unknown,
    };
    let reach = dilation * (k - 1);
    if stride == 1 && begin + end == reach {
        return input.clone();
    }
    let Dim::Fixed(n) = input else {
        return Dim::Unknown;
    };
    let span = n + begin + end - reach - 1;
    if span < 0 || stride <= 0 || (ceil && span % stride != 0) {
        return Dim::Unknown;
    }
    Dim::Fixed(span / stride + 1)
}

/// Strides, dilations and pads of spatial axis `i` of `spatial`.
fn window_params(
    strides: &Option<Vec<i64>>,
    dilations: &Option<Vec<i64>>,
    pads: &Option<Vec<i64>>,
    i: usize,
    spatial: usize,
) -> (i64, i64, (i64, i64)) {
    let at = |v: &Option<Vec<i64>>, i: usize, default: i64| {
        v.as_ref()
            .and_then(|v| v.get(i).copied())
            .unwrap_or(default)
    };
    let pads = (at(pads, i, 0), at(pads, i + spatial, 0));
    (at(strides, i, 1), at(dilations, i, 1), pads)
}

fn conv(x: &[Dim], w: &[Dim], attrs: &ConvAttrs) -> Option<Shape> {
    if x.len() < 3 || w.len() != x.len() {
        return None;
    }
    let spatial = x.len() - 2;
    let auto_pad = attrs.auto_pad.as_deref().unwrap_or("NOTSET");
    let mut out = vec![x[0].clone(), w[0].clone()];
    for i in 0..spatial {
        let (stride, dilation, pads) =
            window_params(&attrs.strides, &attrs.dilations, &attrs.pads, i, spatial);
        let kernel = attrs
            .kernel_shape
            .as_ref()
            .and_then(|k| k.get(i).copied())
            .or(fixed(&w[2 + i]));
        out.push(window(
            &x[2 + i],
            kernel,
            stride,
            dilation,
            pads,
            auto_pad,
            false,
        ));
    }
    Some(out)
}

fn pool(x: &[Dim], attrs: &PoolAttrs) -> Option<Shape> {
    if x.len() < 3 {
        return None;
    }
    let spatial = x.len() - 2;
    let auto_pad = attrs.auto_pad.as_deref().unwrap_or("NOTSET");
    let mut out = x[..2].to_vec();
    for i in 0..spatial {
        let (stride, dilation, pads) =
            window_params(&attrs.strides, &attrs.dilations, &attrs.pads, i, spatial);
        let kernel = attrs.kernel_shape.as_ref().and_then(|k| k.get(i).copied());
        let ceil = attrs.ceil_mode;
        out.push(window(
            &x[2 + i],
            kernel,
            stride,
            dilation,
            pads,
            auto_pad,
            ceil,
        ));
    }
    Some(out)
}

/// Lengths ONNX `Slice` keeps along a dimension of size `dim`.
fn slice_range(dim: i64, start: i64, end: i64, step: i64) -> Option<(i64, i64)> {
    if step == 0 {
        return None;
    }
    let wrap = |v: i64| if v < 0 { v.saturating_add(dim) } else { v };
    let (start, end) = if step > 0 {
        (wrap(start).clamp(0, dim), wrap(end).clamp(0, dim))
    } else {
        (wrap(start).clamp(0, dim - 1), wrap(end).clamp(-1, dim - 1))
    };
    Some((start, end))
}

fn slice_len(dim: i64, start: i64, end: i64, step: i64) -> Option<i64> {
    let (start, end) = slice_range(dim, start, end, step)?;
    let len = if step > 0 {
        (end - start + step - 1) / step
    } else {
        (start - end - step - 1) / -step
    };
    Some(len.max(0))
}

/// Shapes of the outputs of `node`, where they follow from its inputs.
fn output_shapes(g: &Graph, node: &Node, elems: &Elems) -> Vec<Option<Shape>> {
    let shape = |i: usize| node.input(i).and_then(|v| g.value(v).shape.clone());
    let ints = |i: usize| node.input(i).and_then(|v| all_fixed_elems(elems.get(&v)?));
    // The elements of 1-D input `i` as dimensions, as many unknown ones as
    // it has if they are not tracked.
    let dims = |i: usize| match elems.get(&node.input(i)?) {
        Some(e) => Some(e.iter().map(|e| e.to_dim(g)).collect::<Shape>()),
        None => match shape(i)?.as_slice() {
            [Dim::Fixed(n)] => Some(vec![Dim::Unknown; usize::try_from(*n).ok()?]),
            _ => None,
        },
    };
    // A target of input dimensions.
    let target = |dims: &[TargetDim]| -> Shape {
        dims.iter()
            .map(|d| match *d {
                TargetDim::Fixed(n) => Dim::Fixed(n),
                TargetDim::Of { input, axis } => shape(input)
                    .and_then(|s| s.get(axis).cloned())
                    .unwrap_or(Dim::Unknown),
                TargetDim::Infer => Dim::Unknown,
            })
            .collect()
    };
    let n_out = node.outputs().len();
    let first = |s: Option<Shape>| {
        let mut out = vec![None; n_out];
        if let Some(o) = out.first_mut() {
            *o = s;
        }
        out
    };
    match &node.op {
        Op::Unary(_)
        | Op::BatchNormalization { .. }
        | Op::Cast { .. }
        | Op::Clip
        | Op::CumSum { .. }
        | Op::DequantizeLinear(_)
        | Op::FakeQuantizeLinear(_)
        | Op::Gelu { .. }
        | Op::HardSigmoid { .. }
        | Op::Identity
        | Op::LayerNormalization { .. }
        | Op::LeakyRelu { .. }
        | Op::LogSoftmax { .. }
        | Op::QuantizeLinear(_)
        | Op::Silu
        | Op::Softmax { .. } => first(shape(0)),
        Op::DynamicQuantizeLinear => {
            let mut out = vec![shape(0), Some(Vec::new()), Some(Vec::new())];
            out.truncate(n_out);
            out
        }
        Op::Binary(_) | Op::Max | Op::Where => {
            let shapes: Option<Vec<Shape>> = node
                .inputs()
                .iter()
                .map(|v| g.value((*v)?).shape.clone())
                .collect();
            first(shapes.map(|s| broadcast(&s.iter().map(Vec::as_slice).collect::<Vec<_>>())))
        }
        Op::MatMul | Op::MatMulInteger | Op::Linear => {
            first(shape(0).zip(shape(1)).and_then(|(a, b)| matmul(&a, &b)))
        }
        Op::QLinearDynamic { .. } => {
            first(shape(0).zip(shape(1)).and_then(|(a, b)| matmul(&a, &b)))
        }
        Op::QLinearMatMulI8 => first(shape(0).zip(shape(3)).and_then(|(a, b)| matmul(&a, &b))),
        Op::Gemm {
            trans_a, trans_b, ..
        } => first(shape(0).zip(shape(1)).and_then(|(a, b)| {
            let (a, b) = (<[Dim; 2]>::try_from(a).ok()?, <[Dim; 2]>::try_from(b).ok()?);
            let m = a[usize::from(*trans_a)].clone();
            let n = b[usize::from(!*trans_b)].clone();
            Some(vec![m, n])
        })),
        Op::Conv(attrs) | Op::ConvInteger(attrs) | Op::ConvAct { conv: attrs, .. } => first(
            shape(0)
                .zip(shape(1))
                .and_then(|(x, w)| conv(&x, &w, attrs)),
        ),
        Op::MaxPool(attrs) | Op::AveragePool { pool: attrs, .. } => {
            let out = shape(0).and_then(|x| pool(&x, attrs));
            vec![out; n_out]
        }
        Op::GlobalAveragePool => first(shape(0).filter(|x| x.len() >= 2).map(|x| {
            let mut out = x[..2].to_vec();
            out.resize(x.len(), Dim::Fixed(1));
            out
        })),
        Op::Transpose { perm } => first(shape(0).and_then(|x| {
            match perm {
                Some(perm) => perm
                    .iter()
                    .map(|&p| x.get(usize::try_from(p).ok()?).cloned())
                    .collect(),
                None => Some(x.into_iter().rev().collect()),
            }
        })),
        Op::Reshape => first(reshape_target(shape(0).as_deref(), dims(1))),
        Op::ReshapeTo { dims } => first(Some(target(dims))),
        Op::Flatten { axis } => first(shape(0).and_then(|x| {
            let a = if *axis < 0 {
                *axis + x.len() as i64
            } else {
                *axis
            };
            let a = usize::try_from(a).ok().filter(|&a| a <= x.len())?;
            let part = |dims: &[Dim]| product(dims).map_or(Dim::Unknown, Dim::Fixed);
            Some(vec![part(&x[..a]), part(&x[a..])])
        })),
        Op::Unsqueeze { axes } => first(shape(0).and_then(|x| {
            let axes = axes.clone().or_else(|| ints(1))?;
            let rank = x.len() + axes.len();
            let mut axes: Vec<usize> = axes
                .iter()
                .map(|&a| norm_axis(a, rank))
                .collect::<Option<_>>()?;
            axes.sort_unstable();
            let mut out = x;
            for a in axes {
                (a <= out.len()).then(|| out.insert(a, Dim::Fixed(1)))?;
            }
            Some(out)
        })),
        Op::Squeeze { axes } => first(shape(0).and_then(|x| {
            let axes = match axes.clone().or_else(|| ints(1)) {
                Some(axes) => axes
                    .iter()
                    .map(|&a| norm_axis(a, x.len()))
                    .collect::<Option<Vec<_>>>()?,
                // Every axis of size 1, if they are all known.
                None => {
                    if x.iter().any(|d| fixed(d).is_none()) {
                        return None;
                    }
                    (0..x.len()).filter(|&i| x[i] == Dim::Fixed(1)).collect()
                }
            };
            Some(
                x.into_iter()
                    .enumerate()
                    .filter(|(i, _)| !axes.contains(i))
                    .map(|(_, d)| d)
                    .collect(),
            )
        })),
        Op::Concat { axis } => {
            let shapes: Option<Vec<Shape>> = node
                .inputs()
                .iter()
                .map(|v| g.value((*v)?).shape.clone())
                .collect();
            first(shapes.and_then(|shapes| {
                let rank = shapes.first()?.len();
                if shapes.iter().any(|s| s.len() != rank) {
                    return None;
                }
                let a = norm_axis(*axis, rank)?;
                Some(
                    (0..rank)
                        .map(|i| {
                            if i == a {
                                let sizes: Option<Vec<i64>> =
                                    shapes.iter().map(|s| fixed(&s[i])).collect();
                                sizes.map_or(Dim::Unknown, |s| Dim::Fixed(s.iter().sum()))
                            } else {
                                shapes
                                    .iter()
                                    .fold(Dim::Unknown, |acc, s| unify(&acc, &s[i]))
                            }
                        })
                        .collect(),
                )
            }))
        }
        Op::Gather { axis } => first(shape(0).zip(shape(1)).and_then(|(data, indices)| {
            let a = norm_axis(*axis, data.len())?;
            let mut out = data[..a].to_vec();
            out.extend(indices);
            out.extend_from_slice(&data[a + 1..]);
            Some(out)
        })),
        Op::GatherElements { .. } => first(shape(1)),
        Op::Shape { start, end } => first(shape(0).map(|x| {
            let (start, end) = shape_range(x.len(), *start, *end);
            vec![Dim::Fixed(end.saturating_sub(start) as i64)]
        })),
        Op::Size => first(Some(Vec::new())),
        Op::Slice => first(shape(0).and_then(|x| {
            // Starts, ends and steps may be known only in part, or not at
            // all; the axes they apply to have to be.
            let n = match elems.get(&node.input(1)?) {
                Some(starts) => starts.len(),
                None => usize::try_from(fixed(shape(1)?.first()?)?).ok()?,
            };
            let axes = match node.input(3) {
                Some(_) => ints(3)?,
                None => (0..n as i64).collect(),
            };
            let elem = |input: usize, i: usize, default: i64| match node.input(input) {
                Some(v) => elems.get(&v).and_then(|e| e.get(i)?.fixed()),
                None => Some(default),
            };
            let mut out = x.clone();
            for i in 0..n {
                let a = norm_axis(*axes.get(i)?, x.len())?;
                let known = (elem(1, i, 0), elem(2, i, 0), elem(4, i, 1));
                out[a] = match (&x[a], known) {
                    (Dim::Fixed(n), (Some(start), Some(end), Some(step))) => {
                        slice_len(*n, start, end, step).map_or(Dim::Unknown, Dim::Fixed)
                    }
                    // The whole of an unknown dimension.
                    (d, (Some(0), Some(end), Some(1))) if end >= i32::MAX as i64 => d.clone(),
                    _ => Dim::Unknown,
                };
            }
            Some(out)
        })),
        Op::Split { axis, split } => {
            let Some(x) = shape(0) else {
                return vec![None; n_out];
            };
            let Some(a) = norm_axis(*axis, x.len()) else {
                return vec![None; n_out];
            };
            let sizes = match node.input(1) {
                Some(_) => ints(1),
                None => split.clone().or_else(|| {
                    let n = fixed(&x[a])?;
                    let parts = n_out as i64;
                    (parts > 0 && n % parts == 0).then(|| vec![n / parts; n_out])
                }),
            };
            match sizes.filter(|s| s.len() == n_out) {
                Some(sizes) => sizes
                    .into_iter()
                    .map(|s| {
                        let mut out = x.clone();
                        out[a] = Dim::Fixed(s);
                        Some(out)
                    })
                    .collect(),
                None => vec![None; n_out],
            }
        }
        Op::Reduce {
            reduction,
            axes,
            keepdims,
        } => first(shape(0).and_then(|x| {
            let axes = match axes {
                Some(axes) => Some(axes.clone()),
                None if node.input(1).is_some() => Some(ints(1)?),
                None => None,
            };
            let axes: Vec<usize> = match axes {
                Some(axes) if !axes.is_empty() => axes
                    .iter()
                    .map(|&a| norm_axis(a, x.len()))
                    .collect::<Option<_>>()?,
                // lele's ReduceMean and ReduceL2 take no computed axes.
                _ if matches!(reduction, Reduction::Mean | Reduction::L2)
                    && node.input(1).is_some() =>
                {
                    return None;
                }
                _ => (0..x.len()).collect(),
            };
            Some(reduce(&x, &axes, *keepdims))
        })),
        Op::ArgMax { axis, keepdims } => {
            first(shape(0).and_then(|x| Some(reduce(&x, &[norm_axis(*axis, x.len())?], *keepdims))))
        }
        Op::TopK { axis, .. } => {
            let out = shape(0).zip(ints(1)).and_then(|(x, k)| {
                let a = norm_axis(*axis, x.len())?;
                let mut out = x;
                out[a] = Dim::Fixed(*k.first()?);
                Some(out)
            });
            vec![out; n_out]
        }
        Op::Range => {
            let scalar = |i: usize| elems.get(&node.input(i)?)?.first().cloned();
            let len = match (scalar(0), scalar(1), scalar(2)) {
                (Some(Elem::Fixed(start)), Some(Elem::Fixed(limit)), Some(Elem::Fixed(delta)))
                    if delta != 0 =>
                {
                    let span = limit - start;
                    let up = span % delta != 0 && (span < 0) == (delta < 0);
                    Dim::Fixed((span / delta + i64::from(up)).max(0))
                }
                (Some(Elem::Fixed(0)), Some(limit), Some(Elem::Fixed(1))) => limit.to_dim(g),
                _ => Dim::Unknown,
            };
            first(Some(vec![len]))
        }
        Op::ConstantOfShape { .. } => first(dims(0)),
        Op::ConstantOfShapeTo { dims, .. } => first(Some(target(dims))),
        Op::Constant { value } => first(Some(value.dims.iter().map(|&d| Dim::Fixed(d)).collect())),
        Op::Expand => first(
            shape(0)
                .zip(dims(1))
                .map(|(x, target)| broadcast(&[&x, &target])),
        ),
        Op::ExpandTo { dims } => first(shape(0).map(|x| broadcast(&[&x, &target(dims)]))),
        Op::Tile => first(shape(0).and_then(|x| {
            let repeats: Vec<Option<i64>> = match elems.get(&node.input(1)?) {
                Some(r) => r.iter().map(Elem::fixed).collect(),
                None => vec![None; x.len()],
            };
            (x.len() == repeats.len()).then(|| {
                x.iter()
                    .zip(&repeats)
                    .map(|(d, r)| match (d, r) {
                        (_, Some(1)) => d.clone(),
                        (Dim::Fixed(n), Some(r)) => Dim::Fixed(n * r),
                        _ => Dim::Unknown,
                    })
                    .collect()
            })
        })),
        // Without `axes`, which lele does not take.
        Op::Pad { .. } if node.input(3).is_none() => first(shape(0).and_then(|x| {
            let rank = x.len();
            let pads: Vec<Option<i64>> = match elems.get(&node.input(1)?) {
                Some(pads) => pads.iter().map(Elem::fixed).collect(),
                None => vec![None; 2 * rank],
            };
            (pads.len() == 2 * rank).then(|| {
                x.iter()
                    .enumerate()
                    .map(|(i, d)| match (d, pads[i].zip(pads[i + rank])) {
                        (_, Some((0, 0))) => d.clone(),
                        (Dim::Fixed(n), Some((a, b))) => Dim::Fixed(n + a + b),
                        _ => Dim::Unknown,
                    })
                    .collect()
            })
        })),
        Op::If {
            then_branch,
            else_branch,
        } => (0..n_out)
            .map(|i| {
                let branch = |b: &Graph| b.value(*b.outputs.get(i)?).shape.clone();
                let (a, b) = (branch(then_branch)?, branch(else_branch)?);
                (a.len() == b.len()).then(|| {
                    a.iter()
                        .zip(&b)
                        .map(|(a, b)| if a == b { a.clone() } else { Dim::Unknown })
                        .collect()
                })
            })
            .collect(),
        Op::Einsum { equation } if equation.replace(' ', "") == "bs,d->bsd" => {
            first(shape(0).zip(shape(1)).and_then(|(a, b)| {
                Some(vec![
                    a.first()?.clone(),
                    a.get(1)?.clone(),
                    b.first()?.clone(),
                ])
            }))
        }
        Op::EmbeddingConcat { .. } => first(shape(1).zip(dims(0)).and_then(|(w, fill)| {
            let rows = match (fixed(w.first()?), fixed(fill.first()?)) {
                (Some(a), Some(b)) => Dim::Fixed(a + b),
                _ => Dim::Unknown,
            };
            let mut out = w;
            out[0] = rows;
            Some(out)
        })),
        _ => vec![None; n_out],
    }
}

/// `Shape`'s `start` and `end` resolved against rank `rank`.
fn shape_range(rank: usize, start: Option<i64>, end: Option<i64>) -> (usize, usize) {
    let resolve = |v: i64| {
        let v = if v < 0 { v + rank as i64 } else { v };
        v.clamp(0, rank as i64) as usize
    };
    let start = resolve(start.unwrap_or(0));
    let end = end.map_or(rank, resolve);
    (start, end.max(start))
}

fn reduce(x: &[Dim], axes: &[usize], keepdims: bool) -> Shape {
    x.iter()
        .enumerate()
        .filter_map(|(i, d)| match (axes.contains(&i), keepdims) {
            (false, _) => Some(d.clone()),
            (true, true) => Some(Dim::Fixed(1)),
            (true, false) => None,
        })
        .collect()
}

/// The shape a `Reshape` of `x` to `target` produces.
fn reshape_target(x: Option<&[Dim]>, target: Option<Shape>) -> Option<Shape> {
    let target = target?;
    let mut out: Shape = target
        .iter()
        .enumerate()
        .map(|(i, d)| match d {
            // 0 copies the input's dimension.
            Dim::Fixed(0) => x.and_then(|x| x.get(i).cloned()).unwrap_or(Dim::Unknown),
            Dim::Fixed(-1) => Dim::Unknown,
            d => d.clone(),
        })
        .collect();
    let infer = target.iter().position(|d| *d == Dim::Fixed(-1));
    if let (Some(i), Some(total)) = (infer, x.and_then(product)) {
        let others: Vec<Dim> = out
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, d)| d.clone())
            .collect();
        if let Some(rest) = product(&others).filter(|&p| p > 0 && total % p == 0) {
            out[i] = Dim::Fixed(total / rest);
        }
    }
    Some(out)
}

/// Elements of output 0 of `node`, for the integer tensors it computes from
/// tracked ones.
fn output_elems(g: &Graph, node: &Node, elems: &Elems) -> Option<Vec<Elem>> {
    let input = |i: usize| elems.get(&node.input(i)?);
    let ints = |i: usize| all_fixed_elems(input(i)?);
    match &node.op {
        Op::Shape { start, end } => {
            let x = node.input(0)?;
            let shape = g.value(x).shape.as_ref()?;
            let (start, end) = shape_range(shape.len(), *start, *end);
            Some(
                (start..end)
                    .map(|axis| match shape[axis] {
                        Dim::Fixed(n) => Elem::Fixed(n),
                        _ => Elem::Dim(x, axis),
                    })
                    .collect(),
            )
        }
        Op::Size => {
            let x = g.value(node.input(0)?).shape.as_ref()?;
            Some(vec![Elem::Fixed(product(x)?)])
        }
        Op::Cast {
            to: DType::I64 | DType::I32,
        }
        | Op::Identity
        | Op::Reshape
        | Op::Flatten { .. }
        | Op::Squeeze { .. }
        | Op::Unsqueeze { .. } => input(0).cloned(),
        Op::Gather { axis: 0 } => {
            let data = input(0)?;
            let len = data.len() as i64;
            ints(1)?
                .iter()
                .map(|&i| {
                    let i = if i < 0 { i + len } else { i };
                    data.get(usize::try_from(i).ok()?).cloned()
                })
                .collect()
        }
        Op::Concat { axis: 0 | -1 } => {
            let parts: Option<Vec<&Vec<Elem>>> = (0..node.inputs().len()).map(input).collect();
            Some(parts?.into_iter().flatten().cloned().collect())
        }
        Op::Slice => {
            let data = input(0)?;
            let (start, end) = (*ints(1)?.first()?, *ints(2)?.first()?);
            if node.input(3).is_some() && !matches!(ints(3)?.as_slice(), [0] | [-1]) {
                return None;
            }
            let step = match node.input(4) {
                Some(_) => *ints(4)?.first()?,
                None => 1,
            };
            let (start, end) = slice_range(data.len() as i64, start, end, step)?;
            let mut out = Vec::new();
            let mut i = start;
            while (step > 0 && i < end) || (step < 0 && i > end) {
                out.push(data[i as usize].clone());
                i += step;
            }
            Some(out)
        }
        Op::Binary(op @ (Binary::Add | Binary::Sub | Binary::Mul | Binary::Div)) => {
            let (a, b) = (input(0)?, input(1)?);
            let len = a.len().max(b.len());
            if (a.len() != len && a.len() != 1) || (b.len() != len && b.len() != 1) {
                return None;
            }
            let at = |v: &Vec<Elem>, i: usize| v[if v.len() == 1 { 0 } else { i }].clone();
            Some((0..len).map(|i| arith(*op, &at(a, i), &at(b, i))).collect())
        }
        _ => None,
    }
}

/// `a op b` for integers, as far as it is known.
fn arith(op: Binary, a: &Elem, b: &Elem) -> Elem {
    use Elem::Fixed;
    match (op, a, b) {
        (Binary::Add | Binary::Sub, d, Fixed(0))
        | (Binary::Add, Fixed(0), d)
        | (Binary::Mul | Binary::Div, d, Fixed(1))
        | (Binary::Mul, Fixed(1), d) => d.clone(),
        (_, Fixed(x), Fixed(y)) => {
            let r = match op {
                Binary::Add => x.checked_add(*y),
                Binary::Sub => x.checked_sub(*y),
                Binary::Mul => x.checked_mul(*y),
                // Integer division truncates.
                _ => x.checked_div(*y),
            };
            r.map_or(Elem::Unknown, Fixed)
        }
        _ => Elem::Unknown,
    }
}

/// Replaces the integer tensors of `g` whose shape and elements are known
/// by initializers.
fn fold_ints(g: &mut Graph, elems: &Elems) {
    for id in g.node_ids() {
        let node = g.node(id);
        if node.outputs().len() != 1 || matches!(node.op, Op::Constant { .. }) {
            continue;
        }
        let Some(v) = node.output(0) else {
            continue;
        };
        let value = g.value(v);
        let (Some(values), Some(dims)) = (
            elems.get(&v).and_then(|e| all_fixed_elems(e)),
            value.shape.as_deref().and_then(all_fixed),
        ) else {
            continue;
        };
        // Empty tensors are not written to the weights file.
        if values.is_empty()
            || dims.iter().product::<i64>() != values.len() as i64
            || g.outputs.contains(&v)
        {
            continue;
        }
        let tensor = match value.dtype {
            DType::I64 => TensorProto {
                dims,
                data_type: DType::I64.to_onnx(),
                int64_data: values,
                ..Default::default()
            },
            DType::I32 => TensorProto {
                dims,
                data_type: DType::I32.to_onnx(),
                int32_data: values.iter().map(|&x| x as i32).collect(),
                ..Default::default()
            },
            _ => continue,
        };
        g.remove_node(id);
        g.add_initializer(v, tensor);
    }
}

/// Replaces `Reshape`, `Expand` and `ConstantOfShape` nodes whose target is
/// computed at run time but known up to dimensions of values by
/// [`Op::ReshapeTo`], [`Op::ExpandTo`] and [`Op::ConstantOfShapeTo`].
fn rewrite_targets(g: &mut Graph, elems: &Elems) {
    for id in g.node_ids() {
        let node = g.node(id);
        let (data, target) = match node.op {
            Op::Reshape | Op::Expand => match node.input(0) {
                Some(x) => (Some(x), node.input(1)),
                None => continue,
            },
            Op::ConstantOfShape { .. } => (None, node.input(0)),
            _ => continue,
        };
        let Some(target) = target else {
            continue;
        };
        // Constant targets are left as they are.
        let constant = g
            .producer(target)
            .is_some_and(|p| matches!(p.op, Op::Constant { .. }));
        if g.data(target).is_some() || constant {
            continue;
        }
        // Fusion makes `Concat([weight, ConstantOfShape(...)])` an
        // `EmbeddingConcat`.
        let concat = node.output(0).is_some_and(|out| {
            g.value(out)
                .consumers()
                .iter()
                .any(|&c| matches!(g.node(c).op, Op::Concat { .. }))
        });
        if data.is_none() && concat {
            continue;
        }
        let Some(target) = elems.get(&target) else {
            continue;
        };
        let reshape = matches!(node.op, Op::Reshape);
        let mut inputs: Vec<ValueId> = data.into_iter().collect();
        let dims: Option<Vec<TargetDim>> = target
            .iter()
            .enumerate()
            .map(|(i, e)| target_dim(g, e, i, reshape, &mut inputs))
            .collect();
        let Some(dims) = dims else {
            continue;
        };
        if dims.iter().filter(|d| **d == TargetDim::Infer).count() > 1 {
            continue;
        }
        let op = match &node.op {
            Op::Reshape => Op::ReshapeTo { dims },
            Op::Expand => Op::ExpandTo { dims },
            Op::ConstantOfShape { value } => Op::ConstantOfShapeTo {
                value: value.clone(),
                dims,
            },
            _ => unreachable!(),
        };
        let proto = node.proto.clone();
        g.fuse(&[id], op, proto, inputs.into_iter().map(Some).collect());
    }
}

/// Element `i` of a target as a [`TargetDim`], taking dimensions from
/// `inputs` and adding to them the values whose dimensions it needs. The
/// first input is the tensor reshaped or expanded, if any. `reshape` reads
/// the target as `Reshape` does, where 0 copies the input's dimension and -1
/// infers one; elsewhere a dimension that is not known fails.
fn target_dim(
    g: &Graph,
    e: &Elem,
    i: usize,
    reshape: bool,
    inputs: &mut Vec<ValueId>,
) -> Option<TargetDim> {
    let dim = |v: ValueId, axis: usize| {
        let shape = g.value(v).shape.as_ref()?;
        shape.get(axis).cloned()
    };
    // An input with dimension `d`, by value for a symbolic one.
    let find = |inputs: &[ValueId], d: &Dim| {
        inputs.iter().enumerate().find_map(|(input, &v)| {
            let shape = g.value(v).shape.as_ref()?;
            let axis = shape.iter().position(|s| s == d)?;
            Some(TargetDim::Of { input, axis })
        })
    };
    let unknown = reshape.then_some(TargetDim::Infer);
    match e {
        Elem::Fixed(0) if reshape => {
            let x = *inputs.first()?;
            Some(match dim(x, i) {
                Some(Dim::Fixed(n)) => TargetDim::Fixed(n),
                Some(_) => TargetDim::Of { input: 0, axis: i },
                None => TargetDim::Infer,
            })
        }
        Elem::Fixed(n) if *n >= 0 => Some(TargetDim::Fixed(*n)),
        Elem::Fixed(_) => unknown,
        Elem::Dim(v, axis) => {
            if let Some(input) = inputs.iter().position(|u| u == v) {
                return Some(TargetDim::Of { input, axis: *axis });
            }
            // A symbolic dimension an input already has.
            if let Some(d @ Dim::Sym(_)) = dim(*v, *axis)
                && let Some(t) = find(inputs, &d)
            {
                return Some(t);
            }
            inputs.push(*v);
            Some(TargetDim::Of {
                input: inputs.len() - 1,
                axis: *axis,
            })
        }
        Elem::Sym(s) => find(inputs, &Dim::Sym(s.clone())).or(unknown),
        Elem::Unknown => unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::passes::infer_dtypes;
    use crate::model::onnx_proto::{
        AttributeProto, GraphProto, NodeProto, TensorShapeProto, TypeProto, ValueInfoProto,
        tensor_shape_proto, type_proto,
    };

    fn node(op: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
        NodeProto {
            op_type: op.to_string(),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    fn input(name: &str, dims: &[&str]) -> ValueInfoProto {
        let dim = |d: &&str| tensor_shape_proto::Dimension {
            value: Some(match d.parse::<i64>() {
                Ok(n) => tensor_shape_proto::dimension::Value::DimValue(n),
                Err(_) => tensor_shape_proto::dimension::Value::DimParam(d.to_string()),
            }),
            ..Default::default()
        };
        ValueInfoProto {
            name: name.to_string(),
            r#type: Some(TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: DType::F32.to_onnx(),
                    shape: Some(TensorShapeProto {
                        dim: dims.iter().map(dim).collect(),
                    }),
                })),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    /// A graph output of no stated type.
    fn output(name: &str) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_string(),
            ..Default::default()
        }
    }

    fn ints(name: &str, v: &[i64]) -> TensorProto {
        TensorProto {
            name: name.to_string(),
            data_type: DType::I64.to_onnx(),
            dims: vec![v.len() as i64],
            int64_data: v.to_vec(),
            ..Default::default()
        }
    }

    fn run(graph: GraphProto) -> Graph {
        let mut g = Graph::from_onnx(graph);
        infer_dtypes(&mut g);
        fold_shapes(&mut g);
        g
    }

    /// The attention reshape `x[batch, seq, 768] -> [batch, seq, 12, 64]`,
    /// with the target assembled from `Shape(x)`.
    #[test]
    fn reshape_target_from_shape_becomes_static() {
        let mut nodes = vec![node("Shape", &["x"], &["s"])];
        for (i, idx) in ["i0", "i1"].iter().enumerate() {
            let g = format!("g{i}");
            let u = format!("u{i}");
            nodes.push(node("Gather", &["s", idx], &[&g]));
            nodes.push(NodeProto {
                attribute: vec![AttributeProto {
                    name: "axes".to_string(),
                    ints: vec![0],
                    ..Default::default()
                }],
                ..node("Unsqueeze", &[&g], &[&u])
            });
        }
        nodes.push(NodeProto {
            attribute: vec![AttributeProto {
                name: "axis".to_string(),
                i: 0,
                ..Default::default()
            }],
            ..node("Concat", &["u0", "u1", "heads"], &["target"])
        });
        nodes.push(node("Reshape", &["x", "target"], &["y"]));
        let scalar = |name: &str, v: i64| TensorProto {
            dims: vec![],
            ..ints(name, &[v])
        };
        let mut g = run(GraphProto {
            node: nodes,
            input: vec![input("x", &["batch", "seq", "768"])],
            output: vec![output("y")],
            initializer: vec![scalar("i0", 0), scalar("i1", 1), ints("heads", &[12, 64])],
            ..Default::default()
        });
        super::super::eliminate_dead_code(&mut g);
        let ops: Vec<_> = g.nodes().map(|(_, n)| n.op.clone()).collect();
        assert_eq!(ops.len(), 1, "{ops:?}");
        let Op::ReshapeTo { dims } = &ops[0] else {
            panic!("{ops:?}");
        };
        use TargetDim::*;
        let x = |axis| Of { input: 0, axis };
        assert_eq!(dims, &[x(0), x(1), Fixed(12), Fixed(64)]);
        let y = g.find_value("y").unwrap();
        assert_eq!(
            g.value(y).shape,
            Some(vec![
                Dim::Sym("batch".into()),
                Dim::Sym("seq".into()),
                Dim::Fixed(12),
                Dim::Fixed(64)
            ])
        );
    }

    /// A shape computed from fixed dimensions folds to an initializer.
    #[test]
    fn folds_shape_of_fixed_dimensions() {
        let g = run(GraphProto {
            node: vec![
                node("Shape", &["x"], &["s"]),
                node("Mul", &["s", "two"], &["m"]),
                node("Expand", &["x", "m"], &["y"]),
            ],
            input: vec![input("x", &["1", "3"])],
            output: vec![output("y")],
            initializer: vec![ints("two", &[2, 1])],
            ..Default::default()
        });
        let m = g.find_value("m").unwrap();
        assert_eq!(g.data(m).unwrap().int64_data, [2, 3]);
        let y = g.find_value("y").unwrap();
        assert_eq!(g.value(y).shape, Some(vec![Dim::Fixed(2), Dim::Fixed(3)]));
    }

    /// Silero reverses the rows of its `[2, 2]` STFT padding. The elements
    /// of tensors of higher rank are not tracked: they were taken for a
    /// vector and reversed whole.
    #[test]
    fn leaves_matrices_to_constant_folding() {
        let g = run(GraphProto {
            node: vec![
                node("Reshape", &["pads", "rows"], &["m"]),
                node("Slice", &["m", "start", "end", "axis", "step"], &["y"]),
            ],
            output: vec![output("y")],
            initializer: vec![
                ints("pads", &[1, 2, 3, 4]),
                ints("rows", &[-1, 2]),
                ints("start", &[-1]),
                ints("end", &[i64::MIN]),
                ints("axis", &[0]),
                ints("step", &[-1]),
            ],
            ..Default::default()
        });
        let y = g.find_value("y").unwrap();
        assert!(g.data(y).is_none());
        assert_eq!(g.value(y).shape, Some(vec![Dim::Fixed(2), Dim::Fixed(2)]));
    }

    /// An input of shape `[?, 3]`.
    fn input_of_unknown_rows(name: &str) -> ValueInfoProto {
        let mut x = input(name, &["n", "3"]);
        if let Some(type_proto::Value::TensorType(t)) =
            x.r#type.as_mut().and_then(|t| t.value.as_mut())
        {
            t.shape.as_mut().unwrap().dim[0].value = None;
        }
        x
    }

    /// Unknown dimensions stay unknown: `Shape` of an input of shape
    /// `[?, 3]` is not folded.
    #[test]
    fn keeps_shape_of_unknown_dimensions() {
        let g = run(GraphProto {
            node: vec![node("Shape", &["x"], &["s"])],
            input: vec![input_of_unknown_rows("x")],
            output: vec![output("s")],
            ..Default::default()
        });
        assert_eq!(g.nodes().count(), 1);
    }

    /// `Expand` and `ConstantOfShape` to the shape of another tensor take
    /// its dimensions, unknown ones included, instead of calling `Shape`.
    #[test]
    fn expand_and_constant_of_shape_take_dimensions_of_other_inputs() {
        let mut g = run(GraphProto {
            node: vec![
                node("Shape", &["x"], &["s"]),
                node("Expand", &["y", "s"], &["e"]),
                node("ConstantOfShape", &["s"], &["z"]),
            ],
            input: vec![input_of_unknown_rows("x"), input("y", &["1", "1"])],
            output: vec![output("e"), output("z")],
            ..Default::default()
        });
        super::super::eliminate_dead_code(&mut g);
        let (x, y) = (g.find_value("x").unwrap(), g.find_value("y").unwrap());
        let nodes: Vec<_> = g
            .nodes()
            .map(|(_, n)| (n.op.clone(), n.inputs().to_vec()))
            .collect();
        use TargetDim::*;
        let rows = |input| Of { input, axis: 0 };
        assert_eq!(nodes.len(), 2, "{nodes:?}");
        assert!(matches!(&nodes[0].0, Op::ExpandTo { dims } if *dims == [rows(1), Fixed(3)]));
        assert_eq!(nodes[0].1, [Some(y), Some(x)]);
        assert!(
            matches!(&nodes[1].0, Op::ConstantOfShapeTo { dims, .. } if *dims == [rows(0), Fixed(3)])
        );
        assert_eq!(nodes[1].1, [Some(x)]);
        let e = g.find_value("e").unwrap();
        assert_eq!(g.value(e).shape, Some(vec![Dim::Unknown, Dim::Fixed(3)]));
    }

    /// `Range(0, n, 1)` has `n` elements, and a `Slice` with bounds known
    /// only at run time keeps the dimensions it does not slice.
    #[test]
    fn infers_range_and_partly_known_slice() {
        let scalar = |name: &str, v: i64| TensorProto {
            dims: vec![],
            ..ints(name, &[v])
        };
        let g = run(GraphProto {
            node: vec![
                node("Shape", &["x"], &["s"]),
                node("Gather", &["s", "zero"], &["n"]),
                node("Range", &["zero", "n", "one"], &["r"]),
                node("Slice", &["x", "start", "end", "axis"], &["y"]),
            ],
            input: vec![
                input("x", &["n", "8"]),
                input("start", &["1"]),
                input("end", &["1"]),
            ],
            output: vec![output("r"), output("y")],
            initializer: vec![scalar("zero", 0), scalar("one", 1), ints("axis", &[1])],
            ..Default::default()
        });
        let shape = |name: &str| g.value(g.find_value(name).unwrap()).shape.clone().unwrap();
        assert_eq!(shape("r"), [Dim::Sym("n".into())]);
        assert_eq!(shape("y"), [Dim::Sym("n".into()), Dim::Unknown]);
    }

    #[test]
    fn infers_convolution_and_pooling_sizes() {
        let mut conv = node("Conv", &["x", "w"], &["c"]);
        conv.attribute = vec![
            AttributeProto {
                name: "strides".to_string(),
                ints: vec![2, 2],
                ..Default::default()
            },
            AttributeProto {
                name: "pads".to_string(),
                ints: vec![1, 1, 1, 1],
                ..Default::default()
            },
        ];
        let mut pool = node("MaxPool", &["c"], &["p"]);
        pool.attribute = vec![AttributeProto {
            name: "kernel_shape".to_string(),
            ints: vec![2, 2],
            ..Default::default()
        }];
        let w = TensorProto {
            name: "w".to_string(),
            data_type: DType::F32.to_onnx(),
            dims: vec![8, 3, 3, 3],
            float_data: vec![0.0; 8 * 27],
            ..Default::default()
        };
        let g = run(GraphProto {
            node: vec![conv, pool],
            input: vec![input("x", &["1", "3", "32", "seq"])],
            output: vec![output("p")],
            initializer: vec![w],
            ..Default::default()
        });
        let shape = |name: &str| g.value(g.find_value(name).unwrap()).shape.clone().unwrap();
        use Dim::*;
        assert_eq!(shape("c"), [Fixed(1), Fixed(8), Fixed(16), Unknown]);
        assert_eq!(shape("p"), [Fixed(1), Fixed(8), Fixed(15), Unknown]);
    }
}
