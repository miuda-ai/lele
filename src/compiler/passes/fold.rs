//! Constant folding.

use crate::compiler::ir::{Binary, DType, Graph, Op, QuantAttrs, Unary};
use crate::model::onnx_proto::TensorProto;
use crate::model::tensor_to_array;
use crate::tensor::TensorView;
use std::collections::{HashMap, HashSet};

/// A tensor known at compile time. Values are held as f32 whatever the
/// element type; `dtype` is the type they stand for.
#[derive(Clone, Debug)]
struct Const {
    data: Vec<f32>,
    /// The exact values of an int64, int32 or bool tensor, where known: f32
    /// does not hold integers beyond 2^24, so those are only known as read
    /// from the model or computed by integer folding.
    ints: Option<Vec<i64>>,
    shape: Vec<usize>,
    dtype: DType,
}

impl Const {
    /// The contents of `t`, if they can be read.
    fn of(t: &TensorProto) -> Option<Self> {
        let (data, shape) = tensor_to_array(t).ok()?;
        if data.is_empty() {
            return None;
        }
        let dtype = DType::from_onnx(t.data_type);
        let ints = is_integer(dtype).then(|| exact_ints(t)).flatten();
        Some(Const {
            data,
            ints,
            shape,
            dtype,
        })
    }
}

/// Whether lele computes tensors of `t` as integers (in i64).
fn is_integer(t: DType) -> bool {
    matches!(t, DType::I64 | DType::I32 | DType::Bool)
}

/// The values of an int64, int32 or bool tensor.
fn exact_ints(t: &TensorProto) -> Option<Vec<i64>> {
    if !t.int64_data.is_empty() {
        return Some(t.int64_data.clone());
    }
    if !t.int32_data.is_empty() {
        return Some(t.int32_data.iter().map(|&x| x as i64).collect());
    }
    let raw = &t.raw_data;
    Some(match DType::from_onnx(t.data_type) {
        DType::I64 => raw
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&b| i64::from_le_bytes(b))
            .collect(),
        DType::I32 => {
            let words = raw.as_chunks::<4>().0.iter();
            words.map(|&b| i32::from_le_bytes(b) as i64).collect()
        }
        DType::Bool => raw.iter().map(|&b| (b != 0) as i64).collect(),
        _ => return None,
    })
}

/// `data`, computed in f32, as the integers it stands for, if it holds them
/// exactly: integral values of at most 2^24 in magnitude. A larger integer may
/// have been rounded on its way through f32; a smaller one was not.
fn ints_held_exactly(data: &[f32]) -> Option<Vec<i64>> {
    const EXACT: f32 = (1 << 24) as f32;
    data.iter()
        .map(|&x| (x.fract() == 0.0 && x.abs() <= EXACT).then_some(x as i64))
        .collect()
}

/// Known tensors by name, so that subgraphs can look up those of their
/// enclosing graphs.
type Consts = HashMap<String, Const>;

/// Replaces nodes whose inputs are all known at compile time by initializers
/// holding their result.
pub(crate) fn fold_constants(g: &mut Graph) {
    fold_graph(g, &Consts::new());
}

/// Folds `g` given the tensors `inherited` from enclosing graphs, then its
/// subgraphs.
fn fold_graph(g: &mut Graph, inherited: &Consts) {
    let mut consts = inherited.clone();
    // 1. Initializers, overriding inherited ones of the same name.
    for &v in g.initializers() {
        let t = g.data(v).expect("initializer without data");
        if let Some(c) = Const::of(t) {
            consts.insert(g.value(v).name.clone(), c);
        }
    }
    // 2. Constant nodes. They stay in the graph; code generation reads them.
    for (_, node) in g.nodes() {
        if let Op::Constant { value } = &node.op
            && let Some(out) = node.output(0)
            && let Some(c) = Const::of(value)
        {
            consts.insert(g.value(out).name.clone(), c);
        }
    }

    // 3. One pass in execution order folds whole constant chains.
    // An initializer of an input's name is only its default value.
    let graph_inputs: HashSet<_> = g.inputs.iter().copied().collect();
    for id in g.node_ids() {
        let node = g.node(id);
        if matches!(node.op, Op::Constant { .. }) || node.inputs().is_empty() {
            continue;
        }
        let inputs: Option<Vec<&Const>> = node
            .inputs()
            .iter()
            .map(|v| {
                let v = (*v)?;
                if graph_inputs.contains(&v) {
                    return None;
                }
                consts.get(&g.value(v).name)
            })
            .collect();
        let Some(inputs) = inputs else {
            continue;
        };
        let Some(out) = node.output(0) else {
            continue;
        };
        let dtype = output_dtype(&node.op, &inputs);
        let folded = match fold_int(&node.op, &inputs) {
            Some((ints, shape)) => {
                Some((ints.iter().map(|&x| x as f32).collect(), Some(ints), shape))
            }
            None => fold_op(&node.op, &inputs).map(|(data, shape)| {
                let ints = is_integer(dtype)
                    .then(|| ints_held_exactly(&data))
                    .flatten();
                (data, ints, shape)
            }),
        };
        // Empty tensors are not written to the weights file.
        let Some((data, ints, shape)) = folded.filter(|(d, _, _)| !d.is_empty()) else {
            continue;
        };

        let name = g.value(out).name.clone();
        let dims = shape.iter().map(|&s| s as i64).collect();
        let exact = || {
            ints.clone()
                .unwrap_or_else(|| data.iter().map(|&x| x as i64).collect())
        };
        let tensor = if dtype == DType::I64 {
            TensorProto {
                name: name.clone(),
                dims,
                data_type: DType::I64.to_onnx(),
                int64_data: exact(),
                ..Default::default()
            }
        } else if dtype == DType::Bool && ints.is_some() {
            TensorProto {
                name: name.clone(),
                dims,
                data_type: DType::Bool.to_onnx(),
                raw_data: exact().iter().map(|&x| (x != 0) as u8).collect(),
                ..Default::default()
            }
        } else {
            // Every other type is stored as float.
            TensorProto {
                name: name.clone(),
                dims,
                data_type: DType::F32.to_onnx(),
                float_data: data.clone(),
                ..Default::default()
            }
        };
        let c = Const {
            data,
            ints,
            shape,
            dtype,
        };
        consts.insert(name, c);
        g.remove_node(id);
        g.add_initializer(out, tensor);
    }

    // 4. Subgraphs, which see everything known here.
    for id in g.node_ids() {
        for sub in g.node_mut(id).op.subgraphs_mut() {
            fold_graph(sub, &consts);
        }
    }
}

/// ONNX type of `op`'s output: that of the first input unless the operator
/// sets it.
fn output_dtype(op: &Op, inputs: &[&Const]) -> DType {
    match op {
        Op::Shape { .. } => DType::I64,
        Op::Cast { to } => *to,
        // Typed by its fill value, not by the int64 shape it consumes.
        Op::ConstantOfShape { value } => value
            .as_ref()
            .map_or(DType::F32, |t| DType::from_onnx(t.data_type)),
        // QDQ weights fold back to plain float initializers.
        Op::DequantizeLinear(_) => DType::F32,
        Op::QuantizeLinear(attrs) => quantize_output_dtype(attrs, inputs.get(2).copied()),
        Op::Binary(b) if b.is_predicate() => DType::Bool,
        Op::Unary(Unary::Not) => DType::Bool,
        Op::Where => inputs.get(1).map_or(DType::F32, |c| c.dtype),
        _ => inputs.first().map_or(DType::F32, |c| c.dtype),
    }
}

/// Resolves the integer element type a `QuantizeLinear` node saturates to:
/// the `output_dtype` attribute (opset 21+), else the zero-point dtype, else
/// the ONNX default of uint8.
fn quantize_output_dtype(attrs: &QuantAttrs, zero_point: Option<&Const>) -> DType {
    attrs
        .output_dtype
        .or(zero_point.map(|c| c.dtype))
        .unwrap_or(DType::U8)
}

/// Float results up to this many elements are folded even where they are
/// larger than every input.
const MAX_GROWN: usize = 1 << 12;

type UnaryKernel = for<'a, 'b> fn(&TensorView<'b>, &'a mut Vec<f32>) -> TensorView<'a>;
type BinaryKernel =
    for<'a, 'b> fn(&TensorView<'b>, &TensorView<'b>, &'a mut Vec<f32>) -> TensorView<'a>;

/// The kernel code generation calls for float `op`.
fn unary_kernel(op: Unary) -> Option<UnaryKernel> {
    use crate::kernels as k;
    Some(match op {
        Unary::Ceil => k::ceil,
        Unary::Cos => k::cos,
        Unary::Erf => k::erf,
        Unary::Exp => k::exp,
        Unary::Floor => k::floor,
        Unary::Log => k::log,
        Unary::Neg => k::neg,
        Unary::Reciprocal => k::reciprocal,
        Unary::Relu => k::relu,
        Unary::Round => k::round,
        Unary::Sigmoid => k::math::sigmoid,
        Unary::Sin => k::sin,
        Unary::Softplus => k::softplus,
        Unary::Sqrt => k::sqrt,
        Unary::Tanh => k::tanh_kernel,
        Unary::Not => return None,
    })
}

/// The kernel code generation calls for float `op`, for arithmetic.
fn binary_kernel(op: Binary) -> Option<BinaryKernel> {
    use crate::kernels as k;
    Some(match op {
        Binary::Add => k::add,
        Binary::Div => k::div,
        Binary::Mul => k::mul,
        Binary::Pow => k::pow,
        Binary::PRelu => k::prelu,
        Binary::Sub => k::sub,
        _ => return None,
    })
}

fn view(c: &Const) -> TensorView<'_> {
    TensorView::from_slice(c.data.as_slice(), c.shape.clone())
}

/// Evaluates float `op` on known `inputs` with the kernel the generated code
/// would run, so that the result is the same.
fn fold_float(op: &Op, inputs: &[&Const]) -> Option<(Vec<f32>, Vec<usize>)> {
    if inputs.iter().any(|c| c.dtype != DType::F32) {
        return None;
    }
    let mut out = Vec::new();
    let result = match op {
        Op::Unary(op) => unary_kernel(*op)?(&view(inputs.first()?), &mut out),
        Op::Binary(op) => {
            let (a, b) = (inputs.first()?, inputs.get(1)?);
            binary_kernel(*op)?(&view(a), &view(b), &mut out)
        }
        Op::Range => {
            let [start, limit, delta] = inputs else {
                return None;
            };
            let (s, l, d) = (
                start.data.first()?,
                limit.data.first()?,
                delta.data.first()?,
            );
            let len = ((l - s) / d).ceil();
            if !(0.0..=MAX_GROWN as f32).contains(&len) {
                return None;
            }
            crate::kernels::range(&view(start), &view(limit), &view(delta), &mut out)
        }
        _ => return None,
    };
    let largest = inputs.iter().map(|c| c.data.len()).max().unwrap_or(0);
    if result.data.len() > largest.max(MAX_GROWN) {
        return None;
    }
    Some((result.data.to_vec(), result.shape.to_vec()))
}

/// Evaluates `op` on known int64, int32 or bool `inputs` exactly, in i64 as
/// the generated code computes them. Returns the values and shape of its
/// first output, or `None` if the operator is not supported, an input's
/// values are not known exactly, or the result overflows or divides by zero.
fn fold_int(op: &Op, inputs: &[&Const]) -> Option<(Vec<i64>, Vec<usize>)> {
    if !inputs.iter().all(|c| is_integer(c.dtype)) {
        return None;
    }
    let ints = |i: usize| inputs.get(i)?.ints.as_deref();
    let (values, shape) = match op {
        Op::Binary(op) => {
            let f: fn(i64, i64) -> Option<i64> = match op {
                Binary::Add => i64::checked_add,
                Binary::Sub => i64::checked_sub,
                Binary::Mul => i64::checked_mul,
                // Integer division truncates.
                Binary::Div => i64::checked_div,
                Binary::And => |a, b| Some((a != 0 && b != 0) as i64),
                Binary::Or => |a, b| Some((a != 0 || b != 0) as i64),
                Binary::Xor => |a, b| Some(((a != 0) != (b != 0)) as i64),
                Binary::Equal => |a, b| Some((a == b) as i64),
                Binary::Greater => |a, b| Some((a > b) as i64),
                Binary::GreaterOrEqual => |a, b| Some((a >= b) as i64),
                Binary::Less => |a, b| Some((a < b) as i64),
                Binary::LessOrEqual => |a, b| Some((a <= b) as i64),
                _ => return None,
            };
            elementwise(inputs.get(..2)?, |v| f(v[0], v[1]))?
        }
        Op::Unary(Unary::Neg) => elementwise(inputs.get(..1)?, |v| v[0].checked_neg())?,
        Op::Unary(Unary::Not) => elementwise(inputs.get(..1)?, |v| Some((v[0] == 0) as i64))?,
        Op::Max => elementwise(inputs, |v| v.iter().copied().max())?,
        Op::Where => elementwise(inputs.get(..3)?, |v| {
            Some(if v[0] != 0 { v[1] } else { v[2] })
        })?,
        Op::Range => {
            let scalar = |i: usize| ints(i)?.first().copied();
            let (start, limit, delta) = (scalar(0)?, scalar(1)?, scalar(2)?);
            if delta == 0 {
                return None;
            }
            // ceil((limit - start) / delta), or none.
            let (span, step) = (limit as i128 - start as i128, delta as i128);
            let n = ((span + step - step.signum()) / step).max(0) as usize;
            if n > MAX_GROWN {
                return None;
            }
            let values = (0..n as i64).map(|i| start + i * delta).collect();
            (values, vec![n])
        }
        Op::Reshape => {
            let (x, target) = (inputs.first()?, ints(1)?);
            let values = x.ints.clone()?;
            let mut infer = None;
            let mut shape = Vec::with_capacity(target.len());
            for (i, &d) in target.iter().enumerate() {
                shape.push(match d {
                    // 0 copies the input's dimension.
                    0 => *x.shape.get(i)?,
                    -1 if infer.is_none() => {
                        infer = Some(i);
                        1
                    }
                    d => usize::try_from(d).ok()?,
                });
            }
            if let Some(i) = infer {
                let known: usize = shape.iter().product();
                if known == 0 || values.len() % known != 0 {
                    return None;
                }
                shape[i] = values.len() / known;
            }
            if shape.iter().product::<usize>() != values.len() {
                return None;
            }
            (values, shape)
        }
        Op::Slice => {
            let (x, starts, ends) = (inputs.first()?, ints(1)?, ints(2)?);
            let values = x.ints.as_deref()?;
            let rank = x.shape.len() as i64;
            let axes = match inputs.get(3) {
                Some(_) => ints(3)?.to_vec(),
                None => (0..starts.len() as i64).collect(),
            };
            let steps = match inputs.get(4) {
                Some(_) => ints(4)?.to_vec(),
                None => vec![1; starts.len()],
            };
            if [ends.len(), axes.len(), steps.len()] != [starts.len(); 3] {
                return None;
            }
            // First index, step and count along each axis.
            let mut ranges: Vec<(i64, i64, usize)> = x.shape.iter().map(|&d| (0, 1, d)).collect();
            for (((&start, &end), &axis), &step) in starts.iter().zip(ends).zip(&axes).zip(&steps) {
                let axis = if axis < 0 { axis + rank } else { axis };
                let dim = *x.shape.get(usize::try_from(axis).ok()?)? as i64;
                if step == 0 || dim == 0 {
                    return None;
                }
                let wrap = |v: i64| if v < 0 { v.saturating_add(dim) } else { v };
                let (start, end) = if step > 0 {
                    (wrap(start).clamp(0, dim), wrap(end).clamp(0, dim))
                } else {
                    (wrap(start).clamp(0, dim - 1), wrap(end).clamp(-1, dim - 1))
                };
                let (span, step128) = (end as i128 - start as i128, step as i128);
                let n = ((span + step128 - step128.signum()) / step128).max(0) as usize;
                ranges[axis as usize] = (start, step, n);
            }
            let shape: Vec<usize> = ranges.iter().map(|r| r.2).collect();
            let mut strides = vec![1usize; x.shape.len()];
            for d in (0..x.shape.len().saturating_sub(1)).rev() {
                strides[d] = strides[d + 1] * x.shape[d + 1];
            }
            let len: usize = shape.iter().product();
            let mut out = Vec::with_capacity(len);
            for flat in 0..len {
                let (mut rest, mut at) = (flat, 0);
                for d in (0..shape.len()).rev() {
                    let (first, step, n) = ranges[d];
                    let i = first + (rest % n) as i64 * step;
                    rest /= n;
                    at += i as usize * strides[d];
                }
                out.push(values[at]);
            }
            (out, shape)
        }
        _ => return None,
    };
    let largest = inputs.iter().map(|c| c.data.len()).max().unwrap_or(0);
    (values.len() <= largest.max(MAX_GROWN)).then_some((values, shape))
}

/// `f` of the elements of `inputs` at each position of their broadcast
/// shape, if their values are all known exactly, they broadcast, the result
/// is not much larger than they are, and `f` gives a value everywhere.
fn elementwise(
    inputs: &[&Const],
    f: impl Fn(&[i64]) -> Option<i64>,
) -> Option<(Vec<i64>, Vec<usize>)> {
    let values: Vec<&[i64]> = inputs
        .iter()
        .map(|c| c.ints.as_deref())
        .collect::<Option<_>>()?;
    let rank = inputs.iter().map(|c| c.shape.len()).max()?;
    let mut shape = vec![1; rank];
    for c in inputs {
        for (d, &n) in shape[rank - c.shape.len()..].iter_mut().zip(&c.shape) {
            if *d == 1 {
                *d = n;
            } else if n != 1 && n != *d {
                return None;
            }
        }
    }
    let len: usize = shape.iter().product();
    if len > values.iter().map(|v| v.len()).max()?.max(MAX_GROWN) {
        return None;
    }
    // Strides of each input along the output's axes, 0 where it is broadcast.
    let strides: Vec<Vec<usize>> = inputs
        .iter()
        .map(|c| {
            let mut strides = vec![0; rank];
            let mut step = 1;
            for (i, &n) in c.shape.iter().enumerate().rev() {
                if n != 1 {
                    strides[rank - c.shape.len() + i] = step;
                }
                step *= n;
            }
            strides
        })
        .collect();
    let mut out = Vec::with_capacity(len);
    let mut args = vec![0; inputs.len()];
    for flat in 0..len {
        let mut offsets = vec![0; inputs.len()];
        let mut rest = flat;
        for axis in (0..rank).rev() {
            let coord = rest % shape[axis];
            rest /= shape[axis];
            for (o, s) in offsets.iter_mut().zip(&strides) {
                *o += coord * s[axis];
            }
        }
        for ((a, v), o) in args.iter_mut().zip(&values).zip(&offsets) {
            *a = v[*o];
        }
        out.push(f(&args)?);
    }
    Some((out, shape))
}

/// Evaluates `op` on known `inputs`. Returns the data and shape of its first
/// output, or `None` if the operator or this use of it is not supported.
fn fold_op(op: &Op, inputs: &[&Const]) -> Option<(Vec<f32>, Vec<usize>)> {
    match op {
        Op::Unary(_) | Op::Binary(_) | Op::Range => fold_float(op, inputs),
        Op::DequantizeLinear(attrs) | Op::QuantizeLinear(attrs) => {
            let (x, scale) = (inputs.first()?, inputs.get(1)?);
            let zp = inputs.get(2);
            let block_size = attrs.block_size as usize;

            let xv = TensorView::from_slice(x.data.as_slice(), x.shape.clone());
            let sv = TensorView::from_slice(scale.data.as_slice(), scale.shape.clone());
            let zv = zp.map(|c| TensorView::from_slice(c.data.as_slice(), c.shape.clone()));

            let mut out = Vec::new();
            if let Op::DequantizeLinear(_) = op {
                crate::kernels::dequantize_linear(
                    &xv,
                    &sv,
                    zv.as_ref(),
                    attrs.axis,
                    block_size,
                    &mut out,
                );
            } else {
                let out_dtype = quantize_output_dtype(attrs, zp.copied());
                let (qmin, qmax) =
                    crate::kernels::quant_range(out_dtype.to_onnx()).unwrap_or((0.0, 255.0));
                crate::kernels::quantize_linear(
                    &xv,
                    &sv,
                    zv.as_ref(),
                    attrs.axis,
                    block_size,
                    qmin,
                    qmax,
                    &mut out,
                );
            }
            Some((out, x.shape.clone()))
        }
        Op::Shape { start, end } => {
            let shape = &inputs.first()?.shape;
            // opset-15 `start`/`end` slice the reported shape.
            let rank = shape.len() as i64;
            let clamp = |v: i64| (if v < 0 { v + rank } else { v }).clamp(0, rank) as usize;
            let start = clamp(start.unwrap_or(0));
            let end = clamp(end.unwrap_or(rank)).max(start);
            let data: Vec<f32> = shape[start..end].iter().map(|&s| s as f32).collect();
            let out_shape = vec![data.len()];
            Some((data, out_shape))
        }
        Op::Unsqueeze { axes } => {
            let axes = match inputs.get(1) {
                Some(c) => c.data.iter().map(|&x| x as i64).collect(),
                None => axes.clone()?,
            };
            let x = inputs.first()?;
            let mut new_shape = x.shape.clone();
            let mut sorted_axes = axes;
            sorted_axes.sort();
            for ax in sorted_axes {
                let idx = if ax < 0 {
                    (new_shape.len() + 1) as i64 + ax
                } else {
                    ax
                } as usize;
                if idx <= new_shape.len() {
                    new_shape.insert(idx, 1);
                }
            }
            Some((x.data.clone(), new_shape))
        }
        Op::Squeeze { axes } => {
            let axes: Option<Vec<i64>> = match inputs.get(1) {
                Some(c) => Some(c.data.iter().map(|&x| x as i64).collect()),
                None => axes.clone(),
            };
            let x = inputs.first()?;
            let shape = &x.shape;
            let mut new_shape = Vec::new();
            if let Some(axes) = axes {
                let axes_set: HashSet<usize> = axes
                    .iter()
                    .map(|&a| {
                        if a < 0 {
                            (shape.len() as i64 + a) as usize
                        } else {
                            a as usize
                        }
                    })
                    .collect();
                for (idx, &d) in shape.iter().enumerate() {
                    if d != 1 || !axes_set.contains(&idx) {
                        new_shape.push(d);
                    }
                }
            } else {
                new_shape.extend(shape.iter().copied().filter(|&d| d != 1));
            }
            Some((x.data.clone(), new_shape))
        }
        Op::Concat { axis: axis_attr } => {
            let first = inputs.first()?;
            let axis = |shape: &[usize]| {
                if *axis_attr < 0 {
                    (shape.len() as i64 + axis_attr) as usize
                } else {
                    *axis_attr as usize
                }
            };
            let mut target_shape = first.shape.clone();
            let axis0 = axis(&target_shape);
            target_shape[axis0] = inputs.iter().map(|c| c.shape[axis(&c.shape)]).sum();
            // Only appends: 1-D, or 2-D along rows.
            if target_shape.len() == 1 || (target_shape.len() == 2 && axis0 == 0) {
                let merged = inputs.iter().flat_map(|c| c.data.iter().copied()).collect();
                Some((merged, target_shape))
            } else {
                None
            }
        }
        Op::Slice => {
            // Supports 1-D and 2-D data, unit steps.
            let (x, starts, ends) = (inputs.first()?, inputs.get(1)?, inputs.get(2)?);
            if inputs
                .get(4)
                .is_some_and(|steps| steps.data.iter().any(|&s| s != 1.0))
            {
                return None;
            }
            let (data, shape) = (&x.data, &x.shape);
            let ndim = shape.len();
            let num_slices = starts.data.len();
            let axes: Vec<usize> = match inputs.get(3) {
                Some(c) => c
                    .data
                    .iter()
                    .map(|&v| {
                        let a = v as i64;
                        if a < 0 {
                            (ndim as i64 + a) as usize
                        } else {
                            a as usize
                        }
                    })
                    .collect(),
                None => (0..num_slices).collect(),
            };
            let mut dim_start = vec![0usize; ndim];
            let mut dim_end: Vec<usize> = shape.clone();
            for i in 0..num_slices {
                let ax = axes[i];
                let dim = shape[ax] as i64;
                let wrap = |v: i64| (if v < 0 { v + dim } else { v }).clamp(0, dim);
                dim_start[ax] = wrap(starts.data[i] as i64) as usize;
                dim_end[ax] = wrap(ends.data[i] as i64) as usize;
            }
            let out_shape: Vec<usize> = (0..ndim).map(|d| dim_end[d] - dim_start[d]).collect();
            let out_size: usize = out_shape.iter().product();
            if out_size == 0 || ndim > 2 {
                return None;
            }
            let mut result = Vec::with_capacity(out_size);
            if ndim == 1 {
                result.extend_from_slice(&data[dim_start[0]..dim_end[0]]);
            } else {
                // 2D: data is row-major [rows, cols]
                let cols = shape[1];
                for row in dim_start[0]..dim_end[0] {
                    let row_offset = row * cols;
                    result.extend_from_slice(
                        &data[row_offset + dim_start[1]..row_offset + dim_end[1]],
                    );
                }
            }
            Some((result, out_shape))
        }
        Op::Cast { .. } => {
            let x = inputs.first()?;
            Some((x.data.clone(), x.shape.clone()))
        }
        Op::Transpose { perm } => {
            let x = inputs.first()?;
            let xv = TensorView::from_slice(x.data.as_slice(), x.shape.clone());
            let mut out = Vec::new();
            let t = crate::kernels::transpose(&xv, perm.as_deref().unwrap_or(&[]), &mut out);
            Some((t.data.to_vec(), t.shape.to_vec()))
        }
        Op::ConstantOfShape { value } => {
            let val = value
                .as_ref()
                .and_then(|t| {
                    if !t.double_data.is_empty() {
                        return Some(t.double_data[0] as f32);
                    }
                    // PyTorch's exporter stores the fill value in
                    // `raw_data`; reading only the typed fields folded
                    // `ConstantOfShape(value=1)` to zeros.
                    tensor_to_array(t)
                        .ok()
                        .and_then(|(d, _)| d.first().copied())
                })
                .unwrap_or(0.0);
            let shape: Vec<usize> = inputs.first()?.data.iter().map(|&x| x as usize).collect();
            let size: usize = shape.iter().product();
            Some((vec![val; size], shape))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::onnx_proto::{AttributeProto, GraphProto, NodeProto};

    /// `ConstantOfShape([2], value)` with the shape as an int64 initializer.
    fn constant_of_shape(value: TensorProto) -> GraphProto {
        GraphProto {
            node: vec![NodeProto {
                op_type: "ConstantOfShape".to_string(),
                input: vec!["shape".to_string()],
                output: vec!["y".to_string()],
                attribute: vec![AttributeProto {
                    name: "value".to_string(),
                    t: Some(value),
                    ..Default::default()
                }],
                ..Default::default()
            }],
            initializer: vec![TensorProto {
                name: "shape".to_string(),
                data_type: DType::I64.to_onnx(),
                dims: vec![1],
                int64_data: vec![2],
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn fold(g: GraphProto) -> GraphProto {
        let mut ir = Graph::from_onnx(g);
        fold_constants(&mut ir);
        ir.into_onnx()
    }

    fn folded(g: &GraphProto) -> &TensorProto {
        assert!(g.node.is_empty(), "ConstantOfShape was not folded");
        g.initializer.iter().find(|t| t.name == "y").unwrap()
    }

    #[test]
    fn folds_constant_of_shape_with_raw_int64_value() {
        // PyTorch's exporter writes the fill value as raw bytes; reading only
        // the typed fields used to fold this to zeros.
        let g = fold(constant_of_shape(TensorProto {
            data_type: DType::I64.to_onnx(),
            dims: vec![1],
            raw_data: 1i64.to_le_bytes().to_vec(),
            ..Default::default()
        }));
        let y = folded(&g);
        assert_eq!(DType::from_onnx(y.data_type), DType::I64);
        assert_eq!(y.int64_data, [1, 1]);
    }

    #[test]
    fn folds_constant_of_shape_with_float_value_as_float() {
        // The output takes the value's type, not the int64 shape's.
        let g = fold(constant_of_shape(TensorProto {
            data_type: DType::F32.to_onnx(),
            dims: vec![1],
            raw_data: 0.5f32.to_le_bytes().to_vec(),
            ..Default::default()
        }));
        let y = folded(&g);
        assert_eq!(DType::from_onnx(y.data_type), DType::F32);
        assert_eq!(y.float_data, [0.5, 0.5]);
    }

    /// `Slice([0, 1, 2, 3], starts=[0], ends=[4], axes=[0], steps=[step])`.
    fn slice(step: i64) -> GraphProto {
        let ints = |name: &str, v: Vec<i64>| TensorProto {
            name: name.to_string(),
            data_type: DType::I64.to_onnx(),
            dims: vec![v.len() as i64],
            int64_data: v,
            ..Default::default()
        };
        GraphProto {
            node: vec![NodeProto {
                op_type: "Slice".to_string(),
                input: ["x", "starts", "ends", "axes", "steps"]
                    .map(String::from)
                    .to_vec(),
                output: vec!["y".to_string()],
                ..Default::default()
            }],
            initializer: vec![
                TensorProto {
                    name: "x".to_string(),
                    data_type: DType::F32.to_onnx(),
                    dims: vec![4],
                    float_data: vec![0.0, 1.0, 2.0, 3.0],
                    ..Default::default()
                },
                ints("starts", vec![0]),
                ints("ends", vec![4]),
                ints("axes", vec![0]),
                ints("steps", vec![step]),
            ],
            ..Default::default()
        }
    }

    #[test]
    fn folds_slice_with_unit_step() {
        let g = fold(slice(1));
        assert!(g.node.is_empty());
        let y = g.initializer.iter().find(|t| t.name == "y").unwrap();
        assert_eq!(y.float_data, [0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn keeps_slice_with_other_steps() {
        // Folding used to ignore `steps` and return every element.
        let g = fold(slice(2));
        assert_eq!(g.node.len(), 1);
    }

    fn floats(name: &str, dims: &[i64], v: &[f32]) -> TensorProto {
        TensorProto {
            name: name.to_string(),
            data_type: DType::F32.to_onnx(),
            dims: dims.to_vec(),
            float_data: v.to_vec(),
            ..Default::default()
        }
    }

    fn node(op: &str, inputs: &[&str], output: &str) -> NodeProto {
        NodeProto {
            op_type: op.to_string(),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: vec![output.to_string()],
            ..Default::default()
        }
    }

    /// Rotary embedding frequencies: `1 / sqrt(d)` and `Range(0, 4, 1) * 2`
    /// on constants become initializers.
    #[test]
    fn folds_float_arithmetic() {
        let g = fold(GraphProto {
            node: vec![
                node("Sqrt", &["d"], "s"),
                node("Reciprocal", &["s"], "r"),
                node("Range", &["zero", "four", "one"], "i"),
                node("Mul", &["i", "two"], "y"),
            ],
            initializer: vec![
                floats("d", &[1], &[16.0]),
                floats("zero", &[], &[0.0]),
                floats("four", &[], &[4.0]),
                floats("one", &[], &[1.0]),
                floats("two", &[], &[2.0]),
            ],
            ..Default::default()
        });
        assert!(g.node.is_empty(), "{:?}", g.node);
        let get = |name: &str| g.initializer.iter().find(|t| t.name == name).unwrap();
        assert_eq!(get("r").float_data, [0.25]);
        assert_eq!(get("y").float_data, [0.0, 2.0, 4.0, 6.0]);
        assert_eq!(get("y").dims, [4]);
    }

    /// A weight that is stored transposed is transposed once, whatever its
    /// element type.
    #[test]
    fn folds_transpose_of_a_constant() {
        let transposed = |perm: Option<Vec<i64>>| {
            let mut t = node("Transpose", &["x"], "y");
            if let Some(perm) = perm {
                t.attribute = vec![AttributeProto {
                    name: "perm".to_string(),
                    ints: perm,
                    ..Default::default()
                }];
            }
            fold(GraphProto {
                node: vec![t],
                initializer: vec![floats("x", &[2, 3], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])],
                ..Default::default()
            })
        };
        // Without `perm` the axes are reversed.
        for perm in [None, Some(vec![1, 0])] {
            let g = transposed(perm);
            assert!(g.node.is_empty(), "{:?}", g.node);
            let y = g.initializer.iter().find(|t| t.name == "y").unwrap();
            assert_eq!(y.dims, [3, 2]);
            assert_eq!(y.float_data, [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        }
    }

    fn ints(name: &str, dims: &[i64], v: &[i64]) -> TensorProto {
        TensorProto {
            name: name.to_string(),
            data_type: DType::I64.to_onnx(),
            dims: dims.to_vec(),
            int64_data: v.to_vec(),
            ..Default::default()
        }
    }

    /// Integer arithmetic on constants is folded exactly: integer division
    /// truncates, and integers beyond what f32 holds keep their value.
    #[test]
    fn folds_integer_arithmetic_exactly() {
        let big = (1 << 40) + 1;
        let g = fold(GraphProto {
            node: vec![
                node("Div", &["a", "two"], "q"),
                node("Add", &["big", "one"], "s"),
                node("Neg", &["q"], "n"),
                node("Mul", &["n", "row"], "m"),
            ],
            initializer: vec![
                ints("a", &[2, 1], &[7, -7]),
                ints("two", &[], &[2]),
                ints("big", &[], &[big]),
                ints("one", &[1], &[1]),
                ints("row", &[1, 3], &[1, 2, 3]),
            ],
            ..Default::default()
        });
        assert!(g.node.is_empty(), "{:?}", g.node);
        let get = |name: &str| g.initializer.iter().find(|t| t.name == name).unwrap();
        assert_eq!(get("q").int64_data, [3, -3]);
        assert_eq!(get("s").int64_data, [big + 1]);
        // [2, 1] times [1, 3] broadcasts to [2, 3].
        assert_eq!(get("m").int64_data, [-3, -6, -9, 3, 6, 9]);
        assert_eq!(get("m").dims, [2, 3]);
    }

    /// The shape computations exporters leave: comparisons to booleans, a
    /// `Where` of them, an integer `Range` and a `Reshape` of integers.
    #[test]
    fn folds_integer_comparisons_where_range_and_reshape() {
        let g = fold(GraphProto {
            node: vec![
                node("Equal", &["dims", "minus_one"], "e"),
                node("Where", &["e", "one", "dims"], "w"),
                node("Range", &["zero", "five", "two"], "r"),
                node("Reshape", &["w", "target"], "y"),
            ],
            initializer: vec![
                ints("dims", &[3], &[4, -1, 8]),
                ints("minus_one", &[], &[-1]),
                ints("one", &[], &[1]),
                ints("zero", &[], &[0]),
                ints("five", &[], &[5]),
                ints("two", &[], &[2]),
                ints("target", &[2], &[-1, 1]),
            ],
            ..Default::default()
        });
        assert!(g.node.is_empty(), "{:?}", g.node);
        let get = |name: &str| g.initializer.iter().find(|t| t.name == name).unwrap();
        let e = get("e");
        assert_eq!(
            (e.data_type, &e.raw_data[..]),
            (DType::Bool.to_onnx(), &[0, 1, 0][..])
        );
        assert_eq!(get("w").int64_data, [4, 1, 8]);
        assert_eq!(get("r").int64_data, [0, 2, 4]);
        assert_eq!(
            (&get("y").int64_data[..], &get("y").dims[..]),
            (&[4, 1, 8][..], &[3, 1][..])
        );
    }

    /// Slices of integers of any rank, steps and bounds, as ONNX clamps them.
    #[test]
    fn folds_integer_slices() {
        let slice = |starts: &[i64], ends: &[i64], axes: &[i64], steps: &[i64]| {
            let n = starts.len() as i64;
            let g = fold(GraphProto {
                node: vec![node(
                    "Slice",
                    &["x", "starts", "ends", "axes", "steps"],
                    "y",
                )],
                initializer: vec![
                    ints("x", &[2, 1, 3], &[0, 1, 2, 3, 4, 5]),
                    ints("starts", &[n], starts),
                    ints("ends", &[n], ends),
                    ints("axes", &[n], axes),
                    ints("steps", &[n], steps),
                ],
                ..Default::default()
            });
            assert!(g.node.is_empty(), "{:?}", g.node);
            let y = g.initializer.iter().find(|t| t.name == "y").unwrap();
            (y.int64_data.clone(), y.dims.clone())
        };
        assert_eq!(
            slice(&[1], &[i64::MAX], &[-1], &[1]),
            (vec![1, 2, 4, 5], vec![2, 1, 2])
        );
        assert_eq!(
            slice(&[-1], &[i64::MIN], &[2], &[-2]),
            (vec![2, 0, 5, 3], vec![2, 1, 2])
        );
        assert_eq!(
            slice(&[1, 0], &[2, 1], &[0, 2], &[1, 1]),
            (vec![3], vec![1, 1, 1])
        );
    }

    /// Division by zero, overflow, and integers that went through f32 beyond
    /// what it holds exactly are left to run.
    #[test]
    fn keeps_integer_arithmetic_it_cannot_do_exactly() {
        let g = fold(GraphProto {
            node: vec![
                node("Div", &["one", "zero"], "q"),
                node("Mul", &["max", "two"], "m"),
                // Concat folds through f32, which rounds 2^40 + 1.
                node("Concat", &["big", "one"], "c"),
                node("Add", &["c", "one"], "s"),
            ],
            initializer: vec![
                ints("one", &[1], &[1]),
                ints("zero", &[1], &[0]),
                ints("two", &[], &[2]),
                ints("max", &[], &[i64::MAX]),
                ints("big", &[1], &[(1 << 40) + 1]),
            ],
            ..Default::default()
        });
        let ops: Vec<_> = g.node.iter().map(|n| n.op_type.as_str()).collect();
        assert_eq!(ops, ["Div", "Mul", "Add"]);
    }

    #[test]
    fn folds_into_subgraphs_using_outer_constants() {
        // `Shape(c)` inside the branch reads an initializer of the outer
        // graph.
        let branch = GraphProto {
            node: vec![NodeProto {
                op_type: "Shape".to_string(),
                input: vec!["c".to_string()],
                output: vec!["s".to_string()],
                ..Default::default()
            }],
            output: vec![crate::model::onnx_proto::ValueInfoProto {
                name: "s".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        let g = GraphProto {
            node: vec![NodeProto {
                op_type: "If".to_string(),
                input: vec!["cond".to_string()],
                output: vec!["r".to_string()],
                attribute: ["then_branch", "else_branch"]
                    .map(|name| AttributeProto {
                        name: name.to_string(),
                        g: Some(branch.clone()),
                        ..Default::default()
                    })
                    .to_vec(),
                ..Default::default()
            }],
            initializer: vec![TensorProto {
                name: "c".to_string(),
                data_type: DType::F32.to_onnx(),
                dims: vec![3, 2],
                float_data: vec![0.0; 6],
                ..Default::default()
            }],
            ..Default::default()
        };
        let g = fold(g);
        for attr in &g.node[0].attribute {
            let branch = attr.g.as_ref().unwrap();
            assert!(
                branch.node.is_empty(),
                "{}: Shape was not folded",
                attr.name
            );
            assert_eq!(branch.initializer[0].int64_data, [3, 2]);
        }
    }
}
