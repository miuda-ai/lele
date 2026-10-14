//! Constant folding.

use crate::compiler::ir::{DType, Graph, Op, QuantAttrs};
use crate::model::onnx_proto::TensorProto;
use crate::model::tensor_to_array;
use std::collections::{HashMap, HashSet};

/// A tensor known at compile time. Values are held as f32 whatever the
/// element type; `dtype` is the type they stand for.
#[derive(Clone, Debug)]
struct Const {
    data: Vec<f32>,
    shape: Vec<usize>,
    dtype: DType,
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
        if let Ok((data, shape)) = tensor_to_array(t)
            && !data.is_empty()
        {
            let dtype = DType::from_onnx(t.data_type);
            consts.insert(g.value(v).name.clone(), Const { data, shape, dtype });
        }
    }
    // 2. Constant nodes. They stay in the graph; code generation reads them.
    for (_, node) in g.nodes() {
        if let Op::Constant { value } = &node.op
            && let Some(out) = node.output(0)
            && let Ok((data, shape)) = tensor_to_array(value)
            && !data.is_empty()
        {
            let dtype = DType::from_onnx(value.data_type);
            consts.insert(g.value(out).name.clone(), Const { data, shape, dtype });
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
        let Some((data, shape)) = fold_op(&node.op, &inputs) else {
            continue;
        };

        let name = g.value(out).name.clone();
        let tensor = if dtype == DType::I64 {
            TensorProto {
                name: name.clone(),
                dims: shape.iter().map(|&s| s as i64).collect(),
                data_type: DType::I64.to_onnx(),
                int64_data: data.iter().map(|&x| x as i64).collect(),
                ..Default::default()
            }
        } else {
            // Every other type is stored as float.
            TensorProto {
                name: name.clone(),
                dims: shape.iter().map(|&s| s as i64).collect(),
                data_type: DType::F32.to_onnx(),
                float_data: data.clone(),
                ..Default::default()
            }
        };
        consts.insert(name, Const { data, shape, dtype });
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

/// Evaluates `op` on known `inputs`. Returns the data and shape of its first
/// output, or `None` if the operator or this use of it is not supported.
fn fold_op(op: &Op, inputs: &[&Const]) -> Option<(Vec<f32>, Vec<usize>)> {
    match op {
        Op::DequantizeLinear(attrs) | Op::QuantizeLinear(attrs) => {
            use crate::tensor::TensorView;
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
