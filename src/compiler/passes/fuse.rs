//! Fusion of operator chains into single operators that code generation
//! maps onto fused kernels.
//!
//! Each fusion starts from the last node of a chain and walks producers
//! back. Every intermediate tensor must be read by the chain alone, so
//! removing the chain cannot leave another reader without its input.

use crate::compiler::ir::{
    Activation, Binary, DType, Graph, NodeId, Op, Reduction, Unary, ValueId,
};
use crate::model::onnx_proto::{NodeProto, TensorProto};
use crate::model::tensor_to_array;

/// Runs every fusion on `g` and its subgraphs.
pub(crate) fn fuse(g: &mut Graph) {
    fuse_layer_norm(g);
    fuse_scale_norm(g);
    fuse_gelu(g);
    fuse_silu(g);
    fuse_conv_bias(g);
    fuse_conv_act(g);
    fuse_dynamic_quant_linear(g);
    fuse_linear(g);
    fuse_embedding_concat(g);
    for id in g.node_ids() {
        for sub in g.node_mut(id).op.subgraphs_mut() {
            fuse(sub);
        }
    }
}

/// Matches a `binary` node.
fn binary(binary: Binary) -> impl Fn(&Op) -> bool {
    move |op| matches!(op, Op::Binary(b) if *b == binary)
}

/// Matches a `unary` node.
fn unary(unary: Unary) -> impl Fn(&Op) -> bool {
    move |op| matches!(op, Op::Unary(u) if *u == unary)
}

fn reduce_mean(op: &Op) -> bool {
    matches!(
        op,
        Op::Reduce {
            reduction: Reduction::Mean,
            ..
        }
    )
}

/// The node producing `v` if `is_match` accepts its operator and its only
/// reader is the chain being fused.
fn sole_producer(g: &Graph, v: Option<ValueId>, is_match: impl Fn(&Op) -> bool) -> Option<NodeId> {
    let v = v?;
    let id = g.value(v).producer()?;
    (is_match(&g.node(id).op) && g.uses(v) == 1).then_some(id)
}

/// A compile-time constant: an initializer or a `Constant`.
fn constant_tensor(g: &Graph, v: Option<ValueId>) -> Option<&TensorProto> {
    let v = v?;
    match g.data(v) {
        Some(t) => Some(t),
        None => match &g.producer(v)?.op {
            Op::Constant { value } => Some(value),
            _ => None,
        },
    }
}

/// The values of a compile-time constant.
fn constant(g: &Graph, v: Option<ValueId>) -> Option<Vec<f32>> {
    tensor_to_array(constant_tensor(g, v)?).ok().map(|(data, _)| data)
}

/// Whether `v` is a constant scalar close to `expected`.
fn is_scalar(g: &Graph, v: Option<ValueId>, expected: f32) -> bool {
    constant(g, v).is_some_and(|d| d.len() == 1 && (d[0] - expected).abs() <= 1e-3 * expected.abs())
}

/// The operand of binary node `id` that is not `v`, if `v` is one of them.
fn other_operand(g: &Graph, id: NodeId, v: ValueId) -> Option<Option<ValueId>> {
    let node = g.node(id);
    match (node.input(0), node.input(1)) {
        (Some(a), b) if a == v => Some(b),
        (a, Some(b)) if b == v => Some(a),
        _ => None,
    }
}

/// The `op` nodes, whose operands callers try in both orders. A fusion
/// removes nodes, so callers skip those no longer in the graph.
fn binary_nodes(g: &Graph, op: Binary) -> Vec<NodeId> {
    let is_op = binary(op);
    g.nodes()
        .filter(|(_, n)| is_op(&n.op) && n.inputs().len() == 2)
        .map(|(id, _)| id)
        .collect()
}

/// A proto for a fused node, named after `id`.
fn named_after(g: &Graph, id: NodeId) -> NodeProto {
    NodeProto {
        name: g.node(id).proto.name.clone(),
        ..Default::default()
    }
}

/// Whether reduction `id` is over the last axis alone, keeping it.
fn reduces_last_axis(g: &Graph, id: NodeId) -> bool {
    let node = g.node(id);
    let Op::Reduce { axes, keepdims, .. } = &node.op else {
        return false;
    };
    let axes = match axes {
        Some(axes) => Some(axes.iter().map(|&x| x as f32).collect()),
        None => constant(g, node.input(1)),
    };
    axes.as_deref() == Some(&[-1.0]) && *keepdims
}

/// `x - mean(x)`, normalized by `sqrt(mean((x - mean(x))^2) + eps)`, then
/// scaled and shifted, all over the last axis → `LayerNormalization`.
fn fuse_layer_norm(g: &mut Graph) {
    for add in binary_nodes(g, Binary::Add) {
        if !g.contains(add) {
            continue;
        }
        let fused = (0..2).find_map(|k| {
            let node = g.node(add);
            let (scaled, bias) = (node.input(k)?, node.input(1 - k));
            let mul = sole_producer(g, Some(scaled), binary(Binary::Mul))?;
            let normed = g.node(mul).input(0)?;
            let scale = g.node(mul).input(1);
            let div = sole_producer(g, Some(normed), binary(Binary::Div))?;
            let (centered, std) = (g.node(div).input(0)?, g.node(div).input(1));
            let sqrt = sole_producer(g, std, unary(Unary::Sqrt))?;
            let add_eps = sole_producer(g, g.node(sqrt).input(0), binary(Binary::Add))?;
            let (var, eps) = (g.node(add_eps).input(0), g.node(add_eps).input(1));
            let epsilon = constant(g, eps).filter(|e| e.len() == 1)?[0];
            let var_mean = sole_producer(g, var, reduce_mean)?;
            let pow = sole_producer(g, g.node(var_mean).input(0), binary(Binary::Pow))?;
            if g.node(pow).input(0) != Some(centered) || !is_scalar(g, g.node(pow).input(1), 2.0) {
                return None;
            }
            // Read by the Pow and the Div.
            let sub = g.value(centered).producer()?;
            if !binary(Binary::Sub)(&g.node(sub).op) || g.uses(centered) != 2 {
                return None;
            }
            let x = g.node(sub).input(0)?;
            let mean = sole_producer(g, g.node(sub).input(1), reduce_mean)?;
            if g.node(mean).input(0) != Some(x)
                || !reduces_last_axis(g, mean)
                || !reduces_last_axis(g, var_mean)
            {
                return None;
            }
            let nodes = [mean, sub, pow, var_mean, add_eps, sqrt, div, mul, add];
            Some((nodes, [Some(x), scale, bias], epsilon))
        });
        if let Some((nodes, inputs, epsilon)) = fused {
            let op = Op::LayerNormalization { axis: -1, epsilon };
            let proto = named_after(g, add);
            g.fuse(&nodes, op, proto, inputs.to_vec());
        }
    }
}

/// `gamma * (x / (ReduceL2(x) * scale + eps))` over the last axis, with
/// constant scalars `scale` and `eps` and the operands of each `Mul` and
/// `Add` in either order → `ScaleNorm`.
fn fuse_scale_norm(g: &mut Graph) {
    let scalar = |g: &Graph, v| {
        constant(g, v).and_then(|c| match c[..] {
            [c] if c.is_finite() => Some(c),
            _ => None,
        })
    };
    for mul in binary_nodes(g, Binary::Mul) {
        if !g.contains(mul) {
            continue;
        }
        let fused = (0..2).find_map(|k| {
            let node = g.node(mul);
            let (normed, gamma) = (node.input(k), node.input(1 - k));
            let div = sole_producer(g, normed, binary(Binary::Div))?;
            let (x, denom) = (g.node(div).input(0)?, g.node(div).input(1));
            let add = sole_producer(g, denom, binary(Binary::Add))?;
            let (scaled, epsilon) = (0..2).find_map(|j| {
                let eps = scalar(g, g.node(add).input(1 - j))?;
                Some((g.node(add).input(j), eps))
            })?;
            let scale_mul = sole_producer(g, scaled, binary(Binary::Mul))?;
            let (norm, scale) = (0..2).find_map(|j| {
                let scale = scalar(g, g.node(scale_mul).input(1 - j))?;
                Some((g.node(scale_mul).input(j), scale))
            })?;
            let is_l2 = |op: &Op| {
                matches!(
                    op,
                    Op::Reduce {
                        reduction: Reduction::L2,
                        ..
                    }
                )
            };
            let l2 = sole_producer(g, norm, is_l2)?;
            // The kernel takes f32, and a constant gamma of one element per
            // column (rank 1, so it broadcasts along the last axis) or of one.
            let gamma_ok = constant_tensor(g, gamma).is_some_and(|t| {
                t.dims.len() <= 1 || t.dims.iter().product::<i64>() == 1
            });
            if g.node(l2).input(0) != Some(x)
                || !reduces_last_axis(g, l2)
                || g.value(x).dtype != DType::F32
                || !gamma_ok
            {
                return None;
            }
            Some(([l2, scale_mul, add, div, mul], [Some(x), gamma], scale, epsilon))
        });
        if let Some((nodes, inputs, scale, epsilon)) = fused {
            let proto = named_after(g, mul);
            g.fuse(&nodes, Op::ScaleNorm { scale, epsilon }, proto, inputs.to_vec());
        }
    }
}

/// `x * 0.5 * (1 + erf(x / sqrt(2)))`, with the 0.5 applied first or last
/// → `Gelu`.
fn fuse_gelu(g: &mut Graph) {
    for mul2 in binary_nodes(g, Binary::Mul) {
        if !g.contains(mul2) {
            continue;
        }
        let fused = (0..2).find_map(|k| {
            let node = g.node(mul2);
            let (inner, outer) = (node.input(k), node.input(1 - k));
            let mul1 = sole_producer(g, inner, binary(Binary::Mul))?;
            // One Mul takes `1 + erf`, the other two operands are `x` and 0.5.
            let (add, rest) = (0..2).find_map(|j| {
                let add = sole_producer(g, g.node(mul1).input(j), binary(Binary::Add))?;
                Some((add, g.node(mul1).input(1 - j)))
            })?;
            let (erf, one) = (0..2).find_map(|j| {
                let erf = sole_producer(g, g.node(add).input(j), unary(Unary::Erf))?;
                Some((erf, g.node(add).input(1 - j)))
            })?;
            let div = sole_producer(g, g.node(erf).input(0), binary(Binary::Div))?;
            let x = g.node(div).input(0)?;
            let half_ok = (rest == Some(x) && is_scalar(g, outer, 0.5))
                || (outer == Some(x) && is_scalar(g, rest, 0.5));
            if !half_ok
                || !is_scalar(g, one, 1.0)
                || !is_scalar(g, g.node(div).input(1), std::f32::consts::SQRT_2)
            {
                return None;
            }
            Some(([div, erf, add, mul1, mul2], x))
        });
        if let Some((nodes, x)) = fused {
            let op = Op::Gelu { approximate: None };
            let proto = named_after(g, mul2);
            g.fuse(&nodes, op, proto, vec![Some(x)]);
        }
    }
}

/// `x * sigmoid(x)` → `Silu`.
fn fuse_silu(g: &mut Graph) {
    for mul in binary_nodes(g, Binary::Mul) {
        if !g.contains(mul) {
            continue;
        }
        let fused = (0..2).find_map(|k| {
            let node = g.node(mul);
            let sigmoid = sole_producer(g, node.input(k), unary(Unary::Sigmoid))?;
            let x = g.node(sigmoid).input(0)?;
            (node.input(1 - k) == Some(x)).then_some((sigmoid, x))
        });
        if let Some((sigmoid, x)) = fused {
            let proto = named_after(g, mul);
            g.fuse(&[sigmoid, mul], Op::Silu, proto, vec![Some(x)]);
        }
    }
}

/// `Conv` without bias followed by `Add` of a per-channel constant → `Conv`
/// with that bias.
fn fuse_conv_bias(g: &mut Graph) {
    for add in binary_nodes(g, Binary::Add) {
        if !g.contains(add) {
            continue;
        }
        let fused = (0..2).find_map(|k| {
            let conv_out = g.node(add).input(k)?;
            let bias = g.node(add).input(1 - k)?;
            let conv = sole_producer(g, Some(conv_out), |op| matches!(op, Op::Conv(_)))?;
            let node = g.node(conv);
            if node.input(2).is_some() {
                return None;
            }
            let (x, w) = (node.input(0)?, node.input(1)?);
            let w_dims = &g.data(w)?.dims;
            let b_dims = &g.data(bias)?.dims;
            // The bias must broadcast along the channel axis of the
            // `[N, C, spatial...]` output: `C` followed by one 1 per spatial
            // axis, possibly with leading 1s.
            let channels = *w_dims.first()?;
            let spatial = w_dims.len().checked_sub(2)?;
            let numel: i64 = b_dims.iter().product();
            let channel_axis = b_dims.len().checked_sub(spatial + 1)?;
            (numel == channels && b_dims[channel_axis] == channels).then_some((conv, x, w, bias))
        });
        if let Some((conv, x, w, bias)) = fused {
            let Op::Conv(attrs) = g.node(conv).op.clone() else {
                unreachable!()
            };
            let proto = g.node(conv).proto.clone();
            g.fuse(
                &[conv, add],
                Op::Conv(attrs),
                proto,
                vec![Some(x), Some(w), Some(bias)],
            );
        }
    }
}

/// `Conv` followed by `Relu`, or a 2D `Conv` followed by `Silu` → `ConvAct`.
fn fuse_conv_act(g: &mut Graph) {
    let acts: Vec<(NodeId, Activation)> = g
        .nodes()
        .filter_map(|(id, n)| match n.op {
            Op::Unary(Unary::Relu) => Some((id, Activation::Relu)),
            Op::Silu => Some((id, Activation::Silu)),
            _ => None,
        })
        .collect();
    for (act_id, act) in acts {
        if !g.contains(act_id) {
            continue;
        }
        let input = g.node(act_id).input(0);
        let Some(conv) = sole_producer(g, input, |op| matches!(op, Op::Conv(_))) else {
            continue;
        };
        let Op::Conv(attrs) = g.node(conv).op.clone() else {
            unreachable!()
        };
        // Only the 2D kernel fuses SiLU.
        if act == Activation::Silu && attrs.kernel_shape.as_ref().is_none_or(|k| k.len() < 2) {
            continue;
        }
        let inputs = g.node(conv).inputs().to_vec();
        let proto = g.node(conv).proto.clone();
        let op = Op::ConvAct { conv: attrs, act };
        g.fuse(&[conv, act_id], op, proto, inputs);
    }
}

/// `MatMul` followed by `Add` → `Linear`.
fn fuse_linear(g: &mut Graph) {
    for add in binary_nodes(g, Binary::Add) {
        if !g.contains(add) {
            continue;
        }
        let fused = (0..2).find_map(|k| {
            let node = g.node(add);
            let mm = sole_producer(g, node.input(k), |op| matches!(op, Op::MatMul))?;
            // `Linear` computes the product's shape: not an `Add` that broadcasts
            // it to more (when shape inference knows both).
            let shape = |v: Option<ValueId>| v.and_then(|v| g.value(v).shape.clone());
            match (shape(node.input(k)), shape(node.output(0))) {
                (Some(product), Some(sum)) if product != sum => return None,
                _ => {}
            }
            Some((mm, node.input(1 - k)))
        });
        if let Some((mm, bias)) = fused {
            let inputs = vec![g.node(mm).input(0), g.node(mm).input(1), bias];
            let proto = named_after(g, mm);
            g.fuse(&[mm, add], Op::Linear, proto, inputs);
        }
    }
}

/// The dynamically quantized linear layer ONNX Runtime's
/// `quantize_dynamic` emits, optionally followed by `Relu`:
///
/// ```text
/// q, s, z = DynamicQuantizeLinear(x)
/// y = Cast(MatMulInteger(q, w, z, w_zp)) * (s * w_scale) + bias
/// ```
///
/// → `QLinearDynamic`.
fn fuse_dynamic_quant_linear(g: &mut Graph) {
    for add in binary_nodes(g, Binary::Add) {
        if !g.contains(add) {
            continue;
        }
        let fused = (0..2).find_map(|k| {
            let node = g.node(add);
            let (dequant, bias) = (node.input(k), node.input(1 - k));
            let mul = sole_producer(g, dequant, binary(Binary::Mul))?;
            let (cast, scale) = (0..2).find_map(|j| {
                let input = g.node(mul).input(j);
                let cast = sole_producer(g, input, |op| matches!(op, Op::Cast { .. }))?;
                Some((cast, g.node(mul).input(1 - j)))
            })?;
            let input = g.node(cast).input(0);
            let mmi = sole_producer(g, input, |op| matches!(op, Op::MatMulInteger))?;
            let mmi_node = g.node(mmi);
            let (q, w, z, w_zp) = (
                mmi_node.input(0),
                mmi_node.input(1),
                mmi_node.input(2),
                mmi_node.input(3),
            );
            w_zp?;
            let dql = sole_producer(g, q, |op| matches!(op, Op::DynamicQuantizeLinear))?;
            let dql_node = g.node(dql);
            if dql_node.output(2) != z || g.uses(z?) != 1 {
                return None;
            }
            let s = dql_node.output(1)?;
            let scale_mul = sole_producer(g, scale, binary(Binary::Mul))?;
            let w_scale = other_operand(g, scale_mul, s)?;
            if g.uses(s) != 1 {
                return None;
            }
            let x = dql_node.input(0);
            let mut nodes = vec![dql, scale_mul, mmi, cast, mul, add];
            // A ReLU reading nothing but the layer joins it.
            let out = g.node(add).output(0)?;
            let relu = match g.value(out).consumers() {
                [r] if g.uses(out) == 1 && unary(Unary::Relu)(&g.node(*r).op) => Some(*r),
                _ => None,
            };
            nodes.extend(relu);
            Some((nodes, vec![x, w, w_scale, w_zp, bias], relu.is_some()))
        });
        if let Some((nodes, inputs, relu)) = fused {
            let proto = named_after(g, add);
            g.fuse(&nodes, Op::QLinearDynamic { relu }, proto, inputs);
        }
    }
}

/// `Concat([weight, ConstantOfShape(shape)], axis = 0)` → `EmbeddingConcat`.
fn fuse_embedding_concat(g: &mut Graph) {
    let concats: Vec<NodeId> = g
        .nodes()
        .filter(|(_, n)| matches!(n.op, Op::Concat { axis: 0 }) && n.inputs().len() == 2)
        .map(|(id, _)| id)
        .collect();
    for concat in concats {
        if !g.contains(concat) {
            continue;
        }
        let node = g.node(concat);
        let weight = node.input(0);
        let is_cos = |op: &Op| matches!(op, Op::ConstantOfShape { .. });
        let Some(cos) = sole_producer(g, node.input(1), is_cos) else {
            continue;
        };
        let Op::ConstantOfShape { value } = g.node(cos).op.clone() else {
            unreachable!()
        };
        let shape = g.node(cos).input(0);
        let proto = named_after(g, concat);
        let op = Op::EmbeddingConcat { value };
        g.fuse(&[cos, concat], op, proto, vec![shape, weight]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::onnx_proto::{GraphProto, TensorProto, ValueInfoProto};

    fn node(op: &str, inputs: &[&str], output: &str) -> NodeProto {
        NodeProto {
            op_type: op.to_string(),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: vec![output.to_string()],
            ..Default::default()
        }
    }

    fn scalar(name: &str, v: f32) -> TensorProto {
        TensorProto {
            name: name.to_string(),
            data_type: crate::compiler::ir::DType::F32.to_onnx(),
            float_data: vec![v],
            ..Default::default()
        }
    }

    /// Fuses `nodes`, whose last output is the graph output, and returns
    /// the resulting op types.
    fn fused(nodes: Vec<NodeProto>, initializers: Vec<TensorProto>) -> Vec<String> {
        let last = nodes.last().unwrap().output[0].clone();
        let g = GraphProto {
            node: nodes,
            initializer: initializers,
            output: vec![ValueInfoProto {
                name: last,
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut ir = Graph::from_onnx(g);
        crate::compiler::passes::infer_dtypes(&mut ir);
        fuse(&mut ir);
        ir.into_onnx().node.into_iter().map(|n| n.op_type).collect()
    }

    fn gelu_constants() -> Vec<TensorProto> {
        vec![
            scalar("sqrt2", std::f32::consts::SQRT_2),
            scalar("one", 1.0),
            scalar("half", 0.5),
        ]
    }

    /// `(x * (1 + erf(x / sqrt2))) * 0.5`
    #[test]
    fn fuses_gelu_with_scale_applied_last() {
        let nodes = vec![
            node("Div", &["x", "sqrt2"], "d"),
            node("Erf", &["d"], "e"),
            node("Add", &["e", "one"], "a"),
            node("Mul", &["x", "a"], "m1"),
            node("Mul", &["m1", "half"], "y"),
        ];
        assert_eq!(fused(nodes, gelu_constants()), ["Gelu"]);
    }

    /// `x * (0.5 * (1 + erf(x / sqrt2)))`, which is what torch.onnx emits.
    #[test]
    fn fuses_gelu_with_scale_applied_first() {
        let nodes = vec![
            node("Div", &["x", "sqrt2"], "d"),
            node("Erf", &["d"], "e"),
            node("Add", &["e", "one"], "a"),
            node("Mul", &["half", "a"], "m1"),
            node("Mul", &["x", "m1"], "y"),
        ];
        assert_eq!(fused(nodes, gelu_constants()), ["Gelu"]);
    }

    #[test]
    fn keeps_erf_chain_that_never_multiplies_by_the_input() {
        let nodes = vec![
            node("Div", &["x", "sqrt2"], "d"),
            node("Erf", &["d"], "e"),
            node("Add", &["e", "one"], "a"),
            node("Mul", &["half", "a"], "m1"),
            node("Mul", &["m1", "half"], "y"),
        ];
        assert!(!fused(nodes, gelu_constants()).contains(&"Gelu".to_string()));
    }

    #[test]
    fn keeps_erf_chain_with_other_constants() {
        let nodes = vec![
            node("Div", &["x", "sqrt2"], "d"),
            node("Erf", &["d"], "e"),
            node("Add", &["e", "one"], "a"),
            node("Mul", &["x", "a"], "m1"),
            node("Mul", &["m1", "half"], "y"),
        ];
        let mut constants = gelu_constants();
        constants[2] = scalar("half", 0.25);
        assert!(!fused(nodes, constants).contains(&"Gelu".to_string()));
    }

    #[test]
    fn fuses_silu_and_conv_silu() {
        let mut conv = node("Conv", &["x", "w"], "c");
        conv.attribute
            .push(crate::model::onnx_proto::AttributeProto {
                name: "kernel_shape".to_string(),
                ints: vec![3, 3],
                ..Default::default()
            });
        let nodes = vec![
            conv,
            node("Sigmoid", &["c"], "s"),
            node("Mul", &["c", "s"], "y"),
        ];
        assert_eq!(fused(nodes, vec![]), ["ConvAct"]);
    }

    #[test]
    fn keeps_sigmoid_read_elsewhere() {
        let nodes = vec![
            node("Sigmoid", &["x"], "s"),
            node("Mul", &["x", "s"], "m"),
            node("Add", &["m", "s"], "y"),
        ];
        assert_eq!(fused(nodes, vec![]), ["Sigmoid", "Mul", "Add"]);
    }

    /// `MatMul` + `Add` of shaped inputs fuses when the sum has the product's shape,
    /// and not when the `Add` broadcasts the product to more (a bias with extra
    /// leading dimensions), which `Linear` cannot compute.
    #[test]
    fn fuses_linear_only_when_the_bias_keeps_the_product_shape() {
        use crate::model::onnx_proto::{TensorShapeProto, TypeProto, tensor_shape_proto, type_proto};
        let input = |name: &str, dims: &[i64]| ValueInfoProto {
            name: name.to_string(),
            r#type: Some(TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: crate::compiler::ir::DType::F32.to_onnx(),
                    shape: Some(TensorShapeProto {
                        dim: dims
                            .iter()
                            .map(|&d| tensor_shape_proto::Dimension {
                                value: Some(tensor_shape_proto::dimension::Value::DimValue(d)),
                                ..Default::default()
                            })
                            .collect(),
                    }),
                })),
                ..Default::default()
            }),
            ..Default::default()
        };
        let fused = |bias: &[i64]| {
            let g = GraphProto {
                node: vec![node("MatMul", &["x", "w"], "m"), node("Add", &["m", "b"], "y")],
                input: vec![input("x", &[2, 3]), input("w", &[3, 4]), input("b", bias)],
                output: vec![ValueInfoProto {
                    name: "y".to_string(),
                    ..Default::default()
                }],
                ..Default::default()
            };
            let mut ir = Graph::from_onnx(g);
            crate::compiler::passes::infer_dtypes(&mut ir);
            crate::compiler::passes::fold_shapes(&mut ir);
            fuse(&mut ir);
            ir.into_onnx().node.into_iter().map(|n| n.op_type).collect::<Vec<_>>()
        };
        assert_eq!(fused(&[4]), ["Linear"]);
        assert_eq!(fused(&[2, 4]), ["Linear"]);
        assert_eq!(fused(&[5, 1, 4]), ["MatMul", "Add"]);
        assert_eq!(fused(&[1, 2, 4]), ["MatMul", "Add"]);
    }

    #[test]
    fn fuses_matmul_add_into_linear() {
        let nodes = vec![
            node("MatMul", &["x", "w"], "m"),
            node("Relu", &["x"], "unrelated"),
            node("Add", &["b", "m"], "y"),
        ];
        assert_eq!(fused(nodes, vec![]), ["Relu", "Linear"]);
    }

    /// T-one's ScaleNorm: `gamma * (x / (ReduceL2(x) * scale + eps))`, of an
    /// f32 `x` (a Cast, so its type is known), with the operands of each `Mul`
    /// and `Add` in either order.
    #[test]
    fn fuses_scale_norm() {
        let nodes = |swap: bool| {
            let mut cast = node("Cast", &["input"], "x");
            cast.attribute.push(crate::model::onnx_proto::AttributeProto {
                name: "to".to_string(),
                i: 1,
                ..Default::default()
            });
            let mut l2 = node("ReduceL2", &["x"], "norm");
            l2.attribute.push(crate::model::onnx_proto::AttributeProto {
                name: "axes".to_string(),
                ints: vec![-1],
                ..Default::default()
            });
            let order = |a: &'static str, b: &'static str| if swap { [b, a] } else { [a, b] };
            vec![
                cast,
                l2,
                node("Mul", &order("norm", "scale"), "scaled"),
                node("Add", &order("scaled", "eps"), "denom"),
                node("Div", &["x", "denom"], "normed"),
                node("Mul", &order("gamma", "normed"), "y"),
            ]
        };
        let gamma = TensorProto {
            name: "gamma".to_string(),
            data_type: crate::compiler::ir::DType::F32.to_onnx(),
            dims: vec![4],
            float_data: vec![1.0, 2.0, 3.0, 4.0],
            ..Default::default()
        };
        let constants = vec![scalar("scale", 0.5), scalar("eps", 1e-8), gamma];
        for swap in [false, true] {
            assert_eq!(fused(nodes(swap), constants.clone()), ["Cast", "ScaleNorm"]);
        }

        // A scale that is not a constant, or a norm over another axis, is left.
        let mut other = constants.clone();
        other.remove(0);
        assert_eq!(fused(nodes(false), other).len(), 6);
        let mut nodes = nodes(false);
        nodes[1].attribute[0].ints = vec![1];
        assert_eq!(fused(nodes, constants).len(), 6);
    }

    #[test]
    fn fuses_layer_norm() {
        let mut nodes = vec![
            node("ReduceMean", &["x"], "mean"),
            node("Sub", &["x", "mean"], "d"),
            node("Pow", &["d", "two"], "p"),
            node("ReduceMean", &["p"], "var"),
            node("Add", &["var", "eps"], "ve"),
            node("Sqrt", &["ve"], "std"),
            node("Div", &["d", "std"], "n"),
            node("Mul", &["n", "scale"], "s"),
            node("Add", &["s", "bias"], "y"),
        ];
        for i in [0, 3] {
            nodes[i]
                .attribute
                .push(crate::model::onnx_proto::AttributeProto {
                    name: "axes".to_string(),
                    ints: vec![-1],
                    ..Default::default()
                });
        }
        let constants = vec![scalar("two", 2.0), scalar("eps", 1e-6)];
        assert_eq!(
            fused(nodes.clone(), constants.clone()),
            ["LayerNormalization"]
        );

        // Normalizing over another axis is not a LayerNorm over the last.
        nodes[3].attribute[0].ints = vec![1];
        assert_eq!(fused(nodes, constants).len(), 9);
    }
}
