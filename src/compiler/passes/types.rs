//! Element type inference.

use crate::compiler::ir::{DType, Graph, Op, Unary};
use std::collections::HashMap;

/// Sets the element type of every value of `g` and its subgraphs whose type
/// the model does not state, following the ONNX type rules of the producing
/// operator. Values whose type cannot be derived stay
/// [`DType::Undefined`].
pub(crate) fn infer_dtypes(g: &mut Graph) {
    infer_graph(g, &HashMap::new());
}

/// Types of the values of enclosing graphs, by name.
type Scope = HashMap<String, DType>;

fn infer_graph(g: &mut Graph, outer: &Scope) {
    for v in g.captures().collect::<Vec<_>>() {
        if let Some(&t) = outer.get(&g.value(v).name) {
            g.value_mut(v).dtype = t;
        }
    }
    for id in g.node_ids() {
        // Subgraphs see every value typed so far.
        if !g.node(id).op.subgraphs().is_empty() {
            let mut scope = outer.clone();
            scope.extend(
                g.value_ids()
                    .map(|v| (g.value(v).name.clone(), g.value(v).dtype))
                    .filter(|(_, t)| *t != DType::Undefined),
            );
            for sub in g.node_mut(id).op.subgraphs_mut() {
                infer_graph(sub, &scope);
            }
        }
        let node = g.node(id);
        let input = |i: usize| node.input(i).map_or(DType::Undefined, |v| g.value(v).dtype);
        let types: Vec<_> = node
            .outputs()
            .iter()
            .enumerate()
            .filter_map(|(i, v)| {
                let t = match follows_input(&node.op, i) {
                    Some(k) => input(k),
                    None => output_dtype(&node.op, i, &input),
                };
                Some(((*v)?, t))
            })
            .collect();
        for (v, t) in types {
            let value = g.value_mut(v);
            if value.dtype == DType::Undefined {
                value.dtype = t;
            }
        }
    }
}

/// The input whose element type output `i` of `op` has, for operators whose
/// output type is that of an input.
pub(crate) fn follows_input(op: &Op, i: usize) -> Option<usize> {
    match op {
        Op::TopK { .. } if i == 1 => None,
        Op::ArgMax { .. }
        | Op::Cast { .. }
        | Op::Constant { .. }
        | Op::ConstantOfShape { .. }
        | Op::ConstantOfShapeTo { .. }
        | Op::ConvInteger(_)
        | Op::DynamicQuantizeLinear
        | Op::EmbeddingConcat { .. }
        | Op::MatMulInteger
        | Op::Shape { .. }
        | Op::Size
        | Op::Unary(Unary::Not)
        | Op::QuantizeLinear(_)
        | Op::DequantizeLinear(_)
        | Op::QLinearMatMulI8
        | Op::QLinearMatMulAct
        | Op::QWeightMatMul
        | Op::QLinearDynamic { .. }
        | Op::If { .. }
        | Op::Loop { .. } => None,
        Op::Binary(binary) if binary.is_predicate() => None,
        Op::Where => Some(1),
        Op::Onnx { op_type, .. } => match op_type.as_str() {
            "ArgMin" | "NonZero" | "IsNaN" | "IsInf" => None,
            _ => Some(0),
        },
        _ => Some(0),
    }
}

/// ONNX type of output `i` of `op`, for operators [`follows_input`] does not
/// cover. `If` and `Loop` take theirs from their subgraphs.
fn output_dtype(op: &Op, i: usize, input: &dyn Fn(usize) -> DType) -> DType {
    use DType::*;
    let tensor_type = |t: &Option<crate::model::onnx_proto::TensorProto>| {
        t.as_ref().map_or(F32, |t| DType::from_onnx(t.data_type))
    };
    match op {
        Op::Cast { to } => *to,
        Op::Constant { value } => DType::from_onnx(value.data_type),
        Op::ConstantOfShape { value }
        | Op::ConstantOfShapeTo { value, .. }
        | Op::EmbeddingConcat { value } => tensor_type(value),
        Op::Shape { .. } | Op::Size | Op::ArgMax { .. } | Op::TopK { .. } => I64,
        Op::ConvInteger(_) | Op::MatMulInteger => I32,
        Op::DynamicQuantizeLinear => [U8, F32, U8].get(i).copied().unwrap_or(Undefined),
        Op::Binary(_) | Op::Unary(Unary::Not) => Bool,
        Op::QuantizeLinear(attrs) => attrs.output_dtype.unwrap_or(match input(2) {
            Undefined => U8,
            t => t,
        }),
        Op::If { then_branch, .. } => then_branch
            .outputs
            .get(i)
            .map_or(Undefined, |&v| then_branch.value(v).dtype),
        // Outputs are the body's outputs after its condition.
        Op::Loop { body } => body
            .outputs
            .get(i + 1)
            .map_or(Undefined, |&v| body.value(v).dtype),
        Op::Onnx { op_type, .. } => match op_type.as_str() {
            "IsNaN" | "IsInf" => Bool,
            _ => I64,
        },
        _ => F32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::onnx_proto::{
        AttributeProto, GraphProto, NodeProto, TypeProto, ValueInfoProto, type_proto,
    };

    fn node(op: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
        NodeProto {
            op_type: op.to_string(),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    fn info(name: &str, dtype: Option<DType>) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_string(),
            r#type: dtype.map(|t| TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: t.to_onnx(),
                    shape: None,
                })),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    /// Element types of `names` after inference on a graph with float input
    /// `x`.
    fn dtypes(nodes: Vec<NodeProto>, names: &[&str]) -> Vec<DType> {
        let mut g = Graph::from_onnx(GraphProto {
            node: nodes,
            input: vec![info("x", Some(DType::F32))],
            ..Default::default()
        });
        infer_dtypes(&mut g);
        names
            .iter()
            .map(|n| g.value(g.find_value(n).unwrap()).dtype)
            .collect()
    }

    #[test]
    fn follows_operator_type_rules() {
        let mut cast = node("Cast", &["s"], &["c"]);
        cast.attribute.push(AttributeProto {
            name: "to".to_string(),
            i: DType::I32.to_onnx() as i64,
            ..Default::default()
        });
        let types = dtypes(
            vec![
                node("Shape", &["x"], &["s"]),
                cast,
                node("Equal", &["c", "c"], &["e"]),
                node("Where", &["e", "x", "x"], &["w"]),
                node("TopK", &["w", "s"], &["values", "indices"]),
                node("DynamicQuantizeLinear", &["x"], &["q", "scale", "zp"]),
                node("MatMulInteger", &["q", "q"], &["mm"]),
            ],
            &["s", "c", "e", "w", "values", "indices", "q", "scale", "mm"],
        );
        use DType::*;
        assert_eq!(types, [I64, I32, Bool, F32, F32, I64, U8, F32, I32]);
    }

    #[test]
    fn types_values_of_subgraphs() {
        // The branch returns a value of the outer graph, cast to int64.
        let mut cast = node("Cast", &["x"], &["xi"]);
        cast.attribute.push(AttributeProto {
            name: "to".to_string(),
            i: DType::I64.to_onnx() as i64,
            ..Default::default()
        });
        let branch = GraphProto {
            node: vec![node("Neg", &["xi"], &["n"])],
            output: vec![info("n", None)],
            ..Default::default()
        };
        let mut if_node = node("If", &["cond"], &["r"]);
        if_node.attribute = ["then_branch", "else_branch"]
            .map(|name| AttributeProto {
                name: name.to_string(),
                g: Some(branch.clone()),
                ..Default::default()
            })
            .to_vec();
        assert_eq!(dtypes(vec![cast, if_node], &["r"]), [DType::I64]);
    }
}
