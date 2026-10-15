//! Common subexpression elimination.

use crate::compiler::ir::{Graph, NodeId, Op, ValueId};
use prost::Message;
use std::collections::HashMap;

/// Initializers with at most this many elements are compared by contents,
/// so that two copies of a constant count as the same input.
const MAX_COMPARED_ELEMS: i64 = 1024;

/// Operators whose every run gives a different result.
const RANDOM: [&str; 6] = [
    "Bernoulli",
    "Multinomial",
    "RandomNormal",
    "RandomNormalLike",
    "RandomUniform",
    "RandomUniformLike",
];

/// Merges the nodes of `g` and its subgraphs that run the same operator with
/// the same attributes on the same inputs: the readers of a node's outputs
/// read those of the first such node instead, and the node is removed.
///
/// Exported models repeat computations: the rotary embedding of every layer
/// takes the sine and the cosine of two identical products, and each
/// attention block builds the same constants for each of its reshapes.
pub(crate) fn eliminate_common_subexpressions(g: &mut Graph) {
    for id in g.node_ids() {
        for sub in g.node_mut(id).op.subgraphs_mut() {
            eliminate_common_subexpressions(sub);
        }
    }
    let same = same_initializers(g);
    let input = |v: ValueId| same.get(&v).copied().unwrap_or(v);
    let mut seen: HashMap<Vec<u8>, NodeId> = HashMap::new();
    for id in g.node_ids() {
        let node = g.node(id);
        if !node.op.subgraphs().is_empty()
            || matches!(&node.op, Op::Onnx { op_type, .. } if RANDOM.contains(&op_type.as_str()))
        {
            continue;
        }
        let mut proto = g.node_proto(id);
        proto.name.clear();
        proto.doc_string.clear();
        proto.output.clear();
        proto.input = node
            .inputs()
            .iter()
            .map(|v| v.map_or_else(String::new, |v| g.value(input(v)).name.clone()))
            .collect();
        let key = proto.encode_to_vec();
        let Some(&first) = seen.get(&key) else {
            seen.insert(key, id);
            continue;
        };
        if let Some(outputs) = replacements(g, first, id) {
            for (old, new) in outputs {
                replace_uses(g, old, new);
            }
            g.remove_node(id);
        }
    }
}

/// Maps each small initializer to the first one of the same contents.
fn same_initializers(g: &Graph) -> HashMap<ValueId, ValueId> {
    let mut first: HashMap<Vec<u8>, ValueId> = HashMap::new();
    let mut same = HashMap::new();
    for &v in g.initializers() {
        // An initializer of an input's name is only its default value.
        let Some(data) = g.data(v).filter(|_| !g.inputs.contains(&v)) else {
            continue;
        };
        if data.dims.iter().product::<i64>() > MAX_COMPARED_ELEMS {
            continue;
        }
        let mut data = data.clone();
        data.name.clear();
        let canonical = *first.entry(data.encode_to_vec()).or_insert(v);
        same.insert(v, canonical);
    }
    same
}

/// Pairs of an output of `dup` and the output of `first` that replaces it,
/// or `None` if the outputs of `dup` cannot all be replaced: one is a graph
/// output, or a subgraph reads it, or `first` does not produce it.
fn replacements(g: &Graph, first: NodeId, dup: NodeId) -> Option<Vec<(ValueId, ValueId)>> {
    let mut pairs = Vec::new();
    for (i, old) in g.node(dup).outputs().iter().enumerate() {
        let Some(old) = *old else {
            continue;
        };
        let new = g.node(first).output(i)?;
        let read_by_subgraph = g
            .value(old)
            .consumers()
            .iter()
            .any(|&c| !g.node(c).op.subgraphs().is_empty());
        if g.outputs.contains(&old) || read_by_subgraph {
            return None;
        }
        pairs.push((old, new));
    }
    Some(pairs)
}

/// Makes every node reading `old` read `new` instead.
pub(super) fn replace_uses(g: &mut Graph, old: ValueId, new: ValueId) {
    let mut readers = g.value(old).consumers().to_vec();
    readers.sort();
    readers.dedup();
    for r in readers {
        let inputs = g
            .node(r)
            .inputs()
            .iter()
            .map(|&v| if v == Some(old) { Some(new) } else { v })
            .collect();
        g.set_inputs(r, inputs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ir::DType;
    use crate::model::onnx_proto::{
        AttributeProto, GraphProto, NodeProto, TensorProto, ValueInfoProto,
    };

    fn node(op: &str, inputs: &[&str], output: &str) -> NodeProto {
        NodeProto {
            op_type: op.to_string(),
            name: output.to_string(),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: vec![output.to_string()],
            ..Default::default()
        }
    }

    fn ints(name: &str, values: &[i64]) -> TensorProto {
        TensorProto {
            name: name.to_string(),
            data_type: DType::I64.to_onnx(),
            dims: vec![values.len() as i64],
            int64_data: values.to_vec(),
            ..Default::default()
        }
    }

    /// The nodes left of `nodes` with graph outputs `outputs`, as
    /// `op(inputs)`.
    fn cse(nodes: Vec<NodeProto>, initializer: Vec<TensorProto>, outputs: &[&str]) -> Vec<String> {
        let info = |name: &&str| ValueInfoProto {
            name: name.to_string(),
            ..Default::default()
        };
        let mut g = Graph::from_onnx(GraphProto {
            node: nodes,
            initializer,
            input: vec![info(&"x")],
            output: outputs.iter().map(info).collect(),
            ..Default::default()
        });
        eliminate_common_subexpressions(&mut g);
        g.into_onnx()
            .node
            .iter()
            .map(|n| format!("{}({})", n.op_type, n.input.join(", ")))
            .collect()
    }

    #[test]
    fn merges_chains_of_the_same_computation() {
        let nodes = vec![
            node("Sin", &["x"], "a"),
            node("Sin", &["x"], "b"),
            node("Neg", &["a"], "c"),
            node("Neg", &["b"], "d"),
            node("Add", &["c", "d"], "y"),
        ];
        assert_eq!(
            cse(nodes, vec![], &["y"]),
            ["Sin(x)", "Neg(a)", "Add(c, c)"]
        );
    }

    #[test]
    fn compares_small_initializers_by_contents() {
        let nodes = vec![
            node("Mul", &["x", "k1"], "a"),
            node("Mul", &["x", "k2"], "b"),
            node("Mul", &["x", "k3"], "c"),
            node("Concat", &["a", "b", "c"], "y"),
        ];
        let initializers = vec![
            ints("k1", &[2, 3]),
            ints("k2", &[2, 3]),
            ints("k3", &[2, 4]),
        ];
        assert_eq!(
            cse(nodes, initializers, &["y"]),
            ["Mul(x, k1)", "Mul(x, k3)", "Concat(a, a, c)"]
        );
    }

    #[test]
    fn keeps_different_attributes_graph_outputs_and_random_values() {
        let axis = |i: i64| AttributeProto {
            name: "axis".to_string(),
            i,
            r#type: 2, // INT
            ..Default::default()
        };
        let mut softmax = node("Softmax", &["x"], "a");
        softmax.attribute.push(axis(0));
        let mut other_axis = node("Softmax", &["x"], "b");
        other_axis.attribute.push(axis(1));
        let nodes = vec![
            softmax,
            other_axis,
            // A graph output keeps its name.
            node("Exp", &["x"], "c"),
            node("Exp", &["x"], "d"),
            node("RandomNormalLike", &["x"], "e"),
            node("RandomNormalLike", &["x"], "f"),
            node("Concat", &["a", "b", "c", "e", "f"], "y"),
        ];
        assert_eq!(
            cse(nodes, vec![], &["y", "d"]),
            [
                "Softmax(x)",
                "Softmax(x)",
                "Exp(x)",
                "Exp(x)",
                "RandomNormalLike(x)",
                "RandomNormalLike(x)",
                "Concat(a, b, c, e, f)"
            ]
        );
    }
}
