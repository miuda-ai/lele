//! Removal of the nodes that leave their input as it is, and of transposes
//! that do not move data.

use super::cse::replace_uses;
use super::fold::exact_ints;
use super::fuse::{constant, constant_tensor};
use crate::compiler::ir::{Binary, Dim, Graph, NodeId, Op, TargetDim, ValueId};

/// Removes the nodes of `g` and its subgraphs whose output is their input:
/// `Identity`, `Dropout` without its mask, a `Cast` to the type the input
/// has, a product or quotient by ones and a sum or difference with zeros
/// that do not broadcast, a `Pad` by zeros, a `Slice` of everything, an
/// `Expand` to the shape the input has and a `Concat` of one input, once
/// those of its inputs that are empty along its axis are dropped. Merges
/// each transpose of a transpose into one, and makes a transpose that only
/// moves axes of size 1 a reshape, which does not copy.
///
/// Exporters leave such nodes: a residual branch scaled by a weight of 1
/// (SenseVoice), casts of integers to their type (MOSS-TTS-Nano, whose
/// rotary embeddings then differ by name in each layer), attention heads
/// moved past the one position a decoding step takes (Moonshine).
pub(crate) fn simplify(g: &mut Graph) {
    for id in g.node_ids() {
        for sub in g.node_mut(id).op.subgraphs_mut() {
            simplify(sub);
        }
    }
    for id in g.node_ids() {
        if !g.contains(id) {
            continue;
        }
        drop_empty_inputs(g, id);
        if let Some(src) = same_as_input(g, id) {
            bypass(g, id, src);
        } else if matches!(g.node(id).op, Op::Transpose { .. }) {
            simplify_transpose(g, id);
        }
    }
}

/// Removes the inputs of `Concat` node `id` that are empty along its axis,
/// keeping one: Supertonic 3 appends `ConstantOfShape([0])` to constant
/// pads, which then stay a run-time concatenation.
fn drop_empty_inputs(g: &mut Graph, id: NodeId) {
    let node = g.node(id);
    let Op::Concat { axis } = node.op else {
        return;
    };
    let empty = |v: &Option<ValueId>| {
        let shape = v.and_then(|v| g.value(v).shape.as_ref());
        shape.is_some_and(|s| {
            let axis = if axis < 0 {
                axis + s.len() as i64
            } else {
                axis
            };
            usize::try_from(axis).is_ok_and(|a| s.get(a) == Some(&Dim::Fixed(0)))
        })
    };
    let mut kept: Vec<Option<ValueId>> = node
        .inputs()
        .iter()
        .filter(|v| !empty(v))
        .copied()
        .collect();
    if kept.len() == node.inputs().len() {
        return;
    }
    if kept.is_empty() {
        kept.push(node.input(0));
    }
    g.set_inputs(id, kept);
}

/// The input that node `id` outputs unchanged, if it does.
fn same_as_input(g: &Graph, id: NodeId) -> Option<ValueId> {
    let node = g.node(id);
    let x = node.input(0)?;
    let out = node.output(0)?;
    let shape = |v: ValueId| g.value(v).shape.clone();
    let known = |v: ValueId| shape(v).filter(|s| !s.contains(&Dim::Unknown));
    let ints = |i: usize| constant_tensor(g, node.input(i)).and_then(exact_ints);
    match &node.op {
        Op::Identity => Some(x),
        Op::Onnx { op_type, .. } if op_type == "Dropout" => {
            let training = node.input(2).is_some() && ints(2)? != [0];
            let mask = node.output(1).is_some_and(|m| g.uses(m) > 0);
            (!training && !mask).then_some(x)
        }
        Op::Cast { to } => (g.value(x).dtype == *to).then_some(x),
        Op::Binary(op @ (Binary::Add | Binary::Sub | Binary::Mul | Binary::Div)) => {
            let neutral = if matches!(op, Binary::Add | Binary::Sub) {
                0.0
            } else {
                1.0
            };
            // Either operand of a sum or product, the second of a
            // difference or quotient.
            let commutes = matches!(op, Binary::Add | Binary::Mul);
            let (a, b) = (node.input(0)?, node.input(1)?);
            let candidates = if commutes {
                vec![(a, b), (b, a)]
            } else {
                vec![(a, b)]
            };
            candidates.into_iter().find_map(|(x, k)| {
                let values = constant(g, Some(k))?;
                // Of no higher rank, and each dimension 1 or that of `x`,
                // `k` does not broadcast `x` to another shape.
                let k_dims = &constant_tensor(g, Some(k))?.dims;
                let x_dims = shape(x)?;
                let keeps_shape = k_dims.len() <= x_dims.len()
                    && k_dims
                        .iter()
                        .rev()
                        .zip(x_dims.iter().rev())
                        .all(|(&k, x)| k == 1 || *x == Dim::Fixed(k));
                let neutral = values.iter().all(|&v| v == neutral);
                (keeps_shape && neutral && g.value(x).dtype == g.value(out).dtype).then_some(x)
            })
        }
        Op::Pad { .. } => ints(1)?.iter().all(|&p| p == 0).then_some(x),
        Op::Slice => {
            let rank = shape(x)?.len() as i64;
            let (starts, ends) = (ints(1)?, ints(2)?);
            let axes = match node.input(3) {
                Some(_) => ints(3)?,
                None => (0..starts.len() as i64).collect(),
            };
            let steps = match node.input(4) {
                Some(_) => ints(4)?,
                None => vec![1; starts.len()],
            };
            let dims = shape(x)?;
            let everything = (0..starts.len()).all(|i| {
                let axis = (axes[i] + rank) % rank;
                let end_past = match dims[axis as usize] {
                    Dim::Fixed(d) => ends[i] >= d,
                    _ => ends[i] >= i64::MAX / 2,
                };
                steps[i] == 1 && starts[i] == 0 && end_past
            });
            everything.then_some(x)
        }
        Op::Expand => {
            let same = known(x).is_some_and(|s| Some(s) == known(out));
            (constant_tensor(g, node.input(1)).is_some() && same).then_some(x)
        }
        Op::Concat { .. } => (node.inputs().len() == 1).then_some(x),
        _ => None,
    }
}

/// Removes node `id`, whose readers read `src` instead. Where its output is
/// a graph output, the node producing `src` produces that output instead,
/// if nothing else reads `src`. Leaves the node where a subgraph reads its
/// output, by name.
fn bypass(g: &mut Graph, id: NodeId, src: ValueId) {
    let node = g.node(id);
    let Some(out) = node.output(0) else {
        return;
    };
    let read_by_subgraph = g
        .value(out)
        .consumers()
        .iter()
        .any(|&c| !g.node(c).op.subgraphs().is_empty());
    if read_by_subgraph {
        return;
    }
    if !g.outputs.contains(&out) {
        replace_uses(g, out, src);
        g.remove_node(id);
        return;
    }
    let Some(producer) = g.value(src).producer() else {
        return;
    };
    if g.uses(src) != 1 || !g.node(producer).op.subgraphs().is_empty() {
        return;
    }
    let i = g
        .node(producer)
        .outputs()
        .iter()
        .position(|&v| v == Some(src))
        .unwrap();
    g.remove_node(id);
    g.set_output(producer, i, Some(out));
}

/// The permutation of transpose `id` of an input of rank `rank`.
fn perm(g: &Graph, id: NodeId, rank: Option<usize>) -> Option<Vec<usize>> {
    let Op::Transpose { perm } = &g.node(id).op else {
        return None;
    };
    match perm {
        Some(p) => {
            let rank = p.len() as i64;
            p.iter()
                .map(|&a| usize::try_from((a + rank) % rank).ok())
                .collect()
        }
        None => Some((0..rank?).rev().collect()),
    }
}

/// Merges transpose `id` with a transpose producing its input that nothing
/// else reads, and then removes it if it permutes nothing, or makes it a
/// reshape if it only moves axes of size 1.
fn simplify_transpose(g: &mut Graph, id: NodeId) {
    let Some(mut x) = g.node(id).input(0) else {
        return;
    };
    let rank = |g: &Graph, v: ValueId| g.value(v).shape.as_ref().map(Vec::len);
    let Some(mut p) = perm(g, id, rank(g, x)) else {
        return;
    };
    if let Some(inner) = g.value(x).producer().filter(|_| g.uses(x) == 1)
        && let Some(src) = g.node(inner).input(0)
        && let Some(q) = perm(g, inner, rank(g, src))
        && q.len() == p.len()
    {
        // Output axis i is axis p[i] of x, which is axis q[p[i]] of src.
        p = p.iter().map(|&a| q[a]).collect();
        let new_perm = Op::Transpose {
            perm: Some(p.iter().map(|&a| a as i64).collect()),
        };
        g.node_mut(id).op = new_perm;
        g.set_inputs(id, vec![Some(src)]);
        g.remove_node(inner);
        x = src;
    }
    if p.iter().enumerate().all(|(i, &a)| i == a) {
        bypass(g, id, x);
        return;
    }
    let Some(dims) = g.value(x).shape.clone() else {
        return;
    };
    // The axes that are not of size 1 keep their order.
    let moved: Vec<usize> = p
        .iter()
        .copied()
        .filter(|&a| dims[a] != Dim::Fixed(1))
        .collect();
    if !moved.is_sorted() {
        return;
    }
    let target = p
        .iter()
        .map(|&a| match dims[a] {
            Dim::Fixed(n) => TargetDim::Fixed(n),
            _ => TargetDim::Of { input: 0, axis: a },
        })
        .collect();
    let proto = g.node(id).proto.clone();
    g.fuse(&[id], Op::ReshapeTo { dims: target }, proto, vec![Some(x)]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ir::DType;
    use crate::model::onnx_proto::{
        AttributeProto, GraphProto, NodeProto, TensorProto, TensorShapeProto, TypeProto,
        ValueInfoProto, tensor_shape_proto, type_proto,
    };

    fn node(op: &str, input: &[&str], output: &str) -> NodeProto {
        NodeProto {
            op_type: op.to_string(),
            name: output.to_string(),
            input: input.iter().map(|s| s.to_string()).collect(),
            output: vec![output.to_string()],
            ..Default::default()
        }
    }

    fn ints_attr(name: &str, values: &[i64]) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            ints: values.to_vec(),
            r#type: 7, // INTS
            ..Default::default()
        }
    }

    fn transpose(input: &str, perm: &[i64], output: &str) -> NodeProto {
        let mut n = node("Transpose", &[input], output);
        n.attribute.push(ints_attr("perm", perm));
        n
    }

    fn floats(name: &str, dims: &[i64], values: &[f32]) -> TensorProto {
        TensorProto {
            name: name.to_string(),
            data_type: DType::F32.to_onnx(),
            dims: dims.to_vec(),
            float_data: values.to_vec(),
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

    /// An f32 tensor of `shape`, where `0` is a symbolic dimension.
    fn info(name: &str, shape: &[i64]) -> ValueInfoProto {
        let dim = |&d: &i64| tensor_shape_proto::Dimension {
            value: Some(if d == 0 {
                tensor_shape_proto::dimension::Value::DimParam("n".to_string())
            } else {
                tensor_shape_proto::dimension::Value::DimValue(d)
            }),
            ..Default::default()
        };
        ValueInfoProto {
            name: name.to_string(),
            r#type: Some(TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: DType::F32.to_onnx(),
                    shape: Some(TensorShapeProto {
                        dim: shape.iter().map(dim).collect(),
                    }),
                })),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    /// The nodes left of `nodes` on input `x` of `shape`, with graph output
    /// `y`, as `op(inputs) -> output`.
    fn simplified(
        nodes: Vec<NodeProto>,
        initializer: Vec<TensorProto>,
        shape: &[i64],
    ) -> Vec<String> {
        let mut g = Graph::from_onnx(GraphProto {
            node: nodes,
            initializer,
            input: vec![info("x", shape)],
            output: vec![ValueInfoProto {
                name: "y".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        });
        crate::compiler::passes::infer_dtypes(&mut g);
        crate::compiler::passes::fold_shapes(&mut g);
        simplify(&mut g);
        g.into_onnx()
            .node
            .iter()
            .map(|n| {
                format!(
                    "{}({}) -> {}",
                    n.op_type,
                    n.input.join(", "),
                    n.output.join(", ")
                )
            })
            .collect()
    }

    #[test]
    fn nodes_that_change_nothing_go() {
        let mut cast = node("Cast", &["d"], "e");
        cast.attribute.push(AttributeProto {
            name: "to".to_string(),
            i: DType::F32.to_onnx() as i64,
            r#type: 2, // INT
            ..Default::default()
        });
        let nodes = vec![
            node("Identity", &["x"], "a"),
            node("Mul", &["ones", "a"], "a1"),
            node("Mul", &["one", "a1"], "b"),
            node("Add", &["b", "zero"], "c"),
            node("Div", &["c", "one"], "d"),
            cast,
            node("Pad", &["e", "no_pads"], "f"),
            node("Slice", &["f", "start", "end"], "g"),
            node("Concat", &["nothing", "g"], "h"),
            node("Relu", &["h"], "y"),
        ];
        let initializer = vec![
            // A norm's scale of ones, along the last axis.
            floats("ones", &[4], &[1.0; 4]),
            floats("one", &[1, 1], &[1.0]),
            floats("zero", &[], &[0.0]),
            ints("no_pads", &[0, 0, 0, 0]),
            ints("start", &[0]),
            ints("end", &[i64::MAX]),
            // Empty along the concatenation's axis.
            floats("nothing", &[0, 4], &[]),
        ];
        assert_eq!(simplified(nodes, initializer, &[0, 4]), ["Relu(x) -> y"]);
    }

    #[test]
    fn nodes_that_change_their_input_stay() {
        let nodes = vec![
            // Broadcasts x to [3, 4].
            node("Mul", &["x", "ones"], "a"),
            // A difference from zero is a negation.
            node("Sub", &["zero", "a"], "b"),
            node("Add", &["b", "half"], "c"),
            node("Slice", &["c", "start", "end"], "y"),
        ];
        let initializer = vec![
            floats("ones", &[3, 1], &[1.0; 3]),
            floats("zero", &[], &[0.0]),
            floats("half", &[], &[0.5]),
            ints("start", &[1]),
            ints("end", &[i64::MAX]),
        ];
        assert_eq!(
            simplified(nodes, initializer, &[1, 4]),
            [
                "Mul(x, ones) -> a",
                "Sub(zero, a) -> b",
                "Add(b, half) -> c",
                "Slice(c, start, end) -> y"
            ]
        );
    }

    /// The node producing the input of a node that goes produces the graph
    /// output in its place.
    #[test]
    fn a_graph_output_keeps_its_name() {
        let nodes = vec![node("Relu", &["x"], "a"), node("Identity", &["a"], "y")];
        assert_eq!(simplified(nodes, vec![], &[2]), ["Relu(x) -> y"]);
        // Unless the graph reads that input elsewhere too.
        let nodes = vec![
            node("Relu", &["x"], "a"),
            node("Identity", &["a"], "y"),
            node("Neg", &["a"], "z"),
        ];
        let simplified = simplified(nodes, vec![], &[2]);
        assert_eq!(simplified[..2], ["Relu(x) -> a", "Identity(a) -> y"]);
    }

    #[test]
    fn transposes_of_transposes_merge() {
        let nodes = vec![
            transpose("x", &[0, 2, 1, 3], "a"),
            // Undoes `a`.
            transpose("a", &[0, 2, 1, 3], "b"),
            transpose("b", &[0, 2, 1, 3], "c"),
            transpose("c", &[0, 1, 3, 2], "y"),
        ];
        let simplified = simplified(nodes, vec![], &[0, 2, 8, 4]);
        assert_eq!(simplified, ["Transpose(x) -> y"]);
    }

    #[test]
    fn a_transpose_moving_axes_of_size_one_alone_is_a_reshape() {
        let nodes = vec![transpose("x", &[0, 2, 1, 3], "y")];
        assert_eq!(
            simplified(nodes, vec![], &[0, 1, 8, 4]),
            ["ReshapeTo(x) -> y"]
        );
        let nodes = vec![transpose("x", &[0, 2, 1, 3], "y")];
        assert_eq!(
            simplified(nodes, vec![], &[0, 2, 8, 4]),
            ["Transpose(x) -> y"]
        );
    }
}
