//! Fusion of the runs of nodes [`Compiler::with_pattern`](super::Compiler::with_pattern)
//! matchers find.

use super::ir::{Graph, NodeId, Op, ValueId};
use crate::model::onnx_proto::NodeProto;
use std::collections::{HashMap, HashSet};

/// Replaces each run of nodes of `g` that `matcher` finds by one node of
/// operator type `op_type`, as [`Compiler::with_pattern`](super::Compiler::with_pattern)
/// describes.
pub(crate) fn fuse_matches(
    g: &mut Graph,
    op_type: &str,
    matcher: &dyn Fn(&Graph, NodeId) -> Option<Vec<NodeId>>,
) {
    for id in g.node_ids() {
        if !g.contains(id) {
            continue;
        }
        let Some(run) = matcher(g, id) else {
            continue;
        };
        let Some((run, inputs, outputs)) = boundary(g, run) else {
            continue;
        };
        let last = *run.last().unwrap();
        let proto = NodeProto {
            name: g.node(last).proto.name.clone(),
            ..Default::default()
        };
        let op = Op::Onnx {
            op_type: op_type.to_string(),
            domain: String::new(),
            graphs: Vec::new(),
        };
        g.fuse_with_outputs(&run, op, proto, inputs, outputs);
    }
}

/// `run` in execution order, with what it reads from outside and what
/// outside reads of it, if it can run as one node where its last node ran.
#[allow(clippy::type_complexity)]
fn boundary(
    g: &Graph,
    run: Vec<NodeId>,
) -> Option<(Vec<NodeId>, Vec<Option<ValueId>>, Vec<Option<ValueId>>)> {
    let position: HashMap<NodeId, usize> = g
        .node_ids()
        .into_iter()
        .enumerate()
        .map(|(i, id)| (id, i))
        .collect();
    let members: HashSet<NodeId> = run.iter().copied().collect();
    if run.is_empty()
        || members.len() != run.len()
        || run.iter().any(|id| !position.contains_key(id))
    {
        return None;
    }
    let mut run = run;
    run.sort_by_key(|id| position[id]);
    if run.iter().any(|&id| !g.node(id).op.subgraphs().is_empty()) {
        return None;
    }
    let end = position[run.last().unwrap()];
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for &id in &run {
        let node = g.node(id);
        for &v in node.inputs().iter().flatten() {
            let inside = g.value(v).producer().is_some_and(|p| members.contains(&p));
            if !inside && !inputs.contains(&Some(v)) {
                inputs.push(Some(v));
            }
        }
        for &v in node.outputs().iter().flatten() {
            let readers: Vec<NodeId> = g
                .value(v)
                .consumers()
                .iter()
                .copied()
                .filter(|c| !members.contains(c))
                .collect();
            if readers.iter().any(|c| position[c] < end) {
                return None;
            }
            if !readers.is_empty() || g.outputs.contains(&v) {
                outputs.push(Some(v));
            }
        }
    }
    Some((run, inputs, outputs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{Compiler, Optimizations};
    use crate::model::onnx_proto::{GraphProto, TensorProto, ValueInfoProto};
    use std::cell::RefCell;
    use std::rc::Rc;

    fn node(op: &str, inputs: &[&str], output: &str) -> NodeProto {
        NodeProto {
            op_type: op.to_string(),
            name: output.to_string(),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: vec![output.to_string()],
            ..Default::default()
        }
    }

    fn graph(nodes: Vec<NodeProto>, init: Vec<TensorProto>, outputs: &[&str]) -> GraphProto {
        let info = |name: &str| ValueInfoProto {
            name: name.to_string(),
            ..Default::default()
        };
        GraphProto {
            node: nodes,
            initializer: init,
            input: vec![info("x")],
            output: outputs.iter().map(|o| info(o)).collect(),
            ..Default::default()
        }
    }

    fn floats(name: &str, dims: &[i64], v: &[f32]) -> TensorProto {
        TensorProto {
            name: name.to_string(),
            data_type: crate::compiler::ir::DType::F32.to_onnx(),
            dims: dims.to_vec(),
            raw_data: v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            ..Default::default()
        }
    }

    /// The `Relu` a `Neg` reads, with the `Neg`.
    fn relu_neg(g: &Graph, id: NodeId) -> Option<Vec<NodeId>> {
        let neg = g.node(id);
        let relu = g.value(neg.input(0)?).producer()?;
        (neg.op.op_type() == "Neg" && g.node(relu).op.op_type() == "Relu").then(|| vec![relu, id])
    }

    /// A run becomes one node reading what the run reads and producing what
    /// is read of it, the result of its first node included; code
    /// generation hands that node to the pattern's generator.
    #[test]
    fn patterns_replace_runs_by_one_node() {
        let model = graph(
            vec![
                node("Relu", &["x"], "a"),
                node("Neg", &["a"], "b"),
                node("Add", &["b", "w"], "y"),
                node("Mul", &["a", "x"], "z"),
            ],
            vec![floats("w", &[1], &[2.0])],
            &["y", "z"],
        );
        let result = Compiler::new()
            .with_default_optimizations()
            .with_pattern("ReluNeg", relu_neg, |c, w| {
                let (o, i) = (c.outputs(), c.inputs());
                assert_eq!(c.node().op.op_type(), "ReluNeg");
                writeln!(
                    w,
                    "{}let ({}, {}) = relu_neg(&{});",
                    c.tab(),
                    o[0],
                    o[1],
                    i[0]
                )
            })
            .compile(&model)
            .unwrap();
        let code = &result.code;
        assert!(code.contains("let (a, b) = relu_neg(&x);"), "{code}");
        assert!(!code.contains("kernels::relu("), "{code}");
        assert!(!code.contains("kernels::neg("), "{code}");
    }

    /// A run whose result another node reads before the run ends stays.
    #[test]
    fn runs_read_before_they_end_stay() {
        let model = graph(
            vec![
                node("Relu", &["x"], "a"),
                node("Mul", &["a", "x"], "z"),
                node("Neg", &["a"], "b"),
            ],
            Vec::new(),
            &["b", "z"],
        );
        let mut g = Graph::from_onnx(model);
        fuse_matches(&mut g, "ReluNeg", &relu_neg);
        let ops: Vec<&str> = g.nodes().map(|(_, n)| n.op.op_type()).collect();
        assert_eq!(ops, ["Relu", "Mul", "Neg"]);
    }

    /// Passes before fusion see the graph unfused and may rewrite weights;
    /// passes after it see the fused operators.
    #[test]
    fn passes_run_before_and_after_fusion() {
        let model = graph(
            vec![
                node("MatMul", &["x", "w"], "m"),
                node("Add", &["m", "b"], "y"),
            ],
            vec![
                floats("w", &[2, 2], &[1.0, 2.0, 3.0, 4.0]),
                floats("b", &[2], &[0.5, 0.5]),
            ],
            &["y"],
        );
        let seen = Rc::new(RefCell::new(Vec::<Vec<String>>::new()));
        let ops = |g: &Graph| g.nodes().map(|(_, n)| n.op.op_type().to_string()).collect();
        let (before, after) = (seen.clone(), seen.clone());
        let result = Compiler::new()
            .with_optimizations(Optimizations {
                operator_fusion: true,
                ..Default::default()
            })
            .with_pass_before_fusion(move |g| {
                before.borrow_mut().push(ops(g));
                let w = g.find_value("w").unwrap();
                g.set_data(w, floats("", &[2, 2], &[5.0, 6.0, 7.0, 8.0]));
            })
            .with_pass_after_fusion(move |g| after.borrow_mut().push(ops(g)))
            .compile(&model)
            .unwrap();
        assert_eq!(*seen.borrow(), [vec!["MatMul", "Add"], vec!["Linear"]]);
        let bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let contains = |s: &[u8]| result.weights.windows(s.len()).any(|x| x == s);
        assert!(contains(&bytes(&[5.0, 6.0, 7.0, 8.0])));
        assert!(!contains(&bytes(&[1.0, 2.0, 3.0, 4.0])));
    }
}
