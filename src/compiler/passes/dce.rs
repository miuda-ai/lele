//! Dead code elimination.

use crate::compiler::ir::Graph;

/// Removes the nodes of `g` and its subgraphs whose outputs nothing reads.
pub(crate) fn eliminate_dead_code(g: &mut Graph) {
    for id in g.node_ids() {
        for sub in g.node_mut(id).op.subgraphs_mut() {
            eliminate_dead_code(sub);
        }
    }
    // Last to first, so a chain goes in one sweep.
    for id in g.node_ids().into_iter().rev() {
        let node = g.node(id);
        if node.outputs().iter().flatten().all(|&v| g.uses(v) == 0) {
            g.remove_node(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::onnx_proto::{GraphProto, NodeProto, ValueInfoProto};

    #[test]
    fn removes_chains_nothing_reads() {
        let node = |op: &str, input: &str, output: &str| NodeProto {
            op_type: op.to_string(),
            input: vec![input.to_string()],
            output: vec![output.to_string()],
            ..Default::default()
        };
        let mut g = Graph::from_onnx(GraphProto {
            node: vec![
                node("Relu", "x", "a"),
                node("Neg", "a", "b"),
                node("Exp", "x", "y"),
            ],
            output: vec![ValueInfoProto {
                name: "y".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        });
        eliminate_dead_code(&mut g);
        let ops: Vec<_> = g.nodes().map(|(_, n)| n.op.op_type().to_string()).collect();
        assert_eq!(ops, ["Exp"]);
    }
}
