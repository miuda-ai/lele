//! Inlining of the `If` nodes whose condition is known.

use super::fold::exact_ints;
use crate::compiler::ir::{Graph, Op};

/// Replaces each `If` of `g` and its subgraphs whose condition is an
/// initializer or a constant by the branch it takes. Returns whether one was.
///
/// Exporters leave such `If`s where a model checks a setting (Silero's
/// sample rate, the rank of a tensor), and folding and fixed input values
/// resolve the condition.
pub(crate) fn inline_constant_ifs(g: &mut Graph) -> bool {
    let mut inlined = false;
    for id in g.node_ids() {
        for sub in g.node_mut(id).op.subgraphs_mut() {
            inlined |= inline_constant_ifs(sub);
        }
        let node = g.node(id);
        let Op::If {
            then_branch,
            else_branch,
        } = &node.op
        else {
            continue;
        };
        let Some(cond) = node.input(0) else {
            continue;
        };
        let tensor = match g.producer(cond).map(|p| &p.op) {
            Some(Op::Constant { value }) => Some(value),
            _ => g.data(cond),
        };
        let Some(&first) = tensor
            .and_then(exact_ints)
            .as_deref()
            .and_then(<[i64]>::first)
        else {
            continue;
        };
        let branch = if first != 0 { then_branch } else { else_branch }.clone();
        inlined |= g.inline_subgraph(id, &branch);
    }
    inlined
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ir::DType;
    use crate::model::onnx_proto::{
        AttributeProto, GraphProto, NodeProto, TensorProto, ValueInfoProto,
    };

    fn node(op: &str, input: &[&str], output: &str) -> NodeProto {
        NodeProto {
            op_type: op.to_string(),
            input: input.iter().map(|s| s.to_string()).collect(),
            output: vec![output.to_string()],
            ..Default::default()
        }
    }

    fn info(name: &str) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_string(),
            ..Default::default()
        }
    }

    fn flag(name: &str, value: bool) -> TensorProto {
        TensorProto {
            name: name.to_string(),
            data_type: DType::Bool.to_onnx(),
            int32_data: vec![value as i32],
            ..Default::default()
        }
    }

    /// `If(cond)` taking `then` or `else`, each a chain of the operators
    /// given on `x`, its result read by an `Add`.
    fn graph(cond: Option<bool>, then: GraphProto, otherwise: GraphProto) -> Graph {
        let branch = |name: &str, g: GraphProto| AttributeProto {
            name: name.to_string(),
            g: Some(g),
            ..Default::default()
        };
        let mut if_node = node("If", &["cond"], "r");
        if_node.attribute = vec![
            branch("then_branch", then),
            branch("else_branch", otherwise),
        ];
        Graph::from_onnx(GraphProto {
            node: vec![if_node, node("Add", &["r", "x"], "y")],
            initializer: cond.map(|c| flag("cond", c)).into_iter().collect(),
            input: [Some("x"), cond.is_none().then_some("cond")]
                .into_iter()
                .flatten()
                .map(info)
                .collect(),
            output: vec![info("y")],
            ..Default::default()
        })
    }

    /// A branch running `ops` one after another on `x`, its values named
    /// with `prefix`.
    fn chain(prefix: &str, ops: &[&str]) -> GraphProto {
        let names: Vec<String> = (0..=ops.len())
            .map(|i| {
                if i == 0 {
                    "x".to_string()
                } else {
                    format!("{prefix}{i}")
                }
            })
            .collect();
        GraphProto {
            node: ops
                .iter()
                .enumerate()
                .map(|(i, op)| node(op, &[&names[i]], &names[i + 1]))
                .collect(),
            output: vec![info(names.last().unwrap())],
            ..Default::default()
        }
    }

    fn ops(g: &Graph) -> Vec<&str> {
        g.nodes().map(|(_, n)| n.op.op_type()).collect()
    }

    #[test]
    fn an_if_of_known_condition_runs_its_branch_in_place() {
        for (cond, want) in [
            (true, ["Relu", "Neg", "Identity", "Add"].as_slice()),
            (false, &["Exp", "Identity", "Add"]),
        ] {
            let mut g = graph(
                Some(cond),
                chain("t", &["Relu", "Neg"]),
                chain("e", &["Exp"]),
            );
            assert!(inline_constant_ifs(&mut g));
            assert_eq!(ops(&g), want);
            let out = g.value(g.outputs[0]);
            assert_eq!(out.name, "y");
            let r = g.find_value("r").unwrap();
            assert_eq!(g.producer(r).map(|n| n.op.op_type()), Some("Identity"));
        }
    }

    /// Branches are inlined innermost first, so an `If` in the branch taken
    /// goes too.
    #[test]
    fn nested_ifs_of_known_conditions_go_together() {
        let mut inner = node("If", &["inner_cond"], "t1");
        inner.attribute = vec![
            AttributeProto {
                name: "then_branch".to_string(),
                g: Some(chain("a", &["Sigmoid"])),
                ..Default::default()
            },
            AttributeProto {
                name: "else_branch".to_string(),
                g: Some(chain("b", &["Tanh"])),
                ..Default::default()
            },
        ];
        let then = GraphProto {
            node: vec![inner],
            initializer: vec![flag("inner_cond", false)],
            output: vec![info("t1")],
            ..Default::default()
        };
        let mut g = graph(Some(true), then, chain("e", &["Exp"]));
        assert!(inline_constant_ifs(&mut g));
        assert_eq!(ops(&g), ["Tanh", "Identity", "Identity", "Add"]);
    }

    #[test]
    fn ifs_stay_where_the_condition_is_unknown_or_names_clash() {
        let mut g = graph(None, chain("t", &["Relu"]), chain("e", &["Exp"]));
        assert!(!inline_constant_ifs(&mut g));
        assert_eq!(ops(&g), ["If", "Add"]);
        // The branch defines `y`, which the graph defines too.
        let clash = GraphProto {
            node: vec![node("Relu", &["x"], "y")],
            output: vec![info("y")],
            ..Default::default()
        };
        let mut g = graph(Some(true), clash, chain("e", &["Exp"]));
        assert!(!inline_constant_ifs(&mut g));
        assert_eq!(ops(&g), ["If", "Add"]);
    }
}
