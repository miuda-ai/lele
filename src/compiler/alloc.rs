//! Liveness analysis and workspace buffer allocation.

use super::generate::ElemType;
use super::ir::{Graph, Op};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

pub struct Allocator {
    /// Workspace buffer index of each tensor, by name.
    pub tensor_to_buffer: HashMap<String, usize>,
    pub num_buffers: usize,
    /// Element type of each buffer, that of every tensor it holds.
    pub(crate) buffer_types: Vec<ElemType>,
}

pub(crate) struct AnalysisData {
    /// Index of the last node reading each tensor, by name.
    pub death_time: HashMap<String, usize>,
    /// Indices of the nodes reading each tensor, by name.
    pub uses: HashMap<String, Vec<usize>>,
}

/// A node in the order code generation emits nodes: a node, then the nodes
/// of its subgraphs, depth first.
struct Step {
    inputs: Vec<String>,
    outputs: Vec<String>,
}

#[derive(Default)]
struct Schedule {
    steps: Vec<Step>,
    uses: HashMap<String, Vec<usize>>,
    /// Tensors that are views of others: `If` outputs of their branches'
    /// outputs, reshapes and casts of their input.
    aliases: HashMap<String, Vec<String>>,
    /// Tensors that may be views of others or copies in a buffer of their
    /// own, as the kernel finds at run time: slices, gathers, transposes and
    /// splits (see `kernels::manipulation::borrow_run`). They get a buffer,
    /// and keep what they read alive as views do.
    borrows: HashMap<String, String>,
}

impl Schedule {
    fn visit(&mut self, g: &Graph, ty: &dyn Fn(&str) -> ElemType) {
        let name = |v: Option<_>| v.map_or_else(String::new, |v| g.value(v).name.clone());
        for (_, node) in g.nodes() {
            let id = self.steps.len();
            let inputs: Vec<String> = node.inputs().iter().map(|&v| name(v)).collect();
            let outputs: Vec<String> = node.outputs().iter().map(|&v| name(v)).collect();
            for input in &inputs {
                self.uses.entry(input.clone()).or_default().push(id);
            }
            match &node.op {
                Op::If {
                    then_branch,
                    else_branch,
                } => {
                    for (i, out) in outputs.iter().enumerate() {
                        let sources: Vec<String> = [then_branch, else_branch]
                            .iter()
                            .filter_map(|b| Some(b.value(*b.outputs.get(i)?).name.clone()))
                            .collect();
                        if !sources.is_empty() {
                            self.aliases.insert(out.clone(), sources);
                        }
                    }
                }
                // Zero-copy views.
                Op::Squeeze { .. }
                | Op::Unsqueeze { .. }
                | Op::Reshape
                | Op::ReshapeTo { .. }
                | Op::Identity
                | Op::Flatten { .. }
                    if !inputs.is_empty() && !outputs.is_empty() =>
                {
                    self.aliases
                        .insert(outputs[0].clone(), vec![inputs[0].clone()]);
                }
                Op::Cast { to }
                    if !inputs.is_empty()
                        && !outputs.is_empty()
                        && ty(&inputs[0]).cast_is_view(*to) =>
                {
                    self.aliases
                        .insert(outputs[0].clone(), vec![inputs[0].clone()]);
                }
                Op::Slice | Op::Gather { .. } | Op::Transpose { .. } | Op::Split { .. }
                    if !inputs.is_empty() =>
                {
                    for out in outputs.iter().filter(|o| !o.is_empty()) {
                        self.borrows.insert(out.clone(), inputs[0].clone());
                    }
                }
                _ => {}
            }
            self.steps.push(Step { inputs, outputs });
            for sub in node.op.subgraphs() {
                self.visit(sub, ty);
            }
        }
    }
}

/// Assigns workspace buffers to the tensors of `g` and its subgraphs.
///
/// A tensor's buffer is freed after its last reader and handed to later
/// tensors. Views share the buffer of what they view, which therefore lives
/// as long as the longest-lived view. `ty` gives the element type of a tensor by
/// name; a buffer only holds tensors of one type.
pub(crate) fn solve_allocation(
    g: &Graph,
    ty: &dyn Fn(&str) -> ElemType,
) -> (Allocator, AnalysisData) {
    let mut schedule = Schedule::default();
    schedule.visit(g, ty);
    // Graph outputs are read after the last node.
    let end = schedule.steps.len();
    for &out in &g.outputs {
        let name = g.value(out).name.clone();
        schedule.uses.entry(name).or_default().push(end);
    }

    let mut death_time = HashMap::new();
    for (name, uses) in &schedule.uses {
        if let Some(max) = uses.iter().max() {
            death_time.insert(name.clone(), *max);
        }
    }
    // A viewed tensor must live as long as its views.
    let views = schedule
        .aliases
        .iter()
        .flat_map(|(view, sources)| sources.iter().map(move |s| (view, s)));
    let views: Vec<(&String, &String)> = views.chain(schedule.borrows.iter()).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for &(view, source) in &views {
            let view_death = *death_time.get(view).unwrap_or(&0);
            let source_death = *death_time.get(source).unwrap_or(&0);
            if view_death > source_death {
                death_time.insert(source.clone(), view_death);
                changed = true;
            }
        }
    }

    let mut free: HashMap<ElemType, BinaryHeap<Reverse<usize>>> = HashMap::new();
    let mut active: HashMap<String, usize> = HashMap::new();
    let mut all: HashMap<String, usize> = HashMap::new();
    let mut buffer_types = Vec::new();
    for (id, step) in schedule.steps.iter().enumerate() {
        // Release the buffers of tensors read for the last time before.
        let dead: Vec<(String, usize)> = active
            .iter()
            .filter(|(t, _)| *death_time.get(*t).unwrap_or(&usize::MAX) < id)
            .map(|(t, &b)| (t.clone(), b))
            .collect();
        for (t, b) in dead {
            active.remove(&t);
            free.entry(buffer_types[b]).or_default().push(Reverse(b));
        }
        for out in &step.outputs {
            if out.is_empty() {
                continue;
            }
            // A view takes its source's buffer; the source stays its owner.
            let alias = schedule
                .aliases
                .get(out)
                .and_then(|sources| sources.first())
                .and_then(|src| active.get(src));
            if let Some(&b) = alias {
                all.insert(out.clone(), b);
                continue;
            }
            // Never write into a buffer this node reads, nor one the five
            // nodes before it read: a user pattern may generate a whole run
            // of nodes at once.
            let mut reserved = HashSet::new();
            for prev in 0..=5 {
                if let Some(s) = id.checked_sub(prev).map(|i| &schedule.steps[i]) {
                    reserved.extend(s.inputs.iter().filter_map(|n| all.get(n)).copied());
                }
            }
            let t = ty(out);
            let free_buffers = free.entry(t).or_default();
            let mut deferred = Vec::new();
            let mut selected = None;
            while let Some(Reverse(b)) = free_buffers.pop() {
                if reserved.contains(&b) {
                    deferred.push(b);
                } else {
                    selected = Some(b);
                    break;
                }
            }
            free_buffers.extend(deferred.into_iter().map(Reverse));
            let b = selected.unwrap_or_else(|| {
                buffer_types.push(t);
                buffer_types.len() - 1
            });
            active.insert(out.clone(), b);
            all.insert(out.clone(), b);
        }
    }
    (
        Allocator {
            tensor_to_buffer: all,
            num_buffers: buffer_types.len(),
            buffer_types,
        },
        AnalysisData {
            death_time,
            uses: schedule.uses,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::onnx_proto::{GraphProto, NodeProto, ValueInfoProto};

    fn node(op: &str, inputs: &[&str], output: &str) -> NodeProto {
        NodeProto {
            op_type: op.to_string(),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: vec![output.to_string()],
            ..Default::default()
        }
    }

    /// What a slice, gather, transpose or split reads stays in its buffer while the
    /// result is in use, since the result may be a view of it.
    #[test]
    fn test_what_may_be_viewed_lives_as_long_as_the_view() {
        for (op, inputs) in [
            ("Slice", &["a", "starts", "ends"][..]),
            ("Gather", &["a", "index"]),
            ("Transpose", &["a"]),
            ("Split", &["a"]),
            // A copy: `a` is free once read.
            ("Relu", &["a"]),
        ] {
            // A chain long enough to pass the nodes whose inputs are never overwritten.
            let chain: Vec<String> = (0..8).map(|i| format!("b{i}")).collect();
            let mut nodes = vec![node("Relu", &["x"], "a"), node(op, inputs, "v")];
            nodes.push(node("Relu", &["x"], &chain[0]));
            for w in chain.windows(2) {
                nodes.push(node("Relu", &[&w[0]], &w[1]));
            }
            nodes.push(node("Add", &["v", &chain[7]], "y"));
            let g = crate::compiler::ir::Graph::from_onnx(GraphProto {
                node: nodes,
                input: vec![ValueInfoProto {
                    name: "x".to_string(),
                    ..Default::default()
                }],
                output: vec![ValueInfoProto {
                    name: "y".to_string(),
                    ..Default::default()
                }],
                ..Default::default()
            });
            let (alloc, _) = solve_allocation(&g, &|_| ElemType::F32);
            let buffer = |t: &str| alloc.tensor_to_buffer[t];
            let reused = chain.iter().any(|t| buffer(t) == buffer("a"));
            assert_eq!(reused, op == "Relu", "{op}");
        }
    }
}
