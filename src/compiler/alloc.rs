//! Liveness analysis and workspace buffer allocation.

use super::ir::{Graph, Op};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

pub struct Allocator {
    /// Workspace buffer index of each tensor, by name.
    pub tensor_to_buffer: HashMap<String, usize>,
    pub num_buffers: usize,
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
}

impl Schedule {
    fn visit(&mut self, g: &Graph) {
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
                Op::Loop { .. } => {}
                op => {
                    // Zero-copy views.
                    if matches!(
                        op.op_type(),
                        "Squeeze" | "Unsqueeze" | "Reshape" | "Identity" | "Flatten" | "Cast"
                    ) && !inputs.is_empty()
                        && !outputs.is_empty()
                    {
                        self.aliases
                            .insert(outputs[0].clone(), vec![inputs[0].clone()]);
                    }
                }
            }
            self.steps.push(Step { inputs, outputs });
            for sub in node.op.subgraphs() {
                self.visit(sub);
            }
        }
    }
}

/// Assigns workspace buffers to the tensors of `g` and its subgraphs.
///
/// A tensor's buffer is freed after its last reader and handed to later
/// tensors. Views share the buffer of what they view, which therefore lives
/// as long as the longest-lived view.
pub(crate) fn solve_allocation(g: &Graph) -> (Allocator, AnalysisData) {
    let mut schedule = Schedule::default();
    schedule.visit(g);
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
    let mut changed = true;
    while changed {
        changed = false;
        for (alias, sources) in &schedule.aliases {
            let alias_death = *death_time.get(alias).unwrap_or(&0);
            for source in sources {
                let source_death = *death_time.get(source).unwrap_or(&0);
                if alias_death > source_death {
                    death_time.insert(source.clone(), alias_death);
                    changed = true;
                }
            }
        }
    }

    let mut free_buffers: BinaryHeap<Reverse<usize>> = BinaryHeap::new();
    let mut active: HashMap<String, usize> = HashMap::new();
    let mut all: HashMap<String, usize> = HashMap::new();
    let mut num_buffers = 0;
    for (id, step) in schedule.steps.iter().enumerate() {
        // Release the buffers of tensors read for the last time before.
        let dead: Vec<(String, usize)> = active
            .iter()
            .filter(|(t, _)| *death_time.get(*t).unwrap_or(&usize::MAX) < id)
            .map(|(t, &b)| (t.clone(), b))
            .collect();
        for (t, b) in dead {
            active.remove(&t);
            free_buffers.push(Reverse(b));
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
                num_buffers += 1;
                num_buffers - 1
            });
            active.insert(out.clone(), b);
            all.insert(out.clone(), b);
        }
    }
    (
        Allocator {
            tensor_to_buffer: all,
            num_buffers,
        },
        AnalysisData {
            death_time,
            uses: schedule.uses,
        },
    )
}
