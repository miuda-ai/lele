//! Liveness analysis and workspace buffer allocation.

use super::generate::ElemType;
use super::ir::{Dim, Graph, Op};
use std::cmp::Reverse;
use std::collections::{BTreeSet, HashMap, HashSet};

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

/// The number of elements of a tensor: a count times symbolic dimensions.
#[derive(Clone, Debug, PartialEq)]
struct Size {
    count: u64,
    /// Names of the symbolic dimensions, sorted, a name once per dimension.
    symbols: Vec<String>,
}

impl Size {
    /// The size of a tensor of this shape, if no dimension is unknown.
    fn of(shape: &[Dim]) -> Option<Self> {
        let mut count = 1u64;
        let mut symbols = Vec::new();
        for d in shape {
            match d {
                Dim::Fixed(n) => count = count.checked_mul(u64::try_from(*n).ok()?)?,
                Dim::Sym(s) => symbols.push(s.clone()),
                Dim::Unknown => return None,
            }
        }
        symbols.sort();
        Some(Self { count, symbols })
    }

    /// Whether this size is at most `other` whatever the symbolic dimensions
    /// are, taking each to be at least 1.
    fn at_most(&self, other: &Self) -> bool {
        let mut others = other.symbols.iter().peekable();
        self.count <= other.count
            && self.symbols.iter().all(|s| {
                while others.next_if(|o| *o < s).is_some() {}
                others.next_if(|o| *o == s).is_some()
            })
    }

    /// An order of sizes that extends `at_most`.
    fn key(&self) -> (usize, u64) {
        (self.symbols.len(), self.count)
    }
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
    /// Sizes of the tensors whose shape is known.
    sizes: HashMap<String, Size>,
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
            for &v in node.outputs().iter().flatten() {
                let value = g.value(v);
                if let Some(size) = value.shape.as_deref().and_then(Size::of) {
                    self.sizes.insert(value.name.clone(), size);
                }
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
/// name; a buffer only holds tensors of one type. A node never writes into a
/// buffer it reads, nor into one the `lookback` nodes before it read.
pub(crate) fn solve_allocation(
    g: &Graph,
    ty: &dyn Fn(&str) -> ElemType,
    lookback: usize,
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

    let mut free: HashMap<ElemType, BTreeSet<usize>> = HashMap::new();
    let mut active: HashMap<String, usize> = HashMap::new();
    let mut all: HashMap<String, usize> = HashMap::new();
    let mut buffer_types = Vec::new();
    // The largest size each buffer holds, `None` if a size is unknown or two
    // are not comparable.
    let mut capacities: Vec<Option<Size>> = Vec::new();
    for (id, step) in schedule.steps.iter().enumerate() {
        // Release the buffers of tensors read for the last time before.
        let dead: Vec<(String, usize)> = active
            .iter()
            .filter(|(t, _)| *death_time.get(*t).unwrap_or(&usize::MAX) < id)
            .map(|(t, &b)| (t.clone(), b))
            .collect();
        for (t, b) in dead {
            active.remove(&t);
            free.entry(buffer_types[b]).or_default().insert(b);
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
            let mut reserved = HashSet::new();
            for prev in 0..=lookback {
                if let Some(s) = id.checked_sub(prev).map(|i| &schedule.steps[i]) {
                    reserved.extend(s.inputs.iter().filter_map(|n| all.get(n)).copied());
                }
            }
            let t = ty(out);
            let free_buffers = free.entry(t).or_default();
            let size = schedule.sizes.get(out);
            let candidates = free_buffers
                .iter()
                .copied()
                .filter(|b| !reserved.contains(b));
            let selected = choose_buffer(candidates, size, &capacities);
            let b = match selected {
                Some(b) => {
                    free_buffers.remove(&b);
                    b
                }
                None => {
                    buffer_types.push(t);
                    capacities.push(size.cloned());
                    buffer_types.len() - 1
                }
            };
            capacities[b] = match (capacities[b].take(), size) {
                (Some(c), Some(s)) if s.at_most(&c) => Some(c),
                (Some(c), Some(s)) if c.at_most(s) => Some(s.clone()),
                _ => None,
            };
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

/// The buffer among the free `candidates` to hold a tensor of `size`: of the
/// buffers whose capacity is known, the smallest that holds it, else the
/// largest, which grows least; else one whose capacity is unknown already;
/// else the first. Ties go to the lowest index.
fn choose_buffer(
    candidates: impl Iterator<Item = usize>,
    size: Option<&Size>,
    capacities: &[Option<Size>],
) -> Option<usize> {
    let candidates: Vec<usize> = candidates.collect();
    let unknown = candidates
        .iter()
        .copied()
        .find(|&b| capacities[b].is_none());
    let Some(size) = size else {
        return unknown.or(candidates.first().copied());
    };
    let known = || {
        candidates
            .iter()
            .filter_map(|&b| Some((b, capacities[b].as_ref()?)))
    };
    let fits = known()
        .filter(|(_, c)| size.at_most(c))
        .min_by_key(|&(b, c)| (c.key(), b));
    let smaller = known()
        .filter(|(_, c)| c.at_most(size))
        .max_by_key(|&(b, c)| (c.key(), Reverse(b)));
    fits.or(smaller)
        .map(|(b, _)| b)
        .or(unknown)
        .or(candidates.first().copied())
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
            let g = graph(nodes, "y");
            let (alloc, _) = solve_allocation(&g, &|_| ElemType::F32, 5);
            let buffer = |t: &str| alloc.tensor_to_buffer[t];
            let reused = chain.iter().any(|t| buffer(t) == buffer("a"));
            assert_eq!(reused, op == "Relu", "{op}");
        }
    }

    fn graph(nodes: Vec<NodeProto>, output: &str) -> crate::compiler::ir::Graph {
        let info = |name: &str| ValueInfoProto {
            name: name.to_string(),
            ..Default::default()
        };
        crate::compiler::ir::Graph::from_onnx(GraphProto {
            node: nodes,
            input: vec![info("x")],
            output: vec![info(output)],
            ..Default::default()
        })
    }

    /// A buffer is handed to the next node after its tensor's last reader,
    /// unless one of the `lookback` nodes before read it.
    #[test]
    fn test_buffers_are_reused_after_the_lookback() {
        let g = graph(
            vec![
                node("Relu", &["x"], "a"),
                node("Relu", &["a"], "b"),
                node("Relu", &["b"], "c"),
            ],
            "c",
        );
        for (lookback, reused) in [(0, true), (1, false)] {
            let (alloc, _) = solve_allocation(&g, &|_| ElemType::F32, lookback);
            let buffer = |t: &str| alloc.tensor_to_buffer[t];
            assert_eq!(buffer("c") == buffer("a"), reused, "lookback {lookback}");
            assert_ne!(buffer("b"), buffer("a"));
        }
    }

    /// A shape of numbers and symbolic dimensions.
    fn dims(shape: &[&str]) -> Vec<Dim> {
        shape
            .iter()
            .map(|d| {
                d.parse()
                    .map_or_else(|_| Dim::Sym(d.to_string()), Dim::Fixed)
            })
            .collect()
    }

    fn size(shape: &[&str]) -> Size {
        Size::of(&dims(shape)).unwrap()
    }

    /// Symbolic dimensions are at least 1, so a size is at most another
    /// with more of them.
    #[test]
    fn test_sizes_compare_whatever_their_symbolic_dimensions() {
        let at_most = |a: &[&str], b: &[&str]| size(a).at_most(&size(b));
        assert!(at_most(&["4", "2"], &["8"]));
        assert!(!at_most(&["9"], &["8"]));
        assert!(at_most(&["384"], &["T", "384"]));
        assert!(at_most(&["T", "4"], &["2", "T", "2"]));
        assert!(at_most(&["T", "4"], &["T", "S", "4"]));
        assert!(!at_most(&["T", "4"], &["S", "4"]));
        assert!(at_most(&["T", "4"], &["T", "T", "4"]));
        assert!(!at_most(&["T", "T", "4"], &["T", "4"]));
        assert!(!at_most(&["T", "4"], &["T", "2"]));
        assert_eq!(Size::of(&[Dim::Fixed(2), Dim::Unknown]), None);
    }

    /// A tensor takes the smallest free buffer that holds it, and one larger
    /// than any free buffer the largest, rather than the first free buffer.
    #[test]
    fn test_a_tensor_takes_the_buffer_that_fits_it_best() {
        for (big, small) in [(["1000"], ["10"]), (["T"], ["1"])] {
            let mut g = graph(
                vec![
                    node("Relu", &["x"], "a"),
                    node("Relu", &["x"], "b"),
                    node("Add", &["a", "b"], "m"),
                    node("Relu", &["m"], "c"),
                    node("Relu", &["c"], "d"),
                ],
                "d",
            );
            for (name, shape) in [
                ("a", big),
                ("b", small),
                ("m", small),
                ("c", small),
                ("d", big),
            ] {
                let v = g.find_value(name).unwrap();
                g.value_mut(v).shape = Some(dims(&shape));
            }
            let (alloc, _) = solve_allocation(&g, &|_| ElemType::F32, 0);
            let buffer = |t: &str| alloc.tensor_to_buffer[t];
            assert_eq!(buffer("c"), buffer("b"), "{big:?}");
            assert_eq!(buffer("d"), buffer("a"), "{big:?}");
            assert_eq!(alloc.num_buffers, 3, "{big:?}");
        }
    }
}
