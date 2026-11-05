use super::ir::{Binary, DType, Graph, Node, NodeId, Op};
use super::weights::{WeightRef, Weights};
use super::{Allocator, AnalysisData, Compiler, sanitize_name};
use crate::model::onnx_proto::{NodeProto, ValueInfoProto};
use std::collections::{HashMap, HashSet};
use std::io::Write;

pub(crate) struct OpContext<'a> {
    /// The node, lowered.
    pub node: &'a NodeProto,
    /// The graph holding the node, and its id there.
    pub graph: &'a Graph,
    pub id: NodeId,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub buf_expr: String,
    pub indent: usize,
    pub weights: &'a Weights,
    #[allow(dead_code)]
    pub allocator: Option<&'a Allocator>,
    #[allow(dead_code)]
    pub analysis: Option<&'a AnalysisData>,
    pub current_id: &'a mut usize,
    pub compiler: &'a Compiler,
    pub var_types: &'a VarTypes,
}

impl<'a> OpContext<'a> {
    pub fn tab(&self) -> String {
        "    ".repeat(self.indent)
    }

    /// The node in the IR.
    pub fn ir(&self) -> &'a Node {
        self.graph.node(self.id)
    }

    /// The weight input `i` reads, if it reads one.
    pub fn weight(&self, i: usize) -> Option<&'a WeightRef> {
        self.weights.get(self.node.input.get(i)?)
    }

    /// The element type input `i` is held in, whether it is a variable or a
    /// weight.
    pub fn input_type(&self, i: usize) -> ElemType {
        self.node.input.get(i).map_or(ElemType::F32, |name| {
            self.var_types.get(&sanitize_name(name))
        })
    }

    /// Whether input `i` is a variable that nothing reads after this node,
    /// directly or through a view, so that the node may take it by value.
    /// Only known in the top-level graph, whose liveness is analysed.
    pub fn input_dies_here(&self, i: usize) -> bool {
        let (Some(an), Some(name)) = (self.analysis, self.node.input.get(i)) else {
            return false;
        };
        !name.is_empty()
            && !self.weights.contains(name)
            && self.node.input.iter().filter(|n| *n == name).count() == 1
            // `current_id` has moved past this node.
            && an.death_time.get(name) == Some(&(*self.current_id - 1))
    }

    /// The values of input `i`, if it is an int64 or int32 weight.
    pub fn ints(&self, i: usize) -> Option<&'a [i64]> {
        self.weights.ints(self.node.input.get(i)?)
    }

    /// Input `i` as a weight operand of the matrix products (`matmul`,
    /// `linear`, `conv1d_pointwise`): an f16 weight is passed as stored
    /// (packed or not), for the kernel to widen as it reads it.
    pub fn matmul_weight(&self, i: usize) -> String {
        match self.weight(i) {
            Some(w) if w.packed => format!(
                "self.weight_f16_packed({}, {}, &{:?})",
                w.offset, w.len, w.shape
            ),
            Some(w) if w.dtype == DType::F16 => format!(
                "self.weight_f16_raw({}, {}, &{:?})",
                w.offset, w.len, w.shape
            ),
            _ => self.inputs[i].clone(),
        }
    }
}

/// Element type of the `TensorView` holding a value in generated code.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) enum ElemType {
    #[default]
    F32,
    I64,
}

impl ElemType {
    /// The type holding tensors of ONNX type `t`, lele's kernel conventions
    /// aside: int64, int32 and bool tensors are i64, everything else f32.
    pub fn of(t: DType) -> Self {
        match t {
            DType::I64 | DType::I32 | DType::Bool => Self::I64,
            _ => Self::F32,
        }
    }

    /// The Rust element type.
    pub fn rust(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::I64 => "i64",
        }
    }

    /// Whether a `Cast` to `to` of a tensor of this type is generated as a view of
    /// its input rather than a conversion into a buffer of its own.
    pub fn cast_is_view(self, to: DType) -> bool {
        match to {
            DType::I64 | DType::I32 | DType::Bool => false,
            DType::F32 => self == Self::F32,
            _ => true,
        }
    }
}

impl std::fmt::Display for ElemType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.rust())
    }
}

/// [`ElemType`] of each value by sanitized name, f32 for names it does not
/// know.
#[derive(Clone, Debug, Default)]
pub(crate) struct VarTypes(HashMap<String, ElemType>);

impl VarTypes {
    pub fn get(&self, name: &str) -> ElemType {
        self.0.get(name).copied().unwrap_or_default()
    }

    pub fn is_i64(&self, name: &str) -> bool {
        self.get(name) == ElemType::I64
    }
}

/// Expression reading weight `w` from the weights file as `target`.
pub(crate) fn weight_expr(w: &WeightRef, target: ElemType) -> String {
    let accessor = match target {
        ElemType::I64 => match w.dtype {
            DType::I32 => "weight_i32_i64",
            DType::Bool => "weight_bool_i64",
            _ => "weight_i64",
        },
        ElemType::F32 => match w.dtype {
            DType::U8 => "weight_u8",
            DType::I8 => "weight_i8",
            DType::I32 => "weight_i32_f32",
            DType::I64 => "weight_i64_f32",
            DType::F16 => "weight_f16",
            _ => "weight_f32",
        },
    };
    format!("self.{accessor}({}, {}, &{:?})", w.offset, w.len, w.shape)
}

pub(crate) fn collect_recursive_metrics(
    nodes: &[&NodeProto],
    defined: &mut HashSet<String>,
    used: &mut HashSet<String>,
) {
    for node in nodes {
        for out in &node.output {
            if !out.is_empty() {
                defined.insert(out.clone());
            }
        }
        for inp in &node.input {
            if !inp.is_empty() {
                used.insert(inp.clone());
            }
        }
        // For subgraphs (If, Loop), recursively collect what they use
        // This is needed because subgraphs can access outer scope variables
        for attr in &node.attribute {
            if let Some(g) = &attr.g {
                let mut sub_defined = HashSet::new();
                let mut sub_used = HashSet::new();
                // Recursively collect from subgraph
                let sub_nodes: Vec<&NodeProto> = g.node.iter().collect();
                collect_recursive_metrics(&sub_nodes, &mut sub_defined, &mut sub_used);
                // What the subgraph uses but doesn't define itself needs to come from outer scope
                for u in &sub_used {
                    if !sub_defined.contains(u) {
                        used.insert(u.clone());
                    }
                }
                // Note: we don't add sub_defined to defined, because those are scoped to the subgraph
            }
        }
    }
}

/// [`ElemType`] of every value of `g` and its subgraphs. Expects
/// [`infer_dtypes`](super::passes::infer_dtypes) to have run.
///
/// Values take [`ElemType::of`] their ONNX type, with the exceptions of
/// lele's kernels: 8-bit quantized data is held in f32, and so are the int32
/// results of `MatMulInteger` and `ConvInteger`, the indices `TopK` returns
/// and the result of `Mod`. An operator whose output takes the type of an
/// input takes its element type too, so these propagate.
pub(crate) fn variable_types(g: &Graph) -> VarTypes {
    use crate::compiler::ir::ValueId;
    use crate::compiler::passes::follows_input;

    fn walk(g: &Graph, outer: &HashMap<String, ElemType>, out: &mut HashMap<String, ElemType>) {
        let mut repr: HashMap<ValueId, ElemType> = HashMap::new();
        let captures: HashSet<ValueId> = g.captures().collect();
        for v in g.value_ids() {
            let value = g.value(v);
            if value.producer().is_none() {
                let r = outer
                    .get(&value.name)
                    .filter(|_| captures.contains(&v))
                    .copied()
                    .unwrap_or_else(|| ElemType::of(value.dtype));
                repr.insert(v, r);
            }
        }
        for (_, node) in g.nodes() {
            let subgraphs = node.op.subgraphs();
            let mut sub_reprs: Vec<HashMap<String, ElemType>> = Vec::new();
            if !subgraphs.is_empty() {
                let mut scope = outer.clone();
                for (&v, &r) in &repr {
                    scope.insert(g.value(v).name.clone(), r);
                }
                for sub in &subgraphs {
                    let mut sub_out = HashMap::new();
                    walk(sub, &scope, &mut sub_out);
                    sub_reprs.push(sub_out.clone());
                    out.extend(sub_out);
                }
            }
            for (i, v) in node.outputs().iter().enumerate() {
                let Some(v) = *v else { continue };
                // `Mod` of two i64 tensors stays i64: its result goes on to meet i64
                // initializers, which the kernels of f32 and i64 tensors cannot mix.
                let integer_mod = matches!(node.op, Op::Binary(Binary::Mod))
                    && (0..2).all(|k| {
                        node.input(k)
                            .is_some_and(|input| repr.get(&input) == Some(&ElemType::I64))
                    });
                let f32_kernel = !integer_mod
                    && matches!(
                        node.op,
                        Op::Binary(Binary::Mod)
                            | Op::ConvInteger(_)
                            | Op::MatMulInteger
                            | Op::TopK { .. }
                    );
                // If and Loop take the representation of their subgraphs'
                // outputs.
                let from_subgraph = |sub: &Graph, j: usize, reprs: &HashMap<String, ElemType>| {
                    sub.outputs
                        .get(j)
                        .and_then(|&o| reprs.get(&sanitize_name(&sub.value(o).name)))
                        .copied()
                };
                let of_dtype = || ElemType::of(g.value(v).dtype);
                let r = if f32_kernel {
                    ElemType::F32
                } else if let Op::If { then_branch, .. } = &node.op {
                    from_subgraph(then_branch, i, &sub_reprs[0]).unwrap_or_else(of_dtype)
                } else if let Op::Loop { body } = &node.op {
                    from_subgraph(body, i + 1, &sub_reprs[0]).unwrap_or_else(of_dtype)
                } else {
                    match follows_input(&node.op, i).and_then(|k| node.input(k)) {
                        Some(input) => repr[&input],
                        None => of_dtype(),
                    }
                };
                repr.insert(v, r);
            }
        }
        for (v, r) in repr {
            out.insert(sanitize_name(&g.value(v).name), r);
        }
    }
    let mut types = HashMap::new();
    walk(g, &HashMap::new(), &mut types);
    VarTypes(types)
}

/// A run of nodes of one IR graph to generate code for, in execution order:
/// their ids and their lowered protos, which overrides and patterns see.
#[derive(Clone, Copy)]
pub(crate) struct Nodes<'a> {
    pub graph: &'a Graph,
    pub ids: &'a [NodeId],
    pub protos: &'a [&'a NodeProto],
}

impl<'a> Nodes<'a> {
    fn len(&self) -> usize {
        self.ids.len()
    }

    fn slice(&self, range: std::ops::Range<usize>) -> Nodes<'a> {
        Nodes {
            graph: self.graph,
            ids: &self.ids[range.clone()],
            protos: &self.protos[range],
        }
    }
}

/// The ids and lowered protos of `g`'s nodes, for [`Nodes`].
pub(crate) fn lower_nodes(g: &Graph) -> (Vec<NodeId>, Vec<NodeProto>) {
    let ids = g.node_ids();
    let protos = ids.iter().map(|&id| g.node_proto(id)).collect();
    (ids, protos)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_partitioned_graph<W: Write>(
    nodes: Nodes<'_>,               // Top level nodes
    chunk_writer: &mut Vec<String>, // Function definitions
    body_writer: &mut W,            // Forward body calls
    indent: usize,
    weights: &Weights,
    allocator: Option<&Allocator>,
    analysis: Option<&AnalysisData>,
    current_id: &mut usize,
    compiler: &Compiler,
    graph_inputs: &[ValueInfoProto],
    graph_outputs: &[ValueInfoProto],
    var_types: &VarTypes,
) -> std::io::Result<()> {
    let chunk_size = 500;
    let total_nodes = nodes.len();
    let mut i = 0;
    let mut chunk_idx = 0;
    let mut available_vars: HashMap<String, String> = HashMap::new();
    // Initial variables in forward() scope
    for inp in graph_inputs {
        let name = sanitize_name(&inp.name);
        available_vars.insert(name.clone(), name);
    }
    while i < total_nodes {
        let end = std::cmp::min(i + chunk_size, total_nodes);
        let chunk = nodes.slice(i..end);
        let chunk_nodes = chunk.protos;
        let nodes_count = chunk_nodes.len();
        // 1. Identify Inputs: Used in chunk, defined BEFORE chunk (or global)
        let mut chunk_inputs_set = HashSet::new();
        let mut defined_raw = HashSet::new();
        let mut used_raw = HashSet::new();
        collect_recursive_metrics(chunk_nodes, &mut defined_raw, &mut used_raw);
        let mut defined_sanitized_chunk = HashSet::new();
        for d in &defined_raw {
            defined_sanitized_chunk.insert(sanitize_name(d));
        }
        for inp in &used_raw {
            let name = sanitize_name(inp);
            if name.is_empty() || weights.contains(inp) {
                continue;
            }
            if !defined_sanitized_chunk.contains(&name) {
                chunk_inputs_set.insert(name);
            }
        }
        let mut chunk_inputs: Vec<String> = chunk_inputs_set.into_iter().collect();
        chunk_inputs.sort(); // Deterministic order
        // 2. Identify Outputs: Live AFTER chunk
        let chunk_end_id = *current_id + nodes_count;
        let global_outputs: HashSet<String> =
            graph_outputs.iter().map(|o| o.name.clone()).collect();
        let _chunk_start_id = *current_id;
        let mut live_sanitized = HashSet::new();
        // Collect all potential live variables (raw names) from recursive usage
        let mut potent_live = HashSet::new();
        for d in &defined_raw {
            potent_live.insert(d.clone());
        }
        for u in &used_raw {
            potent_live.insert(u.clone());
        }
        if let Some(an) = analysis {
            for raw in potent_live {
                let sanitized = sanitize_name(&raw);
                if weights.contains(&raw) {
                    continue;
                }
                // If defined in this chunk OR is an input arg
                let is_defined_here = chunk_nodes.iter().any(|n| n.output.contains(&raw));
                let is_input_arg = chunk_inputs.contains(&sanitized); // approximate check
                if is_defined_here || is_input_arg {
                    let last_use = *an.death_time.get(&raw).unwrap_or(&0);
                    let is_graph_out = global_outputs.contains(&raw);
                    if is_graph_out || last_use >= chunk_end_id {
                        live_sanitized.insert(sanitized);
                    }
                }
            }
        }
        let mut chunk_outputs: Vec<String> = live_sanitized.into_iter().collect();
        chunk_outputs.sort();
        // 3. Generate Function Code
        let func_name = format!("run_chunk_{}", chunk_idx);
        let mut f = Vec::new();
        let args_sig: Vec<String> = chunk_inputs
            .iter()
            .map(|n| format!("{}: TensorView<'w, {}>", n, var_types.get(n)))
            .collect();
        let ret_sig: Vec<String> = chunk_outputs
            .iter()
            .map(|n| format!("TensorView<'static, {}>", var_types.get(n)))
            .collect();
        let ret_type = if ret_sig.len() == 1 {
            ret_sig[0].clone()
        } else {
            format!("({})", ret_sig.join(", "))
        };
        writeln!(&mut f, "    #[inline(never)]")?; // Prevent inlining large chunks
        writeln!(
            &mut f,
            "    fn {}<'w>(&self, ws: &'w mut {}Workspace, {}) -> {} {{",
            func_name,
            compiler.model_name,
            args_sig.join(", "),
            ret_type
        )?;
        // Generate body
        generate_nodes(
            chunk, &mut f, 2, weights, allocator, analysis, current_id, compiler, var_types,
        )?;
        // Return — detach_or_move (zero-copy) for views of the workspace or the weights,
        // which outlive the chunk, and for tensors owning their data, which move out;
        // to_owned for everything else, views of buffers local to the chunk included,
        // which are freed when it returns. A view of a tensor that moves out stays
        // valid: moving a Vec keeps its elements where they are, and the tensor is
        // returned as long as its views live. Copies are taken first and views
        // detached before what they view, which then is no longer borrowed when it
        // moves.
        let body = String::from_utf8_lossy(&f).into_owned();
        let (mut detached, copied): (Vec<&String>, Vec<&String>) = chunk_outputs
            .iter()
            .partition(|s| outlives_chunk(&body, s, 0));
        detached.sort_by_key(|s| std::cmp::Reverse(definition_line(&body, s)));
        for s in &copied {
            writeln!(&mut f, "        let {s}__ret = {s}.to_owned();")?;
        }
        for s in &detached {
            writeln!(
                &mut f,
                "        let {s}__ret = unsafe {{ {s}.detach_or_move() }};"
            )?;
        }
        let ret_vals: Vec<String> = chunk_outputs.iter().map(|s| format!("{s}__ret")).collect();
        if ret_vals.is_empty() {
            writeln!(&mut f, "        // No outputs")?;
        } else if ret_vals.len() == 1 {
            writeln!(&mut f, "        {}", ret_vals[0])?;
        } else {
            writeln!(&mut f, "        ({})", ret_vals.join(", "))?;
        }
        writeln!(&mut f, "    }}\n")?;
        chunk_writer.push(String::from_utf8(f).unwrap());
        // 4. Generate Call in Forward
        let tab = "    ".repeat(indent);
        let call_args: Vec<String> = chunk_inputs
            .iter()
            .map(|n| {
                if let Some(mapped) = available_vars.get(n) {
                    mapped.clone()
                } else {
                    "lele::tensor::TensorView::empty()".to_string()
                }
            })
            .collect();
        if chunk_outputs.is_empty() {
            writeln!(
                body_writer,
                "{}self.{}(ws, {});",
                tab,
                func_name,
                call_args.join(", ")
            )?;
        } else {
            let out_vars = chunk_outputs.join(", ");
            writeln!(
                body_writer,
                "{}let ({}) = self.{}(ws, {});",
                tab,
                out_vars,
                func_name,
                call_args.join(", ")
            )?;
        }
        // Update variables available in forward
        for out in &chunk_outputs {
            available_vars.insert(out.clone(), out.clone());
        }
        i += nodes_count;
        chunk_idx += 1;
    }
    Ok(())
}

/// Whether chunk output `var` may leave the chunk as a borrowed view, decided from the
/// chunk's generated `body`: its data must be in the workspace or the weights, not in a
/// buffer local to the chunk (an output buffer `buf_*`, a cast's `temp_cast_buf_*`), which
/// is freed when the chunk returns.
///
/// A variable written into `ws.buf_*` qualifies, as does a weight or a chunk argument; a
/// view (`reshape`, `unsqueeze`, `squeeze`, `.clone()`) qualifies if what it views does,
/// and a slice, gather, transpose or split (which may be a view) if both do.
/// Every definition counts (there is one per target under `cfg`). Anything else is
/// returned as a copy.
/// The line of `body` that defines `var`, or `None` if it is an argument of
/// the chunk.
fn definition_line(body: &str, var: &str) -> Option<usize> {
    let prefix = format!("let {var} = ");
    body.lines().map(str::trim_start).position(|l| {
        l.starts_with(&prefix)
            || l.strip_prefix("let (")
                .and_then(|l| l.split_once(") ="))
                .is_some_and(|(names, _)| names.split(',').any(|n| n.trim() == var))
    })
}

fn outlives_chunk(body: &str, var: &str, depth: usize) -> bool {
    if depth > 64 {
        return false;
    }
    // Multi-output kernels (`let (a, b) = topk(..)`, ...) write buffers local to the
    // chunk.
    let in_tuple = body.lines().map(str::trim_start).any(|l| {
        l.strip_prefix("let (")
            .and_then(|l| l.split_once(") ="))
            .is_some_and(|(names, _)| names.split(',').any(|n| n.trim() == var))
    });
    if in_tuple {
        return false;
    }
    let prefix = format!("let {var} = ");
    let defs: Vec<&str> = body
        .lines()
        .map(str::trim_start)
        .filter_map(|l| l.strip_prefix(prefix.as_str()))
        .collect();
    if defs.is_empty() {
        return true;
    }
    // The first operand of `rhs` if it is a call to `f`.
    let operand = |rhs: &str, f: &str| {
        rhs.strip_prefix(&format!("lele::kernels::{f}(&"))
            .map(|s| s.split([',', ')']).next().unwrap_or("").trim().to_string())
    };
    defs.iter().all(|rhs| {
        let rhs = rhs.split(';').next().unwrap_or("");
        // These may return a view of their first operand rather than their buffer
        // (`kernels::manipulation::borrow_run`): both have to outlive the chunk.
        let borrowed = ["slice", "gather", "transpose", "split_one"]
            .iter()
            .find_map(|f| operand(rhs, f));
        if let Some(source) = borrowed {
            return rhs.contains("&mut ws.buf_") && outlives_chunk(body, &source, depth + 1);
        }
        // A tensor owning its data moves out of the chunk.
        if rhs.contains("&mut ws.buf_")
            || rhs.starts_with("self.weight_")
            || rhs.starts_with("lele::kernels::scatter_nd_owned(")
        {
            return true;
        }
        let viewed = ["reshape", "unsqueeze", "squeeze"]
            .iter()
            .find_map(|f| rhs.strip_prefix(&format!("lele::kernels::{f}(&")))
            .or_else(|| rhs.strip_suffix(".clone()"))
            .map(|s| s.split([',', ')']).next().unwrap_or("").trim());
        match viewed {
            Some(source) if !source.is_empty() => outlives_chunk(body, source, depth + 1),
            _ => false,
        }
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_nodes(
    ir_nodes: Nodes<'_>,
    w: &mut dyn Write,
    indent: usize,
    weights: &Weights,
    allocator: Option<&Allocator>,
    analysis: Option<&AnalysisData>,
    current_id: &mut usize,
    compiler: &Compiler,
    var_types: &VarTypes,
) -> std::io::Result<()> {
    let tab = "    ".repeat(indent);
    let nodes = ir_nodes.protos;
    let mut idx = 0;
    while idx < nodes.len() {
        // Try to match patterns first
        let mut matched_len = 0;
        let remaining_nodes = &nodes[idx..];
        for pattern in &compiler.patterns {
            if let Some(len) = (pattern.matcher)(remaining_nodes) {
                let table = weights.pattern_table();
                (pattern.generator)(remaining_nodes, table, allocator, w, indent)?;
                matched_len = len;
                break;
            }
        }
        if matched_len > 0 {
            for _ in 0..matched_len {
                *current_id += 1;
            }
            idx += matched_len;
            continue;
        }
        let node = nodes[idx];
        let id = ir_nodes.ids[idx];
        idx += 1;
        *current_id += 1;
        let op = node.op_type.as_str();
        let ir_op = &ir_nodes.graph.node(id).op;
        let outputs: Vec<String> = node.output.iter().map(|s| sanitize_name(s)).collect();
        // Skip Code Generation for Constants that are already in weights
        if matches!(ir_op, Op::Constant { .. })
            && !outputs.is_empty()
            && weights.get(&node.output[0]).is_some()
        {
            continue;
        }
        // Check if Div/Mod output is i64 (e.g. integer division for shape computation)
        let div_or_mod = matches!(ir_op, Op::Binary(Binary::Div | Binary::Mod));
        let div_mod_output_is_i64 =
            div_or_mod && outputs.first().is_some_and(|o| var_types.is_i64(o));
        let inputs: Vec<String> = node
            .input
            .iter()
            .map(|s| {
                let name = sanitize_name(s);
                if let Some(weight) = weights.get(s) {
                    // For Div/Mod: only force f32 when output is NOT i64.
                    // When output is i64 (integer division), keep i64 types.
                    let force_f32 = div_or_mod && !div_mod_output_is_i64;
                    let target = if force_f32 {
                        ElemType::F32
                    } else {
                        var_types.get(&name)
                    };
                    weight_expr(weight, target)
                } else {
                    name
                }
            })
            .collect();
        // Unused check
        let is_unused = if let Some(an) = analysis {
            if matches!(ir_op, Op::If { .. } | Op::Loop { .. }) {
                false
            }
            // Conservative for control flow
            else {
                node.output.iter().all(|out| {
                    if out.is_empty() {
                        return true;
                    }
                    /* Graph outputs check removed here as we don't have access to graph outputs easily in this Func,
                    but liveness analysis should handle it via infinity use */
                    an.uses.get(out).map(|u| u.is_empty()).unwrap_or(true)
                })
            }
        } else {
            false
        };
        if is_unused && !node.output.is_empty() {
            // writeln!(w, "{}// Op {} {} skipped (unused)", tab, op, outputs.join(","))?;
            continue;
        }
        // For Div and Mod operations, force f32 type for output ONLY when output is not i64.
        // When output is i64 (integer division for shape computation), keep i64.
        let force_f32 = div_or_mod && !div_mod_output_is_i64;
        let is_i64 = !force_f32 && outputs.first().is_some_and(|o| var_types.is_i64(o));
        let buf_expr = if let Some(alloc) = allocator {
            if !node.output.is_empty() {
                if let Some(&idx) = alloc.tensor_to_buffer.get(&node.output[0]) {
                    format!("&mut ws.buf_{}", idx)
                } else {
                    if !outputs.is_empty() {
                        format!("&mut buf_{}", outputs[0])
                    } else {
                        String::new()
                    }
                }
            } else {
                String::new()
            }
        } else {
            if !outputs.is_empty() {
                format!("&mut buf_{}", outputs[0])
            } else {
                String::new()
            }
        };

        if is_i64 && !outputs.is_empty() && buf_expr.starts_with("&mut buf_") {
            writeln!(w, "{}let mut buf_{} = Vec::<i64>::new();", tab, outputs[0])?;
        } else if !is_i64 && !outputs.is_empty() && buf_expr.starts_with("&mut buf_") {
            // Built-in handlers might not need Vec<f32> if they don't use buf_expr,
            // but we'll generate it by default if expected.
            // Some ops like Split/LSTM manage their own buffers.
            if !matches!(
                ir_op,
                Op::DynamicQuantizeLinear | Op::Lstm | Op::Split { .. }
            ) {
                writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, outputs[0])?;
            }
        }
        let mut ctx = OpContext {
            node,
            graph: ir_nodes.graph,
            id,
            inputs: inputs.clone(),
            outputs: outputs.clone(),
            buf_expr,
            indent,
            weights,
            allocator,
            analysis,
            current_id,
            compiler,
            var_types,
        };

        // 1. Override
        if let Some(handler) = compiler.overrides.get(op) {
            handler(node, &inputs, &outputs, &ctx.buf_expr, w, indent)?;
            continue;
        }

        // 2. Built-in
        if super::ops::dispatch_builtin(&mut ctx, w)? {
            continue;
        }

        // 3. Fallback
        eprintln!(
            "Warning: Unrecognized operator '{}' (node: {})",
            op, node.name
        );
        for (idx, out_name) in outputs.iter().enumerate() {
            writeln!(
                w,
                "{}let {} = lele::tensor::TensorView::empty(); // Unimplemented {} out {}",
                tab, out_name, op, idx
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::Compiler;
    use crate::model::onnx_proto::NodeProto;

    /// SenseVoice's attention mask came out of a cast into a chunk-local buffer and was
    /// returned as a borrowed view of it: every later chunk read freed memory.
    #[test]
    fn test_only_views_of_the_workspace_or_weights_leave_a_chunk_borrowed() {
        let body = "\
        let mut temp_cast_buf_m = Vec::<f32>::new();
        let m = lele::kernels::utils::cast_to_f32(&less, &mut temp_cast_buf_m);
        let m2 = m.clone(); // Cast f32->f32 is no-op
        let m3 = lele::kernels::unsqueeze(&m2, &[1]);
        let x = lele::kernels::add(&a, &b, &mut ws.buf_3);
        let x2 = lele::kernels::reshape(&x, &[-1, 512]);
        let w = self.weight_f32(0, 4, &[]);
        let y = lele::kernels::relu(&x, &mut buf_y);
        #[cfg(target_arch = \"aarch64\")]
        let z = lele::kernels::add(&a, &b, &mut ws.buf_4);
        #[cfg(not(target_arch = \"aarch64\"))]
        let z = lele::kernels::add(&a, &b, &mut buf_z);
        let (values, indices) = lele::kernels::topk(&x, 5, 1, true, true, &mut buf_v, &mut buf_i);
        let picked = lele::kernels::squeeze(&indices, None);
        let part = lele::kernels::slice(&x, &[0], &[1], &[0], &[1], &mut ws.buf_5);
        let part_of_y = lele::kernels::slice(&y, &[0], &[1], &[0], &[1], &mut ws.buf_6);
        let layer = lele::kernels::gather(&arg, &self.weight_i64(0, 8, &[]), 0, &mut ws.buf_7);
        let swapped = lele::kernels::transpose(&x, &[1, 0], &mut buf_t);
        let half = lele::kernels::split_one(&m3, 1, &[0, 0], 0, &mut ws.buf_8);
";
        for (var, safe) in [
            ("values", false),
            ("indices", false),
            ("picked", false),
            ("m", false),
            ("m2", false),
            ("m3", false),
            ("x", true),
            ("x2", true),
            ("w", true),
            ("y", false),
            ("z", false),
            ("chunk_argument", true),
            // May be views of their first operand.
            ("part", true),
            ("part_of_y", false),
            ("layer", true),
            ("swapped", false),
            ("half", false),
        ] {
            assert_eq!(outlives_chunk(body, var, 0), safe, "{var}");
        }
    }

    /// Code generated for `node` alone.
    fn generate_one(
        node: &NodeProto,
        weights: &Weights,
        compiler: &Compiler,
        var_types: &VarTypes,
    ) -> String {
        let g = Graph::from_onnx(crate::model::onnx_proto::GraphProto {
            node: vec![node.clone()],
            ..Default::default()
        });
        let (ids, protos) = lower_nodes(&g);
        let protos: Vec<&NodeProto> = protos.iter().collect();
        let nodes = Nodes {
            graph: &g,
            ids: &ids,
            protos: &protos,
        };
        let mut output = Vec::new();
        generate_nodes(
            nodes,
            &mut output,
            0,
            weights,
            None,
            None,
            &mut 0,
            compiler,
            var_types,
        )
        .unwrap();
        String::from_utf8(output).unwrap()
    }

    #[test]
    fn test_custom_op_override() {
        let mut compiler = Compiler::new();
        compiler = compiler.with_override("CustomOp", |_node, inputs, outputs, _buf, w, indent| {
            let tab = "    ".repeat(indent);
            writeln!(w, "{}// Custom implementation for {}", tab, outputs[0])?;
            writeln!(
                w,
                "{}let {} = custom_kernel(&{});",
                tab, outputs[0], inputs[0]
            )
        });

        let node = NodeProto {
            input: vec!["input1".to_string()],
            output: vec!["output1".to_string()],
            op_type: "CustomOp".to_string(),
            ..Default::default()
        };

        let result = generate_one(&node, &Weights::default(), &compiler, &VarTypes::default());
        assert!(result.contains("// Custom implementation for output1"));
        assert!(result.contains("let output1 = custom_kernel(&input1);"));
    }

    /// yolo26n-seg splits the flat TopK indices into (anchor, class) with an
    /// integer Div by an i64 constant (#28). `topk` returns its indices as f32,
    /// so the Div has to run in f32 on the constant cast to f32, and truncate.
    #[test]
    fn test_integer_div_of_topk_indices_runs_in_f32() {
        let node = |op: &str, input: &[&str], output: &[&str]| NodeProto {
            input: input.iter().map(|s| s.to_string()).collect(),
            output: output.iter().map(|s| s.to_string()).collect(),
            op_type: op.to_string(),
            ..Default::default()
        };
        let topk = node("TopK", &["scores", "k"], &["values", "indices"]);
        let div = node("Div", &["indices", "classes"], &["anchor"]);
        let int64 = |name: &str, v: i64| crate::model::onnx_proto::TensorProto {
            name: name.to_string(),
            data_type: DType::I64.to_onnx(),
            int64_data: vec![v],
            ..Default::default()
        };
        let graph = crate::model::onnx_proto::GraphProto {
            node: vec![topk, div.clone()],
            initializer: vec![int64("k", 5), int64("classes", 80)],
            ..Default::default()
        };
        let mut ir = crate::compiler::ir::Graph::from_onnx(graph);
        crate::compiler::passes::infer_dtypes(&mut ir);
        let var_types = variable_types(&ir);
        let weights = Weights::collect(&ir, false);
        assert_eq!(var_types.get("indices"), ElemType::F32);
        assert_eq!(var_types.get("anchor"), ElemType::F32);

        let result = generate_one(&div, &weights, &Compiler::new(), &var_types);
        assert!(
            result.contains(
                "let anchor = lele::kernels::div_trunc(&indices, &self.weight_i64_f32(16, 8, &[]), &mut buf_anchor);"
            ),
            "{result}"
        );
    }

    #[test]
    fn test_builtin_dispatch() {
        let compiler = Compiler::new();
        let node = NodeProto {
            input: vec!["a".to_string(), "b".to_string()],
            output: vec!["c".to_string()],
            op_type: "Add".to_string(),
            ..Default::default()
        };

        let result = generate_one(&node, &Weights::default(), &compiler, &VarTypes::default());
        assert!(result.contains("let mut buf_c = Vec::<f32>::new();"));
        assert!(result.contains("let c = lele::kernels::add(&a, &b, &mut buf_c);"));
    }

    /// f16 weights read only as the right operand of a product, with a
    /// multiple of 16 columns, or as the weights of a pointwise conv with a
    /// multiple of 16 output channels (transposed), are stored packed and read
    /// as such; others are stored as they are.
    #[test]
    fn test_packs_f16_matmul_weights() {
        use crate::compiler::Optimizations;
        use crate::model::onnx_proto::{GraphProto, TensorProto, ValueInfoProto};
        let info = |name: &str| ValueInfoProto {
            name: name.to_string(),
            ..Default::default()
        };
        let halves = |k: usize, n: usize, first: usize| -> Vec<half::f16> {
            (0..k * n).map(|i| half::f16::from_f32((first + i) as f32)).collect()
        };
        let f16_dims = |name: &str, dims: &[usize], first: usize| TensorProto {
            name: name.to_string(),
            data_type: DType::F16.to_onnx(),
            dims: dims.iter().map(|&d| d as i64).collect(),
            raw_data: halves(dims.iter().product(), 1, first)
                .iter()
                .flat_map(|h| h.to_le_bytes())
                .collect(),
            ..Default::default()
        };
        let f16 = |name: &str, k: usize, n: usize, first: usize| f16_dims(name, &[k, n], first);
        let node = |op: &str, input: &[&str], output: &str| NodeProto {
            op_type: op.to_string(),
            input: input.iter().map(|s| s.to_string()).collect(),
            output: vec![output.to_string()],
            ..Default::default()
        };
        let graph = GraphProto {
            node: vec![
                node("MatMul", &["x", "packed"], "y1"),
                node("MatMul", &["x", "narrow"], "y2"),
                node("MatMul", &["x", "shared"], "y3"),
                node("Neg", &["shared"], "y4"),
                node("Conv", &["z", "conv"], "y5"),
                node("Conv", &["z", "conv_narrow"], "y6"),
            ],
            initializer: vec![
                f16("packed", 8, 32, 0),
                f16("narrow", 8, 24, 300),
                f16("shared", 8, 16, 600),
                f16_dims("conv", &[32, 8, 1], 900),
                f16_dims("conv_narrow", &[24, 8, 1], 1200),
            ],
            input: vec![info("x"), info("z")],
            output: ["y1", "y2", "y3", "y4", "y5", "y6"].map(info).to_vec(),
            ..Default::default()
        };
        let compile = |weight_packing| {
            Compiler::new()
                .with_optimizations(Optimizations {
                    weight_packing,
                    ..Default::default()
                })
                .compile(&graph)
                .unwrap()
        };
        let result = compile(true);
        let code = &result.code;
        assert_eq!(code.matches("self.weight_f16_packed(").count(), 2, "{code}");
        assert_eq!(code.matches("self.weight_f16_raw(").count(), 3, "{code}");
        let bytes = |h: &[half::f16]| -> Vec<u8> { h.iter().flat_map(|h| h.to_le_bytes()).collect() };
        let pack_f16 = crate::kernels::matmul::pack_f16;
        let packed = bytes(&pack_f16(&halves(8, 32, 0), 8, 32));
        let raw = bytes(&halves(8, 32, 0));
        // The conv weights [32, 8, 1] are packed as their transpose [8, 32].
        let conv = halves(32, 8, 900);
        let transposed: Vec<_> = (0..8 * 32).map(|i| conv[i % 32 * 8 + i / 32]).collect();
        let conv_packed = bytes(&pack_f16(&transposed, 8, 32));
        let contains = |w: &[u8], s: &[u8]| w.windows(s.len()).any(|x| x == s);
        assert!(contains(&result.weights, &packed));
        assert!(!contains(&result.weights, &raw));
        assert!(contains(&result.weights, &conv_packed));
        assert!(!contains(&result.weights, &bytes(&conv)));

        let unpacked = compile(false);
        assert!(!unpacked.code.contains("weight_f16_packed"));
        assert!(contains(&unpacked.weights, &raw));
        assert!(contains(&unpacked.weights, &bytes(&conv)));
    }

    /// i64 tensors, casts and every output of a Split are written into the workspace,
    /// each into a buffer of its own type, not into vectors allocated on every call.
    #[test]
    fn test_workspace_holds_i64_tensors_casts_and_splits() {
        use crate::model::onnx_proto::{AttributeProto, GraphProto, TensorProto, ValueInfoProto};
        let info = |name: &str| ValueInfoProto {
            name: name.to_string(),
            ..Default::default()
        };
        let node = |op: &str, input: &[&str], output: &[&str], attribute: Vec<AttributeProto>| {
            NodeProto {
                op_type: op.to_string(),
                input: input.iter().map(|s| s.to_string()).collect(),
                output: output.iter().map(|s| s.to_string()).collect(),
                attribute,
                ..Default::default()
            }
        };
        let int = |name: &str, i: i64| AttributeProto {
            name: name.to_string(),
            i,
            ..Default::default()
        };
        let graph = GraphProto {
            node: vec![
                node("Shape", &["x"], &["s"], vec![]),
                node("Gather", &["s", "idx"], &["g"], vec![]),
                node("Cast", &["g"], &["c"], vec![int("to", DType::F32.to_onnx() as i64)]),
                node("Split", &["x"], &["a", "b"], vec![int("axis", 0)]),
            ],
            initializer: vec![TensorProto {
                name: "idx".to_string(),
                data_type: DType::I64.to_onnx(),
                dims: vec![1],
                raw_data: 0i64.to_le_bytes().to_vec(),
                ..Default::default()
            }],
            input: vec![info("x")],
            output: ["c", "a", "b"].map(info).to_vec(),
            ..Default::default()
        };
        let code = Compiler::new().compile(&graph).unwrap().code;
        assert!(!code.contains("Vec::<"), "{code}");
        let buffer = |call: &str| -> String {
            let line = code.lines().find(|l| l.contains(call)).unwrap_or_else(|| panic!("{code}"));
            let at = line.find("&mut ws.buf_").unwrap_or_else(|| panic!("{line}"));
            let digits = &line[at + "&mut ws.buf_".len()..];
            let n: String = digits.chars().take_while(char::is_ascii_digit).collect();
            format!("pub buf_{n}: ")
        };
        let field = |call: &str| {
            let name = buffer(call);
            code.lines()
                .find_map(|l| l.trim().strip_prefix(&name))
                .unwrap_or_else(|| panic!("{code}"))
                .to_string()
        };
        assert_eq!(field("kernels::gather("), "Vec<i64>,");
        assert_eq!(field("cast_to_f32("), "Vec<f32>,");
        assert_eq!(field("split_one(&x, 0, &[0, 0], 0,"), "Vec<f32>,");
        assert_ne!(
            buffer("split_one(&x, 0, &[0, 0], 0,"),
            buffer("split_one(&x, 0, &[0, 0], 1,")
        );
    }

    /// Trilu, ScatterND and InstanceNormalization have code, a Trilu without its
    /// `k` takes the main diagonal, and a Reshape target keeps its 0s, which copy
    /// the input's dimension, beside a -1.
    #[test]
    fn test_trilu_scatter_nd_instance_norm_and_reshape_zero() {
        use crate::model::onnx_proto::{AttributeProto, GraphProto, TensorProto, ValueInfoProto};
        let info = |name: &str| ValueInfoProto {
            name: name.to_string(),
            ..Default::default()
        };
        let node =
            |op: &str, input: &[&str], output: &[&str], attribute: Vec<AttributeProto>| NodeProto {
                op_type: op.to_string(),
                input: input.iter().map(|s| s.to_string()).collect(),
                output: output.iter().map(|s| s.to_string()).collect(),
                attribute,
                ..Default::default()
            };
        let constant = |name: &str, dims: Vec<i64>, values: &[f32]| TensorProto {
            name: name.to_string(),
            data_type: DType::F32.to_onnx(),
            dims,
            raw_data: values.iter().flat_map(|x| x.to_le_bytes()).collect(),
            ..Default::default()
        };
        let ints = |name: &str, values: &[i64]| TensorProto {
            name: name.to_string(),
            data_type: DType::I64.to_onnx(),
            dims: vec![values.len() as i64],
            raw_data: values.iter().flat_map(|x| x.to_le_bytes()).collect(),
            ..Default::default()
        };
        let attribute = |name: &str, f: f32| AttributeProto {
            name: name.to_string(),
            f,
            ..Default::default()
        };
        let graph = GraphProto {
            node: vec![
                node("Trilu", &["x"], &["t"], vec![]),
                node("ScatterND", &["x", "idx", "upd"], &["s"], vec![]),
                node(
                    "InstanceNormalization",
                    &["x", "g", "b"],
                    &["n"],
                    vec![attribute("epsilon", 1e-3)],
                ),
                node("Reshape", &["x", "target"], &["r"], vec![]),
            ],
            initializer: vec![
                ints("idx", &[1]),
                constant("upd", vec![1], &[0.0]),
                constant("g", vec![1], &[1.0]),
                constant("b", vec![1], &[0.0]),
                ints("target", &[0, 1, -1]),
            ],
            input: vec![info("x")],
            output: ["t", "s", "n", "r"].map(info).to_vec(),
            ..Default::default()
        };
        let code = Compiler::new().compile(&graph).unwrap().code;
        assert!(code.contains("kernels::trilu(&x, 0, true,"), "{code}");
        assert!(code.contains("kernels::scatter_nd(&x,"), "{code}");
        assert!(
            code.contains("kernels::instance_norm(&x,") && code.contains("0.001"),
            "{code}"
        );
        assert!(code.contains("kernels::reshape(&x, &[0, 1, -1])"), "{code}");
    }

    /// A state that a ScatterND reads last is updated in place and moves out
    /// of `forward`; one read again afterwards is copied.
    #[test]
    fn test_scatter_nd_takes_a_state_it_reads_last() {
        use crate::model::onnx_proto::{GraphProto, TensorProto, ValueInfoProto};
        let info = |name: &str| ValueInfoProto {
            name: name.to_string(),
            ..Default::default()
        };
        let node = |op: &str, input: &[&str], output: &str| NodeProto {
            op_type: op.to_string(),
            input: input.iter().map(|s| s.to_string()).collect(),
            output: vec![output.to_string()],
            ..Default::default()
        };
        let graph = |nodes: Vec<NodeProto>, outputs: &[&str]| GraphProto {
            node: nodes,
            initializer: vec![TensorProto {
                name: "idx".to_string(),
                data_type: DType::I64.to_onnx(),
                dims: vec![1, 1],
                int64_data: vec![1],
                ..Default::default()
            }],
            input: vec![info("cache"), info("upd")],
            output: outputs.iter().map(|o| info(o)).collect(),
            ..Default::default()
        };
        let code = |g: &GraphProto| Compiler::new().compile(g).unwrap().code;
        let scatter = |out: &str| node("ScatterND", &["cache", "idx", "upd"], out);

        let moved = code(&graph(vec![scatter("next")], &["next"]));
        assert!(moved.contains("kernels::scatter_nd_owned(cache, "), "{moved}");
        assert!(moved.contains("next.detach_or_move()"), "{moved}");
        assert!(moved.contains("res.into_owned()"), "{moved}");

        let read_again = code(&graph(
            vec![scatter("next"), node("Add", &["cache", "next"], "sum")],
            &["next", "sum"],
        ));
        assert!(read_again.contains("kernels::scatter_nd(&cache, "), "{read_again}");
    }

    /// Elu, IsNaN and ScatterElements have code, with the attributes they were given.
    #[test]
    fn test_elu_is_nan_and_scatter_elements() {
        use crate::model::onnx_proto::{AttributeProto, GraphProto, TensorProto, ValueInfoProto};
        let info = |name: &str| ValueInfoProto {
            name: name.to_string(),
            ..Default::default()
        };
        let node =
            |op: &str, input: &[&str], output: &[&str], attribute: Vec<AttributeProto>| NodeProto {
                op_type: op.to_string(),
                input: input.iter().map(|s| s.to_string()).collect(),
                output: output.iter().map(|s| s.to_string()).collect(),
                attribute,
                ..Default::default()
            };
        let graph = GraphProto {
            node: vec![
                node("Elu", &["x"], &["e"], vec![]),
                node(
                    "Elu",
                    &["x"],
                    &["e2"],
                    vec![AttributeProto {
                        name: "alpha".to_string(),
                        f: 0.5,
                        ..Default::default()
                    }],
                ),
                node("IsNaN", &["x"], &["n"], vec![]),
                node(
                    "ScatterElements",
                    &["x", "idx", "x"],
                    &["s"],
                    vec![AttributeProto {
                        name: "axis".to_string(),
                        i: 1,
                        ..Default::default()
                    }],
                ),
            ],
            initializer: vec![TensorProto {
                name: "idx".to_string(),
                data_type: DType::I64.to_onnx(),
                dims: vec![1],
                raw_data: 0i64.to_le_bytes().to_vec(),
                ..Default::default()
            }],
            input: vec![info("x")],
            output: ["e", "e2", "n", "s"].map(info).to_vec(),
            ..Default::default()
        };
        let code = Compiler::new().compile(&graph).unwrap().code;
        // A whole alpha is written as a float.
        assert!(code.contains("kernels::elu(&x, 1.0,"), "{code}");
        assert!(code.contains("kernels::elu(&x, 0.5,"), "{code}");
        assert!(code.contains("kernels::is_nan(&x,"), "{code}");
        assert!(
            code.contains("kernels::scatter_elements(&x,") && code.contains(", 1, "),
            "{code}"
        );
    }

    /// A 1D Conv fused with its Relu borrows its input, which later nodes may
    /// still read (it used to be moved), and keeps both of its pads.
    #[test]
    fn test_fused_conv1d_relu_borrows_its_input() {
        use crate::model::onnx_proto::{AttributeProto, GraphProto, TensorProto, ValueInfoProto};
        let info = |name: &str| ValueInfoProto {
            name: name.to_string(),
            ..Default::default()
        };
        let node = |op: &str, input: &[&str], output: &[&str], attribute: Vec<AttributeProto>| {
            NodeProto {
                op_type: op.to_string(),
                input: input.iter().map(|s| s.to_string()).collect(),
                output: output.iter().map(|s| s.to_string()).collect(),
                attribute,
                ..Default::default()
            }
        };
        let pads = AttributeProto {
            name: "pads".to_string(),
            ints: vec![0, 1],
            ..Default::default()
        };
        let graph = GraphProto {
            node: vec![
                node("Conv", &["x", "w"], &["c"], vec![pads]),
                node("Relu", &["c"], &["r"], vec![]),
                node("Mul", &["x", "x"], &["m"], vec![]),
            ],
            initializer: vec![TensorProto {
                name: "w".to_string(),
                data_type: DType::F32.to_onnx(),
                dims: vec![2, 2, 2],
                raw_data: [0.5f32; 8].iter().flat_map(|x| x.to_le_bytes()).collect(),
                ..Default::default()
            }],
            input: vec![info("x")],
            output: ["r", "m"].map(info).to_vec(),
            ..Default::default()
        };
        let code = Compiler::new()
            .with_default_optimizations()
            .compile(&graph)
            .unwrap()
            .code;
        let line = code.lines().find(|l| l.contains("let r = ")).unwrap_or_else(|| panic!("{code}"));
        assert!(line.contains("lele::kernels::conv1d_fused(&x, "), "{line}");
        assert!(line.contains("&[0, 1], &[1], true, "), "{line}");
    }

    /// An `If` branch that returns one of its initializers reads it through
    /// the weight accessor of its type; it used to call a nonexistent
    /// `self.weight`.
    #[test]
    fn test_if_branch_returning_a_weight() {
        use crate::model::onnx_proto::{AttributeProto, GraphProto, TensorProto, ValueInfoProto};
        let info = |name: &str| ValueInfoProto {
            name: name.to_string(),
            ..Default::default()
        };
        let branch = |name: &str, dtype: DType, raw: Vec<u8>| GraphProto {
            initializer: vec![TensorProto {
                name: name.to_string(),
                data_type: dtype.to_onnx(),
                dims: vec![2],
                raw_data: raw,
                ..Default::default()
            }],
            output: vec![info(name)],
            ..Default::default()
        };
        let graph = GraphProto {
            node: vec![NodeProto {
                op_type: "If".to_string(),
                input: vec!["cond".to_string()],
                output: vec!["r".to_string()],
                attribute: vec![
                    AttributeProto {
                        name: "then_branch".to_string(),
                        g: Some(branch(
                            "w_f32",
                            DType::F32,
                            [1.0f32, 2.0].map(f32::to_le_bytes).concat(),
                        )),
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "else_branch".to_string(),
                        g: Some(branch("w_i8", DType::I8, vec![3, 4])),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            input: vec![info("cond")],
            output: vec![info("r")],
            ..Default::default()
        };
        let code = Compiler::new().compile(&graph).unwrap().code;
        assert!(!code.contains("self.weight("), "{code}");
        assert!(code.contains("self.weight_f32("), "{code}");
        assert!(code.contains("self.weight_i8("), "{code}");
    }
}
