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

    /// The values of input `i`, if it is an int64 or int32 weight.
    pub fn ints(&self, i: usize) -> Option<&'a [i64]> {
        self.weights.ints(self.node.input.get(i)?)
    }
}

/// Element type of the `TensorView` holding a value in generated code.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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
                let f32_kernel = matches!(
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
        // Return — detach_or_own (zero-copy) for f32 views of the workspace or the weights,
        // which outlive the chunk; to_owned for everything else, i64 and views of buffers
        // local to the chunk included, which are freed when it returns.
        let body = String::from_utf8_lossy(&f).into_owned();
        let ret_vals: Vec<String> = chunk_outputs
            .iter()
            .map(|s| {
                if var_types.get(s) == ElemType::F32 && outlives_chunk(&body, s, 0) {
                    format!("unsafe {{ {}.detach_or_own() }}", s)
                } else {
                    format!("{}.to_owned()", s)
                }
            })
            .collect();
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
/// view (`reshape`, `unsqueeze`, `squeeze`, `.clone()`) qualifies if what it views does.
/// Every definition counts (there is one per target under `cfg`). Anything else is
/// returned as a copy.
fn outlives_chunk(body: &str, var: &str, depth: usize) -> bool {
    if depth > 64 {
        return false;
    }
    // Multi-output kernels (`let (a, b) = topk(..)`, `split`, ...) write
    // buffers local to the chunk.
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
    defs.iter().all(|rhs| {
        let rhs = rhs.split(';').next().unwrap_or("");
        if rhs.contains("&mut ws.buf_") || rhs.starts_with("self.weight_") {
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
                    if is_i64 {
                        format!("&mut buf_{}", outputs[0])
                    } else {
                        format!("&mut ws.buf_{}", idx)
                    }
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
        let weights = Weights::collect(&ir);
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
