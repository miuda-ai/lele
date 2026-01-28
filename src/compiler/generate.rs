use super::{sanitize_name, Allocator, AnalysisData, Compiler};
use crate::model::onnx_proto::{NodeProto, ValueInfoProto};
use std::collections::{HashMap, HashSet};
use std::io::Write;

pub(crate) struct OpContext<'a> {
    pub node: &'a NodeProto,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub buf_expr: String,
    pub indent: usize,
    pub known_weights: &'a HashMap<String, (usize, usize, Vec<usize>)>,
    pub int64_map: &'a HashMap<String, (Vec<i64>, Vec<usize>)>,
    #[allow(dead_code)]
    pub allocator: Option<&'a Allocator>,
    #[allow(dead_code)]
    pub analysis: Option<&'a AnalysisData>,
    pub current_id: &'a mut usize,
    pub compiler: &'a Compiler,
}

impl<'a> OpContext<'a> {
    pub fn tab(&self) -> String {
        "    ".repeat(self.indent)
    }
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

pub(crate) fn generate_partitioned_graph<W: Write>(
    nodes: &[&NodeProto],           // Top level nodes
    chunk_writer: &mut Vec<String>, // Function definitions
    body_writer: &mut W,            // Forward body calls
    indent: usize,
    known_weights: &HashMap<String, (usize, usize, Vec<usize>)>,
    int64_map: &HashMap<String, (Vec<i64>, Vec<usize>)>,
    allocator: Option<&Allocator>,
    analysis: Option<&AnalysisData>,
    current_id: &mut usize,
    compiler: &Compiler,
    graph_inputs: &[ValueInfoProto],
    graph_outputs: &[ValueInfoProto],
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
        let chunk_nodes = &nodes[i..end];
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
            if name.is_empty() {
                continue;
            }
            if known_weights.contains_key(&name) {
                continue;
            }
            if int64_map.contains_key(inp) {
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
                // Filter weights
                if known_weights.contains_key(&sanitized) {
                    continue;
                }
                if int64_map.contains_key(&raw) {
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
            .map(|n| format!("{}: TensorView<'w>", n))
            .collect();
        let ret_sig: Vec<String> = chunk_outputs
            .iter()
            .map(|_| "TensorView<'static>".to_string())
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
            chunk_nodes,
            &mut f,
            2,
            known_weights,
            int64_map,
            allocator,
            analysis,
            current_id,
            compiler,
        )?;
        // Return
        let ret_vals: Vec<String> = chunk_outputs
            .iter()
            .map(|s| format!("{}.to_owned()", s))
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
                "{}self.{}(&mut ws, {});",
                tab,
                func_name,
                call_args.join(", ")
            )?;
        } else {
            let out_vars = chunk_outputs.join(", ");
            writeln!(
                body_writer,
                "{}let ({}) = self.{}(&mut ws, {});",
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

pub(crate) fn generate_nodes(
    nodes: &[&NodeProto],
    w: &mut dyn Write,
    indent: usize,
    known_weights: &HashMap<String, (usize, usize, Vec<usize>)>,
    int64_map: &HashMap<String, (Vec<i64>, Vec<usize>)>,
    allocator: Option<&Allocator>,
    analysis: Option<&AnalysisData>,
    current_id: &mut usize,
    compiler: &Compiler,
) -> std::io::Result<()> {
    let tab = "    ".repeat(indent);
    let mut idx = 0;
    while idx < nodes.len() {
        // Try to match patterns first
        let mut matched_len = 0;
        let remaining_nodes = &nodes[idx..];
        for pattern in &compiler.patterns {
            if let Some(len) = (pattern.matcher)(remaining_nodes) {
                (pattern.generator)(remaining_nodes, known_weights, allocator, w, indent)?;
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
        idx += 1;
        *current_id += 1;
        let op = node.op_type.as_str();
        let outputs: Vec<String> = node.output.iter().map(|s| sanitize_name(s)).collect();
        // Skip Code Generation for Constants that are already in weights
        if op == "Constant" && !outputs.is_empty() && known_weights.contains_key(&outputs[0]) {
            continue;
        }
        let inputs: Vec<String> = node
            .input
            .iter()
            .map(|s| {
                let name = sanitize_name(s);
                if let Some((offset, len, shape)) = known_weights.get(&name) {
                    format!("self.weight({}, {}, &{:?})", offset, len, shape)
                } else {
                    name
                }
            })
            .collect();
        // Unused check
        let is_unused = if let Some(an) = analysis {
            if op == "If" || op == "Loop" {
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
        let mut ctx = OpContext {
            node,
            inputs: inputs.clone(),
            outputs: outputs.clone(),
            buf_expr,
            indent,
            known_weights,
            int64_map,
            allocator,
            analysis,
            current_id,
            compiler,
        };

        // 1. Override
        if let Some(handler) = compiler.overrides.get(op) {
            if !ctx.buf_expr.is_empty() && !ctx.buf_expr.starts_with("&mut ws.buf_") {
                writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, outputs[0])?;
            }
            handler(node, &inputs, &outputs, &ctx.buf_expr, w, indent)?;
            continue;
        }

        // 2. Built-in
        if !ctx.buf_expr.is_empty()
            && !ctx.buf_expr.starts_with("&mut ws.buf_")
            && op != "DynamicQuantizeLinear"
            && op != "LSTM"
            && op != "Split"
        {
            writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, outputs[0])?;
        }

        if super::ops::dispatch_builtin(&mut ctx, w)? {
            continue;
        }

        // 3. Fallback
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
    use std::collections::HashMap;

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

        let mut output = Vec::new();
        let mut current_id = 0;
        let known_weights = HashMap::new();
        let int64_map = HashMap::new();

        generate_nodes(
            &[&node],
            &mut output,
            0,
            &known_weights,
            &int64_map,
            None,
            None,
            &mut current_id,
            &compiler,
        )
        .unwrap();

        let result = String::from_utf8(output).unwrap();
        assert!(result.contains("// Custom implementation for output1"));
        assert!(result.contains("let output1 = custom_kernel(&input1);"));
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

        let mut output = Vec::new();
        let mut current_id = 0;
        let known_weights = HashMap::new();
        let int64_map = HashMap::new();

        generate_nodes(
            &[&node],
            &mut output,
            0,
            &known_weights,
            &int64_map,
            None,
            None,
            &mut current_id,
            &compiler,
        )
        .unwrap();

        let result = String::from_utf8(output).unwrap();
        assert!(result.contains("let mut buf_c = Vec::<f32>::new();"));
        assert!(result.contains("let c = lele::kernels::add(&a, &b, &mut buf_c);"));
    }
}
