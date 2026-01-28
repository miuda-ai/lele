use super::{sanitize_name, Allocator, AnalysisData, Compiler};
use crate::model::onnx_proto::{NodeProto, ValueInfoProto};
use crate::model::tensor_to_array;
use std::collections::{HashMap, HashSet};
use std::io::Write;

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

pub(crate) fn generate_nodes<W: Write>(
    nodes: &[&NodeProto],
    w: &mut W,
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
        // Override
        if let Some(handler) = compiler.overrides.get(op) {
            if !buf_expr.is_empty() && !buf_expr.starts_with("&mut ws.buf_") {
                writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, outputs[0])?;
            }
            handler(node, &inputs, &outputs, &buf_expr, w, indent)?;
            continue;
        }
        match op {
            "Constant" => {
                let name_raw = &node.output[0];
                if let Some((offset, len, shape)) = known_weights.get(&outputs[0]) {
                    writeln!(
                        w,
                        "{}let {} = self.weight({}, {}, &{:?});",
                        tab, outputs[0], offset, len, shape
                    )?;
                } else if let Some((ints, shape)) = int64_map.get(name_raw) {
                    let floats: Vec<String> =
                        ints.iter().map(|&x| format!("{:.1}", x as f32)).collect();
                    writeln!(
                        w,
                        "{}let {} = lele::tensor::TensorView::from_owned(vec![{}], vec!{:?});",
                        tab,
                        outputs[0],
                        floats.join(", "),
                        shape
                    )?;
                } else {
                    let val = node
                        .attribute
                        .iter()
                        .find(|a| a.name == "value")
                        .and_then(|a| a.t.as_ref());
                    if let Some(t) = val {
                        if !t.float_data.is_empty() {
                            if t.float_data.len() > 100 {
                                writeln!(
                                    w,
                                    "{}let {} = lele::tensor::TensorView::empty(); // Large",
                                    tab, outputs[0]
                                )?;
                            } else {
                                writeln!(w, "{}let {} = lele::tensor::TensorView::from_owned(vec!{:?}, vec!{:?});", tab, outputs[0], t.float_data, t.dims)?;
                            }
                        } else if !t.raw_data.is_empty() {
                            let floats: Vec<f32> = t
                                .raw_data
                                .chunks(4)
                                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                                .collect();
                            if floats.len() > 100 {
                                writeln!(
                                    w,
                                    "{}let {} = lele::tensor::TensorView::empty(); // Large",
                                    tab, outputs[0]
                                )?;
                            } else {
                                writeln!(w, "{}let {} = lele::tensor::TensorView::from_owned(vec!{:?}, vec!{:?});", tab, outputs[0], floats, t.dims)?;
                            }
                        } else {
                            writeln!(
                                w,
                                "{}let {} = lele::tensor::TensorView::empty();",
                                tab, outputs[0]
                            )?;
                        }
                    } else {
                        writeln!(
                            w,
                            "{}let {} = lele::tensor::TensorView::empty();",
                            tab, outputs[0]
                        )?;
                    }
                }
            }
            "Identity" => {
                if let Some((ints, shape)) = int64_map.get(&node.input[0]) {
                    let floats: Vec<String> =
                        ints.iter().map(|&x| format!("{:.1}", x as f32)).collect();
                    writeln!(
                        w,
                        "{}let {} = lele::tensor::TensorView::from_owned(vec![{}], vec!{:?});",
                        tab,
                        outputs[0],
                        floats.join(", "),
                        shape
                    )?;
                } else {
                    writeln!(w, "{}let {} = {}.clone();", tab, outputs[0], inputs[0])?;
                }
            }
            "If" => {
                let then_branch = node
                    .attribute
                    .iter()
                    .find(|a| a.name == "then_branch")
                    .unwrap()
                    .g
                    .as_ref()
                    .unwrap();
                let else_branch = node
                    .attribute
                    .iter()
                    .find(|a| a.name == "else_branch")
                    .unwrap()
                    .g
                    .as_ref()
                    .unwrap();
                let cond = &inputs[0];
                let out_vars = outputs.join(", ");
                writeln!(
                    w,
                    "{}let ({}) = if {}.data.get(0).map(|v| *v != 0.0).unwrap_or(false) {{",
                    tab, out_vars, cond
                )?;
                let then_nodes: Vec<&NodeProto> = then_branch.node.iter().collect();
                // Don't pass analysis data to subgraphs - they have their own scope and the parent's liveness info doesn't apply
                generate_nodes(
                    &then_nodes,
                    w,
                    indent + 1,
                    known_weights,
                    int64_map,
                    allocator,
                    None,
                    current_id,
                    compiler,
                )?;
                // Collect variables defined in then branch
                let mut then_defined = HashSet::new();
                for n in &then_nodes {
                    for out in &n.output {
                        if !out.is_empty() {
                            then_defined.insert(sanitize_name(out));
                        }
                    }
                }
                let then_outs: Vec<String> = then_branch
                    .output
                    .iter()
                    .map(|o| {
                        let name = sanitize_name(&o.name);
                        if let Some((offset, len, shape)) = known_weights.get(&name) {
                            format!("self.weight({}, {}, &{:?}).to_owned()", offset, len, shape)
                        } else if then_defined.contains(&name) {
                            format!("{}.to_owned()", name)
                        } else {
                            // Output not defined in branch - must be from outer scope or input
                            format!("{}.to_owned()", name)
                        }
                    })
                    .collect();
                writeln!(w, "{}    ({})", tab, then_outs.join(", "))?;
                writeln!(w, "{}}} else {{", tab)?;
                let else_nodes: Vec<&NodeProto> = else_branch.node.iter().collect();
                // Don't pass analysis data to subgraphs - they have their own scope and the parent's liveness info doesn't apply
                generate_nodes(
                    &else_nodes,
                    w,
                    indent + 1,
                    known_weights,
                    int64_map,
                    allocator,
                    None,
                    current_id,
                    compiler,
                )?;
                // Collect variables defined in else branch
                let mut else_defined = HashSet::new();
                for n in &else_nodes {
                    for out in &n.output {
                        if !out.is_empty() {
                            else_defined.insert(sanitize_name(out));
                        }
                    }
                }
                let else_outs: Vec<String> = else_branch
                    .output
                    .iter()
                    .map(|o| {
                        let name = sanitize_name(&o.name);
                        if let Some((offset, len, shape)) = known_weights.get(&name) {
                            format!("self.weight({}, {}, &{:?}).to_owned()", offset, len, shape)
                        } else if else_defined.contains(&name) {
                            format!("{}.to_owned()", name)
                        } else {
                            // Output not defined in branch - must be from outer scope or input
                            format!("{}.to_owned()", name)
                        }
                    })
                    .collect();
                writeln!(w, "{}    ({})", tab, else_outs.join(", "))?;
                writeln!(w, "{}}};", tab)?;
            }
            _ => {
                if !buf_expr.is_empty() && !buf_expr.starts_with("&mut ws.buf_") {
                    writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, outputs[0])?;
                }
                match op {
                    "Add" => writeln!(
                        w,
                        "{}let {} = lele::kernels::add(&{}, &{}, {});",
                        tab, outputs[0], inputs[0], inputs[1], buf_expr
                    )?,
                    "Sub" => writeln!(
                        w,
                        "{}let {} = lele::kernels::sub(&{}, &{}, {});",
                        tab, outputs[0], inputs[0], inputs[1], buf_expr
                    )?,
                    "Mul" => writeln!(
                        w,
                        "{}let {} = lele::kernels::mul(&{}, &{}, {});",
                        tab, outputs[0], inputs[0], inputs[1], buf_expr
                    )?,
                    "Div" => writeln!(
                        w,
                        "{}let {} = lele::kernels::div(&{}, &{}, {});",
                        tab, outputs[0], inputs[0], inputs[1], buf_expr
                    )?,
                    "MatMul" => writeln!(
                        w,
                        "{}let {} = lele::kernels::matmul(&{}, &{}, {});",
                        tab, outputs[0], inputs[0], inputs[1], buf_expr
                    )?,
                    "MatMulInteger" => {
                        let a_zp = if inputs.len() > 2 {
                            format!("Some(&{})", inputs[2])
                        } else {
                            "None".to_string()
                        };
                        let b_zp = if inputs.len() > 3 {
                            format!("Some(&{})", inputs[3])
                        } else {
                            "None".to_string()
                        };
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::mat_mul_integer(&{}, &{}, {}, {}, {});",
                            tab, outputs[0], inputs[0], inputs[1], a_zp, b_zp, buf_expr
                        )?;
                    }
                    "DynamicQuantizeLinear" => {
                        writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, outputs[1])?;
                        writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, outputs[2])?;
                        writeln!(w, "{}let ({}, {}, {}) = lele::kernels::dynamic_quantize_linear(&{}, {}, &mut buf_{}, &mut buf_{});", tab, outputs[0], outputs[1], outputs[2], inputs[0], buf_expr, outputs[1], outputs[2])?;
                    }
                    "Relu" => writeln!(
                        w,
                        "{}let {} = lele::kernels::relu(&{}, {});",
                        tab, outputs[0], inputs[0], buf_expr
                    )?,
                    "Sigmoid" => writeln!(
                        w,
                        "{}let {} = lele::kernels::sigmoid(&{}, {});",
                        tab, outputs[0], inputs[0], buf_expr
                    )?,
                    "Softmax" => {
                        let axis = node
                            .attribute
                            .iter()
                            .find(|a| a.name == "axis")
                            .map(|a| a.i)
                            .unwrap_or(-1);
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::softmax(&{}, {}, {});",
                            tab, outputs[0], inputs[0], axis, buf_expr
                        )?;
                    }
                    "LayerNormalization" => {
                        let epsilon = node
                            .attribute
                            .iter()
                            .find(|a| a.name == "epsilon")
                            .map(|a| a.f)
                            .unwrap_or(1e-5);
                        let axis = node
                            .attribute
                            .iter()
                            .find(|a| a.name == "axis")
                            .map(|a| a.i)
                            .unwrap_or(-1);
                        let scale = if node.input.len() > 1
                            && !node.input[1].is_empty()
                            && !inputs[1].is_empty()
                        {
                            format!("&{}", inputs[1])
                        } else {
                            "&lele::tensor::TensorView::empty()".to_string()
                        };
                        let bias = if node.input.len() > 2
                            && !node.input[2].is_empty()
                            && !inputs[2].is_empty()
                        {
                            format!("&{}", inputs[2])
                        } else {
                            "&lele::tensor::TensorView::empty()".to_string()
                        };
                        if outputs.len() > 1 {
                            let fillers = vec!["_"; outputs.len() - 1].join(", ");
                            let dummy_tensors =
                                vec!["lele::tensor::TensorView::empty()"; outputs.len() - 1]
                                    .join(", ");
                            writeln!(w, "{}let ({}, {}) = (lele::kernels::layer_norm(&{}, {}, {}, {}, {}, {}), {});", 
                                    tab, outputs[0], fillers, inputs[0], scale, bias, axis, epsilon, buf_expr, dummy_tensors)?;
                        } else {
                            writeln!(
                                w,
                                "{}let {} = lele::kernels::layer_norm(&{}, {}, {}, {}, {}, {});",
                                tab, outputs[0], inputs[0], scale, bias, axis, epsilon, buf_expr
                            )?;
                        }
                    }
                    "Transpose" => {
                        let perm = node
                            .attribute
                            .iter()
                            .find(|a| a.name == "perm")
                            .map(|a| a.ints.clone())
                            .unwrap_or(vec![]);
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::transpose(&{}, &{:?}, {});",
                            tab, outputs[0], inputs[0], perm, buf_expr
                        )?;
                    }
                    "Reshape" => {
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::reshape(&{}, &{});",
                            tab, outputs[0], inputs[0], inputs[1]
                        )?;
                    }
                    "Unsqueeze" => {
                        let axes = if node.input.len() > 1
                            && !node.input[1].is_empty()
                            && !inputs[1].is_empty()
                        {
                            format!("&lele::kernels::to_i64_vec(&{})", inputs[1])
                        } else {
                            let axes_attr = node
                                .attribute
                                .iter()
                                .find(|a| a.name == "axes")
                                .map(|a| a.ints.clone())
                                .unwrap_or(vec![]);
                            format!("&{:?}", axes_attr)
                        };
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::unsqueeze(&{}, {});",
                            tab, outputs[0], inputs[0], axes
                        )?;
                    }
                    "Squeeze" => {
                        let axes = if node.input.len() > 1
                            && !node.input[1].is_empty()
                            && !inputs[1].is_empty()
                        {
                            format!("Some(&lele::kernels::to_i64_vec(&{}))", inputs[1])
                        } else {
                            let axes_attr = node
                                .attribute
                                .iter()
                                .find(|a| a.name == "axes")
                                .map(|a| a.ints.clone());
                            if let Some(a) = axes_attr {
                                format!("Some(&{:?})", a)
                            } else {
                                "None".to_string()
                            }
                        };
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::squeeze(&{}, {});",
                            tab, outputs[0], inputs[0], axes
                        )?;
                    }
                    "Concat" => {
                        let axis = node
                            .attribute
                            .iter()
                            .find(|a| a.name == "axis")
                            .map(|a| a.i)
                            .unwrap_or(0);
                        let args = inputs
                            .iter()
                            .map(|s| format!("&{}", s))
                            .collect::<Vec<_>>()
                            .join(", ");
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::concat(&[{}], {}, {});",
                            tab, outputs[0], args, axis, buf_expr
                        )?;
                    }
                    "Gather" => {
                        let axis = node
                            .attribute
                            .iter()
                            .find(|a| a.name == "axis")
                            .map(|a| a.i)
                            .unwrap_or(0);
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::gather(&{}, &{}, {}, {});",
                            tab, outputs[0], inputs[0], inputs[1], axis, buf_expr
                        )?;
                    }
                    "Shape" => {
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::shape(&{});",
                            tab, outputs[0], inputs[0]
                        )?;
                    }
                    "Cast" => {
                        writeln!(
                            w,
                            "{}let {} = {}.clone(); // Cast ignored",
                            tab, outputs[0], inputs[0]
                        )?;
                    }
                    "ReduceMean" => {
                        let axes = node
                            .attribute
                            .iter()
                            .find(|a| a.name == "axes")
                            .map(|a| a.ints.clone())
                            .unwrap_or(vec![]);
                        let keepdims = node
                            .attribute
                            .iter()
                            .find(|a| a.name == "keepdims")
                            .map(|a| a.i)
                            .unwrap_or(1)
                            != 0;
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::reduce_mean(&{}, &{:?}, {}, {});",
                            tab, outputs[0], inputs[0], axes, keepdims, buf_expr
                        )?;
                    }
                    "Equal" => writeln!(
                        w,
                        "{}let {} = lele::kernels::equal(&{}, &{}, {});",
                        tab, outputs[0], inputs[0], inputs[1], buf_expr
                    )?,
                    "Not" => writeln!(
                        w,
                        "{}let {} = lele::kernels::not(&{}, {});",
                        tab, outputs[0], inputs[0], buf_expr
                    )?,
                    "ConstantOfShape" => {
                        let val = node
                            .attribute
                            .iter()
                            .find(|a| a.name == "value")
                            .and_then(|a| a.t.as_ref())
                            .map(|t| {
                                if let Ok((data, _)) = tensor_to_array(t) {
                                    if !data.is_empty() {
                                        return data[0];
                                    }
                                }
                                0.0
                            })
                            .unwrap_or(0.0);
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::constant_of_shape(&{}, {:.1}, {});",
                            tab, outputs[0], inputs[0], val, buf_expr
                        )?;
                    }
                    "Slice" => {
                        let axes = if node.input.len() > 3 {
                            format!("&lele::kernels::to_i64_vec(&{})", inputs[3])
                        } else {
                            "&[]".to_string()
                        };
                        let steps = if node.input.len() > 4 {
                            format!("&lele::kernels::to_i64_vec(&{})", inputs[4])
                        } else {
                            "&[]".to_string()
                        };
                        writeln!(w, "{}let {} = lele::kernels::slice(&{}, &lele::kernels::to_i64_vec(&{}), &lele::kernels::to_i64_vec(&{}), {}, {}, {});", 
                            tab, outputs[0], inputs[0], inputs[1], inputs[2], axes, steps, buf_expr)?;
                    }
                    "ArgMax" => {
                        let axis = node
                            .attribute
                            .iter()
                            .find(|a| a.name == "axis")
                            .map(|a| a.i)
                            .unwrap_or(0);
                        let keepdims = node
                            .attribute
                            .iter()
                            .find(|a| a.name == "keepdims")
                            .map(|a| a.i)
                            .unwrap_or(1);
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::argmax(&{}, {}, {}, {});",
                            tab, outputs[0], inputs[0], axis, keepdims, buf_expr
                        )?;
                    }
                    "Conv" => {
                        let dilations = node
                            .attribute
                            .iter()
                            .find(|a| a.name == "dilations")
                            .map(|a| a.ints.clone())
                            .unwrap_or(vec![]);
                        let group = node
                            .attribute
                            .iter()
                            .find(|a| a.name == "group")
                            .map(|a| a.i)
                            .unwrap_or(1);
                        let pads = node
                            .attribute
                            .iter()
                            .find(|a| a.name == "pads")
                            .map(|a| a.ints.clone())
                            .unwrap_or(vec![]);
                        let strides = node
                            .attribute
                            .iter()
                            .find(|a| a.name == "strides")
                            .map(|a| a.ints.clone())
                            .unwrap_or(vec![]);
                        let bias_arg = if inputs.len() > 2 {
                            format!("Some(&{})", inputs[2])
                        } else {
                            "None".to_string()
                        };
                        writeln!(w, "{}let {} = lele::kernels::conv1d(&{}, &{}, {}, &{:?}, {}, &{:?}, &{:?}, {});", 
                                tab, outputs[0], inputs[0], inputs[1], bias_arg, dilations, group, pads, strides, buf_expr)?;
                    }
                    "LSTM" => {
                        let bias = if inputs.len() > 3 && !node.input[3].is_empty() {
                            format!("Some(&{})", inputs[3])
                        } else {
                            "None".to_string()
                        };
                        let seq_lens = if inputs.len() > 4 && !node.input[4].is_empty() {
                            format!("Some(&{})", inputs[4])
                        } else {
                            "None".to_string()
                        };
                        let initial_h = if inputs.len() > 5 && !node.input[5].is_empty() {
                            format!("Some(&{})", inputs[5])
                        } else {
                            "None".to_string()
                        };
                        let initial_c = if inputs.len() > 6 && !node.input[6].is_empty() {
                            format!("Some(&{})", inputs[6])
                        } else {
                            "None".to_string()
                        };
                        writeln!(
                            w,
                            "{}let mut buf_{}_h = Vec::<f32>::new();",
                            tab, outputs[0]
                        )?;
                        writeln!(
                            w,
                            "{}let mut buf_{}_c = Vec::<f32>::new();",
                            tab, outputs[0]
                        )?;
                        writeln!(w, "{}let ({}, {}, {}) = lele::kernels::lstm(&{}, &{}, &{}, {}, {}, {}, {}, {}, &mut buf_{}_h, &mut buf_{}_c);", 
                                tab, outputs[0], outputs[1], outputs[2], inputs[0], inputs[1], inputs[2], bias, seq_lens, initial_h, initial_c, buf_expr, outputs[0], outputs[0])?;
                    }
                    "Size" => {
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::size(&{});",
                            tab, outputs[0], inputs[0]
                        )?;
                    }
                    "Pad" => {
                        let mode = node
                            .attribute
                            .iter()
                            .find(|a| a.name == "mode")
                            .map(|a| String::from_utf8_lossy(&a.s))
                            .unwrap_or("constant".into());
                        let constant_value = if inputs.len() > 2
                            && !node.input[2].is_empty()
                            && !inputs[2].is_empty()
                        {
                            format!("Some(&{})", inputs[2])
                        } else {
                            "None".to_string()
                        };
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::pad(&{}, &{}, {}, {:?}, {});",
                            tab, outputs[0], inputs[0], inputs[1], constant_value, mode, buf_expr
                        )?;
                    }
                    "Pow" => {
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::pow(&{}, &{}, {});",
                            tab, outputs[0], inputs[0], inputs[1], buf_expr
                        )?;
                    }
                    "Sqrt" => {
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::sqrt(&{}, {});",
                            tab, outputs[0], inputs[0], buf_expr
                        )?;
                    }
                    "Split" => {
                        // Split can have 'split' as attribute or input
                        let axis = node
                            .attribute
                            .iter()
                            .find(|a| a.name == "axis")
                            .map(|a| a.i)
                            .unwrap_or(0);
                        let splits = if node.input.len() > 1 && !node.input[1].is_empty() {
                            format!("lele::kernels::to_i64_vec(&{})", inputs[1])
                        } else {
                            let split_attr = node
                                .attribute
                                .iter()
                                .find(|a| a.name == "split")
                                .map(|a| a.ints.clone())
                                .unwrap_or_else(|| {
                                    // If no split attribute, divide evenly
                                    let num_outputs = outputs.len();
                                    vec![0; num_outputs] // Placeholder, needs runtime shape info
                                });
                            format!("vec!{:?}", split_attr)
                        };
                        // Allocate buffers for each output
                        for out_name in &outputs {
                            writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, out_name)?;
                        }
                        // Create array binding first
                        let buf_names: Vec<String> =
                            outputs.iter().map(|n| format!("buf_{}", n)).collect();
                        writeln!(
                            w,
                            "{}let mut split_buffers = [{}];",
                            tab,
                            buf_names.join(", ")
                        )?;
                        writeln!(w, "{}let splits_vec = {};", tab, splits)?;
                        writeln!(w, "{}let split_results = lele::kernels::split(&{}, {}, &splits_vec, &mut split_buffers);", tab, inputs[0], axis)?;
                        // Assign results
                        for (i, out_name) in outputs.iter().enumerate() {
                            writeln!(w, "{}let {} = split_results[{}].clone();", tab, out_name, i)?;
                        }
                    }
                    "Where" => {
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::where_op(&{}, &{}, &{}, {});",
                            tab, outputs[0], inputs[0], inputs[1], inputs[2], buf_expr
                        )?;
                    }
                    "Range" => {
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::range(&{}, &{}, &{}, {});",
                            tab, outputs[0], inputs[0], inputs[1], inputs[2], buf_expr
                        )?;
                    }
                    "Sin" => {
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::sin(&{}, {});",
                            tab, outputs[0], inputs[0], buf_expr
                        )?;
                    }
                    "Cos" => {
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::cos(&{}, {});",
                            tab, outputs[0], inputs[0], buf_expr
                        )?;
                    }
                    "Exp" => {
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::exp(&{}, {});",
                            tab, outputs[0], inputs[0], buf_expr
                        )?;
                    }
                    "Neg" => {
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::neg(&{}, {});",
                            tab, outputs[0], inputs[0], buf_expr
                        )?;
                    }
                    "Less" => {
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::less(&{}, &{}, {});",
                            tab, outputs[0], inputs[0], inputs[1], buf_expr
                        )?;
                    }
                    "Expand" => {
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::expand(&{}, &{}, {});",
                            tab, outputs[0], inputs[0], inputs[1], buf_expr
                        )?;
                    }
                    "Tile" => {
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::tile(&{}, &{}, {});",
                            tab, outputs[0], inputs[0], inputs[1], buf_expr
                        )?;
                    }
                    "Reciprocal" => writeln!(
                        w,
                        "{}let {} = lele::kernels::reciprocal(&{}, {});",
                        tab, outputs[0], inputs[0], buf_expr
                    )?,
                    "Erf" => writeln!(
                        w,
                        "{}let {} = lele::kernels::erf(&{}, {});",
                        tab, outputs[0], inputs[0], buf_expr
                    )?,
                    "Softplus" => writeln!(
                        w,
                        "{}let {} = lele::kernels::softplus(&{}, {});",
                        tab, outputs[0], inputs[0], buf_expr
                    )?,
                    "Tanh" => writeln!(
                        w,
                        "{}let {} = lele::kernels::tanh_kernel(&{}, {});",
                        tab, outputs[0], inputs[0], buf_expr
                    )?,
                    "ReduceSum" => {
                        let axes = if node.input.len() > 1
                            && !node.input[1].is_empty()
                            && !inputs[1].is_empty()
                        {
                            format!("&lele::kernels::to_i64_vec(&{})", inputs[1])
                        } else {
                            let axes_attr = node
                                .attribute
                                .iter()
                                .find(|a| a.name == "axes")
                                .map(|a| a.ints.clone())
                                .unwrap_or(vec![]);
                            format!("&{:?}", axes_attr)
                        };
                        let keepdims = node
                            .attribute
                            .iter()
                            .find(|a| a.name == "keepdims")
                            .map(|a| a.i)
                            .unwrap_or(1)
                            != 0;
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::reduce_sum(&{}, {}, {}, {});",
                            tab, outputs[0], inputs[0], axes, keepdims, buf_expr
                        )?;
                    }
                    "Clip" => {
                        let min = if inputs.len() > 1 && !inputs[1].is_empty() {
                            format!("Some(&{})", inputs[1])
                        } else {
                            "None".to_string()
                        };
                        let max = if inputs.len() > 2 && !inputs[2].is_empty() {
                            format!("Some(&{})", inputs[2])
                        } else {
                            "None".to_string()
                        };
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::clip(&{}, {}, {}, {});",
                            tab, outputs[0], inputs[0], min, max, buf_expr
                        )?;
                    }
                    "PRelu" => {
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::prelu(&{}, &{}, {});",
                            tab, outputs[0], inputs[0], inputs[1], buf_expr
                        )?;
                    }
                    "BatchNormalization" => {
                        let epsilon = node
                            .attribute
                            .iter()
                            .find(|a| a.name == "epsilon")
                            .map(|a| a.f)
                            .unwrap_or(1e-5);
                        writeln!(w, "{}let {} = lele::kernels::batch_norm(&{}, &{}, &{}, &{}, &{}, {:?}, {});", 
                            tab, outputs[0], inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon, buf_expr)?;
                    }
                    "Gemm" => {
                        let alpha = node
                            .attribute
                            .iter()
                            .find(|a| a.name == "alpha")
                            .map(|a| a.f)
                            .unwrap_or(1.0);
                        let beta = node
                            .attribute
                            .iter()
                            .find(|a| a.name == "beta")
                            .map(|a| a.f)
                            .unwrap_or(1.0);
                        let trans_a = node
                            .attribute
                            .iter()
                            .find(|a| a.name == "transA")
                            .map(|a| a.i)
                            .unwrap_or(0)
                            != 0;
                        let trans_b = node
                            .attribute
                            .iter()
                            .find(|a| a.name == "transB")
                            .map(|a| a.i)
                            .unwrap_or(0)
                            != 0;
                        let c = if inputs.len() > 2 && !inputs[2].is_empty() {
                            format!("Some(&{})", inputs[2])
                        } else {
                            "None".to_string()
                        };
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::gemm(&{}, &{}, {}, {:?}, {:?}, {}, {}, {});",
                            tab,
                            outputs[0],
                            inputs[0],
                            inputs[1],
                            c,
                            alpha,
                            beta,
                            trans_a,
                            trans_b,
                            buf_expr
                        )?;
                    }
                    _ => {
                        for (idx, out_name) in outputs.iter().enumerate() {
                            writeln!(w, "{}let {} = lele::tensor::TensorView::empty(); // Unimplemented {} out {}", tab, out_name, op, idx)?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
