use super::super::{generate::{OpContext, generate_nodes}, sanitize_name};
use std::collections::HashSet;
use std::io::Write;

pub(crate) fn handle_control_flow_ops(ctx: &mut OpContext, w: &mut dyn Write) -> std::io::Result<bool> {
    let op = ctx.node.op_type.as_str();
    let tab = ctx.tab();
    let inputs = &ctx.inputs;
    let outputs = &ctx.outputs;

    match op {
        "If" => {
            let then_branch = ctx.node.attribute.iter().find(|a| a.name == "then_branch").unwrap().g.as_ref().unwrap();
            let else_branch = ctx.node.attribute.iter().find(|a| a.name == "else_branch").unwrap().g.as_ref().unwrap();
            let cond = &inputs[0];
            let out_vars = outputs.join(", ");
            writeln!(w, "{}let ({}) = if {}.data.get(0).map(|v| *v != 0.0).unwrap_or(false) {{", tab, out_vars, cond)?;
            
            let then_nodes: Vec<&crate::model::onnx_proto::NodeProto> = then_branch.node.iter().collect();
            generate_nodes(
                &then_nodes,
                w,
                ctx.indent + 1,
                ctx.known_weights,
                ctx.int64_map,
                ctx.allocator,
                None, // Subgraphs have their own scope
                ctx.current_id,
                ctx.compiler,
            )?;

            // Collect variables defined in then branch
            let mut then_defined = HashSet::new();
            for n in &then_nodes {
                for out in &n.output {
                    if !out.is_empty() { then_defined.insert(sanitize_name(out)); }
                }
            }
            let then_outs: Vec<String> = then_branch.output.iter().map(|o| {
                let name = sanitize_name(&o.name);
                if let Some((offset, len, shape)) = ctx.known_weights.get(&name) {
                    format!("self.weight({}, {}, &{:?}).to_owned()", offset, len, shape)
                } else if then_defined.contains(&name) {
                    format!("{}.to_owned()", name)
                } else {
                    format!("{}.to_owned()", name)
                }
            }).collect();
            writeln!(w, "{}    ({})", tab, then_outs.join(", "))?;
            writeln!(w, "{}}} else {{", tab)?;

            let else_nodes: Vec<&crate::model::onnx_proto::NodeProto> = else_branch.node.iter().collect();
            generate_nodes(
                &else_nodes,
                w,
                ctx.indent + 1,
                ctx.known_weights,
                ctx.int64_map,
                ctx.allocator,
                None,
                ctx.current_id,
                ctx.compiler,
            )?;

            let mut else_defined = HashSet::new();
            for n in &else_nodes {
                for out in &n.output {
                    if !out.is_empty() { else_defined.insert(sanitize_name(out)); }
                }
            }
            let else_outs: Vec<String> = else_branch.output.iter().map(|o| {
                let name = sanitize_name(&o.name);
                if let Some((offset, len, shape)) = ctx.known_weights.get(&name) {
                    format!("self.weight({}, {}, &{:?}).to_owned()", offset, len, shape)
                } else if else_defined.contains(&name) {
                    format!("{}.to_owned()", name)
                } else {
                    format!("{}.to_owned()", name)
                }
            }).collect();
            writeln!(w, "{}    ({})", tab, else_outs.join(", "))?;
            writeln!(w, "{}}};", tab)?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}
