use super::super::{
    generate::{Nodes, OpContext, generate_nodes, lower_nodes, weight_expr},
    ir::{Graph, Op},
    sanitize_name,
};
use crate::model::onnx_proto::NodeProto;
use std::io::Write;

pub(crate) fn handle_control_flow_ops(
    ctx: &mut OpContext,
    w: &mut dyn Write,
) -> std::io::Result<bool> {
    let op = ctx.node.op_type.as_str();
    let tab = ctx.tab();
    let inputs = &ctx.inputs;
    let outputs = &ctx.outputs;

    match op {
        "If" => {
            let Op::If {
                then_branch,
                else_branch,
            } = &ctx.ir().op
            else {
                panic!("If node {} without both branches", ctx.node.name);
            };
            // A branch output is either a weight or a variable of the branch.
            let (known_weights, var_types) = (ctx.known_weights, ctx.var_types);
            let branch_outputs = |branch: &Graph| -> Vec<String> {
                branch
                    .outputs
                    .iter()
                    .map(|&o| {
                        let name = sanitize_name(&branch.value(o).name);
                        match known_weights.get(&name) {
                            Some((offset, len, shape, data_type)) => {
                                let ty = var_types.get(&name).map_or("f32", |s| s.as_str());
                                let weight = weight_expr(*offset, *len, shape, *data_type, ty);
                                format!("{weight}.to_owned()")
                            }
                            None => format!("{name}.to_owned()"),
                        }
                    })
                    .collect()
            };
            let cond = &inputs[0];
            let out_vars = outputs.join(", ");
            let cond_is_i64 = ctx.var_types.get(cond).map(|t| t == "i64").unwrap_or(false);
            if cond_is_i64 {
                writeln!(
                    w,
                    "{}let ({}) = if {}.data.first().is_some_and(|&v| v != 0) {{",
                    tab, out_vars, cond
                )?;
            } else {
                writeln!(
                    w,
                    "{}let ({}) = if {}.data.first().is_some_and(|&v| v != 0.0) {{",
                    tab, out_vars, cond
                )?;
            }

            for (i, branch) in [then_branch, else_branch].into_iter().enumerate() {
                if i == 1 {
                    writeln!(w, "{}}} else {{", tab)?;
                }
                let (ids, protos) = lower_nodes(branch);
                let protos: Vec<&NodeProto> = protos.iter().collect();
                let nodes = Nodes {
                    graph: branch,
                    ids: &ids,
                    protos: &protos,
                };
                generate_nodes(
                    nodes,
                    w,
                    ctx.indent + 1,
                    ctx.known_weights,
                    ctx.int64_map,
                    ctx.allocator,
                    None, // Subgraphs have their own scope
                    ctx.current_id,
                    ctx.compiler,
                    ctx.var_types,
                )?;
                writeln!(w, "{}    ({})", tab, branch_outputs(branch).join(", "))?;
            }
            writeln!(w, "{}}};", tab)?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}
