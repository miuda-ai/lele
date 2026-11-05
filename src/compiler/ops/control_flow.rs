//! Code generation for control flow.

use super::super::{
    generate::{ElemType, Nodes, OpContext, generate_nodes, lower_nodes},
    ir::{Graph, Op},
    sanitize_name,
};
use crate::model::onnx_proto::NodeProto;
use std::io::Write;

pub(super) fn generate(ctx: &mut OpContext, w: &mut dyn Write) -> std::io::Result<()> {
    let tab = ctx.tab();
    let inputs = &ctx.inputs;
    let outputs = &ctx.outputs;

    let Op::If {
        then_branch,
        else_branch,
    } = &ctx.ir().op
    else {
        unreachable!("{} is not a control flow operator", ctx.ir().op.op_type());
    };
    // A branch output is either a weight or a variable of the branch.
    let (weights, var_types) = (ctx.weights, ctx.var_types);
    let branch_outputs = |branch: &Graph| -> Vec<String> {
        branch
            .outputs
            .iter()
            .map(|&o| {
                let name = sanitize_name(&branch.value(o).name);
                match weights.get(&name) {
                    Some(weight) => {
                        let weight = weight.expr(var_types.get(&name));
                        format!("{weight}.to_owned()")
                    }
                    None => format!("{name}.to_owned()"),
                }
            })
            .collect()
    };
    let cond = &inputs[0];
    let out_vars = outputs.join(", ");
    if ctx.input_type(0) == ElemType::I64 {
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
            ctx.weights,
            ctx.allocator,
            None, // Subgraphs have their own scope
            ctx.current_id,
            ctx.compiler,
            ctx.var_types,
        )?;
        writeln!(w, "{}    ({})", tab, branch_outputs(branch).join(", "))?;
    }
    writeln!(w, "{}}};", tab)?;
    Ok(())
}
