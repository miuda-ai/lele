use super::super::generate::OpContext;
use std::io::Write;

pub(crate) fn handle_activation_ops(ctx: &mut OpContext, w: &mut dyn Write) -> std::io::Result<bool> {
    let op = ctx.node.op_type.as_str();
    let tab = ctx.tab();
    let inputs = &ctx.inputs;
    let outputs = &ctx.outputs;
    let buf_expr = &ctx.buf_expr;

    match op {
        "Relu" => writeln!(w, "{}let {} = lele::kernels::relu(&{}, {});", tab, outputs[0], inputs[0], buf_expr)?,
        "Sigmoid" => writeln!(w, "{}let {} = lele::kernels::sigmoid(&{}, {});", tab, outputs[0], inputs[0], buf_expr)?,
        "Softmax" => {
            let axis = ctx.node.attribute.iter().find(|a| a.name == "axis").map(|a| a.i).unwrap_or(-1);
            writeln!(w, "{}let {} = lele::kernels::softmax(&{}, {}, {});", tab, outputs[0], inputs[0], axis, buf_expr)?;
        }
        "ArgMax" => {
            let axis = ctx.node.attribute.iter().find(|a| a.name == "axis").map(|a| a.i).unwrap_or(0);
            let keepdims = ctx.node.attribute.iter().find(|a| a.name == "keepdims").map(|a| a.i).unwrap_or(1);
            writeln!(w, "{}let {} = lele::kernels::argmax(&{}, {}, {}, {});", tab, outputs[0], inputs[0], axis, keepdims, buf_expr)?;
        }
        "Tanh" => writeln!(w, "{}let {} = lele::kernels::tanh_kernel(&{}, {});", tab, outputs[0], inputs[0], buf_expr)?,
        _ => return Ok(false),
    }
    Ok(true)
}
