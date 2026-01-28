use super::super::generate::OpContext;
use std::io::Write;

pub(crate) fn handle_math_ops(ctx: &mut OpContext, w: &mut dyn Write) -> std::io::Result<bool> {
    let op = ctx.node.op_type.as_str();
    let tab = ctx.tab();
    let inputs = &ctx.inputs;
    let outputs = &ctx.outputs;
    let buf_expr = &ctx.buf_expr;

    match op {
        "Add" => writeln!(w, "{}let {} = lele::kernels::add(&{}, &{}, {});", tab, outputs[0], inputs[0], inputs[1], buf_expr)?,
        "Sub" => writeln!(w, "{}let {} = lele::kernels::sub(&{}, &{}, {});", tab, outputs[0], inputs[0], inputs[1], buf_expr)?,
        "Mul" => writeln!(w, "{}let {} = lele::kernels::mul(&{}, &{}, {});", tab, outputs[0], inputs[0], inputs[1], buf_expr)?,
        "Div" => writeln!(w, "{}let {} = lele::kernels::div(&{}, &{}, {});", tab, outputs[0], inputs[0], inputs[1], buf_expr)?,
        "MatMul" => writeln!(w, "{}let {} = lele::kernels::matmul(&{}, &{}, {});", tab, outputs[0], inputs[0], inputs[1], buf_expr)?,
        "MatMulInteger" => {
            let a_zp = if inputs.len() > 2 { format!("Some(&{})", inputs[2]) } else { "None".to_string() };
            let b_zp = if inputs.len() > 3 { format!("Some(&{})", inputs[3]) } else { "None".to_string() };
            writeln!(w, "{}let {} = lele::kernels::mat_mul_integer(&{}, &{}, {}, {}, {});", tab, outputs[0], inputs[0], inputs[1], a_zp, b_zp, buf_expr)?;
        }
        "Pow" => writeln!(w, "{}let {} = lele::kernels::pow(&{}, &{}, {});", tab, outputs[0], inputs[0], inputs[1], buf_expr)?,
        "Sqrt" => writeln!(w, "{}let {} = lele::kernels::sqrt(&{}, {});", tab, outputs[0], inputs[0], buf_expr)?,
        "Neg" => writeln!(w, "{}let {} = lele::kernels::neg(&{}, {});", tab, outputs[0], inputs[0], buf_expr)?,
        "Reciprocal" => writeln!(w, "{}let {} = lele::kernels::reciprocal(&{}, {});", tab, outputs[0], inputs[0], buf_expr)?,
        "Erf" => writeln!(w, "{}let {} = lele::kernels::erf(&{}, {});", tab, outputs[0], inputs[0], buf_expr)?,
        "Softplus" => writeln!(w, "{}let {} = lele::kernels::softplus(&{}, {});", tab, outputs[0], inputs[0], buf_expr)?,
        "Exp" => writeln!(w, "{}let {} = lele::kernels::exp(&{}, {});", tab, outputs[0], inputs[0], buf_expr)?,
        "Sin" => writeln!(w, "{}let {} = lele::kernels::sin(&{}, {});", tab, outputs[0], inputs[0], buf_expr)?,
        "Cos" => writeln!(w, "{}let {} = lele::kernels::cos(&{}, {});", tab, outputs[0], inputs[0], buf_expr)?,
        "Equal" => writeln!(w, "{}let {} = lele::kernels::equal(&{}, &{}, {});", tab, outputs[0], inputs[0], inputs[1], buf_expr)?,
        "Less" => writeln!(w, "{}let {} = lele::kernels::less(&{}, &{}, {});", tab, outputs[0], inputs[0], inputs[1], buf_expr)?,
        "Not" => writeln!(w, "{}let {} = lele::kernels::not(&{}, {});", tab, outputs[0], inputs[0], buf_expr)?,
        "PRelu" => writeln!(w, "{}let {} = lele::kernels::prelu(&{}, &{}, {});", tab, outputs[0], inputs[0], inputs[1], buf_expr)?,
        "ReduceSum" => {
            let axes = if inputs.len() > 1 && !ctx.node.input[1].is_empty() && !inputs[1].is_empty() {
                format!("&lele::kernels::to_i64_vec(&{})", inputs[1])
            } else {
                let axes_attr = ctx.node.attribute.iter().find(|a| a.name == "axes").map(|a| a.ints.clone()).unwrap_or(vec![]);
                format!("&{:?}", axes_attr)
            };
            let keepdims = ctx.node.attribute.iter().find(|a| a.name == "keepdims").map(|a| a.i).unwrap_or(1) != 0;
            writeln!(w, "{}let {} = lele::kernels::reduce_sum(&{}, {}, {}, {});", tab, outputs[0], inputs[0], axes, keepdims, buf_expr)?;
        }
        "ReduceMean" => {
            let axes = ctx.node.attribute.iter().find(|a| a.name == "axes").map(|a| a.ints.clone()).unwrap_or(vec![]);
            let keepdims = ctx.node.attribute.iter().find(|a| a.name == "keepdims").map(|a| a.i).unwrap_or(1) != 0;
            writeln!(w, "{}let {} = lele::kernels::reduce_mean(&{}, &{:?}, {}, {});", tab, outputs[0], inputs[0], axes, keepdims, buf_expr)?;
        }
        "Range" => {
            writeln!(w, "{}let {} = lele::kernels::range(&{}, &{}, &{}, {});", tab, outputs[0], inputs[0], inputs[1], inputs[2], buf_expr)?;
        }
        "Clip" => {
            let min = if inputs.len() > 1 && !inputs[1].is_empty() { format!("Some(&{})", inputs[1]) } else { "None".to_string() };
            let max = if inputs.len() > 2 && !inputs[2].is_empty() { format!("Some(&{})", inputs[2]) } else { "None".to_string() };
            writeln!(w, "{}let {} = lele::kernels::clip(&{}, {}, {}, {});", tab, outputs[0], inputs[0], min, max, buf_expr)?;
        }
        "Greater" => writeln!(w, "{}let {} = lele::kernels::less(&{}, &{}, {});", tab, outputs[0], inputs[1], inputs[0], buf_expr)?,
        _ => return Ok(false),
    }
    Ok(true)
}
