use super::super::generate::OpContext;
use crate::compiler::sanitize_name;
use std::io::Write;

pub(crate) fn handle_math_ops(ctx: &mut OpContext, w: &mut dyn Write) -> std::io::Result<bool> {
    let op = ctx.node.op_type.as_str();
    let tab = ctx.tab();
    let inputs = &ctx.inputs;
    let outputs = &ctx.outputs;
    let buf_expr = &ctx.buf_expr;

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
            // MatMulInteger expects u8 inputs.
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
        "Pow" => writeln!(
            w,
            "{}let {} = lele::kernels::pow(&{}, &{}, {});",
            tab, outputs[0], inputs[0], inputs[1], buf_expr
        )?,
        "Sqrt" => writeln!(
            w,
            "{}let {} = lele::kernels::sqrt(&{}, {});",
            tab, outputs[0], inputs[0], buf_expr
        )?,
        "Neg" => writeln!(
            w,
            "{}let {} = lele::kernels::neg(&{}, {});",
            tab, outputs[0], inputs[0], buf_expr
        )?,
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
        "Exp" => writeln!(
            w,
            "{}let {} = lele::kernels::exp(&{}, {});",
            tab, outputs[0], inputs[0], buf_expr
        )?,
        "Sin" => writeln!(
            w,
            "{}let {} = lele::kernels::sin(&{}, {});",
            tab, outputs[0], inputs[0], buf_expr
        )?,
        "Cos" => writeln!(
            w,
            "{}let {} = lele::kernels::cos(&{}, {});",
            tab, outputs[0], inputs[0], buf_expr
        )?,
        "Equal" => {
            let in0_name = sanitize_name(&ctx.node.input[0]);
            let in1_name = sanitize_name(&ctx.node.input[1]);
            let in0_i64 = ctx.var_types.get(&in0_name).map(|t| t == "i64").unwrap_or(false);
            let in1_i64 = ctx.var_types.get(&in1_name).map(|t| t == "i64").unwrap_or(false);
            
            let (arg0, arg1) = if in0_i64 == in1_i64 {
                (inputs[0].clone(), inputs[1].clone())
            } else if in0_i64 {
                writeln!(w, "{}let mut temp_cast_eq_0_{} = Vec::<f32>::new();", tab, outputs[0])?;
                writeln!(w, "{}let arg0_{} = lele::kernels::utils::cast_to_f32(&{}, &mut temp_cast_eq_0_{});", tab, outputs[0], inputs[0], outputs[0])?;
                (format!("arg0_{}", outputs[0]), inputs[1].clone())
            } else {
                writeln!(w, "{}let mut temp_cast_eq_1_{} = Vec::<f32>::new();", tab, outputs[0])?;
                writeln!(w, "{}let arg1_{} = lele::kernels::utils::cast_to_f32(&{}, &mut temp_cast_eq_1_{});", tab, outputs[0], inputs[1], outputs[0])?;
                (inputs[0].clone(), format!("arg1_{}", outputs[0]))
            };

            writeln!(
                w,
                "{}let {} = lele::kernels::equal_i64(&{}, &{}, {});",
                tab, outputs[0], arg0, arg1, buf_expr
            )?;
        }
        "Less" => {
            let in0_name = sanitize_name(&ctx.node.input[0]);
            let in1_name = sanitize_name(&ctx.node.input[1]);
            let in0_i64 = ctx.var_types.get(&in0_name).map(|t| t == "i64").unwrap_or(false);
            let in1_i64 = ctx.var_types.get(&in1_name).map(|t| t == "i64").unwrap_or(false);

            let (arg0, arg1) = if in0_i64 == in1_i64 {
                (inputs[0].clone(), inputs[1].clone())
            } else if in0_i64 {
                writeln!(w, "{}let mut temp_cast_lt_0_{} = Vec::<f32>::new();", tab, outputs[0])?;
                writeln!(w, "{}let arg0_{} = lele::kernels::utils::cast_to_f32(&{}, &mut temp_cast_lt_0_{});", tab, outputs[0], inputs[0], outputs[0])?;
                (format!("arg0_{}", outputs[0]), inputs[1].clone())
            } else {
                writeln!(w, "{}let mut temp_cast_lt_1_{} = Vec::<f32>::new();", tab, outputs[0])?;
                writeln!(w, "{}let arg1_{} = lele::kernels::utils::cast_to_f32(&{}, &mut temp_cast_lt_1_{});", tab, outputs[0], inputs[1], outputs[0])?;
                (inputs[0].clone(), format!("arg1_{}", outputs[0]))
            };

            writeln!(
                w,
                "{}let {} = lele::kernels::less_i64(&{}, &{}, {});",
                tab, outputs[0], arg0, arg1, buf_expr
            )?;
        }
        "Not" => writeln!(
            w,
            "{}let {} = lele::kernels::not(&{}, {});",
            tab, outputs[0], inputs[0], buf_expr
        )?,
        "PRelu" => writeln!(
            w,
            "{}let {} = lele::kernels::prelu(&{}, &{}, {});",
            tab, outputs[0], inputs[0], inputs[1], buf_expr
        )?,
        "ReduceSum" => {
            let axes = if ctx.node.input.len() > 1 && !ctx.node.input[1].is_empty() {
                let name = &ctx.node.input[1];
                if let Some((ints, _)) = ctx.int64_map.get(name) {
                    format!("&{:?}", ints)
                } else {
                    format!("&lele::kernels::to_i64_vec(&{})", inputs[1])
                }
            } else {
                let axes_attr = ctx
                    .node
                    .attribute
                    .iter()
                    .find(|a| a.name == "axes")
                    .map(|a| a.ints.clone())
                    .unwrap_or(vec![]);
                format!("&{:?}", axes_attr)
            };
            let keepdims = ctx
                .node
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
        "ReduceMax" => {
            let axes = if ctx.node.input.len() > 1 && !ctx.node.input[1].is_empty() {
                let name = &ctx.node.input[1];
                if let Some((ints, _)) = ctx.int64_map.get(name) {
                    format!("&{:?}", ints)
                } else {
                    format!("&lele::kernels::to_i64_vec(&{})", inputs[1])
                }
            } else {
                let axes_attr = ctx
                    .node
                    .attribute
                    .iter()
                    .find(|a| a.name == "axes")
                    .map(|a| a.ints.clone())
                    .unwrap_or(vec![]);
                format!("&{:?}", axes_attr)
            };
            let keepdims = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "keepdims")
                .map(|a| a.i)
                .unwrap_or(1)
                != 0;
            writeln!(
                w,
                "{}let {} = lele::kernels::reduce_max(&{}, {}, {}, {});",
                tab, outputs[0], inputs[0], axes, keepdims, buf_expr
            )?;
        }
        "ReduceMean" => {
            let axes = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "axes")
                .map(|a| a.ints.clone())
                .unwrap_or(vec![]);
            let keepdims = ctx
                .node
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
        "Range" => {
            let is_i64 = ctx
                .var_types
                .get(&inputs[0])
                .map(|t| t == "i64")
                .unwrap_or(false)
                || ctx
                    .var_types
                    .get(&inputs[1])
                    .map(|t| t == "i64")
                    .unwrap_or(false)
                || ctx
                    .var_types
                    .get(&inputs[2])
                    .map(|t| t == "i64")
                    .unwrap_or(false);
            if is_i64 {
                writeln!(w, "{}let mut buf_{} = Vec::<i64>::new();", tab, outputs[0])?;
                writeln!(
                    w,
                    "{}let {} = lele::kernels::range_i64(&{}, &{}, &{}, &mut buf_{});",
                    tab, outputs[0], inputs[0], inputs[1], inputs[2], outputs[0]
                )?;
            } else {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::range(&{}, &{}, &{}, {});",
                    tab, outputs[0], inputs[0], inputs[1], inputs[2], buf_expr
                )?;
            }
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
        "Greater" => {
            let in0_name = sanitize_name(&ctx.node.input[0]);
            let in1_name = sanitize_name(&ctx.node.input[1]);
            let in0_i64 = ctx.var_types.get(&in0_name).map(|t| t == "i64").unwrap_or(false);
            let in1_i64 = ctx.var_types.get(&in1_name).map(|t| t == "i64").unwrap_or(false);

            let (arg0, arg1) = if in0_i64 == in1_i64 {
                (inputs[0].clone(), inputs[1].clone())
            } else if in0_i64 {
                writeln!(w, "{}let mut temp_cast_gt_0_{} = Vec::<f32>::new();", tab, outputs[0])?;
                writeln!(w, "{}let arg0_{} = lele::kernels::utils::cast_to_f32(&{}, &mut temp_cast_gt_0_{});", tab, outputs[0], inputs[0], outputs[0])?;
                (format!("arg0_{}", outputs[0]), inputs[1].clone())
            } else {
                writeln!(w, "{}let mut temp_cast_gt_1_{} = Vec::<f32>::new();", tab, outputs[0])?;
                writeln!(w, "{}let arg1_{} = lele::kernels::utils::cast_to_f32(&{}, &mut temp_cast_gt_1_{});", tab, outputs[0], inputs[1], outputs[0])?;
                (inputs[0].clone(), format!("arg1_{}", outputs[0]))
            };

            writeln!(
                w,
                "{}let {} = lele::kernels::less_i64(&{}, &{}, {});",
                tab, outputs[0], arg1, arg0, buf_expr
            )?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}