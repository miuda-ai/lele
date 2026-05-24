use super::super::generate::OpContext;
use super::super::sanitize_name;
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
        "Mod" => writeln!(
            w,
            "{}let {} = lele::kernels::mod_f32(&{}, &{}, {});",
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
            // Check if B is a static weight — use cached ARM path
            let b_weight_name = super::super::sanitize_name(&ctx.node.input[1]);
            if let Some((o, l, sh, dt)) = ctx.known_weights.get(&b_weight_name) {
                if (*dt == 2 || *dt == 3) && sh.len() == 2 {
                    let k = sh[0];
                    let n = sh[1];
                    writeln!(
                        w,
                        "{}#[cfg(target_arch = \"aarch64\")]",
                        tab
                    )?;
                    writeln!(
                        w,
                        "{}let {} = self.mat_mul_integer_arm(&{}, {}, {}, {}, {}, {}, {}, {});",
                        tab, outputs[0], inputs[0], *o, *l, k, n, a_zp, b_zp, buf_expr
                    )?;
                    writeln!(
                        w,
                        "{}#[cfg(not(target_arch = \"aarch64\"))]",
                        tab
                    )?;
                    writeln!(
                        w,
                        "{}let {} = lele::kernels::mat_mul_integer(&{}, &{}, {}, {}, {});",
                        tab, outputs[0], inputs[0], inputs[1], a_zp, b_zp, buf_expr
                    )?;
                } else {
                    writeln!(
                        w,
                        "{}let {} = lele::kernels::mat_mul_integer(&{}, &{}, {}, {}, {});",
                        tab, outputs[0], inputs[0], inputs[1], a_zp, b_zp, buf_expr
                    )?;
                }
            } else {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::mat_mul_integer(&{}, &{}, {}, {}, {});",
                    tab, outputs[0], inputs[0], inputs[1], a_zp, b_zp, buf_expr
                )?;
            }
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
        "Log" => writeln!(
            w,
            "{}let {} = lele::kernels::log(&{}, {});",
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
            let is_i64 = ctx
                .var_types
                .get(&ctx.outputs[0])
                .map(|t| t == "i64")
                .unwrap_or(false)
                || ctx
                    .var_types
                    .get(&ctx.inputs[0])
                    .map(|t| t == "i64")
                    .unwrap_or(false);
            if is_i64 {
                let a_is_weight_f32 = inputs[0].contains("weight_f32");
                let b_is_weight_f32 = inputs.len() > 1 && inputs[1].contains("weight_f32");
                let raw_a = sanitize_name(&ctx.node.input[0]);
                let type_a = ctx.var_types.get(&raw_a).map(|s| s.as_str()).unwrap_or("f32");
                if b_is_weight_f32 {
                    let func = if type_a == "i64" { "equal_i64_f32_r_i64" } else { "equal_i64_f32_r" };
                    writeln!(
                        w,
                        "{}let {} = lele::kernels::{}(&{}, &{}, {});",
                        tab, outputs[0], func, inputs[0], inputs[1], buf_expr
                    )?;
                } else if a_is_weight_f32 {
                    writeln!(
                        w,
                        "{}let {} = lele::kernels::equal_i64_f32_lhs(&{}, &{}, {});",
                        tab, outputs[0], inputs[0], inputs[1], buf_expr
                    )?;
                } else {
                    writeln!(
                        w,
                        "{}let {} = lele::kernels::equal_i64(&{}, &{}, {});",
                        tab, outputs[0], inputs[0], inputs[1], buf_expr
                    )?;
                }
            } else {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::equal(&{}, &{}, {});",
                    tab, outputs[0], inputs[0], inputs[1], buf_expr
                )?;
            }
        }
        "Less" => {
            let is_i64 = ctx
                .var_types
                .get(&inputs[0])
                .map(|t| t == "i64")
                .unwrap_or(false)
                || ctx
                    .var_types
                    .get(&inputs[1])
                    .map(|t| t == "i64")
                    .unwrap_or(false);
            if is_i64 {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::less_i64(&{}, &{}, {});",
                    tab, outputs[0], inputs[0], inputs[1], buf_expr
                )?;
            } else {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::less(&{}, &{}, {});",
                    tab, outputs[0], inputs[0], inputs[1], buf_expr
                )?;
            }
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
            let mut axes = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "axes")
                .map(|a| a.ints.clone())
                .unwrap_or(vec![]);
            // If axes attribute is empty, check second input
            if axes.is_empty() && inputs.len() > 1 && !ctx.node.input[1].is_empty() {
                if let Some((data, _shape)) = ctx.int64_map.get(&ctx.node.input[1]) {
                    axes = data.clone();
                }
            }
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
        "ReduceL2" => {
            let mut axes = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "axes")
                .map(|a| a.ints.clone())
                .unwrap_or(vec![]);
            // If axes attribute is empty, check second input
            if axes.is_empty() && inputs.len() > 1 && !ctx.node.input[1].is_empty() {
                if let Some((data, _shape)) = ctx.int64_map.get(&ctx.node.input[1]) {
                    axes = data.clone();
                }
            }
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
                "{}let {} = lele::kernels::reduce_l2(&{}, &{:?}, {}, {});",
                tab, outputs[0], inputs[0], axes, keepdims, buf_expr
            )?;
        }
        "Max" => {
            // ONNX Max: element-wise maximum of two or more inputs
            if inputs.len() == 2 {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::max(&{}, &{}, {});",
                    tab, outputs[0], inputs[0], inputs[1], buf_expr
                )?;
            } else {
                // Chain: max(a, max(b, max(c, d)))
                let mut result = inputs[0].clone();
                for (i, input) in inputs.iter().enumerate().skip(1) {
                    if i < inputs.len() - 1 {
                        let tmp = format!("max_tmp_{}_{}", outputs[0], i);
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::max(&{}, &{}, {});",
                            tab, tmp, result, input, buf_expr
                        )?;
                        result = tmp;
                    } else {
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::max(&{}, &{}, {});",
                            tab, outputs[0], result, input, buf_expr
                        )?;
                    }
                }
            }
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
            let is_i64 = ctx
                .var_types
                .get(&inputs[0])
                .map(|t| t == "i64")
                .unwrap_or(false)
                || ctx
                    .var_types
                    .get(&inputs[1])
                    .map(|t| t == "i64")
                    .unwrap_or(false);
            if is_i64 {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::less_i64(&{}, &{}, {});",
                    tab, outputs[0], inputs[1], inputs[0], buf_expr
                )?;
            } else {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::less(&{}, &{}, {});",
                    tab, outputs[0], inputs[1], inputs[0], buf_expr
                )?;
            }
        }
        "STFT" => {
            // ONNX STFT: inputs[0]=signal, inputs[1]=frame_step, inputs[2]=window, inputs[3]=frame_length
            let frame_step = if ctx.node.input.len() > 1 {
                if let Some((ints, _)) = ctx.int64_map.get(&ctx.node.input[1]) {
                    ints[0] as usize
                } else {
                    160
                }
            } else {
                160
            };
            let n_fft = if ctx.node.input.len() > 3 && !ctx.node.input[3].is_empty() {
                if let Some((ints, _)) = ctx.int64_map.get(&ctx.node.input[3]) {
                    ints[0] as usize
                } else {
                    512
                }
            } else {
                512
            };
            let window = if ctx.node.input.len() > 2 && !ctx.node.input[2].is_empty() {
                let wname = sanitize_name(&ctx.node.input[2]);
                if let Some((o, l, s, _dt)) = ctx.known_weights.get(&wname) {
                    format!("Some(&self.weight_f32({}, {}, &{:?}))", o, l, s)
                } else {
                    format!("Some(&{})", inputs[2])
                }
            } else {
                "None".to_string()
            };
            writeln!(
                w,
                "{}let {} = lele::kernels::stft(&{}, {}, {}, {}, {}, {});",
                tab, outputs[0], inputs[0], n_fft, frame_step, n_fft, window, buf_expr
            )?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}
