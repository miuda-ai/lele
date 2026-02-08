use super::super::generate::OpContext;
use crate::compiler::sanitize_name;
use std::io::Write;

pub(crate) fn handle_tensor_ops(ctx: &mut OpContext, w: &mut dyn Write) -> std::io::Result<bool> {
    let op = ctx.node.op_type.as_str();
    let tab = ctx.tab();
    let inputs = &ctx.inputs;
    let outputs = &ctx.outputs;
    let buf_expr = &ctx.buf_expr;

    // Helper to resolve i64 values from inputs
    // Returns the expression and whether a temp variable was created
    let resolve_i64_with_temp =
        |idx: usize, ctx: &OpContext, w: &mut dyn Write, tab: &str| -> std::io::Result<String> {
            if idx >= ctx.node.input.len() || ctx.node.input[idx].is_empty() {
                return Ok("&[]".to_string());
            }
            let name = &ctx.node.input[idx];
            if let Some((ints, _)) = ctx.int64_map.get(name) {
                Ok(format!("&{:?}", ints))
            } else {
                let is_i64 = ctx
                    .var_types
                    .get(&ctx.inputs[idx])
                    .map(|t| t == "i64")
                    .unwrap_or(false);
                if is_i64 {
                    // For i64 tensors, use .data directly
                    Ok(format!("&{}.data[..]", ctx.inputs[idx]))
                } else {
                    // Need to convert to i64
                    let temp_name = format!("temp_i64_{}", idx);
                    writeln!(
                        w,
                        "{}let {} = lele::kernels::to_i64_vec(&{});",
                        tab, temp_name, ctx.inputs[idx]
                    )?;
                    Ok(format!("&{}", temp_name))
                }
            }
        };

    let resolve_i64_opt_with_temp =
        |idx: usize, ctx: &OpContext, w: &mut dyn Write, tab: &str| -> std::io::Result<String> {
            if idx >= ctx.node.input.len() || ctx.node.input[idx].is_empty() {
                return Ok("None".to_string());
            }
            let name = &ctx.node.input[idx];
            if let Some((ints, _)) = ctx.int64_map.get(name) {
                Ok(format!("Some(&{:?})", ints))
            } else {
                let is_i64 = ctx
                    .var_types
                    .get(&ctx.inputs[idx])
                    .map(|t| t == "i64")
                    .unwrap_or(false);
                if is_i64 {
                    Ok(format!("Some(&{}.data[..])", ctx.inputs[idx]))
                } else {
                    let temp_name = format!("temp_i64_{}", idx);
                    writeln!(
                        w,
                        "{}let {} = lele::kernels::to_i64_vec(&{});",
                        tab, temp_name, ctx.inputs[idx]
                    )?;
                    Ok(format!("Some(&{})", temp_name))
                }
            }
        };

    match op {
        "Transpose" => {
            let perm = ctx
                .node
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
            let shape = resolve_i64_with_temp(1, ctx, w, &tab)?;
            writeln!(
                w,
                "{}let {} = lele::kernels::reshape(&{}, {});",
                tab, outputs[0], inputs[0], shape
            )?;
        }
        "Unsqueeze" => {
            let axes = if ctx.node.input.len() > 1 && !ctx.node.input[1].is_empty() {
                resolve_i64_with_temp(1, ctx, w, &tab)?
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
            writeln!(
                w,
                "{}let {} = lele::kernels::unsqueeze(&{}, {});",
                tab, outputs[0], inputs[0], axes
            )?;
        }
        "Squeeze" => {
            let axes = if ctx.node.input.len() > 1 && !ctx.node.input[1].is_empty() {
                resolve_i64_opt_with_temp(1, ctx, w, &tab)?
            } else {
                let axes_attr = ctx
                    .node
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
            let axis = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "axis")
                .map(|a| a.i)
                .unwrap_or(0);

            let out_i64 = ctx
                .var_types
                .get(&outputs[0])
                .map(|t| t == "i64")
                .unwrap_or(false);

            let mut arg_exprs = Vec::new();
            for (i, inp_name_raw) in ctx.node.input.iter().enumerate() {
                if inp_name_raw.is_empty() {
                    continue;
                }
                let inp_name = sanitize_name(inp_name_raw);
                let inp_i64 = ctx
                    .var_types
                    .get(&inp_name)
                    .map(|t| t == "i64")
                    .unwrap_or(false);

                if inp_i64 == out_i64 {
                    arg_exprs.push(format!("&{}", inputs[i]));
                } else if out_i64 {
                    let temp_name = format!("temp_cast_concat_{}_{}", i, outputs[0]);
                    writeln!(w, "{}let mut {}_buf = Vec::<i64>::new();", tab, temp_name)?;
                    writeln!(
                        w,
                        "{}let {} = lele::kernels::utils::cast_to_i64(&{}, &mut {}_buf);",
                        tab, temp_name, inputs[i], temp_name
                    )?;
                    arg_exprs.push(format!("&{}", temp_name));
                } else {
                    let temp_name = format!("temp_cast_concat_{}_{}", i, outputs[0]);
                    writeln!(w, "{}let mut {}_buf = Vec::<f32>::new();", tab, temp_name)?;
                    writeln!(
                        w,
                        "{}let {} = lele::kernels::utils::cast_to_f32(&{}, &mut {}_buf);",
                        tab, temp_name, inputs[i], temp_name
                    )?;
                    arg_exprs.push(format!("&{}", temp_name));
                }
            }

            let args = arg_exprs.join(", ");
            writeln!(
                w,
                "{}let {} = lele::kernels::concat(&[{}], {}, {});",
                tab, outputs[0], args, axis, buf_expr
            )?;
        }
        "Where" => {
            writeln!(
                w,
                "{}let {} = lele::kernels::where_op(&{}, &{}, &{}, {});",
                tab, outputs[0], inputs[0], inputs[1], inputs[2], buf_expr
            )?;
        }
        "Gather" => {
            let axis = ctx
                .node
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
            let start = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "start")
                .map(|a| a.i)
                .unwrap_or(0);
            let end = ctx.node.attribute.iter().find(|a| a.name == "end").map(|a| a.i);

            if start != 0 || end.is_some() {
                let end_str = if let Some(e) = end {
                    format!("Some({})", e)
                } else {
                    "None".to_string()
                };
                writeln!(
                    w,
                    "{}let {} = lele::kernels::shape_slicing(&{}, {}, {});",
                    tab, outputs[0], inputs[0], start, end_str
                )?;
            } else {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::shape(&{});",
                    tab, outputs[0], inputs[0]
                )?;
            }
        }
        "Size" => writeln!(
            w,
            "{}let {} = lele::kernels::size(&{});",
            tab, outputs[0], inputs[0]
        )?,
        "Cast" => {
            let to = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "to")
                .map(|a| a.i);

            // Always use a temporary buffer for Cast to avoid borrow conflicts
            // Cast operations typically convert between types, and reusing buffers
            // can lead to borrowing issues when the input TensorView holds a reference
            // to the same buffer we're trying to mutably borrow for output

            if to == Some(7) || to == Some(6) || to == Some(9) {
                writeln!(
                    w,
                    "{}let mut temp_cast_buf_{} = Vec::<i64>::new();",
                    tab, outputs[0]
                )?;
                writeln!(
                    w,
                    "{}let {} = lele::kernels::utils::cast_to_i64(&{}, &mut temp_cast_buf_{});",
                    tab, outputs[0], inputs[0], outputs[0]
                )?;
            } else if to == Some(1) {
                writeln!(
                    w,
                    "{}let mut temp_cast_buf_{} = Vec::<f32>::new();",
                    tab, outputs[0]
                )?;
                writeln!(
                    w,
                    "{}let {} = lele::kernels::utils::cast_to_f32(&{}, &mut temp_cast_buf_{});",
                    tab, outputs[0], inputs[0], outputs[0]
                )?;
            } else {
                writeln!(
                    w,
                    "{}let {} = {}.clone(); // Cast ignored",
                    tab, outputs[0], inputs[0]
                )?;
            }
        }
        "ConstantOfShape" => {
            let is_i64 = ctx
                .var_types
                .get(&ctx.outputs[0])
                .map(|s| s == "i64")
                .unwrap_or(false);
            let val = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "value")
                .and_then(|a| a.t.as_ref())
                .map(|t| {
                    if let Ok((data, _)) = crate::model::tensor_to_array(t) {
                        if !data.is_empty() {
                            return data[0];
                        }
                    }
                    0.0
                })
                .unwrap_or(0.0);
            if is_i64 {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::constant_of_shape(&{}, {}i64, {});",
                    tab, outputs[0], inputs[0], val as i64, buf_expr
                )?;
            } else {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::constant_of_shape(&{}, {:.1}, {});",
                    tab, outputs[0], inputs[0], val, buf_expr
                )?;
            }
        }
        "Slice" => {
            let starts = resolve_i64_with_temp(1, ctx, w, &tab)?;
            let ends = resolve_i64_with_temp(2, ctx, w, &tab)?;
            let axes = resolve_i64_with_temp(3, ctx, w, &tab)?;
            let steps = resolve_i64_with_temp(4, ctx, w, &tab)?;
            writeln!(
                w,
                "{}let {} = lele::kernels::slice(&{}, {}, {}, {}, {}, {});",
                tab, outputs[0], inputs[0], starts, ends, axes, steps, buf_expr
            )?;
        }
        "Expand" => {
            let shape = resolve_i64_with_temp(1, ctx, w, &tab)?;
            writeln!(
                w,
                "{}let {} = lele::kernels::expand(&{}, {}, {});",
                tab, outputs[0], inputs[0], shape, buf_expr
            )?;
        }
        "Tile" => {
            let repeats = resolve_i64_with_temp(1, ctx, w, &tab)?;
            writeln!(
                w,
                "{}let {} = lele::kernels::tile(&{}, {}, {});",
                tab, outputs[0], inputs[0], repeats, buf_expr
            )?;
        }
        "Split" => {
            let axis = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "axis")
                .map(|a| a.i)
                .unwrap_or(0);
            let splits = if ctx.node.input.len() > 1 && !ctx.node.input[1].is_empty() {
                resolve_i64_with_temp(1, ctx, w, &tab)?
            } else {
                let split_attr = ctx
                    .node
                    .attribute
                    .iter()
                    .find(|a| a.name == "split")
                    .map(|a| a.ints.clone())
                    .unwrap_or_else(|| {
                        let num_outputs = outputs.len();
                        vec![0; num_outputs]
                    });
                format!("&{:?}", split_attr)
            };
            for out_name in outputs {
                writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, out_name)?;
            }
            let buf_names: Vec<String> = outputs.iter().map(|n| format!("buf_{}", n)).collect();
            writeln!(
                w,
                "{}let mut split_buffers = [{}];",
                tab,
                buf_names.join(", ")
            )?;
            writeln!(w, "{}let splits_slice = {};", tab, splits)?;
            writeln!(
                w,
                "{}let split_results = lele::kernels::split(&{}, {}, splits_slice, &mut split_buffers);",
                tab, inputs[0], axis
            )?;
            for (i, out_name) in outputs.iter().enumerate() {
                writeln!(w, "{}let {} = split_results[{}].clone();", tab, out_name, i)?;
            }
        }
        "Pad" => {
            let mode = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "mode")
                .map(|a| String::from_utf8_lossy(&a.s))
                .unwrap_or("constant".into());
            let constant_value =
                if inputs.len() > 2 && !ctx.node.input[2].is_empty() && !inputs[2].is_empty() {
                    format!("Some(&{})", inputs[2])
                } else {
                    "None".to_string()
                };
            // For pad operation, we need pads as &[i64]
            let pads_expr = if ctx.node.input.len() > 1 && !ctx.node.input[1].is_empty() {
                let name = &ctx.node.input[1];
                if let Some((ints, _)) = ctx.int64_map.get(name) {
                    format!("&{:?}", ints)
                } else {
                    // pads is a computed tensor, we need to extract data
                    let is_i64 = ctx
                        .var_types
                        .get(&ctx.inputs[1])
                        .map(|t| t == "i64")
                        .unwrap_or(false);
                    if is_i64 {
                        format!("&{}.data[..]", inputs[1])
                    } else {
                        // Need to convert f32 to i64
                        writeln!(
                            w,
                            "{}let pads_i64 = lele::kernels::to_i64_vec(&{});",
                            tab, inputs[1]
                        )?;
                        "&pads_i64".to_string()
                    }
                }
            } else {
                "&[]".to_string()
            };
            writeln!(
                w,
                "{}let {} = lele::kernels::pad(&{}, {}, {}, {:?}, {});",
                tab, outputs[0], inputs[0], pads_expr, constant_value, mode, buf_expr
            )?;
        }
        "DynamicQuantizeLinear" => {
            writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, outputs[1])?;
            writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, outputs[2])?;
            writeln!(
                w,
                "{}let ({}, {}, {}) = lele::kernels::dynamic_quantize_linear(&{}, {}, &mut buf_{}, &mut buf_{});",
                tab,
                outputs[0],
                outputs[1],
                outputs[2],
                inputs[0],
                buf_expr,
                outputs[1],
                outputs[2]
            )?;
        }
        "Identity" | "Constant" => {
            if let Some((offset, len, shape, dt)) = ctx.known_weights.get(&outputs[0]) {
                let weight_fn = match dt {
                    1 => "weight_f32",
                    2 => "weight_u8",
                    3 => "weight_i8",
                    6 => "weight_i32",
                    7 => "weight_i64",
                    10 => "weight_f16",
                    _ => "weight_f32",
                };
                writeln!(
                    w,
                    "{}let {} = self.{}({}, {}, &{:?});",
                    tab, outputs[0], weight_fn, offset, len, shape
                )?;
            } else if let Some((ints, shape)) = ctx.int64_map.get(&ctx.node.output[0]) {
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
            } else if op == "Identity" {
                writeln!(w, "{}let {} = {}.clone();", tab, outputs[0], inputs[0])?;
            } else {
                // More complex Constant logic
                let val = ctx
                    .node
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
                            writeln!(
                                w,
                                "{}let {} = lele::tensor::TensorView::from_owned(vec!{:?}, vec!{:?});",
                                tab, outputs[0], t.float_data, t.dims
                            )?;
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
                            writeln!(
                                w,
                                "{}let {} = lele::tensor::TensorView::from_owned(vec!{:?}, vec!{:?});",
                                tab, outputs[0], floats, t.dims
                            )?;
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
        _ => return Ok(false),
    }
    Ok(true)
}
