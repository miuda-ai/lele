use super::super::generate::OpContext;
use std::io::Write;

pub(crate) fn handle_tensor_ops(ctx: &mut OpContext, w: &mut dyn Write) -> std::io::Result<bool> {
    let op = ctx.node.op_type.as_str();
    let tab = ctx.tab();
    let inputs = &ctx.inputs;
    let outputs = &ctx.outputs;
    let buf_expr = &ctx.buf_expr;

    match op {
        "Transpose" => {
            let perm = ctx.node.attribute.iter().find(|a| a.name == "perm").map(|a| a.ints.clone()).unwrap_or(vec![]);
            writeln!(w, "{}let {} = lele::kernels::transpose(&{}, &{:?}, {});", tab, outputs[0], inputs[0], perm, buf_expr)?;
        }
        "Reshape" => writeln!(w, "{}let {} = lele::kernels::reshape(&{}, &{});", tab, outputs[0], inputs[0], inputs[1])?,
        "Unsqueeze" => {
            let axes = if ctx.node.input.len() > 1 && !ctx.node.input[1].is_empty() && !inputs[1].is_empty() {
                format!("&lele::kernels::to_i64_vec(&{})", inputs[1])
            } else {
                let axes_attr = ctx.node.attribute.iter().find(|a| a.name == "axes").map(|a| a.ints.clone()).unwrap_or(vec![]);
                format!("&{:?}", axes_attr)
            };
            writeln!(w, "{}let {} = lele::kernels::unsqueeze(&{}, {});", tab, outputs[0], inputs[0], axes)?;
        }
        "Squeeze" => {
            let axes = if ctx.node.input.len() > 1 && !ctx.node.input[1].is_empty() && !inputs[1].is_empty() {
                format!("Some(&lele::kernels::to_i64_vec(&{}))", inputs[1])
            } else {
                let axes_attr = ctx.node.attribute.iter().find(|a| a.name == "axes").map(|a| a.ints.clone());
                if let Some(a) = axes_attr { format!("Some(&{:?})", a) } else { "None".to_string() }
            };
            writeln!(w, "{}let {} = lele::kernels::squeeze(&{}, {});", tab, outputs[0], inputs[0], axes)?;
        }
        "Concat" => {
            let axis = ctx.node.attribute.iter().find(|a| a.name == "axis").map(|a| a.i).unwrap_or(0);
            let args = inputs.iter().map(|s| format!("&{}", s)).collect::<Vec<_>>().join(", ");
            writeln!(w, "{}let {} = lele::kernels::concat(&[{}], {}, {});", tab, outputs[0], args, axis, buf_expr)?;
        }
        "Where" => {
            writeln!(w, "{}let {} = lele::kernels::where_op(&{}, &{}, &{}, {});", tab, outputs[0], inputs[0], inputs[1], inputs[2], buf_expr)?;
        }
        "Gather" => {
            let axis = ctx.node.attribute.iter().find(|a| a.name == "axis").map(|a| a.i).unwrap_or(0);
            writeln!(w, "{}let {} = lele::kernels::gather(&{}, &{}, {}, {});", tab, outputs[0], inputs[0], inputs[1], axis, buf_expr)?;
        }
        "Shape" => writeln!(w, "{}let {} = lele::kernels::shape(&{});", tab, outputs[0], inputs[0])?,
        "Size" => writeln!(w, "{}let {} = lele::kernels::size(&{});", tab, outputs[0], inputs[0])?,
        "Cast" => writeln!(w, "{}let {} = {}.clone(); // Cast ignored", tab, outputs[0], inputs[0])?,
        "ConstantOfShape" => {
            let val = ctx.node.attribute.iter().find(|a| a.name == "value").and_then(|a| a.t.as_ref()).map(|t| {
                if let Ok((data, _)) = crate::model::tensor_to_array(t) {
                    if !data.is_empty() { return data[0]; }
                }
                0.0
            }).unwrap_or(0.0);
            writeln!(w, "{}let {} = lele::kernels::constant_of_shape(&{}, {:.1}, {});", tab, outputs[0], inputs[0], val, buf_expr)?;
        }
        "Slice" => {
            let axes = if ctx.node.input.len() > 3 { format!("&lele::kernels::to_i64_vec(&{})", inputs[3]) } else { "&[]".to_string() };
            let steps = if ctx.node.input.len() > 4 { format!("&lele::kernels::to_i64_vec(&{})", inputs[4]) } else { "&[]".to_string() };
            writeln!(w, "{}let {} = lele::kernels::slice(&{}, &lele::kernels::to_i64_vec(&{}), &lele::kernels::to_i64_vec(&{}), {}, {}, {});", 
                tab, outputs[0], inputs[0], inputs[1], inputs[2], axes, steps, buf_expr)?;
        }
        "Expand" => writeln!(w, "{}let {} = lele::kernels::expand(&{}, &{}, {});", tab, outputs[0], inputs[0], inputs[1], buf_expr)?,
        "Tile" => writeln!(w, "{}let {} = lele::kernels::tile(&{}, &{}, {});", tab, outputs[0], inputs[0], inputs[1], buf_expr)?,
        "Split" => {
            let axis = ctx.node.attribute.iter().find(|a| a.name == "axis").map(|a| a.i).unwrap_or(0);
            let splits = if ctx.node.input.len() > 1 && !ctx.node.input[1].is_empty() {
                format!("lele::kernels::to_i64_vec(&{})", inputs[1])
            } else {
                let split_attr = ctx.node.attribute.iter().find(|a| a.name == "split").map(|a| a.ints.clone()).unwrap_or_else(|| {
                    let num_outputs = outputs.len();
                    vec![0; num_outputs]
                });
                format!("vec!{:?}", split_attr)
            };
            for out_name in outputs {
                writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, out_name)?;
            }
            let buf_names: Vec<String> = outputs.iter().map(|n| format!("buf_{}", n)).collect();
            writeln!(w, "{}let mut split_buffers = [{}];", tab, buf_names.join(", "))?;
            writeln!(w, "{}let splits_vec = {};", tab, splits)?;
            writeln!(w, "{}let split_results = lele::kernels::split(&{}, {}, &splits_vec, &mut split_buffers);", tab, inputs[0], axis)?;
            for (i, out_name) in outputs.iter().enumerate() {
                writeln!(w, "{}let {} = split_results[{}].clone();", tab, out_name, i)?;
            }
        }
        "Pad" => {
            let mode = ctx.node.attribute.iter().find(|a| a.name == "mode").map(|a| String::from_utf8_lossy(&a.s)).unwrap_or("constant".into());
            let constant_value = if inputs.len() > 2 && !ctx.node.input[2].is_empty() && !inputs[2].is_empty() { format!("Some(&{})", inputs[2]) } else { "None".to_string() };
            writeln!(w, "{}let {} = lele::kernels::pad(&{}, &{}, {}, {:?}, {});", tab, outputs[0], inputs[0], inputs[1], constant_value, mode, buf_expr)?;
        }
        "DynamicQuantizeLinear" => {
            writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, outputs[1])?;
            writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, outputs[2])?;
            writeln!(w, "{}let ({}, {}, {}) = lele::kernels::dynamic_quantize_linear(&{}, {}, &mut buf_{}, &mut buf_{});", tab, outputs[0], outputs[1], outputs[2], inputs[0], buf_expr, outputs[1], outputs[2])?;
        }
        "Identity" | "Constant" => {
            if let Some((offset, len, shape)) = ctx.known_weights.get(&outputs[0]) {
                writeln!(w, "{}let {} = self.weight({}, {}, &{:?});", tab, outputs[0], offset, len, shape)?;
            } else if let Some((ints, shape)) = ctx.int64_map.get(&ctx.node.output[0]) {
                let floats: Vec<String> = ints.iter().map(|&x| format!("{:.1}", x as f32)).collect();
                writeln!(w, "{}let {} = lele::tensor::TensorView::from_owned(vec![{}], vec!{:?});", tab, outputs[0], floats.join(", "), shape)?;
            } else if op == "Identity" {
                 writeln!(w, "{}let {} = {}.clone();", tab, outputs[0], inputs[0])?;
            } else {
                // More complex Constant logic
                let val = ctx.node.attribute.iter().find(|a| a.name == "value").and_then(|a| a.t.as_ref());
                if let Some(t) = val {
                    if !t.float_data.is_empty() {
                        if t.float_data.len() > 100 {
                            writeln!(w, "{}let {} = lele::tensor::TensorView::empty(); // Large", tab, outputs[0])?;
                        } else {
                            writeln!(w, "{}let {} = lele::tensor::TensorView::from_owned(vec!{:?}, vec!{:?});", tab, outputs[0], t.float_data, t.dims)?;
                        }
                    } else if !t.raw_data.is_empty() {
                        let floats: Vec<f32> = t.raw_data.chunks(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
                        if floats.len() > 100 {
                            writeln!(w, "{}let {} = lele::tensor::TensorView::empty(); // Large", tab, outputs[0])?;
                        } else {
                            writeln!(w, "{}let {} = lele::tensor::TensorView::from_owned(vec!{:?}, vec!{:?});", tab, outputs[0], floats, t.dims)?;
                        }
                    } else {
                        writeln!(w, "{}let {} = lele::tensor::TensorView::empty();", tab, outputs[0])?;
                    }
                } else {
                    writeln!(w, "{}let {} = lele::tensor::TensorView::empty();", tab, outputs[0])?;
                }
            }
        }
        _ => return Ok(false),
    }
    Ok(true)
}
