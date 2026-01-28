use super::super::generate::OpContext;
use std::io::Write;

pub(crate) fn handle_nn_ops(ctx: &mut OpContext, w: &mut dyn Write) -> std::io::Result<bool> {
    let op = ctx.node.op_type.as_str();
    let tab = ctx.tab();
    let inputs = &ctx.inputs;
    let outputs = &ctx.outputs;
    let buf_expr = &ctx.buf_expr;

    match op {
        "Conv" => {
            let dilations = ctx.node.attribute.iter().find(|a| a.name == "dilations").map(|a| a.ints.clone()).unwrap_or(vec![]);
            let group = ctx.node.attribute.iter().find(|a| a.name == "group").map(|a| a.i).unwrap_or(1);
            let pads = ctx.node.attribute.iter().find(|a| a.name == "pads").map(|a| a.ints.clone()).unwrap_or(vec![]);
            let strides = ctx.node.attribute.iter().find(|a| a.name == "strides").map(|a| a.ints.clone()).unwrap_or(vec![]);
            let bias_arg = if inputs.len() > 2 { format!("Some(&{})", inputs[2]) } else { "None".to_string() };
            writeln!(w, "{}let {} = lele::kernels::conv1d(&{}, &{}, {}, &{:?}, {}, &{:?}, &{:?}, {});", 
                    tab, outputs[0], inputs[0], inputs[1], bias_arg, dilations, group, pads, strides, buf_expr)?;
        }
        "Gemm" => {
            let alpha = ctx.node.attribute.iter().find(|a| a.name == "alpha").map(|a| a.f).unwrap_or(1.0);
            let beta = ctx.node.attribute.iter().find(|a| a.name == "beta").map(|a| a.f).unwrap_or(1.0);
            let trans_a = ctx.node.attribute.iter().find(|a| a.name == "transA").map(|a| a.i).unwrap_or(0) != 0;
            let trans_b = ctx.node.attribute.iter().find(|a| a.name == "transB").map(|a| a.i).unwrap_or(0) != 0;
            let c = if inputs.len() > 2 && !inputs[2].is_empty() { format!("Some(&{})", inputs[2]) } else { "None".to_string() };
            writeln!(w, "{}let {} = lele::kernels::gemm(&{}, &{}, {}, {:?}, {:?}, {}, {}, {});",
                tab, outputs[0], inputs[0], inputs[1], c, alpha, beta, trans_a, trans_b, buf_expr)?;
        }
        "LSTM" => {
            let bias = if inputs.len() > 3 && !ctx.node.input[3].is_empty() { format!("Some(&{})", inputs[3]) } else { "None".to_string() };
            let seq_lens = if inputs.len() > 4 && !ctx.node.input[4].is_empty() { format!("Some(&{})", inputs[4]) } else { "None".to_string() };
            let initial_h = if inputs.len() > 5 && !ctx.node.input[5].is_empty() { format!("Some(&{})", inputs[5]) } else { "None".to_string() };
            let initial_c = if inputs.len() > 6 && !ctx.node.input[6].is_empty() { format!("Some(&{})", inputs[6]) } else { "None".to_string() };
            writeln!(w, "{}let mut buf_{}_h = Vec::<f32>::new();", tab, outputs[0])?;
            writeln!(w, "{}let mut buf_{}_c = Vec::<f32>::new();", tab, outputs[0])?;
            writeln!(w, "{}let ({}, {}, {}) = lele::kernels::lstm(&{}, &{}, &{}, {}, {}, {}, {}, {}, &mut buf_{}_h, &mut buf_{}_c);", 
                    tab, outputs[0], outputs[1], outputs[2], inputs[0], inputs[1], inputs[2], bias, seq_lens, initial_h, initial_c, buf_expr, outputs[0], outputs[0])?;
        }
        "LayerNormalization" => {
            let epsilon = ctx.node.attribute.iter().find(|a| a.name == "epsilon").map(|a| a.f).unwrap_or(1e-5);
            let axis = ctx.node.attribute.iter().find(|a| a.name == "axis").map(|a| a.i).unwrap_or(-1);
            let scale = if ctx.node.input.len() > 1 && !ctx.node.input[1].is_empty() && !inputs[1].is_empty() { format!("&{}", inputs[1]) } else { "&lele::tensor::TensorView::empty()".to_string() };
            let bias = if ctx.node.input.len() > 2 && !ctx.node.input[2].is_empty() && !inputs[2].is_empty() { format!("&{}", inputs[2]) } else { "&lele::tensor::TensorView::empty()".to_string() };
            if outputs.len() > 1 {
                let fillers = vec!["_"; outputs.len() - 1].join(", ");
                let dummy_tensors = vec!["lele::tensor::TensorView::empty()"; outputs.len() - 1].join(", ");
                writeln!(w, "{}let ({}, {}) = (lele::kernels::layer_norm(&{}, {}, {}, {}, {}, {}), {});", 
                        tab, outputs[0], fillers, inputs[0], scale, bias, axis, epsilon, buf_expr, dummy_tensors)?;
            } else {
                writeln!(w, "{}let {} = lele::kernels::layer_norm(&{}, {}, {}, {}, {}, {});",
                    tab, outputs[0], inputs[0], scale, bias, axis, epsilon, buf_expr)?;
            }
        }
        "BatchNormalization" => {
            let epsilon = ctx.node.attribute.iter().find(|a| a.name == "epsilon").map(|a| a.f).unwrap_or(1e-5);
            writeln!(w, "{}let {} = lele::kernels::batch_norm(&{}, &{}, &{}, &{}, &{}, {:?}, {});", 
                tab, outputs[0], inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon, buf_expr)?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}
