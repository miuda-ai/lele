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
            let dilations = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "dilations")
                .map(|a| a.ints.clone())
                .unwrap_or(vec![]);
            let group = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "group")
                .map(|a| a.i)
                .unwrap_or(1);
            let pads = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "pads")
                .map(|a| a.ints.clone())
                .unwrap_or(vec![]);
            let strides = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "strides")
                .map(|a| a.ints.clone())
                .unwrap_or(vec![]);
            let bias_arg = if inputs.len() > 2 {
                format!("Some(&{})", inputs[2])
            } else {
                "None".to_string()
            };
            // Determine conv1d vs conv2d based on weight tensor rank
            let weight_name = &ctx.node.input[1];
            let weight_name_s = super::super::sanitize_name(weight_name);
            let weight_rank = ctx
                .known_weights
                .get(&weight_name_s)
                .map(|(_, _, shape, _)| shape.len())
                .unwrap_or(3);
            let kernel_fn = if weight_rank >= 4 { "conv2d" } else { "conv1d" };
            writeln!(
                w,
                "{}let {} = lele::kernels::{}(&{}, &{}, {}, &{:?}, {}, &{:?}, &{:?}, {});",
                tab,
                outputs[0],
                kernel_fn,
                inputs[0],
                inputs[1],
                bias_arg,
                dilations,
                group,
                pads,
                strides,
                buf_expr
            )?;
        }
        "Gemm" => {
            let alpha = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "alpha")
                .map(|a| a.f)
                .unwrap_or(1.0);
            let beta = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "beta")
                .map(|a| a.f)
                .unwrap_or(1.0);
            let trans_a = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "transA")
                .map(|a| a.i)
                .unwrap_or(0)
                != 0;
            let trans_b = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "transB")
                .map(|a| a.i)
                .unwrap_or(0)
                != 0;
            let c = if inputs.len() > 2 && !inputs[2].is_empty() {
                format!("Some(&{})", inputs[2])
            } else {
                "None".to_string()
            };
            writeln!(
                w,
                "{}let {} = lele::kernels::gemm(&{}, &{}, {}, {:?}, {:?}, {}, {}, {});",
                tab, outputs[0], inputs[0], inputs[1], c, alpha, beta, trans_a, trans_b, buf_expr
            )?;
        }
        "LSTM" => {
            let bias = if inputs.len() > 3 && !ctx.node.input[3].is_empty() {
                format!("Some(&{})", inputs[3])
            } else {
                "None".to_string()
            };
            let seq_lens = if inputs.len() > 4 && !ctx.node.input[4].is_empty() {
                format!("Some(&{})", inputs[4])
            } else {
                "None".to_string()
            };
            let initial_h = if inputs.len() > 5 && !ctx.node.input[5].is_empty() {
                format!("Some(&{})", inputs[5])
            } else {
                "None".to_string()
            };
            let initial_c = if inputs.len() > 6 && !ctx.node.input[6].is_empty() {
                format!("Some(&{})", inputs[6])
            } else {
                "None".to_string()
            };
            writeln!(
                w,
                "{}let mut buf_{}_h = Vec::<f32>::new();",
                tab, outputs[0]
            )?;
            writeln!(
                w,
                "{}let mut buf_{}_c = Vec::<f32>::new();",
                tab, outputs[0]
            )?;
            writeln!(
                w,
                "{}let ({}, {}, {}) = lele::kernels::lstm(&{}, &{}, &{}, {}, {}, {}, {}, {}, &mut buf_{}_h, &mut buf_{}_c);",
                tab,
                outputs[0],
                outputs[1],
                outputs[2],
                inputs[0],
                inputs[1],
                inputs[2],
                bias,
                seq_lens,
                initial_h,
                initial_c,
                buf_expr,
                outputs[0],
                outputs[0]
            )?;
        }
        "LayerNormalization" => {
            let epsilon = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "epsilon")
                .map(|a| a.f)
                .unwrap_or(1e-5);
            let axis = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "axis")
                .map(|a| a.i)
                .unwrap_or(-1);
            let scale = if ctx.node.input.len() > 1
                && !ctx.node.input[1].is_empty()
                && !inputs[1].is_empty()
            {
                format!("&{}", inputs[1])
            } else {
                "&lele::tensor::TensorView::empty()".to_string()
            };
            let bias = if ctx.node.input.len() > 2
                && !ctx.node.input[2].is_empty()
                && !inputs[2].is_empty()
            {
                format!("&{}", inputs[2])
            } else {
                "&lele::tensor::TensorView::empty()".to_string()
            };
            if outputs.len() > 1 {
                let fillers = vec!["_"; outputs.len() - 1].join(", ");
                let dummy_tensors =
                    vec!["lele::tensor::TensorView::empty()"; outputs.len() - 1].join(", ");
                writeln!(
                    w,
                    "{}let ({}, {}) = (lele::kernels::layer_norm(&{}, {}, {}, {}, {}, {}), {});",
                    tab,
                    outputs[0],
                    fillers,
                    inputs[0],
                    scale,
                    bias,
                    axis,
                    epsilon,
                    buf_expr,
                    dummy_tensors
                )?;
            } else {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::layer_norm(&{}, {}, {}, {}, {}, {});",
                    tab, outputs[0], inputs[0], scale, bias, axis, epsilon, buf_expr
                )?;
            }
        }
        "ConvInteger" => {
            let dilations = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "dilations")
                .map(|a| a.ints.clone())
                .unwrap_or(vec![]);
            let group = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "group")
                .map(|a| a.i)
                .unwrap_or(1);
            let pads = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "pads")
                .map(|a| a.ints.clone())
                .unwrap_or(vec![]);
            let strides = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "strides")
                .map(|a| a.ints.clone())
                .unwrap_or(vec![]);
            let x_zero_point = if inputs.len() > 2 && !ctx.node.input[2].is_empty() {
                format!("Some(&{})", inputs[2])
            } else {
                "None".to_string()
            };
            let w_zero_point = if inputs.len() > 3 && !ctx.node.input[3].is_empty() {
                format!("Some(&{})", inputs[3])
            } else {
                "None".to_string()
            };
            writeln!(
                w,
                "{}let {} = lele::kernels::conv_integer(&{}, &{}, {}, {}, &{:?}, {}, &{:?}, &{:?}, {});",
                tab,
                outputs[0],
                inputs[0],
                inputs[1],
                x_zero_point,
                w_zero_point,
                dilations,
                group,
                pads,
                strides,
                buf_expr
            )?;
        }
        "BatchNormalization" => {
            let epsilon = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "epsilon")
                .map(|a| a.f)
                .unwrap_or(1e-5);
            writeln!(
                w,
                "{}let {} = lele::kernels::batch_norm(&{}, &{}, &{}, &{}, &{}, {:?}, {});",
                tab,
                outputs[0],
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                epsilon,
                buf_expr
            )?;
        }
        "MaxPool" => {
            let kernel_shape = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "kernel_shape")
                .map(|a| a.ints.clone())
                .unwrap_or(vec![]);
            let strides = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "strides")
                .map(|a| a.ints.clone())
                .unwrap_or(vec![]);
            let pads = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "pads")
                .map(|a| a.ints.clone())
                .unwrap_or(vec![]);
            let dilations = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "dilations")
                .map(|a| a.ints.clone())
                .unwrap_or(vec![]);
            let ceil_mode = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "ceil_mode")
                .map(|a| a.i)
                .unwrap_or(0)
                != 0;
            writeln!(
                w,
                "{}let {} = lele::kernels::max_pool2d(&{}, &{:?}, &{:?}, &{:?}, &{:?}, {}, {});",
                tab,
                outputs[0],
                inputs[0],
                kernel_shape,
                strides,
                pads,
                dilations,
                ceil_mode,
                buf_expr
            )?;
        }
        "Resize" => {
            // ONNX Resize op: inputs are [X, roi, scales] or [X, roi, scales, sizes]
            // For YOLO, we use scales mode with nearest interpolation
            let coord_transform = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "coordinate_transformation_mode")
                .and_then(|a| std::str::from_utf8(&a.s).ok().map(|s| s.to_string()))
                .unwrap_or("half_pixel".to_string());

            // Check if sizes input is provided (input[3])
            let has_sizes = inputs.len() > 3 && !ctx.node.input[3].is_empty();
            // Check if scales input is provided (input[2])
            let has_scales = inputs.len() > 2 && !ctx.node.input[2].is_empty();

            if has_sizes {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::resize_nearest(&{}, None, Some(&{}.data.iter().map(|&v| v as i64).collect::<Vec<_>>()), \"{}\", {});",
                    tab, outputs[0], inputs[0], inputs[3], coord_transform, buf_expr
                )?;
            } else if has_scales {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::resize_nearest(&{}, Some(&{}.data), None, \"{}\", {});",
                    tab, outputs[0], inputs[0], inputs[2], coord_transform, buf_expr
                )?;
            } else {
                panic!("Resize: neither scales nor sizes provided");
            }
        }
        _ => return Ok(false),
    }
    Ok(true)
}
