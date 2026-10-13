use super::super::{
    generate::OpContext,
    ir::{DType, Op},
};
use std::io::Write;

/// Compute explicit pads from auto_pad attribute for SAME_UPPER/SAME_LOWER.
/// Returns Some([pad_top, pad_left, pad_bottom, pad_right]) or None.
fn compute_same_pads(
    kernel_shape: &[i64],
    strides: &[i64],
    dilations: &[i64],
    auto_pad: &str,
) -> Option<Vec<i64>> {
    if auto_pad == "NOTSET" || auto_pad.is_empty() {
        return None;
    }
    let kh = kernel_shape.first().copied().unwrap_or(1);
    let kw = kernel_shape.get(1).copied().unwrap_or(kh);
    let sh = strides.first().copied().unwrap_or(1);
    let sw = strides.get(1).copied().unwrap_or(sh);
    let dh = dilations.first().copied().unwrap_or(1);
    let dw = dilations.get(1).copied().unwrap_or(dh);

    let eff_kh = (kh - 1) * dh + 1;
    let eff_kw = (kw - 1) * dw + 1;

    // For SAME padding with stride 1, pad_total = effective_kernel - 1
    // For stride > 1, we can't know exact padding without input size,
    // but SAME_UPPER typically uses pad_total = effective_kernel - 1 for stride 1
    // For general stride, ONNX spec: output = ceil(input / stride)
    // pad_total = max((output-1)*stride + eff_kernel - input, 0)
    // Since input is dynamic, we use the stride-1 formula as approximation
    // which works for the common case
    let pad_total_h = eff_kh - 1;
    let pad_total_w = eff_kw - 1;

    let (pad_top, pad_bottom) = if auto_pad == "SAME_LOWER" {
        (pad_total_h - pad_total_h / 2, pad_total_h / 2)
    } else {
        (pad_total_h / 2, pad_total_h - pad_total_h / 2)
    };
    let (pad_left, pad_right) = if auto_pad == "SAME_LOWER" {
        (pad_total_w - pad_total_w / 2, pad_total_w / 2)
    } else {
        (pad_total_w / 2, pad_total_w - pad_total_w / 2)
    };

    // For stride > 1, we need to account for the extra padding needed
    // For SAME, pad_total = (output-1)*stride + eff_kernel - input
    // where output = ceil(input/stride). For dynamic input, we use:
    // pad_total_h and pad_total_w are for stride 1.
    // For stride > 1, the pad depends on input size which we don't have at compile time.
    // However, the kernel functions handle this by using the explicit pads we provide.
    // For stride=1, this is exact. For stride>1, the user should use explicit pads.
    let _ = (sh, sw);

    Some(vec![pad_top, pad_left, pad_bottom, pad_right])
}

pub(super) fn generate(ctx: &mut OpContext, w: &mut dyn Write) -> std::io::Result<()> {
    let tab = ctx.tab();
    let inputs = &ctx.inputs;
    let outputs = &ctx.outputs;
    let buf_expr = &ctx.buf_expr;

    match &ctx.ir().op {
        Op::Conv(conv) => {
            let dilations = conv.dilations.clone().unwrap_or_default();
            let group = conv.group.unwrap_or(1);
            let auto_pad = conv.auto_pad.as_deref().unwrap_or("NOTSET");
            let kernel_shape: Vec<i64> = ctx
                .weight(1)
                .and_then(|weight| {
                    let shape = &weight.shape;
                    if shape.len() >= 4 {
                        Some(vec![shape[2] as i64, shape[3] as i64])
                    } else if shape.len() == 3 {
                        Some(vec![shape[2] as i64])
                    } else {
                        None
                    }
                })
                .unwrap_or_default();
            let pads = if auto_pad != "NOTSET" && !auto_pad.is_empty() {
                compute_same_pads(&kernel_shape, &[1, 1], &dilations, auto_pad).unwrap_or_default()
            } else {
                conv.pads.clone().unwrap_or_default()
            };
            let strides = conv.strides.clone().unwrap_or_default();
            let bias_arg = if ctx.ir().input(2).is_some() {
                format!("Some(&{})", inputs[2])
            } else {
                "None".to_string()
            };
            // An f16 pointwise conv1d (the case `conv1d` computes as a matrix
            // product) takes its weights as stored, widening them as it reads them.
            let ones = |v: &[i64]| v.iter().all(|&x| x == 1);
            if let Some(weight) = ctx.weight(1)
                && weight.dtype == DType::F16
                && matches!(weight.shape[..], [_, in_channels, 1] if in_channels > 1)
                && group == 1
                && ones(&strides)
                && ones(&dilations)
                && pads.iter().all(|&p| p == 0)
            {
                return writeln!(
                    w,
                    "{}let {} = lele::kernels::conv1d_pointwise(&{}, &{}, {}, false, {});",
                    tab,
                    outputs[0],
                    inputs[0],
                    ctx.matmul_weight(1),
                    bias_arg,
                    buf_expr
                );
            }
            // Determine conv1d vs conv2d based on weight tensor rank
            let weight_rank = ctx.weight(1).map(|weight| weight.shape.len()).unwrap_or(3);
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
        Op::Gemm {
            alpha,
            beta,
            trans_a,
            trans_b,
        } => {
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
        Op::Lstm => {
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
        Op::Gru {
            linear_before_reset,
        } => {
            let bias = if inputs.len() > 3 && !ctx.node.input[3].is_empty() {
                format!("Some(&{})", inputs[3])
            } else {
                "None".to_string()
            };
            let initial_h = if inputs.len() > 5 && !ctx.node.input[5].is_empty() {
                format!("Some(&{})", inputs[5])
            } else {
                "None".to_string()
            };
            writeln!(
                w,
                "{}let mut buf_{}_h = Vec::<f32>::new();",
                tab, outputs[0]
            )?;
            if outputs.len() >= 2 {
                writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, outputs[0])?;
                writeln!(
                    w,
                    "{}let ({}, {}) = lele::kernels::gru(&{}, &{}, &{}, {}, {}, {}, &mut buf_{}, &mut buf_{}_h);",
                    tab,
                    outputs[0],
                    outputs[1],
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    bias,
                    initial_h,
                    linear_before_reset,
                    outputs[0],
                    outputs[0]
                )?;
            } else {
                writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, outputs[0])?;
                writeln!(
                    w,
                    "{}let ({}, _) = lele::kernels::gru(&{}, &{}, &{}, {}, {}, {}, &mut buf_{}, &mut buf_{}_h);",
                    tab,
                    outputs[0],
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    bias,
                    initial_h,
                    linear_before_reset,
                    outputs[0],
                    outputs[0]
                )?;
            }
        }
        Op::LayerNormalization { axis, epsilon } => {
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
        Op::ConvInteger(conv) => {
            let dilations = conv.dilations.clone().unwrap_or_default();
            let group = conv.group.unwrap_or(1);
            let pads = conv.pads.clone().unwrap_or_default();
            let strides = conv.strides.clone().unwrap_or_default();
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
            // A static weight is prepared once, on first use.
            match ctx.weight(1) {
                Some(weight)
                    if matches!(weight.dtype, DType::U8 | DType::I8) && weight.shape.len() == 4 =>
                {
                    writeln!(
                        w,
                        "{}let {} = self.conv_integer_packed(&{}, {}, {}, {:?}, {}, {}, {}, {}, &{:?}, &{:?}, &{:?}, {});",
                        tab,
                        outputs[0],
                        inputs[0],
                        weight.offset,
                        weight.len,
                        weight.shape,
                        weight.dtype == DType::I8,
                        group,
                        x_zero_point,
                        w_zero_point,
                        dilations,
                        pads,
                        strides,
                        buf_expr
                    )?
                }
                _ => writeln!(
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
                )?,
            }
        }
        Op::BatchNormalization { epsilon } => {
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
        Op::MaxPool(pool) => {
            let kernel_shape = pool.kernel_shape.clone().unwrap_or_default();
            let strides = pool.strides.clone().unwrap_or_default();
            let auto_pad = pool.auto_pad.as_deref().unwrap_or("NOTSET");
            let dilations = pool.dilations.clone().unwrap_or_default();
            let pads = if auto_pad != "NOTSET" && !auto_pad.is_empty() {
                compute_same_pads(&kernel_shape, &strides, &dilations, auto_pad).unwrap_or_default()
            } else {
                pool.pads.clone().unwrap_or_default()
            };
            let ceil_mode = pool.ceil_mode;
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
        Op::Resize {
            coordinate_transformation_mode,
            ..
        } => {
            // ONNX Resize op: inputs are [X, roi, scales] or [X, roi, scales, sizes]
            // For YOLO, we use scales mode with nearest interpolation
            let coord_transform = coordinate_transformation_mode
                .as_deref()
                .unwrap_or("half_pixel");

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
        Op::GlobalAveragePool => {
            writeln!(
                w,
                "{}let {} = lele::kernels::global_average_pool(&{}, {});",
                tab, outputs[0], inputs[0], buf_expr
            )?;
        }
        Op::AveragePool {
            pool,
            count_include_pad,
        } => {
            let kernel_shape = pool.kernel_shape.clone().unwrap_or_default();
            let strides = pool.strides.clone().unwrap_or_default();
            let pads = pool.pads.clone().unwrap_or_default();
            writeln!(
                w,
                "{}let {} = lele::kernels::average_pool2d(&{}, &{:?}, &{:?}, &{:?}, {}, {});",
                tab,
                outputs[0],
                inputs[0],
                kernel_shape,
                strides,
                pads,
                count_include_pad,
                buf_expr
            )?;
        }
        Op::ConvTranspose(conv) => {
            let dilations = conv.dilations.clone().unwrap_or_default();
            let group = conv.group.unwrap_or(1);
            let pads = conv.pads.clone().unwrap_or_default();
            let strides = conv.strides.clone().unwrap_or_default();
            let bias_arg = if inputs.len() > 2 {
                format!("Some(&{})", inputs[2])
            } else {
                "None".to_string()
            };
            writeln!(
                w,
                "{}let {} = lele::kernels::conv_transpose(&{}, &{}, {}, &{:?}, {}, &{:?}, &{:?}, {});",
                tab,
                outputs[0],
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
        op => unreachable!("{} is not a neural network operator", op.op_type()),
    }
    Ok(())
}
