//! Code generation for the operators the fusion passes create.

use super::super::{
    generate::{ElemType, OpContext, weight_expr},
    ir::{Activation, ConvAttrs, DType, Op},
};
use std::io::Write;

impl ConvAttrs {
    /// Strides and dilations with one entry per spatial axis of a 2D
    /// convolution (a single value covers both, none means 1); each axis
    /// keeps its own, so a stride of `[2, 1]` stays `[2, 1]`.
    fn per_axis(ints: &Option<Vec<i64>>) -> Vec<i64> {
        match ints.as_deref() {
            None | Some([]) => vec![1, 1],
            Some([v]) => vec![*v, *v],
            Some(ints) => ints.to_vec(),
        }
    }
}

pub(super) fn generate(ctx: &mut OpContext, w: &mut dyn Write) -> std::io::Result<()> {
    let tab = ctx.tab();
    let (inputs, outputs, buf) = (&ctx.inputs, &ctx.outputs, &ctx.buf_expr);
    match &ctx.ir().op {
        Op::Silu => writeln!(
            w,
            "{tab}let {} = lele::kernels::silu(&{}, {buf});",
            outputs[0], inputs[0]
        )?,
        Op::Gelu { approximate } => {
            let kernel = match approximate.as_deref() {
                None | Some("none") => "gelu_erf",
                Some("tanh") => "fast_gelu",
                Some(other) => panic!("Gelu approximate={other:?} is not supported"),
            };
            writeln!(
                w,
                "{tab}let {} = lele::kernels::{kernel}(&{}, {buf});",
                outputs[0], inputs[0]
            )?;
        }
        Op::Linear => writeln!(
            w,
            "{tab}let {} = self.linear(&{}, &{}, &{}, {buf});",
            outputs[0], inputs[0], inputs[1], inputs[2]
        )?,
        Op::ConvAct { conv, act } => {
            let weight_shape = ctx
                .weight(1)
                .map(|weight| weight.shape.clone())
                .unwrap_or(vec![3, 3, 3, 3]);
            let bias = match ctx.ir().input(2) {
                Some(_) => format!("Some(&{})", inputs[2]),
                None => "None".to_string(),
            };
            if *act == Activation::Silu {
                // Attributes as written, with ONNX defaults.
                let ints = |v: &Option<Vec<i64>>, default: &[i64]| {
                    v.clone().unwrap_or_else(|| default.to_vec())
                };
                writeln!(
                    w,
                    "{tab}let {} = lele::kernels::conv2d_silu(&{}, &{}, {bias}, &{:?}, {}, &{:?}, &{:?}, {buf});",
                    outputs[0],
                    inputs[0],
                    inputs[1],
                    ints(&conv.dilations, &[1, 1]),
                    conv.group.unwrap_or(1),
                    ints(&conv.pads, &[0, 0, 0, 0]),
                    ints(&conv.strides, &[1, 1]),
                )?;
                return Ok(());
            }
            let strides = ConvAttrs::per_axis(&conv.strides);
            let dilations = ConvAttrs::per_axis(&conv.dilations);
            let groups = conv.group.unwrap_or(1) as usize;
            let mut pads = conv.pads.clone().unwrap_or_default();
            let auto_pad = conv.auto_pad.as_deref().unwrap_or("");
            if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") && pads.is_empty() {
                let kh = weight_shape.get(2).copied().unwrap_or(1) as i64;
                let kw = weight_shape.get(3).copied().unwrap_or(kh as usize) as i64;
                let total_h = (kh - 1) * dilations[0];
                let total_w = (kw - 1) * dilations[1];
                let split = |total: i64| {
                    if auto_pad == "SAME_LOWER" {
                        (total - total / 2, total / 2)
                    } else {
                        (total / 2, total - total / 2)
                    }
                };
                let ((pt, pb), (pl, pr)) = (split(total_h), split(total_w));
                pads = vec![pt, pl, pb, pr];
            }
            if pads.is_empty() {
                pads = vec![0, 0, 0, 0];
            }
            if weight_shape.len() >= 4 {
                writeln!(
                    w,
                    "{tab}let {} = lele::kernels::conv2d_fused(&{}, &{}, {bias}, &{:?}, {groups}, &{:?}, &{:?}, true, {buf});",
                    outputs[0], inputs[0], inputs[1], dilations, pads, strides,
                )?;
            } else {
                writeln!(
                    w,
                    "{tab}let {} = self.conv1d_relu({}, {}, {bias}, {}, {}, {groups}, {}, {buf});",
                    outputs[0], inputs[0], inputs[1], strides[0], dilations[0], pads[0] as usize,
                )?;
            }
        }
        Op::QLinearDynamic { relu } => {
            let (x, scale, zero_point, bias) = (&inputs[0], &inputs[2], &inputs[3], &inputs[4]);
            // A static weight is packed once for the integer GEMM.
            match ctx.weight(1) {
                Some(weight)
                    if matches!(weight.dtype, DType::U8 | DType::I8) && weight.shape.len() == 2 =>
                {
                    writeln!(
                        w,
                        "{tab}let {} = self.linear_quantized_packed(&{x}, {}, {}, {}, {}, {}, {scale}, {zero_point}, {bias}, {relu}, {buf});",
                        outputs[0],
                        weight.offset,
                        weight.len,
                        weight.shape[0],
                        weight.shape[1],
                        weight.dtype == DType::I8,
                    )?
                }
                _ => {
                    let method = if *relu {
                        "linear_quantized_relu"
                    } else {
                        "linear_quantized"
                    };
                    writeln!(
                        w,
                        "{tab}let {} = self.{method}(&{x}, {}, {scale}, {zero_point}, {bias}, {buf});",
                        outputs[0], inputs[1],
                    )?
                }
            }
        }
        Op::EmbeddingConcat { value } => {
            let int_value = value.as_ref().and_then(|t| {
                let int = matches!(DType::from_onnx(t.data_type), DType::I32 | DType::I64)
                    || !t.int64_data.is_empty()
                    || !t.int32_data.is_empty();
                int.then(|| {
                    t.int64_data
                        .first()
                        .copied()
                        .or(t.int32_data.first().map(|&v| v as i64))
                        .unwrap_or(0)
                })
            });
            // Weights are read as the type of the fill value.
            let operand = |i: usize, ty: ElemType| match ctx.weight(i) {
                Some(weight) => weight_expr(weight, ty),
                None => inputs[i].clone(),
            };
            let shape = operand(0, ElemType::I64);
            match int_value {
                Some(v) => writeln!(
                    w,
                    "{tab}let {} = self.embedding_concat_i64(&{shape}, {v}, {}, {buf});",
                    outputs[0],
                    operand(1, ElemType::I64),
                )?,
                None => {
                    let v = value
                        .as_ref()
                        .and_then(|t| t.float_data.first())
                        .copied()
                        .unwrap_or(0.0);
                    writeln!(
                        w,
                        "{tab}let {} = self.embedding_concat(&{shape}, {v:?}, {}, {buf});",
                        outputs[0],
                        operand(1, ElemType::F32),
                    )?;
                }
            }
        }
        op => unreachable!("{} is not a fused operator", op.op_type()),
    }
    Ok(())
}
