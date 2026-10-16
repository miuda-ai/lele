//! Code generation for the operators the fusion passes create.

use super::super::{
    generate::{ElemType, OpContext},
    ir::{Activation, DType, Op},
};
use std::io::Write;

/// `let {out} = {product};`, `product(buf)` being the call that writes the
/// product to `buf`; with an activation, the call goes through
/// `lele::kernels::activated`, which applies it in place.
pub(super) fn write_product(
    w: &mut dyn Write,
    tab: &str,
    out: &str,
    buf: &str,
    act: Option<Activation>,
    product: impl Fn(&str) -> String,
) -> std::io::Result<()> {
    match act {
        None => writeln!(w, "{tab}let {out} = {};", product(buf)),
        Some(act) => writeln!(
            w,
            "{tab}let {out} = lele::kernels::activated(lele::kernels::Act::{}, {buf}, |buf| {});",
            act.name(),
            product("buf")
        ),
    }
}

/// An activation as the kernels take it: `None` or `Some(lele::kernels::Act::…)`.
pub(super) fn act_arg(act: Option<Activation>) -> String {
    match act {
        None => "None".to_string(),
        Some(act) => format!("Some(lele::kernels::Act::{})", act.name()),
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
        // `{:?}` writes each float so that it reads back the same.
        Op::ScaleNorm { scale, epsilon } => writeln!(
            w,
            "{tab}let {} = lele::kernels::scale_norm(&{}, &{}, {scale:?}, {epsilon:?}, {buf});",
            outputs[0], inputs[0], inputs[1]
        )?,
        Op::ScaleNormPair {
            scale,
            epsilon,
            next_scale,
            next_epsilon,
        } => {
            // Each result into its workspace buffer, or one local to the chunk.
            let mut bufs = Vec::new();
            for (raw, out) in ctx.node.output.iter().zip(outputs) {
                bufs.push(
                    match ctx.allocator.and_then(|a| a.tensor_to_buffer.get(raw)) {
                        Some(idx) => format!("&mut ws.buf_{idx}"),
                        None => {
                            writeln!(w, "{tab}let mut buf_{out} = Vec::<f32>::new();")?;
                            format!("&mut buf_{out}")
                        }
                    },
                );
            }
            writeln!(
                w,
                "{tab}let ({}, {}) = lele::kernels::scale_norm_pair(&{}, &{}, {scale:?}, {epsilon:?}, &{}, {next_scale:?}, {next_epsilon:?}, {}, {});",
                outputs[0], outputs[1], inputs[0], inputs[1], inputs[2], bufs[0], bufs[1]
            )?;
        }
        Op::RmsNorm { epsilon } => writeln!(
            w,
            "{tab}let {} = lele::kernels::rms_norm(&{}, &{}, -1, {epsilon:?}, {buf});",
            outputs[0], inputs[0], inputs[1]
        )?,
        Op::Snake => writeln!(
            w,
            "{tab}let {} = lele::kernels::snake(&{}, &{}, &{}, {buf});",
            outputs[0], inputs[0], inputs[1], inputs[2]
        )?,
        Op::SimAm {
            axes,
            lambda,
            scale,
            shift,
        } => writeln!(
            w,
            "{tab}let {} = lele::kernels::simam(&{}, &{}, {axes}, {lambda:?}, {scale:?}, {shift:?}, {buf});",
            outputs[0], inputs[0], inputs[1]
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
        // The product kernels apply the activation as they write the output.
        Op::Linear { act } => {
            let (x, weight, out) = (&inputs[0], ctx.matmul_weight(1), &outputs[0]);
            let call = match (ctx.ir().input(2), act) {
                (Some(_), None) => format!("self.linear(&{x}, &{weight}, &{}", inputs[2]),
                (Some(_), Some(_)) => format!(
                    "self.linear_act(&{x}, &{weight}, &{}, {}",
                    inputs[2],
                    act_arg(*act)
                ),
                (None, None) => format!("lele::kernels::matmul(&{x}, &{weight}"),
                (None, Some(_)) => format!(
                    "lele::kernels::matmul_act(&{x}, &{weight}, {}",
                    act_arg(*act)
                ),
            };
            writeln!(w, "{tab}let {out} = {call}, {buf});")?
        }
        Op::SwiGlu => writeln!(
            w,
            "{tab}let {} = lele::kernels::linear_swiglu(&{}, &{}, &{}, {buf});",
            outputs[0],
            inputs[0],
            ctx.matmul_weight(1),
            inputs[2]
        )?,
        Op::ConvAct { conv, act } => super::nn::write_conv(ctx, w, conv, Some(*act))?,
        Op::QLinearDynamic { act } => {
            let (x, scale, zero_point, bias) = (&inputs[0], &inputs[2], &inputs[3], &inputs[4]);
            let (out, act) = (&outputs[0], act_arg(*act));
            // A static weight with static scales and zero points is packed for
            // the integer GEMM when the model is created.
            let at_load = (
                ctx.weight_at_load(2, "lele::scalar(1.0f32)"),
                ctx.weight_at_load(3, "lele::scalar(0.0f32)"),
            );
            match (ctx.weight(1), at_load) {
                (Some(weight), (Some(scale), Some(zero_point)))
                    if matches!(weight.dtype, DType::U8 | DType::I8) && weight.shape.len() == 2 =>
                {
                    let n = ctx.weights.qweights(format!(
                        "weights.pack_qweights({}, {}, {}, {}, &{zero_point}, &{scale})",
                        weight.bytes(),
                        weight.shape[0],
                        weight.shape[1],
                        weight.dtype == DType::I8,
                    ));
                    writeln!(
                        w,
                        "{tab}let {out} = self.linear_quantized_packed(&{x}, self.qweights({n}), {bias}, {act}, {buf});"
                    )?
                }
                _ => writeln!(
                    w,
                    "{tab}let {out} = self.linear_quantized(&{x}, {}, {scale}, {zero_point}, {bias}, {act}, {buf});",
                    inputs[1]
                )?,
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
                Some(weight) => weight.expr(ty),
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
