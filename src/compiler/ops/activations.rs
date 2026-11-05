//! Code generation for activations with parameters, softmax and argmax. The
//! plain ones (`Relu`, `Sigmoid`, ...) are [`Unary`](super::super::ir::Unary)
//! operators.

use super::super::{generate::OpContext, ir::Op};
use std::io::Write;

pub(super) fn generate(ctx: &mut OpContext, w: &mut dyn Write) -> std::io::Result<()> {
    let tab = ctx.tab();
    let inputs = &ctx.inputs;
    let outputs = &ctx.outputs;
    let buf_expr = &ctx.buf_expr;

    match ctx.ir().op {
        Op::LeakyRelu { alpha } => writeln!(
            w,
            "{}let {} = lele::kernels::leaky_relu(&{}, {}, {});",
            tab, outputs[0], inputs[0], alpha, buf_expr
        )?,
        Op::Softmax { axis } | Op::LogSoftmax { axis } => {
            // The opset 13+ default; before 13 it was 1.
            let axis = axis.unwrap_or(-1);
            let kernel = if let Op::Softmax { .. } = ctx.ir().op {
                "softmax"
            } else {
                "log_softmax"
            };
            writeln!(
                w,
                "{}let {} = lele::kernels::{}(&{}, {}, {});",
                tab, outputs[0], kernel, inputs[0], axis, buf_expr
            )?;
        }
        Op::ArgMax { axis, keepdims } => writeln!(
            w,
            "{}let {} = lele::kernels::argmax(&{}, {}, {}, {});",
            tab, outputs[0], inputs[0], axis, keepdims as i64, buf_expr
        )?,
        Op::HardSigmoid { alpha, beta } => writeln!(
            w,
            "{}let {} = lele::kernels::hard_sigmoid(&{}, {:?}, {:?}, {});",
            tab, outputs[0], inputs[0], alpha, beta, buf_expr
        )?,
        ref op => unreachable!("{} is not an activation", op.op_type()),
    }
    Ok(())
}
