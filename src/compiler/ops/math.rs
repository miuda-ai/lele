//! Code generation for elementwise arithmetic, comparisons, reductions and
//! the other numeric operators.

use super::super::generate::{ElemType, OpContext};
use super::super::ir::{Binary, DType, Op, Reduction, Unary};
use std::io::Write;

pub(super) fn generate(ctx: &mut OpContext, w: &mut dyn Write) -> std::io::Result<()> {
    let tab = ctx.tab();
    let inputs = &ctx.inputs;
    let outputs = &ctx.outputs;
    let buf_expr = &ctx.buf_expr;

    match &ctx.ir().op {
        Op::Unary(unary) => writeln!(
            w,
            "{}let {} = lele::kernels::{}(&{}, {});",
            tab,
            outputs[0],
            unary_kernel(*unary),
            inputs[0],
            buf_expr
        )?,
        Op::Binary(binary) => generate_binary(ctx, *binary, w)?,
        Op::IsNaN => writeln!(
            w,
            "{}let {} = lele::kernels::is_nan(&{}, {});",
            tab, outputs[0], inputs[0], buf_expr
        )?,
        Op::MatMul => writeln!(
            w,
            "{}let {} = lele::kernels::matmul(&{}, &{}, {});",
            tab,
            outputs[0],
            inputs[0],
            ctx.matmul_weight(1),
            buf_expr
        )?,
        Op::MatMulInteger => {
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
            // A static B is packed once for the integer GEMM.
            match ctx.weight(1) {
                Some(b) if matches!(b.dtype, DType::U8 | DType::I8) && b.shape.len() == 2 => {
                    writeln!(
                        w,
                        "{}let {} = self.mat_mul_integer_packed(&{}, {}, {}, {}, {}, {}, {}, {}, {});",
                        tab,
                        outputs[0],
                        inputs[0],
                        b.offset,
                        b.len,
                        b.shape[0],
                        b.shape[1],
                        b.dtype == DType::I8,
                        a_zp,
                        b_zp,
                        buf_expr
                    )?;
                }
                _ => writeln!(
                    w,
                    "{}let {} = lele::kernels::mat_mul_integer(&{}, &{}, {}, {}, {});",
                    tab, outputs[0], inputs[0], inputs[1], a_zp, b_zp, buf_expr
                )?,
            }
        }
        Op::Reduce {
            reduction,
            axes,
            keepdims,
        } => {
            let attr_axes = format!("&{:?}", axes.as_deref().unwrap_or_default());
            let axes = match reduction {
                // These kernels take computed axes too.
                Reduction::Sum | Reduction::Prod | Reduction::Max => {
                    if ctx.node.input.len() > 1 && !ctx.node.input[1].is_empty() {
                        match ctx.ints(1) {
                            Some(ints) => format!("&{:?}", ints),
                            None => format!("&lele::kernels::to_i64_vec(&{})", inputs[1]),
                        }
                    } else {
                        attr_axes
                    }
                }
                Reduction::Mean | Reduction::L2 => match ctx.ints(1) {
                    Some(ints) if axes.as_ref().is_none_or(|a| a.is_empty()) => {
                        format!("&{:?}", ints)
                    }
                    _ => attr_axes,
                },
            };
            let kernel = match reduction {
                Reduction::L2 => "reduce_l2",
                Reduction::Max => "reduce_max",
                Reduction::Mean => "reduce_mean",
                Reduction::Prod => "reduce_prod",
                Reduction::Sum => "reduce_sum",
            };
            writeln!(
                w,
                "{}let {} = lele::kernels::{}(&{}, {}, {}, {});",
                tab, outputs[0], kernel, inputs[0], axes, keepdims, buf_expr
            )?;
        }
        Op::Max => {
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
        Op::Range => {
            let is_i64 = (0..3).any(|i| ctx.input_type(i) == ElemType::I64);
            if is_i64 {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::range_i64(&{}, &{}, &{}, {});",
                    tab, outputs[0], inputs[0], inputs[1], inputs[2], buf_expr
                )?;
            } else {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::range(&{}, &{}, &{}, {});",
                    tab, outputs[0], inputs[0], inputs[1], inputs[2], buf_expr
                )?;
            }
        }
        Op::Clip => {
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
        Op::Stft => {
            // ONNX STFT: inputs[0]=signal, inputs[1]=frame_step, inputs[2]=window, inputs[3]=frame_length
            let frame_step = ctx.ints(1).map_or(160, |ints| ints[0] as usize);
            let n_fft = ctx.ints(3).map_or(512, |ints| ints[0] as usize);
            let window = if ctx.node.input.len() > 2 && !ctx.node.input[2].is_empty() {
                if let Some(window) = ctx.weight(2) {
                    let (o, l, s) = (window.offset, window.len, &window.shape);
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
        Op::CumSum { exclusive, reverse } => {
            let axis = ctx.ints(1).map_or(0, |ints| ints[0]);
            writeln!(
                w,
                "{}let {} = lele::kernels::cumsum(&{}, {}, {}, {}, {});",
                tab, outputs[0], inputs[0], axis, exclusive, reverse, buf_expr
            )?;
        }
        Op::Einsum { equation } => {
            let normalized = equation.replace(' ', "");
            if normalized != "bs,d->bsd" {
                writeln!(w, "{}// Unsupported Einsum: {}", tab, normalized)?;
            }
            writeln!(
                w,
                "{}let {} = lele::kernels::einsum_bs_d_bsd(&{}, &{}, {});",
                tab, outputs[0], inputs[0], inputs[1], buf_expr
            )?;
        }
        op => unreachable!("{} is not a math operator", op.op_type()),
    }
    Ok(())
}

/// The kernel computing `op`, which takes the operand and the output buffer.
fn unary_kernel(op: Unary) -> &'static str {
    match op {
        Unary::Ceil => "ceil",
        Unary::Cos => "cos",
        Unary::Erf => "erf",
        Unary::Exp => "exp",
        Unary::Floor => "floor",
        Unary::Log => "log",
        Unary::Neg => "neg",
        Unary::Not => "not",
        Unary::Reciprocal => "reciprocal",
        Unary::Relu => "relu",
        Unary::Round => "round",
        Unary::Sigmoid => "sigmoid",
        Unary::Sin => "sin",
        Unary::Softplus => "softplus",
        Unary::Sqrt => "sqrt",
        Unary::Tanh => "tanh_kernel",
    }
}

fn generate_binary(ctx: &OpContext, op: Binary, w: &mut dyn Write) -> std::io::Result<()> {
    let tab = ctx.tab();
    let (a, b) = (&ctx.inputs[0], &ctx.inputs[1]);
    let (out, buf_expr) = (&ctx.outputs[0], &ctx.buf_expr);
    let types = ctx.var_types;
    let any_i64 = ctx.input_type(0) == ElemType::I64 || ctx.input_type(1) == ElemType::I64;
    let kernel = match op {
        Binary::Add => "add",
        Binary::And => "logical_and",
        Binary::Mod if types.is_i64(out) => "mod_i64",
        Binary::Mod => "mod_f32",
        Binary::Mul => "mul",
        Binary::Or => "logical_or",
        Binary::Pow => "pow",
        Binary::PRelu => "prelu",
        Binary::Sub => "sub",
        Binary::Xor => "logical_xor",
        Binary::Div => {
            // ONNX Div takes operands of one type, so an integer initializer
            // makes it an integer division. When it runs in f32 (e.g. on the
            // f32 indices of TopK) the quotient still has to be truncated.
            let integer_in_f32 = !types.is_i64(out)
                && (0..ctx.node.input.len()).any(|i| {
                    ctx.weight(i)
                        .is_some_and(|w| matches!(w.dtype, DType::I32 | DType::I64))
                });
            if integer_in_f32 { "div_trunc" } else { "div" }
        }
        Binary::Less if any_i64 => "less_i64",
        Binary::Less => "less",
        Binary::LessOrEqual if any_i64 => "less_or_equal_i64",
        Binary::LessOrEqual => "less_or_equal",
        Binary::GreaterOrEqual if any_i64 => "greater_or_equal_i64",
        Binary::GreaterOrEqual => "greater_or_equal",
        // a > b is b < a.
        Binary::Greater => {
            let kernel = if any_i64 { "less_i64" } else { "less" };
            return writeln!(
                w,
                "{}let {} = lele::kernels::{}(&{}, &{}, {});",
                tab, out, kernel, b, a, buf_expr
            );
        }
        Binary::Equal => {
            let (type_a, type_b) = (ctx.input_type(0), ctx.input_type(1));
            if !(types.is_i64(out) || type_a == ElemType::I64) {
                "equal"
            } else {
                let a_is_weight_f32 = a.contains("weight_f32");
                let b_is_weight_f32 = b.contains("weight_f32");
                if b_is_weight_f32 {
                    if type_a == ElemType::I64 {
                        "equal_i64_f32_r_i64"
                    } else {
                        "equal_i64_f32_r"
                    }
                } else if a_is_weight_f32 || type_a == ElemType::F32 && type_b == ElemType::I64 {
                    "equal_i64_f32_lhs"
                } else {
                    "equal_i64"
                }
            }
        }
    };
    writeln!(
        w,
        "{}let {} = lele::kernels::{}(&{}, &{}, {});",
        tab, out, kernel, a, b, buf_expr
    )
}
