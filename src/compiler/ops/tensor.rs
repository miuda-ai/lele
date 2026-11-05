//! Code generation for shape and layout operators, casts, constants and
//! quantization.

use super::super::{
    generate::{ElemType, OpContext, weight_expr},
    ir::{DType, Op, TargetDim},
};
use crate::compiler::passes::{self, ACT_REQUANT, I8_BIAS, I8_REQUANT};
use crate::model::onnx_proto::TensorProto;
use std::io::Write;

pub(super) fn generate(ctx: &mut OpContext, w: &mut dyn Write) -> std::io::Result<()> {
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
            if let Some(ints) = ctx.ints(idx) {
                Ok(format!("&{:?}", ints))
            } else {
                if ctx.input_type(idx) == ElemType::I64 {
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
            if let Some(ints) = ctx.ints(idx) {
                Ok(format!("Some(&{:?})", ints))
            } else {
                if ctx.input_type(idx) == ElemType::I64 {
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

    match &ctx.ir().op {
        Op::Transpose { perm } => {
            let perm = perm.as_deref().unwrap_or_default();
            writeln!(
                w,
                "{}let {} = lele::kernels::transpose(&{}, &{:?}, {});",
                tab, outputs[0], inputs[0], perm, buf_expr
            )?;
        }
        Op::Reshape => {
            // A 0 in the target copies the input's dimension, which the kernel reads.
            let shape = resolve_i64_with_temp(1, ctx, w, &tab)?;
            writeln!(
                w,
                "{}let {} = lele::kernels::reshape(&{}, {});",
                tab, outputs[0], inputs[0], shape
            )?;
        }
        Op::ReshapeTo { dims } => {
            writeln!(
                w,
                "{}let {} = lele::kernels::reshape(&{}, &[{}]);",
                tab,
                outputs[0],
                inputs[0],
                target(dims, inputs, " as i64")
            )?;
        }
        Op::ExpandTo { dims } => {
            writeln!(
                w,
                "{}let {} = lele::kernels::expand(&{}, &[{}], {});",
                tab,
                outputs[0],
                inputs[0],
                target(dims, inputs, " as i64"),
                buf_expr
            )?;
        }
        Op::ConstantOfShapeTo { value, dims } => {
            writeln!(
                w,
                "{}let {} = lele::kernels::full(&[{}], {}, {});",
                tab,
                outputs[0],
                target(dims, inputs, ""),
                fill_value(value, ctx.var_types.is_i64(&outputs[0])),
                buf_expr
            )?;
        }
        Op::Flatten { axis } => {
            // Flatten: reshape to 2D [batch_size, -1]
            // axis parameter determines where to flatten (default is 1)
            let axis = *axis as usize;
            writeln!(
                w,
                "{}let {} = lele::kernels::flatten(&{}, {});",
                tab, outputs[0], inputs[0], axis
            )?;
        }
        Op::Unsqueeze { axes } => {
            let axes = if ctx.ir().input(1).is_some() {
                resolve_i64_with_temp(1, ctx, w, &tab)?
            } else {
                format!("&{:?}", axes.as_deref().unwrap_or_default())
            };
            writeln!(
                w,
                "{}let {} = lele::kernels::unsqueeze(&{}, {});",
                tab, outputs[0], inputs[0], axes
            )?;
        }
        Op::Squeeze { axes } => {
            let axes = match axes {
                _ if ctx.ir().input(1).is_some() => resolve_i64_opt_with_temp(1, ctx, w, &tab)?,
                Some(axes) => format!("Some(&{axes:?})"),
                None => "None".to_string(),
            };
            writeln!(
                w,
                "{}let {} = lele::kernels::squeeze(&{}, {});",
                tab, outputs[0], inputs[0], axes
            )?;
        }
        Op::Concat { axis } => {
            let args = inputs
                .iter()
                .map(|s| format!("&{}", s))
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(
                w,
                "{}let {} = lele::kernels::concat(&[{}], {}, {});",
                tab, outputs[0], args, axis, buf_expr
            )?;
        }
        Op::Where => {
            writeln!(
                w,
                "{}let {} = lele::kernels::where_op(&{}, &{}, &{}, {});",
                tab, outputs[0], inputs[0], inputs[1], inputs[2], buf_expr
            )?;
        }
        Op::Gather { .. } if let Some(index) = passes::selected_index(ctx.graph, ctx.ir()) => {
            writeln!(
                w,
                "{tab}let {} = lele::kernels::select(&{}, {index});",
                outputs[0], inputs[0]
            )?;
        }
        // An 8-bit table, as a quantized embedding, is read as stored and only
        // the rows taken widened.
        Op::Gather { axis }
            if let Some(table) = ctx.weight(0)
                && let Some(ty) = match table.dtype {
                    DType::U8 => Some("u8"),
                    DType::I8 => Some("i8"),
                    _ => None,
                } =>
        {
            writeln!(
                w,
                "{tab}let {} = lele::kernels::gather_widened(&self.weight_{ty}_stored({}, {}, &{:?}), &{}, {axis}, {buf_expr});",
                outputs[0], table.offset, table.len, table.shape, inputs[1],
            )?;
        }
        Op::Gather { axis } => {
            writeln!(
                w,
                "{}let {} = lele::kernels::gather(&{}, &{}, {}, {});",
                tab, outputs[0], inputs[0], inputs[1], axis, buf_expr
            )?;
        }
        Op::Shape { start, end } => {
            if start.is_none() && end.is_none() {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::shape(&{});",
                    tab, outputs[0], inputs[0]
                )?;
            } else {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::shape_slice(&{}, {}, {:?});",
                    tab,
                    outputs[0],
                    inputs[0],
                    start.unwrap_or(0),
                    end
                )?;
            }
        }
        Op::Size => writeln!(
            w,
            "{}let {} = lele::kernels::size(&{});",
            tab, outputs[0], inputs[0]
        )?,
        Op::Cast { to } => {
            // The allocator gives a conversion a buffer of the target type that
            // the input is not in, and a view its input's buffer.
            if ctx.input_type(0).cast_is_view(*to) {
                writeln!(
                    w,
                    "{}let {} = {}.clone();",
                    tab, outputs[0], inputs[0]
                )?;
            } else if let DType::I64 | DType::I32 | DType::Bool = to {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::utils::cast_to_i64(&{}, {});",
                    tab, outputs[0], inputs[0], buf_expr
                )?;
            } else {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::utils::cast_to_f32(&{}, {});",
                    tab, outputs[0], inputs[0], buf_expr
                )?;
            }
        }
        Op::ConstantOfShape { value } => {
            writeln!(
                w,
                "{}let {} = lele::kernels::constant_of_shape(&{}, {}, {});",
                tab,
                outputs[0],
                inputs[0],
                fill_value(value, ctx.var_types.is_i64(&outputs[0])),
                buf_expr
            )?;
        }
        Op::Slice => {
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
        Op::Expand => {
            let shape = resolve_i64_with_temp(1, ctx, w, &tab)?;
            writeln!(
                w,
                "{}let {} = lele::kernels::expand(&{}, {}, {});",
                tab, outputs[0], inputs[0], shape, buf_expr
            )?;
        }
        Op::Tile => {
            let repeats = resolve_i64_with_temp(1, ctx, w, &tab)?;
            writeln!(
                w,
                "{}let {} = lele::kernels::tile(&{}, {}, {});",
                tab, outputs[0], inputs[0], repeats, buf_expr
            )?;
        }
        Op::Split { axis, split } => {
            let splits = if ctx.node.input.len() > 1 && !ctx.node.input[1].is_empty() {
                resolve_i64_with_temp(1, ctx, w, &tab)?
            } else {
                // Without sizes, the kernel splits evenly.
                let split = split.clone().unwrap_or_else(|| vec![0; outputs.len()]);
                format!("&{:?}", split)
            };
            // Each output into its workspace buffer, or one local to the chunk.
            for (i, (raw, out)) in ctx.node.output.iter().zip(outputs).enumerate() {
                let buf = match ctx.allocator.and_then(|a| a.tensor_to_buffer.get(raw)) {
                    Some(idx) => format!("&mut ws.buf_{idx}"),
                    None => {
                        let ty = ctx.var_types.get(out);
                        writeln!(w, "{tab}let mut buf_{out} = Vec::<{ty}>::new();")?;
                        format!("&mut buf_{out}")
                    }
                };
                writeln!(
                    w,
                    "{tab}let {out} = lele::kernels::split_one(&{}, {axis}, {splits}, {i}, {buf});",
                    inputs[0]
                )?;
            }
        }
        Op::Pad { mode } => {
            let mode = mode.as_deref().unwrap_or("constant");
            let constant_value =
                if inputs.len() > 2 && !ctx.node.input[2].is_empty() && !inputs[2].is_empty() {
                    format!("Some(&{})", inputs[2])
                } else {
                    "None".to_string()
                };
            // For pad operation, we need pads as &[i64]
            let pads_expr = if ctx.node.input.len() > 1 && !ctx.node.input[1].is_empty() {
                if let Some(ints) = ctx.ints(1) {
                    format!("&{:?}", ints)
                } else {
                    // pads is a computed tensor, we need to extract data
                    if ctx.input_type(1) == ElemType::I64 {
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
        op @ (Op::QuantizeLinear(attrs)
        | Op::DequantizeLinear(attrs)
        | Op::FakeQuantizeLinear(attrs)) => {
            let (axis, block_size) = (attrs.axis, attrs.block_size);
            let zero_point = ctx.ir().input(2);
            let zp_expr = match zero_point {
                Some(_) => format!("Some(&{})", inputs[2]),
                None => "None".to_string(),
            };
            if let Op::DequantizeLinear(_) = op {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::dequantize_linear(&{}, &{}, {}, {}, {}, {});",
                    tab, outputs[0], inputs[0], inputs[1], zp_expr, axis, block_size, buf_expr
                )?;
            } else {
                // Saturation bounds come from `output_dtype` (opset 21+) or the
                // zero-point dtype; ONNX defaults to uint8 when neither is given.
                let zp_dt = zero_point
                    .map(|zp| ctx.graph.value(zp).dtype)
                    .filter(|&t| t != DType::Undefined);
                let out_dt = match (attrs.output_dtype, zp_dt, zero_point) {
                    (Some(t), _, _) | (None, Some(t), _) => t,
                    (None, None, None) => DType::U8,
                    // Guessing uint8 would silently clamp every negative code
                    // to 0 on an int8 model, so refuse to compile instead.
                    (None, None, Some(zp)) => panic!(
                        "QuantizeLinear '{}': the type of zero point '{}' is unknown and no \
                         output_dtype attribute is present; add output_dtype so the saturation \
                         bounds are known",
                        ctx.node.name,
                        ctx.graph.value(zp).name
                    ),
                };
                let (qmin, qmax) =
                    crate::kernels::quant_range(out_dt.to_onnx()).unwrap_or((0.0, 255.0));
                let kernel = if let Op::FakeQuantizeLinear(_) = op {
                    "fake_quantize_linear"
                } else {
                    "quantize_linear"
                };
                writeln!(
                    w,
                    "{}let {} = lele::kernels::{}(&{}, &{}, {}, {}, {}, {:?}, {:?}, {});",
                    tab,
                    outputs[0],
                    kernel,
                    inputs[0],
                    inputs[1],
                    zp_expr,
                    axis,
                    block_size,
                    qmin,
                    qmax,
                    buf_expr
                )?;
            }
        }
        Op::QLinearMatMulI8 => {
            // Produced by `Compiler::fuse_qdq_matmul`, so the weight is always
            // an int8 initializer, and its scale one too. The runtime packs it
            // into an architecture-specific layout when the model is created.
            let (Some(weight), Some(scale)) = (ctx.weight(3), ctx.weight_at_load(4, "")) else {
                panic!(
                    "QLinearMatMulI8 weight {} or its scale is not a known initializer",
                    ctx.node.input[3]
                );
            };
            let n = ctx.weights.qweights(format!(
                "weights.pack_qweights({}, {}, {}, {}, true, &lele::scalar(0.0f32), &{scale})",
                weight.offset, weight.len, weight.shape[0], weight.shape[1],
            ));
            // `fold_qdq_into_matmuls` may have given it the pair after the
            // product, the bias added after that, or both.
            let requant = requant_expr(ctx, inputs, I8_REQUANT);
            let bias = match ctx.ir().input(I8_BIAS) {
                Some(_) => format!("Some(&{})", inputs[I8_BIAS]),
                None => "None".to_string(),
            };
            writeln!(
                w,
                "{tab}let {} = self.qmatmul_i8(&{}, &{}, &{}, self.qweights({n}), {requant}, {bias}, {buf_expr});",
                outputs[0], inputs[0], inputs[1], inputs[2],
            )?;
        }
        Op::QLinearMatMulAct => {
            // Produced by `fuse_qdq_matmul`: the scales are scalar f32 and the
            // zero points scalar uint8.
            writeln!(
                w,
                "{tab}let {} = lele::kernels::qmatmul_acts(&{}, {}.data[0], {}.data[0] as i32, &{}, {}.data[0], {}.data[0] as i32, {}, {});",
                outputs[0],
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
                requant_expr(ctx, inputs, ACT_REQUANT),
                buf_expr
            )?;
        }
        Op::QWeightMatMul => {
            // Produced by `fuse_weight_only_matmul`, so the weight is always a
            // 2-D int8 or uint8 initializer, read from its bytes in place.
            let Some(weight) = ctx.weight(1) else {
                panic!(
                    "QWeightMatMul weight {} is not a known initializer",
                    ctx.node.input[1]
                );
            };
            let zero_point = match ctx.ir().input(3) {
                Some(_) => format!("Some(&{})", inputs[3]),
                None => "None".to_string(),
            };
            writeln!(
                w,
                "{tab}let {} = self.weight_only_matmul(&{}, {}, {}, {}, ({}, {}), &{}, {zero_point}, {buf_expr});",
                outputs[0],
                inputs[0],
                weight.offset,
                weight.len,
                weight.dtype == DType::I8,
                weight.shape[0],
                weight.shape[1],
                inputs[2],
            )?;
        }
        Op::DynamicQuantizeLinear => {
            writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, outputs[0])?;
            writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, outputs[1])?;
            writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, outputs[2])?;
            writeln!(
                w,
                "{}let ({}_ref, {}_ref, {}_ref) = lele::kernels::dynamic_quantize_linear(&{}, &mut buf_{}, &mut buf_{}, &mut buf_{});",
                tab,
                outputs[0],
                outputs[1],
                outputs[2],
                inputs[0],
                outputs[0],
                outputs[1],
                outputs[2]
            )?;
            // Own the outputs to release borrows on local buffers immediately.
            // scale and zero_point are tiny scalars, so copying is cheap.
            // quantized data is large but we need to release the buffer borrow.
            writeln!(
                w,
                "{}let {} = {}_ref.to_owned();",
                tab, outputs[0], outputs[0]
            )?;
            writeln!(
                w,
                "{}let {} = {}_ref.to_owned();",
                tab, outputs[1], outputs[1]
            )?;
            writeln!(
                w,
                "{}let {} = {}_ref.to_owned();",
                tab, outputs[2], outputs[2]
            )?;
        }
        op @ (Op::Identity | Op::Constant { .. }) => {
            if let Some(weight) = ctx.weights.get(&ctx.node.output[0])
                && matches!(weight.dtype, DType::I64 | DType::F32 | DType::F16)
                && weight.scalar.is_some()
            {
                let weight = weight_expr(weight, ElemType::of(weight.dtype));
                writeln!(w, "{}let {} = {};", tab, outputs[0], weight)?;
            } else if let Some(weight) = ctx.weights.get(&ctx.node.output[0]) {
                let weight_fn = match weight.dtype {
                    DType::U8 => "weight_u8",
                    DType::I8 => "weight_i8",
                    DType::I32 => "weight_i32",
                    DType::I64 => "weight_i64",
                    DType::F16 => "weight_f16",
                    _ => "weight_f32",
                };
                writeln!(
                    w,
                    "{}let {} = self.{}({}, {}, &{:?});",
                    tab, outputs[0], weight_fn, weight.offset, weight.len, weight.shape
                )?;
            } else if let Some((ints, shape)) = ctx.weights.int64.get(&ctx.node.output[0]) {
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
            } else if let Op::Identity = op {
                writeln!(w, "{}let {} = {}.clone();", tab, outputs[0], inputs[0])?;
            } else {
                // More complex Constant logic
                let val = match op {
                    Op::Constant { value } => Some(value),
                    _ => None,
                };
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
        Op::TopK {
            axis,
            largest,
            sorted,
        } => {
            // inputs[1] is k (as a tensor). We need its value at codegen time.
            // It's typically a constant weight.
            writeln!(
                w,
                "{}let mut buf_{}_values = Vec::<f32>::new();",
                tab, outputs[0]
            )?;
            writeln!(
                w,
                "{}let mut buf_{}_indices = Vec::<f32>::new();",
                tab, outputs[0]
            )?;
            writeln!(
                w,
                "{}let ({}, {}) = lele::kernels::topk(&{}, {}.data[0] as usize, {}, {}, {}, &mut buf_{}_values, &mut buf_{}_indices);",
                tab,
                outputs[0],
                outputs[1],
                inputs[0],
                inputs[1],
                axis,
                largest,
                sorted,
                outputs[0],
                outputs[0]
            )?;
        }
        Op::GatherElements { axis } => {
            writeln!(
                w,
                "{}let {} = lele::kernels::gather_elements(&{}, &{}, {}, {});",
                tab, outputs[0], inputs[0], inputs[1], axis, buf_expr
            )?;
        }
        Op::ScatterElements { axis } => {
            writeln!(
                w,
                "{}let {} = lele::kernels::scatter_elements(&{}, &{}, &{}, {}, {});",
                tab, outputs[0], inputs[0], inputs[1], inputs[2], axis, buf_expr
            )?;
        }
        Op::ScatterND => {
            // A state, a graph input or what an earlier ScatterND made of one,
            // read for the last time is updated in place when it owns its
            // elements: a state passed in by value and returned is not copied.
            // Other data is in the workspace, and copied into it.
            let ir = ctx.ir();
            let state = ir.input(0).is_some_and(|v| {
                ctx.graph.inputs.contains(&v)
                    || ctx
                        .graph
                        .producer(v)
                        .is_some_and(|p| matches!(p.op, Op::ScatterND))
            });
            if state && ctx.input_dies_here(0) {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::scatter_nd_owned({}, &{}, &{});",
                    tab, outputs[0], inputs[0], inputs[1], inputs[2]
                )?;
            } else {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::scatter_nd(&{}, &{}, &{}, {});",
                    tab, outputs[0], inputs[0], inputs[1], inputs[2], buf_expr
                )?;
            }
        }
        Op::Trilu { upper } => {
            // `k` is optional and defaults to the main diagonal.
            let k = match ctx.ir().input(1) {
                Some(_) => format!("{}.data[0] as i64", inputs[1]),
                None => "0".to_string(),
            };
            writeln!(
                w,
                "{}let {} = lele::kernels::trilu(&{}, {}, {}, {});",
                tab, outputs[0], inputs[0], k, upper, buf_expr
            )?;
        }
        op => unreachable!("{} is not a tensor operator", op.op_type()),
    }
    Ok(())
}

/// The elements of a [`TargetDim`] target, input dimensions cast with
/// `cast`.
/// The `Option<Requant>` of an integer matmul whose inputs `slot` and `slot + 1`
/// are the scale and zero point of the pair after its product, if it has them.
fn requant_expr(ctx: &OpContext, inputs: &[String], slot: usize) -> String {
    match ctx.ir().input(slot) {
        Some(_) => format!(
            "Some(lele::kernels::Requant {{ scale: {}.data[0], zero_point: {}.data[0] as i32 }})",
            inputs[slot],
            inputs[slot + 1]
        ),
        None => "None".to_string(),
    }
}

fn target(dims: &[TargetDim], inputs: &[String], cast: &str) -> String {
    dims.iter()
        .map(|d| match *d {
            TargetDim::Fixed(n) => n.to_string(),
            TargetDim::Of { input, axis } => format!("{}.shape[{axis}]{cast}", inputs[input]),
            TargetDim::Infer => "-1".to_string(),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// The fill value of a `ConstantOfShape` as a literal, 0 by default.
fn fill_value(value: &Option<TensorProto>, is_i64: bool) -> String {
    let val = value
        .as_ref()
        .and_then(|t| crate::model::tensor_to_array(t).ok())
        .and_then(|(data, _)| data.first().copied())
        .unwrap_or(0.0);
    if is_i64 {
        format!("{}i64", val as i64)
    } else if val.is_finite() {
        format!("{val:?}")
    } else if val.is_nan() {
        "f32::NAN".to_string()
    } else if val > 0.0 {
        "f32::INFINITY".to_string()
    } else {
        "f32::NEG_INFINITY".to_string()
    }
}
