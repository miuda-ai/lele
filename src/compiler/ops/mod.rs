use super::generate::OpContext;
use super::ir::Op;
use std::io::Write;

pub mod activations;
pub mod control_flow;
pub mod fused;
pub mod math;
pub mod nn;
pub mod tensor;

/// Generates the node of `ctx` with lele's kernels. Returns `false` for an
/// operator lele has no code for: one without a typed variant, or `Loop`.
pub(crate) fn dispatch_builtin(ctx: &mut OpContext, w: &mut dyn Write) -> std::io::Result<bool> {
    match &ctx.ir().op {
        Op::Binary(_)
        | Op::Clip
        | Op::CumSum { .. }
        | Op::Einsum { .. }
        | Op::MatMul
        | Op::MatMulInteger
        | Op::Max
        | Op::Range
        | Op::Reduce { .. }
        | Op::Stft
        | Op::Unary(_) => math::generate(ctx, w)?,
        Op::AveragePool { .. }
        | Op::BatchNormalization { .. }
        | Op::Conv(_)
        | Op::ConvInteger(_)
        | Op::ConvTranspose(_)
        | Op::Gemm { .. }
        | Op::GlobalAveragePool
        | Op::Gru { .. }
        | Op::LayerNormalization { .. }
        | Op::Lstm
        | Op::MaxPool(_)
        | Op::Resize { .. } => nn::generate(ctx, w)?,
        Op::Cast { .. }
        | Op::Concat { .. }
        | Op::Constant { .. }
        | Op::ConstantOfShape { .. }
        | Op::ConstantOfShapeTo { .. }
        | Op::DequantizeLinear(_)
        | Op::DynamicQuantizeLinear
        | Op::Expand
        | Op::ExpandTo { .. }
        | Op::FakeQuantizeLinear(_)
        | Op::Flatten { .. }
        | Op::Gather { .. }
        | Op::GatherElements { .. }
        | Op::Identity
        | Op::Pad { .. }
        | Op::QLinearMatMulI8
        | Op::QuantizeLinear(_)
        | Op::Reshape
        | Op::ReshapeTo { .. }
        | Op::Shape { .. }
        | Op::Size
        | Op::Slice
        | Op::Split { .. }
        | Op::Squeeze { .. }
        | Op::Tile
        | Op::TopK { .. }
        | Op::Transpose { .. }
        | Op::Unsqueeze { .. }
        | Op::Where => tensor::generate(ctx, w)?,
        Op::ArgMax { .. }
        | Op::HardSigmoid { .. }
        | Op::LeakyRelu { .. }
        | Op::LogSoftmax { .. }
        | Op::Softmax { .. } => activations::generate(ctx, w)?,
        Op::If { .. } => control_flow::generate(ctx, w)?,
        Op::ConvAct { .. }
        | Op::EmbeddingConcat { .. }
        | Op::Gelu { .. }
        | Op::Linear
        | Op::QLinearDynamic { .. }
        | Op::ScaleNorm { .. }
        | Op::Silu => fused::generate(ctx, w)?,
        Op::Loop { .. } | Op::Onnx { .. } => return Ok(false),
    }
    Ok(true)
}
