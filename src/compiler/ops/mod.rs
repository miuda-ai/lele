use super::generate::OpContext;
use std::io::Write;

pub mod activations;
pub mod control_flow;
pub mod math;
pub mod nn;
pub mod tensor;

pub(crate) fn dispatch_builtin(ctx: &mut OpContext, w: &mut dyn Write) -> std::io::Result<bool> {
    // Try math ops
    if math::handle_math_ops(ctx, w)? {
        return Ok(true);
    }
    // Try nn ops
    if nn::handle_nn_ops(ctx, w)? {
        return Ok(true);
    }
    // Try tensor ops
    if tensor::handle_tensor_ops(ctx, w)? {
        return Ok(true);
    }
    // Try activation ops
    if activations::handle_activation_ops(ctx, w)? {
        return Ok(true);
    }
    // Try control flow
    if control_flow::handle_control_flow_ops(ctx, w)? {
        return Ok(true);
    }

    Ok(false)
}
