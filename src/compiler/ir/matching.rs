//! Helpers for passes matching chains of nodes: lele's fusions, and the
//! passes and pattern matchers given to [`Compiler`](crate::compiler::Compiler).

use super::{DType, Graph, NodeId, Op, ValueId};
use crate::model::onnx_proto::TensorProto;
use crate::model::tensor_to_array;

/// The node producing `v` if `is_match` accepts its operator and `v` has no
/// other reader than the chain being matched.
pub fn sole_producer(
    g: &Graph,
    v: Option<ValueId>,
    is_match: impl Fn(&Op) -> bool,
) -> Option<NodeId> {
    let v = v?;
    let id = g.value(v).producer()?;
    (is_match(&g.node(id).op) && g.uses(v) == 1).then_some(id)
}

/// A compile-time constant: an initializer or a `Constant`.
pub fn constant_tensor(g: &Graph, v: Option<ValueId>) -> Option<&TensorProto> {
    let v = v?;
    match g.data(v) {
        Some(t) => Some(t),
        None => match &g.producer(v)?.op {
            Op::Constant { value } => Some(value),
            _ => None,
        },
    }
}

/// The values of a compile-time constant, float16 ones included.
pub fn constant(g: &Graph, v: Option<ValueId>) -> Option<Vec<f32>> {
    let t = constant_tensor(g, v)?;
    if DType::from_onnx(t.data_type) == DType::F16 {
        return Some(f16_values(t));
    }
    tensor_to_array(t).ok().map(|(data, _)| data)
}

/// The operand of binary node `id` that is not `v`, if `v` is one of them.
pub fn other_operand(g: &Graph, id: NodeId, v: ValueId) -> Option<Option<ValueId>> {
    let node = g.node(id);
    match (node.input(0), node.input(1)) {
        (Some(a), b) if a == v => Some(b),
        (a, Some(b)) if b == v => Some(a),
        _ => None,
    }
}

/// The values of a float16 tensor, which `tensor_to_array` does not read:
/// its bits, raw or one per element of `int32_data`.
pub(crate) fn f16_values(t: &TensorProto) -> Vec<f32> {
    let bits: Vec<u16> = if t.raw_data.is_empty() {
        t.int32_data.iter().map(|&x| x as u16).collect()
    } else {
        t.raw_data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| u16::from_le_bytes(b))
            .collect()
    };
    bits.into_iter()
        .map(|b| half::f16::from_bits(b).to_f32())
        .collect()
}
