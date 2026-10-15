//! Graph rewrites on the [`ir`](super::ir).

mod branches;
mod cse;
mod dce;
mod fold;
mod fuse;
mod pack;
mod qdq;
mod shapes;
mod simplify;
mod types;

pub(crate) use branches::inline_constant_ifs;
pub(crate) use cse::eliminate_common_subexpressions;
pub(crate) use dce::eliminate_dead_code;
pub(crate) use fold::fold_constants;
pub(crate) use fuse::fuse;
pub(crate) use pack::pack_weights;
pub(crate) use qdq::{
    ACT_REQUANT, I8_BIAS, I8_REQUANT, fold_qdq_into_matmuls, fuse_qdq, fuse_qdq_matmul,
    fuse_weight_only_matmul,
};
pub(crate) use shapes::fold_shapes;
pub(crate) use simplify::{bypass_view_casts, simplify};
pub(crate) use types::{follows_input, infer_dtypes};

use super::ir::{DType, Graph, Node, Op};

/// The index of a `Gather` of one constant index along the outermost axis,
/// whose result is one slab of its input: code reads it as a view of that
/// slab ([`select`](crate::kernels::select)), with no buffer of its own. Not
/// of an 8-bit weight, of which code widens only the slab
/// ([`gather_widened`](crate::kernels::gather_widened)).
pub(crate) fn selected_index(g: &Graph, node: &Node) -> Option<i64> {
    if !matches!(node.op, Op::Gather { axis: 0 }) {
        return None;
    }
    // An initializer or the value of a `Constant`.
    let constant = |v| match g.producer(v) {
        Some(Node {
            op: Op::Constant { value },
            ..
        }) => Some(value),
        _ => g.data(v),
    };
    let table = constant(node.input(0)?);
    if table.is_some_and(|t| matches!(DType::from_onnx(t.data_type), DType::U8 | DType::I8)) {
        return None;
    }
    let t = constant(node.input(1)?)?;
    match (t.dims.is_empty(), fold::exact_ints(t)?.as_slice()) {
        (true, &[i]) => Some(i),
        _ => None,
    }
}
