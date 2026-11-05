//! Graph rewrites on the [`ir`](super::ir).

mod branches;
mod cse;
mod dce;
mod fold;
mod fuse;
mod qdq;
mod shapes;
mod types;

pub(crate) use branches::inline_constant_ifs;
pub(crate) use cse::eliminate_common_subexpressions;
pub(crate) use dce::eliminate_dead_code;
pub(crate) use fold::fold_constants;
pub(crate) use fuse::fuse;
pub(crate) use qdq::{
    ACT_REQUANT, I8_BIAS, I8_REQUANT, fold_qdq_into_matmuls, fuse_qdq, fuse_qdq_matmul,
    fuse_weight_only_matmul,
};
pub(crate) use shapes::fold_shapes;
pub(crate) use types::{follows_input, infer_dtypes};
