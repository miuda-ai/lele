//! Graph rewrites on the [`ir`](super::ir).

mod dce;
mod fold;
mod fuse;
mod qdq;
mod shapes;
mod types;

pub(crate) use dce::eliminate_dead_code;
pub(crate) use fold::fold_constants;
pub(crate) use fuse::fuse;
pub(crate) use qdq::{fuse_qdq, fuse_qdq_matmul};
pub(crate) use shapes::fold_shapes;
pub(crate) use types::{follows_input, infer_dtypes};
