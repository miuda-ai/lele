//! Graph rewrites on the [`ir`](super::ir).

mod fold;
mod fuse;
mod qdq;
mod types;

pub(crate) use fold::fold_constants;
pub(crate) use fuse::fuse;
pub(crate) use qdq::{fuse_qdq, fuse_qdq_matmul};
pub(crate) use types::{follows_input, infer_dtypes};
