pub mod activations;
#[cfg(target_arch = "x86_64")]
pub mod avx;
pub mod conv1d;
pub mod gemm;
pub mod manipulation;
pub mod math;
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
pub mod neon;
pub mod norm;
pub mod pooling;
pub mod quantization;
pub mod rnn;
pub mod shape;
pub mod utils;
pub use conv1d::conv1d;
pub use conv1d::conv1d_fused;
pub use gemm::{gemm, matmul, matmul_fused_add};
pub use manipulation::*;
pub use manipulation::{split, where_op};
pub use math::*;
pub use math::{cos, exp, expand, less, min_max, neg, range, sin, tile};
pub use norm::*;
pub use pooling::*;
pub use quantization::*;
pub use rnn::*;
pub use shape::*;
