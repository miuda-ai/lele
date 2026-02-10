pub mod activations;
#[cfg(target_arch = "x86_64")]
pub mod avx;
pub mod conv1d;
pub mod conv2d;
pub mod gemm;
pub mod manipulation;
pub mod math;
#[cfg(target_arch = "aarch64")]
pub mod neon;
pub mod norm;
pub mod pooling;
pub mod quantization;
pub mod rnn;
pub mod shape;
pub mod utils;
pub use conv1d::conv1d;
pub use conv1d::conv1d_fused;
pub use conv2d::{conv_integer, conv_integer_from_f32, conv_integer_from_f32_multi, fused_scale_bias, fused_scale_bias_silu, gather_elements, max_pool2d, print_conv_stats, reset_conv_stats, resize_nearest, topk};
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
