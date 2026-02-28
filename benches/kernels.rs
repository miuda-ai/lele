//! Kernel-level benchmarks for lele operators
//!
//! Run with: cargo bench --bench kernels

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lele::tensor::TensorView;
use std::borrow::Cow;

// ============================================================================
// GEMM Benchmarks
// ============================================================================

fn bench_gemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm");

    // Common sizes from SenseVoice and Supertonic models
    let sizes = [
        // (M, K, N) - attention projections
        (4, 512, 512),
        (8, 512, 512),
        (16, 256, 256),
        // FFN layers
        (1, 512, 2048),
        (1, 2048, 512),
        // Large attention
        (64, 64, 64),
        (128, 128, 128),
    ];

    for &(m, k, n) in &sizes {
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 % 10.0) * 0.1).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32 % 10.0) * 0.1).collect();
        let mut out_buf = vec![0.0f32; m * n];

        let a_shape = vec![m, k];
        let b_shape = vec![k, n];
        let a = TensorView { data: Cow::Borrowed(&a_data), shape: Cow::Borrowed(&a_shape) };
        let b = TensorView { data: Cow::Borrowed(&b_data), shape: Cow::Borrowed(&b_shape) };

        group.throughput(Throughput::Elements((m * k * n) as u64));
        group.bench_with_input(BenchmarkId::new("matmul", format!("{}x{}x{}", m, k, n)), &(m, k, n), |bencher, _| {
            bencher.iter(|| {
                let _ = lele::kernels::gemm::matmul(
                    black_box(&a),
                    black_box(&b),
                    &mut out_buf,
                );
            });
        });
    }

    group.finish();
}

// ============================================================================
// Softmax Benchmarks
// ============================================================================

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");

    // Common axis sizes from models
    let sizes = [
        (1, 512),      // Single attention head
        (4, 512),      // Batch of 4
        (16, 256),     // More heads
        (64, 128),     // Large batch
        (128, 80),     // SenseVoice vocabulary
        (1, 1024),     // Large vocabulary
    ];

    for &(batch, axis_size) in &sizes {
        let input_data: Vec<f32> = (0..batch * axis_size)
            .map(|i| ((i % 20) as f32 - 10.0) * 0.1)
            .collect();
        let mut out_buf = vec![0.0f32; batch * axis_size];

        let input_shape = vec![batch, axis_size];
        let input = TensorView { data: Cow::Borrowed(&input_data), shape: Cow::Borrowed(&input_shape) };

        group.throughput(Throughput::Elements((batch * axis_size) as u64));
        group.bench_with_input(
            BenchmarkId::new("softmax", format!("{}x{}", batch, axis_size)),
            &(batch, axis_size),
            |bencher, _| {
                bencher.iter(|| {
                    let _ = lele::kernels::norm::softmax(
                        black_box(&input),
                        -1,
                        &mut out_buf,
                    );
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// LayerNorm Benchmarks
// ============================================================================

fn bench_layer_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("layer_norm");

    let sizes = [
        (1, 512),
        (4, 512),
        (16, 256),
        (64, 128),
        (128, 512),
    ];

    for &(outer, norm_size) in &sizes {
        let input_data: Vec<f32> = (0..outer * norm_size)
            .map(|i| ((i % 20) as f32 - 10.0) * 0.1)
            .collect();
        let gamma: Vec<f32> = (0..norm_size).map(|i| 1.0 + ((i % 5) as f32) * 0.1).collect();
        let beta: Vec<f32> = (0..norm_size).map(|i| ((i % 5) as f32) * 0.01).collect();
        let mut out_buf = vec![0.0f32; outer * norm_size];

        let input_shape = vec![outer, norm_size];
        let scale_shape = vec![norm_size];
        let input = TensorView { data: Cow::Borrowed(&input_data), shape: Cow::Borrowed(&input_shape) };
        let scale = TensorView { data: Cow::Borrowed(&gamma), shape: Cow::Borrowed(&scale_shape) };
        let bias = TensorView { data: Cow::Borrowed(&beta), shape: Cow::Borrowed(&scale_shape) };

        group.throughput(Throughput::Elements((outer * norm_size) as u64));
        group.bench_with_input(
            BenchmarkId::new("layer_norm", format!("{}x{}", outer, norm_size)),
            &(outer, norm_size),
            |bencher, _| {
                bencher.iter(|| {
                    let _ = lele::kernels::norm::layer_norm(
                        black_box(&input),
                        black_box(&scale),
                        black_box(&bias),
                        -1,
                        1e-5,
                        &mut out_buf,
                    );
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Transpose Benchmarks
// ============================================================================

fn bench_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose");

    let sizes = [
        // (dim0, dim1, dim2) - transpose (0, 2, 1)
        (4, 64, 64),
        (8, 32, 64),
        (16, 64, 32),
        (1, 512, 512),
        (4, 128, 128),
    ];

    for &(d0, d1, d2) in &sizes {
        let input_data: Vec<f32> = (0..d0 * d1 * d2)
            .map(|i| (i as f32 % 10.0) * 0.1)
            .collect();
        let mut out_buf = vec![0.0f32; d0 * d1 * d2];

        let input_shape = vec![d0, d1, d2];
        let input = TensorView { data: Cow::Borrowed(&input_data), shape: Cow::Borrowed(&input_shape) };
        let perm: Vec<i64> = vec![0, 2, 1];

        group.throughput(Throughput::Elements((d0 * d1 * d2) as u64));
        group.bench_with_input(
            BenchmarkId::new("transpose", format!("{}x{}x{}", d0, d1, d2)),
            &(d0, d1, d2),
            |bencher, _| {
                bencher.iter(|| {
                    let _ = lele::kernels::manipulation::transpose(
                        black_box(&input),
                        black_box(&perm),
                        &mut out_buf,
                    );
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Element-wise Operations Benchmarks
// ============================================================================

fn bench_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise");

    let sizes = [512, 1024, 2048, 4096, 8192];

    for &size in &sizes {
        let a_data: Vec<f32> = (0..size).map(|i| (i as f32 % 10.0) * 0.1).collect();
        let b_data: Vec<f32> = (0..size).map(|i| (i as f32 % 10.0) * 0.1 + 0.1).collect();
        let mut out_buf = vec![0.0f32; size];

        let shape = vec![size];
        let a = TensorView { data: Cow::Borrowed(&a_data), shape: Cow::Borrowed(&shape) };
        let b = TensorView { data: Cow::Borrowed(&b_data), shape: Cow::Borrowed(&shape) };

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("add", size), &size, |bencher, _| {
            bencher.iter(|| {
                let _ = lele::kernels::math::add(black_box(&a), black_box(&b), &mut out_buf);
            });
        });

        group.bench_with_input(BenchmarkId::new("mul", size), &size, |bencher, _| {
            bencher.iter(|| {
                let _ = lele::kernels::math::mul(black_box(&a), black_box(&b), &mut out_buf);
            });
        });
    }

    group.finish();
}

// ============================================================================
// Activation Functions Benchmarks
// ============================================================================

#[cfg(target_arch = "aarch64")]
fn bench_activations(c: &mut Criterion) {
    let mut group = c.benchmark_group("activations");

    let sizes = [512, 1024, 2048, 4096];

    for &size in &sizes {
        let input_data: Vec<f32> = (0..size).map(|i| ((i % 20) as f32 - 10.0) * 0.1).collect();
        let mut out_buf = vec![0.0f32; size];

        let shape = vec![size];
        let input = TensorView { data: Cow::Borrowed(&input_data), shape: Cow::Borrowed(&shape) };

        group.throughput(Throughput::Elements(size as u64));

        // Tanh
        group.bench_with_input(BenchmarkId::new("tanh", size), &size, |bencher, _| {
            bencher.iter(|| {
                out_buf.fill(0.0);
                let _ = lele::kernels::math::tanh_kernel(black_box(&input), &mut out_buf);
            });
        });

        // Sigmoid
        group.bench_with_input(BenchmarkId::new("sigmoid", size), &size, |bencher, _| {
            bencher.iter(|| {
                out_buf.fill(0.0);
                let _ = lele::kernels::math::sigmoid(black_box(&input), &mut out_buf);
            });
        });

        // Exp
        group.bench_with_input(BenchmarkId::new("exp", size), &size, |bencher, _| {
            bencher.iter(|| {
                out_buf.fill(0.0);
                let _ = lele::kernels::math::exp(black_box(&input), &mut out_buf);
            });
        });

        // ERF (used in GELU)
        group.bench_with_input(BenchmarkId::new("erf", size), &size, |bencher, _| {
            bencher.iter(|| {
                out_buf.fill(0.0);
                let _ = lele::kernels::math::erf(black_box(&input), &mut out_buf);
            });
        });

        // ReLU
        group.bench_with_input(BenchmarkId::new("relu", size), &size, |bencher, _| {
            bencher.iter(|| {
                out_buf.fill(0.0);
                let _ = lele::kernels::math::relu(black_box(&input), &mut out_buf);
            });
        });

        // SiLU (Swish)
        group.bench_with_input(BenchmarkId::new("silu", size), &size, |bencher, _| {
            bencher.iter(|| {
                out_buf.fill(0.0);
                let _ = lele::kernels::math::silu(black_box(&input), &mut out_buf);
            });
        });
    }

    group.finish();
}

// ============================================================================
// GEMM with Bias (matmul_fused_add)
// ============================================================================

fn bench_matmul_fused_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_fused_add");

    let sizes = [
        (1, 512, 512),
        (4, 512, 512),
        (8, 256, 256),
        (1, 512, 2048),
    ];

    for &(m, k, n) in &sizes {
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 % 10.0) * 0.1).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32 % 10.0) * 0.1).collect();
        let bias_data: Vec<f32> = (0..n).map(|i| ((i % 10) as f32) * 0.01).collect();
        let mut out_buf = vec![0.0f32; m * n];

        let a_shape = vec![m, k];
        let b_shape = vec![k, n];
        let bias_shape = vec![n];
        let a = TensorView { data: Cow::Borrowed(&a_data), shape: Cow::Borrowed(&a_shape) };
        let b = TensorView { data: Cow::Borrowed(&b_data), shape: Cow::Borrowed(&b_shape) };
        let bias = TensorView { data: Cow::Borrowed(&bias_data), shape: Cow::Borrowed(&bias_shape) };

        group.throughput(Throughput::Elements((m * k * n + m * n) as u64));
        group.bench_with_input(
            BenchmarkId::new("matmul_fused_add", format!("{}x{}x{}", m, k, n)),
            &(m, k, n),
            |bencher, _| {
                bencher.iter(|| {
                    let _ = lele::kernels::gemm::matmul_fused_add(
                        black_box(&a),
                        black_box(&b),
                        black_box(&bias),
                        &mut out_buf,
                    );
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_gemm,
    bench_softmax,
    bench_layer_norm,
    bench_transpose,
    bench_elementwise,
    bench_matmul_fused_add,
);

#[cfg(target_arch = "aarch64")]
criterion_group!(neon_benches, bench_activations);

#[cfg(target_arch = "aarch64")]
criterion_main!(benches, neon_benches);

#[cfg(not(target_arch = "aarch64"))]
criterion_main!(benches);
