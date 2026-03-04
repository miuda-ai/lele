//! Lele kernel benchmark without ORT dependency
//!
//! Run with: cargo bench --bench lele_vs_ort

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lele::tensor::TensorView;
use std::borrow::Cow;

// ============================================================================
// MatMul Benchmark
// ============================================================================

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    let sizes = [
        (4, 512, 512),
        (8, 512, 512),
        (16, 256, 256),
        (1, 512, 2048),
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

        group.bench_with_input(
            BenchmarkId::new("lele", format!("{}x{}x{}", m, k, n)),
            &(m, k, n),
            |bencher, _| {
                bencher.iter(|| {
                    let _ = lele::kernels::gemm::matmul(
                        black_box(&a),
                        black_box(&b),
                        &mut out_buf,
                    );
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Softmax Benchmark
// ============================================================================

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");

    let sizes = [
        (1, 512),
        (4, 512),
        (16, 256),
        (64, 128),
    ];

    for &(batch, size) in &sizes {
        let input_data: Vec<f32> = (0..batch * size)
            .map(|i| ((i % 20) as f32 - 10.0) * 0.1)
            .collect();
        let mut out_buf = vec![0.0f32; batch * size];

        let input_shape = vec![batch, size];
        let input = TensorView { data: Cow::Borrowed(&input_data), shape: Cow::Borrowed(&input_shape) };

        group.throughput(Throughput::Elements((batch * size) as u64));

        group.bench_with_input(
            BenchmarkId::new("lele", format!("{}x{}", batch, size)),
            &(batch, size),
            |bencher, _| {
                bencher.iter(|| {
                    let _ = lele::kernels::norm::softmax(
                        black_box(&input),
                        black_box(-1),
                        &mut out_buf
                    );
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// LayerNorm Benchmark
// ============================================================================

fn bench_layernorm(c: &mut Criterion) {
    let mut group = c.benchmark_group("layernorm");

    let sizes = [
        (1, 512),
        (4, 512),
        (16, 256),
        (128, 512),
    ];

    for &(outer, norm_size) in &sizes {
        let input_data: Vec<f32> = (0..outer * norm_size)
            .map(|i| ((i % 20) as f32 - 10.0) * 0.1)
            .collect();
        let scale_data: Vec<f32> = (0..norm_size).map(|i| 1.0 + ((i % 5) as f32) * 0.1).collect();
        let bias_data: Vec<f32> = (0..norm_size).map(|i| ((i % 5) as f32) * 0.01).collect();
        let mut out_buf = vec![0.0f32; outer * norm_size];

        let input_shape = vec![outer, norm_size];
        let scale_shape = vec![norm_size];
        let input = TensorView { data: Cow::Borrowed(&input_data), shape: Cow::Borrowed(&input_shape) };
        let scale = TensorView { data: Cow::Borrowed(&scale_data), shape: Cow::Borrowed(&scale_shape) };
        let bias = TensorView { data: Cow::Borrowed(&bias_data), shape: Cow::Borrowed(&scale_shape) };

        group.throughput(Throughput::Elements((outer * norm_size) as u64));

        group.bench_with_input(
            BenchmarkId::new("lele", format!("{}x{}", outer, norm_size)),
            &(outer, norm_size),
            |bencher, _| {
                bencher.iter(|| {
                    let _ = lele::kernels::norm::layer_norm(
                        black_box(&input),
                        black_box(&scale),
                        black_box(&bias),
                        black_box(-1),
                        1e-5,
                        &mut out_buf
                    );
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// RMSNorm Benchmark (LLM-style)
// ============================================================================

fn bench_rmsnorm(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmsnorm");

    let sizes = [
        (1, 4096),
        (1, 2048),
        (4, 1024),
        (16, 512),
    ];

    for &(batch, hidden) in &sizes {
        let input_data: Vec<f32> = (0..batch * hidden)
            .map(|i| ((i % 20) as f32 - 10.0) * 0.1)
            .collect();
        let weight_data: Vec<f32> = (0..hidden).map(|i| 1.0 + ((i % 5) as f32) * 0.1).collect();
        let mut out_buf = vec![0.0f32; batch * hidden];

        let input_shape = vec![batch, hidden];
        let weight_shape = vec![hidden];
        let input = TensorView { data: Cow::Borrowed(&input_data), shape: Cow::Borrowed(&input_shape) };
        let weight = TensorView { data: Cow::Borrowed(&weight_data), shape: Cow::Borrowed(&weight_shape) };

        group.throughput(Throughput::Elements((batch * hidden) as u64));

        group.bench_with_input(
            BenchmarkId::new("lele", format!("{}x{}", batch, hidden)),
            &(batch, hidden),
            |bencher, _| {
                bencher.iter(|| {
                    let _ = lele::kernels::norm::rms_norm(
                        black_box(&input),
                        black_box(&weight),
                        black_box(-1),
                        1e-5,
                        &mut out_buf
                    );
                });
            },
        );
    }

    group.finish()
}

// ============================================================================
// GEMM with fused bias
// ============================================================================

fn bench_gemm_fused(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_fused");

    let sizes = [
        (1, 512, 512),
        (4, 512, 512),
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
            BenchmarkId::new("lele", format!("{}x{}x{}", m, k, n)),
            &(m, k, n),
            |bencher, _| {
                bencher.iter(|| {
                    let _ = lele::kernels::gemm::matmul_fused_add(
                        black_box(&a),
                        black_box(&b),
                        black_box(&bias),
                        &mut out_buf
                    );
                });
            },
        );
    }

    group.finish()
}

criterion_group!(
    benches,
    bench_matmul,
    bench_softmax,
    bench_layernorm,
    bench_rmsnorm,
    bench_gemm_fused,
);

criterion_main!(benches);
