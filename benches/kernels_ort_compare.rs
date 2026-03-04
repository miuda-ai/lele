//! Performance comparison between lele kernels and ONNX Runtime
//!
//! Run with: cargo bench --bench kernels_ort_compare

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lele::tensor::TensorView;
use std::borrow::Cow;

// ============================================================================
// GEMM Comparison
// ============================================================================

fn bench_gemm_vs_ort(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_vs_ort");

    let sizes = [
        (1, 512, 512),
        (4, 512, 512),
        (16, 256, 256),
        (1, 512, 2048),
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

        // Lele kernel benchmark
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

        // TODO: Add ORT comparison when ort crate is available
        // group.bench_with_input(
        //     BenchmarkId::new("ort", format!("{}x{}x{}", m, k, n)),
        //     &(m, k, n),
        //     |bencher, _| {
        //         // ORT matmul benchmark
        //     },
        // );
    }

    group.finish();
}

// ============================================================================
// Softmax Comparison
// ============================================================================

fn bench_softmax_vs_ort(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_vs_ort");

    let sizes = [
        (1, 512),
        (4, 512),
        (16, 256),
        (128, 80),
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
            BenchmarkId::new("lele", format!("{}x{}", batch, axis_size)),
            &(batch, axis_size),
            |bencher, _| {
                bencher.iter(|| {
                    let _ = lele::kernels::norm::softmax(
                        black_box(&input),
                        black_box(-1),
                        &mut out_buf,
                    );
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// LayerNorm Comparison
// ============================================================================

fn bench_layernorm_vs_ort(c: &mut Criterion) {
    let mut group = c.benchmark_group("layernorm_vs_ort");

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
            BenchmarkId::new("lele", format!("{}x{}", outer, norm_size)),
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
// RMSNorm Comparison (LLM inference)
// ============================================================================

fn bench_rmsnorm_vs_ort(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmsnorm_vs_ort");

    // Typical sizes for LLM inference
    let sizes = [
        (1, 4096),   // LLaMA-like
        (1, 2048),   // Smaller model
        (4, 1024),   // Batched
        (16, 512),   // Large batch
    ];

    for &(batch, hidden_size) in &sizes {
        let input_data: Vec<f32> = (0..batch * hidden_size)
            .map(|i| ((i % 20) as f32 - 10.0) * 0.1)
            .collect();
        let weight: Vec<f32> = (0..hidden_size).map(|i| 1.0 + ((i % 5) as f32) * 0.1).collect();
        let mut out_buf = vec![0.0f32; batch * hidden_size];

        let input_shape = vec![batch, hidden_size];
        let weight_shape = vec![hidden_size];
        let input = TensorView { data: Cow::Borrowed(&input_data), shape: Cow::Borrowed(&input_shape) };
        let w = TensorView { data: Cow::Borrowed(&weight), shape: Cow::Borrowed(&weight_shape) };

        group.throughput(Throughput::Elements((batch * hidden_size) as u64));

        group.bench_with_input(
            BenchmarkId::new("lele", format!("{}x{}", batch, hidden_size)),
            &(batch, hidden_size),
            |bencher, _| {
                bencher.iter(|| {
                    let _ = lele::kernels::norm::rms_norm(
                        black_box(&input),
                        black_box(&w),
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
// Quantized MatMul (int8)
// ============================================================================

fn bench_qgemm_vs_ort(c: &mut Criterion) {
    let mut group = c.benchmark_group("qgemm_vs_ort");

    // Typical quantized model sizes
    let sizes = [
        (1, 512, 512),
        (1, 512, 2048),
        (4, 256, 256),
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
            BenchmarkId::new("lele_int8", format!("{}x{}x{}", m, k, n)),
            &(m, k, n),
            |bencher, _| {
                bencher.iter(|| {
                    let _ = lele::kernels::quantization::mat_mul_integer(
                        black_box(&a),
                        black_box(&b),
                        None,
                        None,
                        &mut out_buf,
                    );
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Fused Operations
// ============================================================================

fn bench_fused_gemm_bias(c: &mut Criterion) {
    let mut group = c.benchmark_group("fused_gemm_bias");

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
    bench_gemm_vs_ort,
    bench_softmax_vs_ort,
    bench_layernorm_vs_ort,
    bench_rmsnorm_vs_ort,
    bench_qgemm_vs_ort,
    bench_fused_gemm_bias,
);

criterion_main!(benches);
