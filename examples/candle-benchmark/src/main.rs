use anyhow::Result;
use candle_nn::ops as candle_ops;
use std::time::Instant;

fn benchmark_matmul() -> Result<()> {
    println!("\n=== Matrix Multiplication (GEMM) ===");

    // Test different matrix sizes
    let test_cases = vec![
        (128, 128, 128, "Small (128x128x128)"),
        (256, 256, 256, "Medium (256x256x256)"),
        (512, 512, 512, "Large (512x512x512)"),
        (1024, 256, 512, "Tall (1024x256x512)"),
        (256, 1024, 512, "Wide (256x1024x512)"),
    ];

    let warmup = 3;
    let iterations = 10;

    for (m, k, n, name) in test_cases {
        println!("\n{}", name);

        // Prepare lele tensors
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
        let a_shape = vec![m, k];
        let b_shape = vec![k, n];

        let a_lele = lele::tensor::TensorView::new(&a_data, &a_shape);
        let b_lele = lele::tensor::TensorView::new(&b_data, &b_shape);

        // Prepare candle tensors
        let a_candle = candle_core::Tensor::from_slice(&a_data, (m, k), &candle_core::Device::Cpu)?;
        let b_candle = candle_core::Tensor::from_slice(&b_data, (k, n), &candle_core::Device::Cpu)?;

        // Warmup lele
        for _ in 0..warmup {
            let mut out = Vec::new();
            let _ = lele::kernels::gemm::matmul(&a_lele, &b_lele, &mut out);
        }

        // Benchmark lele
        let start = Instant::now();
        for _ in 0..iterations {
            let mut out = Vec::new();
            let _ = lele::kernels::gemm::matmul(&a_lele, &b_lele, &mut out);
        }
        let lele_time = start.elapsed().as_secs_f64() / iterations as f64;

        // Warmup candle
        for _ in 0..warmup {
            let _ = a_candle.matmul(&b_candle)?;
        }

        // Benchmark candle
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = a_candle.matmul(&b_candle)?;
        }
        let candle_time = start.elapsed().as_secs_f64() / iterations as f64;

        let speedup = candle_time / lele_time;
        println!("  Lele:   {:.3} ms", lele_time * 1000.0);
        println!("  Candle: {:.3} ms", candle_time * 1000.0);
        println!(
            "  Speedup: {:.2}x ({})",
            speedup,
            if speedup > 1.0 {
                "lele faster"
            } else {
                "candle faster"
            }
        );

        // Calculate GFLOPS
        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        let lele_gflops = flops / lele_time / 1e9;
        let candle_gflops = flops / candle_time / 1e9;
        println!(
            "  GFLOPS - Lele: {:.2}, Candle: {:.2}",
            lele_gflops, candle_gflops
        );
    }

    Ok(())
}

fn benchmark_mul() -> Result<()> {
    println!("\n\n=== Element-wise Multiplication ===");

    let test_cases = vec![
        (1000, "Small (1K)"),
        (10000, "Medium (10K)"),
        (100000, "Large (100K)"),
        (1000000, "XLarge (1M)"),
        (10000000, "XXLarge (10M)"),
    ];

    let warmup = 5;
    let iterations = 20;

    for (size, name) in test_cases {
        println!("\n{}", name);

        let a_data: Vec<f32> = (0..size)
            .map(|i| ((i as f32) / size as f32) * 2.0)
            .collect();
        let b_data: Vec<f32> = (0..size)
            .map(|i| ((i as f32) / size as f32) * 0.5 + 1.0)
            .collect();
        let a_shape = vec![size];
        let b_shape = vec![size];

        // Lele
        let a_lele = lele::tensor::TensorView::new(&a_data, &a_shape);
        let b_lele = lele::tensor::TensorView::new(&b_data, &b_shape);

        // Candle
        let a_candle = candle_core::Tensor::from_slice(&a_data, size, &candle_core::Device::Cpu)?;
        let b_candle = candle_core::Tensor::from_slice(&b_data, size, &candle_core::Device::Cpu)?;

        // Warmup lele
        for _ in 0..warmup {
            let mut out = Vec::new();
            let _ = lele::kernels::math::mul(&a_lele, &b_lele, &mut out);
        }

        // Benchmark lele
        let start = Instant::now();
        for _ in 0..iterations {
            let mut out = Vec::new();
            let _ = lele::kernels::math::mul(&a_lele, &b_lele, &mut out);
        }
        let lele_time = start.elapsed().as_secs_f64() / iterations as f64;

        // Warmup candle
        for _ in 0..warmup {
            let _ = (&a_candle * &b_candle)?;
        }

        // Benchmark candle
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = (&a_candle * &b_candle)?;
        }
        let candle_time = start.elapsed().as_secs_f64() / iterations as f64;

        let speedup = candle_time / lele_time;
        println!("  Lele:   {:.3} ms", lele_time * 1000.0);
        println!("  Candle: {:.3} ms", candle_time * 1000.0);
        println!(
            "  Speedup: {:.2}x ({})",
            speedup,
            if speedup > 1.0 {
                "lele faster"
            } else {
                "candle faster"
            }
        );

        // Throughput GB/s (reading 2 arrays + writing 1 array)
        let bytes = size as f64 * 4.0 * 3.0; // f32 * (read a + read b + write)
        let lele_gbps = bytes / lele_time / 1e9;
        let candle_gbps = bytes / candle_time / 1e9;
        println!(
            "  Throughput - Lele: {:.2} GB/s, Candle: {:.2} GB/s",
            lele_gbps, candle_gbps
        );
    }

    Ok(())
}

fn benchmark_silu() -> Result<()> {
    println!("\n\n=== SiLU Activation ===");

    let test_cases = vec![
        (1000, "Small (1K)"),
        (10000, "Medium (10K)"),
        (100000, "Large (100K)"),
        (1000000, "XLarge (1M)"),
    ];

    let warmup = 5;
    let iterations = 20;

    for (size, name) in test_cases {
        println!("\n{}", name);

        let data: Vec<f32> = (0..size)
            .map(|i| ((i as f32) / size as f32) * 2.0 - 1.0)
            .collect();
        let shape = vec![size];

        // Lele
        let input_lele = lele::tensor::TensorView::new(&data, &shape);

        // Candle
        let input_candle = candle_core::Tensor::from_slice(&data, size, &candle_core::Device::Cpu)?;

        // Warmup lele
        for _ in 0..warmup {
            let mut out = Vec::new();
            let _ = lele::kernels::math::silu(&input_lele, &mut out);
        }

        // Benchmark lele
        let start = Instant::now();
        for _ in 0..iterations {
            let mut out = Vec::new();
            let _ = lele::kernels::math::silu(&input_lele, &mut out);
        }
        let lele_time = start.elapsed().as_secs_f64() / iterations as f64;

        // Warmup candle
        for _ in 0..warmup {
            let _ = input_candle.silu()?;
        }

        // Benchmark candle
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = input_candle.silu()?;
        }
        let candle_time = start.elapsed().as_secs_f64() / iterations as f64;

        let speedup = candle_time / lele_time;
        println!("  Lele:   {:.3} ms", lele_time * 1000.0);
        println!("  Candle: {:.3} ms", candle_time * 1000.0);
        println!(
            "  Speedup: {:.2}x ({})",
            speedup,
            if speedup > 1.0 {
                "lele faster"
            } else {
                "candle faster"
            }
        );

        // Throughput GB/s (reading + writing)
        let bytes = size as f64 * 4.0 * 2.0; // f32 * read+write
        let lele_gbps = bytes / lele_time / 1e9;
        let candle_gbps = bytes / candle_time / 1e9;
        println!(
            "  Throughput - Lele: {:.2} GB/s, Candle: {:.2} GB/s",
            lele_gbps, candle_gbps
        );
    }

    Ok(())
}

fn benchmark_softmax() -> Result<()> {
    println!("\n\n=== Softmax ===");

    let test_cases = vec![
        ((32, 128), "Batch=32, Dim=128"),
        ((128, 256), "Batch=128, Dim=256"),
        ((256, 512), "Batch=256, Dim=512"),
        ((512, 1024), "Batch=512, Dim=1024"),
    ];

    let warmup = 5;
    let iterations = 20;

    for ((batch, dim), name) in test_cases {
        println!("\n{}", name);

        let size = batch * dim;
        let data: Vec<f32> = (0..size)
            .map(|i| ((i as f32) / size as f32) * 4.0 - 2.0)
            .collect();
        let shape = vec![batch, dim];

        // Lele
        let input_lele = lele::tensor::TensorView::new(&data, &shape);

        // Candle
        let input_candle =
            candle_core::Tensor::from_slice(&data, (batch, dim), &candle_core::Device::Cpu)?;

        // Warmup lele
        for _ in 0..warmup {
            let mut out = Vec::new();
            let _ = lele::kernels::norm::softmax(&input_lele, -1, &mut out);
        }

        // Benchmark lele
        let start = Instant::now();
        for _ in 0..iterations {
            let mut out = Vec::new();
            let _ = lele::kernels::norm::softmax(&input_lele, -1, &mut out);
        }
        let lele_time = start.elapsed().as_secs_f64() / iterations as f64;

        // Warmup candle
        for _ in 0..warmup {
            let _ = candle_ops::softmax(&input_candle, 1)?;
        }

        // Benchmark candle
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = candle_ops::softmax(&input_candle, 1)?;
        }
        let candle_time = start.elapsed().as_secs_f64() / iterations as f64;

        let speedup = candle_time / lele_time;
        println!("  Lele:   {:.3} ms", lele_time * 1000.0);
        println!("  Candle: {:.3} ms", candle_time * 1000.0);
        println!(
            "  Speedup: {:.2}x ({})",
            speedup,
            if speedup > 1.0 {
                "lele faster"
            } else {
                "candle faster"
            }
        );
    }

    Ok(())
}

fn benchmark_conv1d() -> Result<()> {
    println!("\n\n=== Conv1D ===");

    let test_cases = vec![
        (
            1,
            32,
            100,
            64,
            3,
            1,
            1,
            "Speech Feature (32->64, k=3, L=100)",
        ),
        (1, 64, 200, 128, 5, 1, 2, "Audio (64->128, k=5, L=200)"),
    ];

    let warmup = 3;
    let iterations = 10;

    for (batch, in_c, length, out_c, k, stride, padding, name) in test_cases {
        println!("\n{}", name);

        let input_size = batch * in_c * length;
        let weight_size = out_c * in_c * k;

        let input: Vec<f32> = (0..input_size).map(|i| (i as f32) * 0.01).collect();
        let weight: Vec<f32> = (0..weight_size).map(|i| (i as f32) * 0.001).collect();
        let bias: Vec<f32> = vec![0.1; out_c];

        let input_shape = vec![batch, in_c, length];
        let weight_shape = vec![out_c, in_c, k];
        let bias_shape = vec![out_c];

        // Lele
        let input_lele = lele::tensor::TensorView::new(&input, &input_shape);
        let weight_lele = lele::tensor::TensorView::new(&weight, &weight_shape);
        let bias_lele = lele::tensor::TensorView::new(&bias, &bias_shape);

        // Candle
        let input_candle = candle_core::Tensor::from_slice(
            &input,
            (batch, in_c, length),
            &candle_core::Device::Cpu,
        )?;
        let weight_candle =
            candle_core::Tensor::from_slice(&weight, (out_c, in_c, k), &candle_core::Device::Cpu)?;

        // Warmup lele
        for _ in 0..warmup {
            let mut out = Vec::new();
            let _ = lele::kernels::conv1d::conv1d(
                &input_lele,
                &weight_lele,
                Some(&bias_lele),
                &[1],              // dilations
                1,                 // group
                &[padding as i64], // pads (only left padding for conv1d)
                &[stride as i64],  // strides
                &mut out,
            );
        }

        // Benchmark lele
        let start = Instant::now();
        for _ in 0..iterations {
            let mut out = Vec::new();
            let _ = lele::kernels::conv1d::conv1d(
                &input_lele,
                &weight_lele,
                Some(&bias_lele),
                &[1],              // dilations
                1,                 // group
                &[padding as i64], // pads
                &[stride as i64],  // strides
                &mut out,
            );
        }
        let lele_time = start.elapsed().as_secs_f64() / iterations as f64;

        // Warmup candle
        for _ in 0..warmup {
            let _ = weight_candle.conv1d(&input_candle, padding, stride, 1, 1)?;
        }

        // Benchmark candle
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = weight_candle.conv1d(&input_candle, padding, stride, 1, 1)?;
        }
        let candle_time = start.elapsed().as_secs_f64() / iterations as f64;

        let speedup = candle_time / lele_time;
        println!("  Lele:   {:.3} ms", lele_time * 1000.0);
        println!("  Candle: {:.3} ms", candle_time * 1000.0);
        println!(
            "  Speedup: {:.2}x ({})",
            speedup,
            if speedup > 1.0 {
                "lele faster"
            } else {
                "candle faster"
            }
        );

        let out_len = (length + padding - (k - 1) - 1) / stride + 1;
        let flops = 2.0 * in_c as f64 * k as f64 * out_len as f64 * out_c as f64 * batch as f64;
        let lele_gflops = flops / lele_time / 1e9;
        let candle_gflops = flops / candle_time / 1e9;
        println!(
            "  GFLOPS - Lele: {:.2}, Candle: {:.2}",
            lele_gflops, candle_gflops
        );
    }

    Ok(())
}

fn benchmark_layer_norm() -> Result<()> {
    println!("\n\n=== Layer Normalization ===");

    let test_cases = vec![
        // SenseVoice-style: encoder hidden_size=560, typical seq_len varies
        ((1, 560), "SenseVoice single (1, 560)"),
        ((32, 560), "SenseVoice (32, 560)"),
        ((128, 560), "SenseVoice long (128, 560)"),
        // Supertonic TextEncoder/DurationPredictor: dim=512
        ((32, 512), "Supertonic (32, 512)"),
        // Supertonic VectorEstimator: dim=512, larger batch
        ((128, 512), "Supertonic VE (128, 512)"),
        // Transformer-style: dim=768, 1024
        ((128, 768), "BERT-base (128, 768)"),
        ((256, 1024), "Large Model (256, 1024)"),
    ];

    let warmup = 10;
    let iterations = 50;

    for ((batch, dim), name) in test_cases {
        println!("\n{}", name);

        let size = batch * dim;
        let data: Vec<f32> = (0..size)
            .map(|i| ((i as f32) / size as f32) * 4.0 - 2.0)
            .collect();
        let scale: Vec<f32> = vec![1.0; dim];
        let bias: Vec<f32> = vec![0.0; dim];

        let input_shape = vec![batch, dim];
        let scale_shape = vec![dim];
        let bias_shape = vec![dim];

        // Lele
        let input_lele = lele::tensor::TensorView::new(&data, &input_shape);
        let scale_lele = lele::tensor::TensorView::new(&scale, &scale_shape);
        let bias_lele = lele::tensor::TensorView::new(&bias, &bias_shape);

        // Candle
        let input_candle =
            candle_core::Tensor::from_slice(&data, (batch, dim), &candle_core::Device::Cpu)?;
        let scale_candle = candle_core::Tensor::from_slice(&scale, dim, &candle_core::Device::Cpu)?;
        let bias_candle = candle_core::Tensor::from_slice(&bias, dim, &candle_core::Device::Cpu)?;

        // Warmup lele
        for _ in 0..warmup {
            let mut out = Vec::new();
            let _ = lele::kernels::norm::layer_norm(
                &input_lele,
                &scale_lele,
                &bias_lele,
                -1,
                1e-5,
                &mut out,
            );
        }

        // Benchmark lele
        let start = Instant::now();
        for _ in 0..iterations {
            let mut out = Vec::new();
            let _ = lele::kernels::norm::layer_norm(
                &input_lele,
                &scale_lele,
                &bias_lele,
                -1,
                1e-5,
                &mut out,
            );
        }
        let lele_time = start.elapsed().as_secs_f64() / iterations as f64;

        // Warmup candle
        for _ in 0..warmup {
            let _ = candle_nn::ops::layer_norm(&input_candle, &scale_candle, &bias_candle, 1e-5)?;
        }

        // Benchmark candle
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = candle_nn::ops::layer_norm(&input_candle, &scale_candle, &bias_candle, 1e-5)?;
        }
        let candle_time = start.elapsed().as_secs_f64() / iterations as f64;

        let speedup = candle_time / lele_time;
        println!("  Lele:   {:.3} ms", lele_time * 1000.0);
        println!("  Candle: {:.3} ms", candle_time * 1000.0);
        println!(
            "  Speedup: {:.2}x ({})",
            speedup,
            if speedup > 1.0 {
                "lele faster"
            } else {
                "candle faster"
            }
        );
    }

    Ok(())
}

fn benchmark_conv2d() -> Result<()> {
    println!("\n\n=== Conv2D ===");

    let test_cases = vec![(1, 8, 7, 7, 16, 3, 1, 1, "Minimal (8->16, 3x3, 7x7)")];

    let warmup = 0; // Skip warmup
    let iterations = 1; // Only 1 iteration

    for (batch, in_c, h, w, out_c, k, stride, padding, name) in test_cases {
        println!("\n{}", name);

        let input_size = batch * in_c * h * w;
        let weight_size = out_c * in_c * k * k;

        let input: Vec<f32> = (0..input_size).map(|i| (i as f32) * 0.01).collect();
        let weight: Vec<f32> = (0..weight_size).map(|i| (i as f32) * 0.001).collect();
        let bias: Vec<f32> = vec![0.1; out_c];

        let out_h = (h + 2 * padding - k) / stride + 1;
        let out_w = (w + 2 * padding - k) / stride + 1;

        let input_shape = vec![batch, in_c, h, w];
        let weight_shape = vec![out_c, in_c, k, k];
        let bias_shape = vec![out_c];

        // Lele
        let input_lele = lele::tensor::TensorView::new(&input, &input_shape);
        let weight_lele = lele::tensor::TensorView::new(&weight, &weight_shape);
        let bias_lele = lele::tensor::TensorView::new(&bias, &bias_shape);

        // Candle
        let input_candle = candle_core::Tensor::from_slice(
            &input,
            (batch, in_c, h, w),
            &candle_core::Device::Cpu,
        )?;
        let weight_candle = candle_core::Tensor::from_slice(
            &weight,
            (out_c, in_c, k, k),
            &candle_core::Device::Cpu,
        )?;

        // Warmup lele
        println!("  Warming up lele...");
        for _ in 0..warmup {
            let mut out = Vec::new();
            let _ = lele::kernels::conv2d::conv2d(
                &input_lele,
                &weight_lele,
                Some(&bias_lele),
                &[1, 1],
                1,
                &[padding as i64, padding as i64],
                &[stride as i64, stride as i64],
                &mut out,
            );
        }

        // Benchmark lele
        println!("  Benchmarking lele...");
        let start = Instant::now();
        for _ in 0..iterations {
            let mut out = Vec::new();
            let _ = lele::kernels::conv2d::conv2d(
                &input_lele,
                &weight_lele,
                Some(&bias_lele),
                &[1, 1],
                1,
                &[padding as i64, padding as i64],
                &[stride as i64, stride as i64],
                &mut out,
            );
        }
        let lele_time = start.elapsed().as_secs_f64() / iterations as f64;

        // Warmup candle
        for _ in 0..warmup {
            let _ = weight_candle.conv2d(&input_candle, padding, stride, 1, 1)?;
        }

        // Benchmark candle
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = weight_candle.conv2d(&input_candle, padding, stride, 1, 1)?;
        }
        let candle_time = start.elapsed().as_secs_f64() / iterations as f64;

        let speedup = candle_time / lele_time;
        println!("  Lele:   {:.3} ms", lele_time * 1000.0);
        println!("  Candle: {:.3} ms", candle_time * 1000.0);
        println!(
            "  Speedup: {:.2}x ({})",
            speedup,
            if speedup > 1.0 {
                "lele faster"
            } else {
                "candle faster"
            }
        );

        let flops = 2.0
            * in_c as f64
            * k as f64
            * k as f64
            * out_h as f64
            * out_w as f64
            * out_c as f64
            * batch as f64;
        let lele_gflops = flops / lele_time / 1e9;
        let candle_gflops = flops / candle_time / 1e9;
        println!(
            "  GFLOPS - Lele: {:.2}, Candle: {:.2}",
            lele_gflops, candle_gflops
        );
    }

    Ok(())
}

fn benchmark_lstm() -> Result<()> {
    println!("\n\n=== LSTM ===");

    let test_cases = vec![
        (20, 64, 128, "Small (seq=20, in=64, hidden=128)"),
        (10, 32, 64, "Tiny (seq=10, in=32, hidden=64)"),
    ];

    let warmup = 1;
    let iterations = 2;

    for (seq_len, input_size, hidden_size, name) in test_cases {
        println!("\n{}", name);

        let batch_size = 1;
        let input: Vec<f32> = (0..seq_len * batch_size * input_size)
            .map(|i| (i as f32) * 0.01)
            .collect();
        let w: Vec<f32> = (0..4 * hidden_size * input_size)
            .map(|i| (i as f32) * 0.001)
            .collect();
        let r: Vec<f32> = (0..4 * hidden_size * hidden_size)
            .map(|i| (i as f32) * 0.001)
            .collect();
        let bias: Vec<f32> = vec![0.0; 8 * hidden_size];

        let input_shape = vec![seq_len, batch_size, input_size];
        let w_shape = vec![1, 4 * hidden_size, input_size];
        let r_shape = vec![1, 4 * hidden_size, hidden_size];
        let bias_shape = vec![1, 8 * hidden_size];

        // Lele
        let input_lele = lele::tensor::TensorView::new(&input, &input_shape);
        let w_lele = lele::tensor::TensorView::new(&w, &w_shape);
        let r_lele = lele::tensor::TensorView::new(&r, &r_shape);
        let bias_lele = lele::tensor::TensorView::new(&bias, &bias_shape);

        // Candle (LSTM is more complex in candle, we'll use a simpler comparison)
        let input_candle = candle_core::Tensor::from_slice(
            &input,
            (seq_len, batch_size, input_size),
            &candle_core::Device::Cpu,
        )?;
        let w_candle = candle_core::Tensor::from_slice(
            &w,
            (4 * hidden_size, input_size),
            &candle_core::Device::Cpu,
        )?;
        let r_candle = candle_core::Tensor::from_slice(
            &r,
            (4 * hidden_size, hidden_size),
            &candle_core::Device::Cpu,
        )?;

        // Warmup lele
        for _ in 0..warmup {
            let mut out_y = Vec::new();
            let mut out_h = Vec::new();
            let mut out_c = Vec::new();
            let _ = lele::kernels::rnn::lstm(
                &input_lele,
                &w_lele,
                &r_lele,
                Some(&bias_lele),
                None,
                None,
                None,
                &mut out_y,
                &mut out_h,
                &mut out_c,
            );
        }

        // Benchmark lele
        let start = Instant::now();
        for _ in 0..iterations {
            let mut out_y = Vec::new();
            let mut out_h = Vec::new();
            let mut out_c = Vec::new();
            let _ = lele::kernels::rnn::lstm(
                &input_lele,
                &w_lele,
                &r_lele,
                Some(&bias_lele),
                None,
                None,
                None,
                &mut out_y,
                &mut out_h,
                &mut out_c,
            );
        }
        let lele_time = start.elapsed().as_secs_f64() / iterations as f64;

        // For candle, we'll measure the equivalent matmul operations
        // LSTM has 2 matmuls per timestep: W*x + R*h
        let mut h_state = candle_core::Tensor::zeros(
            (batch_size, hidden_size),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )?;

        // Warmup candle
        for _ in 0..warmup {
            let mut h = h_state.clone();
            for t in 0..seq_len {
                let x_t = input_candle
                    .narrow(0, t, 1)?
                    .reshape((batch_size, input_size))?;
                let _ = x_t.matmul(&w_candle.t()?)?;
                let _ = h.matmul(&r_candle.t()?)?;
            }
        }

        // Benchmark candle
        let start = Instant::now();
        for _ in 0..iterations {
            let mut h = h_state.clone();
            for t in 0..seq_len {
                let x_t = input_candle
                    .narrow(0, t, 1)?
                    .reshape((batch_size, input_size))?;
                let _ = x_t.matmul(&w_candle.t()?)?;
                let _ = h.matmul(&r_candle.t()?)?;
            }
        }
        let candle_time = start.elapsed().as_secs_f64() / iterations as f64;

        let speedup = candle_time / lele_time;
        println!("  Lele:   {:.3} ms", lele_time * 1000.0);
        println!("  Candle: {:.3} ms (matmul only)", candle_time * 1000.0);
        println!(
            "  Speedup: {:.2}x ({})",
            speedup,
            if speedup > 1.0 {
                "lele faster"
            } else {
                "candle faster"
            }
        );

        // FLOPS: 2 matmuls per timestep: (2*input*hidden*4 + 2*hidden*hidden*4) * seq_len
        let flops = seq_len as f64
            * (2.0 * input_size as f64 * hidden_size as f64 * 4.0
                + 2.0 * hidden_size as f64 * hidden_size as f64 * 4.0);
        let lele_gflops = flops / lele_time / 1e9;
        let candle_gflops = flops / candle_time / 1e9;
        println!(
            "  GFLOPS - Lele: {:.2}, Candle: {:.2}",
            lele_gflops, candle_gflops
        );
    }

    Ok(())
}

fn benchmark_conv2d_old() -> Result<()> {
    println!("\n\n=== Conv2D ===");

    let test_cases = vec![
        (
            1,
            3,
            224,
            224,
            64,
            7,
            2,
            3,
            "ImageNet First Layer (3->64, 7x7, stride=2)",
        ),
        (
            1,
            64,
            56,
            56,
            64,
            3,
            1,
            1,
            "ResNet Block (64->64, 3x3, stride=1)",
        ),
        (1, 128, 28, 28, 256, 1, 1, 0, "1x1 Conv (128->256)"),
        (
            1,
            256,
            14,
            14,
            512,
            3,
            2,
            1,
            "Downsample (256->512, 3x3, stride=2)",
        ),
    ];

    let warmup = 3;
    let iterations = 5;

    for (batch, in_c, h, w, out_c, k, stride, padding, name) in test_cases {
        println!("\n{}", name);

        let input_size = batch * in_c * h * w;
        let weight_size = out_c * in_c * k * k;

        let input: Vec<f32> = (0..input_size).map(|i| (i as f32) * 0.01).collect();
        let weight: Vec<f32> = (0..weight_size).map(|i| (i as f32) * 0.001).collect();
        let bias: Vec<f32> = vec![0.1; out_c];

        let out_h = (h + 2 * padding - k) / stride + 1;
        let out_w = (w + 2 * padding - k) / stride + 1;

        let input_shape = vec![batch, in_c, h, w];
        let weight_shape = vec![out_c, in_c, k, k];
        let bias_shape = vec![out_c];

        // Lele
        let input_lele = lele::tensor::TensorView::new(&input, &input_shape);
        let weight_lele = lele::tensor::TensorView::new(&weight, &weight_shape);
        let bias_lele = lele::tensor::TensorView::new(&bias, &bias_shape);

        // Candle
        let input_candle = candle_core::Tensor::from_slice(
            &input,
            (batch, in_c, h, w),
            &candle_core::Device::Cpu,
        )?;
        let weight_candle = candle_core::Tensor::from_slice(
            &weight,
            (out_c, in_c, k, k),
            &candle_core::Device::Cpu,
        )?;

        // Warmup lele
        for _ in 0..warmup {
            let mut out = Vec::new();
            let _ = lele::kernels::conv2d::conv2d(
                &input_lele,
                &weight_lele,
                Some(&bias_lele),
                &[1, 1],                           // dilations
                1,                                 // group
                &[padding as i64, padding as i64], // pads
                &[stride as i64, stride as i64],   // strides
                &mut out,
            );
        }

        // Benchmark lele
        let start = Instant::now();
        for _ in 0..iterations {
            let mut out = Vec::new();
            let _ = lele::kernels::conv2d::conv2d(
                &input_lele,
                &weight_lele,
                Some(&bias_lele),
                &[1, 1],                           // dilations
                1,                                 // group
                &[padding as i64, padding as i64], // pads
                &[stride as i64, stride as i64],   // strides
                &mut out,
            );
        }
        let lele_time = start.elapsed().as_secs_f64() / iterations as f64;

        // Warmup candle
        for _ in 0..warmup {
            let _ = weight_candle.conv2d(&input_candle, padding, stride, 1, 1)?;
        }

        // Benchmark candle
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = weight_candle.conv2d(&input_candle, padding, stride, 1, 1)?;
        }
        let candle_time = start.elapsed().as_secs_f64() / iterations as f64;

        let speedup = candle_time / lele_time;
        println!("  Lele:   {:.3} ms", lele_time * 1000.0);
        println!("  Candle: {:.3} ms", candle_time * 1000.0);
        println!(
            "  Speedup: {:.2}x ({})",
            speedup,
            if speedup > 1.0 {
                "lele faster"
            } else {
                "candle faster"
            }
        );

        // Calculate GFLOPS (2 * in_c * k * k * out_h * out_w * out_c * batch)
        let flops = 2.0
            * in_c as f64
            * k as f64
            * k as f64
            * out_h as f64
            * out_w as f64
            * out_c as f64
            * batch as f64;
        let lele_gflops = flops / lele_time / 1e9;
        let candle_gflops = flops / candle_time / 1e9;
        println!(
            "  GFLOPS - Lele: {:.2}, Candle: {:.2}",
            lele_gflops, candle_gflops
        );
    }

    Ok(())
}

fn benchmark_erf() -> Result<()> {
    println!("\n\n=== Erf (GELU component, used in Supertonic2) ===");

    // Supertonic2 uses erf extensively: 6 calls per submodel in GELU activation
    // Typical sizes: (batch, seq_len, hidden_dim) — hidden dims 512, 768, 1024
    let test_cases = vec![
        (32 * 512, "Small (32x512 = 16K)"),
        (128 * 512, "Medium (128x512 = 64K)"),
        (128 * 768, "BERT-like (128x768 = 98K)"),
        (256 * 1024, "Large (256x1024 = 256K)"),
    ];

    let warmup = 10;
    let iterations = 50;

    for (size, name) in test_cases {
        println!("\n{}", name);

        let data: Vec<f32> = (0..size)
            .map(|i| ((i as f32) / size as f32) * 4.0 - 2.0)
            .collect();
        let shape = vec![size];

        // Lele
        let input_lele = lele::tensor::TensorView::new(&data, &shape);

        // Candle
        let input_candle = candle_core::Tensor::from_slice(&data, size, &candle_core::Device::Cpu)?;

        // Warmup lele
        for _ in 0..warmup {
            let mut out = Vec::new();
            let _ = lele::kernels::math::erf(&input_lele, &mut out);
        }

        // Benchmark lele
        let start = Instant::now();
        for _ in 0..iterations {
            let mut out = Vec::new();
            let _ = lele::kernels::math::erf(&input_lele, &mut out);
        }
        let lele_time = start.elapsed().as_secs_f64() / iterations as f64;

        // Warmup candle
        for _ in 0..warmup {
            let _ = input_candle.erf()?;
        }

        // Benchmark candle
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = input_candle.erf()?;
        }
        let candle_time = start.elapsed().as_secs_f64() / iterations as f64;

        let speedup = candle_time / lele_time;
        let bytes = size as f64 * 4.0 * 2.0; // read + write
        let lele_bw = bytes / lele_time / 1e9;
        let candle_bw = bytes / candle_time / 1e9;
        println!("  Lele:   {:.3} ms", lele_time * 1000.0);
        println!("  Candle: {:.3} ms", candle_time * 1000.0);
        println!(
            "  Speedup: {:.2}x ({})",
            speedup,
            if speedup > 1.0 {
                "lele faster"
            } else {
                "candle faster"
            }
        );
        println!(
            "  Throughput - Lele: {:.2} GB/s, Candle: {:.2} GB/s",
            lele_bw, candle_bw
        );
    }

    Ok(())
}

fn benchmark_int8_gemm() -> Result<()> {
    println!("\n\n=== Int8 Quantized GEMM (SenseVoice workload) ===");

    // SenseVoice actual GEMM shapes: M=93 (seq_len after feature extraction)
    let test_cases: Vec<(usize, usize, usize, usize, bool, &str)> = vec![
        (93, 512, 512, 69, false, "Attn Out Proj (93×512×512) ×69"),
        (93, 512, 2048, 69, true, "FFN Up+ReLU (93×512×2048) ×69"),
        (93, 2048, 512, 68, false, "FFN Down (93×2048×512) ×68"),
        (93, 512, 1536, 67, false, "Attn QKV (93×512×1536) ×67"),
        (93, 512, 25055, 1, false, "Final Logits (93×512×25055) ×1"),
    ];

    let warmup = 5;
    let iterations = 20;

    // Also compute total estimated time for all calls combined
    let mut total_time_all_calls = 0.0f64;

    for (m, k, n, num_calls, apply_relu, name) in &test_cases {
        let m = *m;
        let k = *k;
        let n = *n;
        let num_calls = *num_calls;
        let apply_relu = *apply_relu;
        println!("\n{}", name);

        // Generate random-ish u8 weights [K, N] and f32 input [M, K]
        let b_u8: Vec<u8> = (0..k * n).map(|i| ((i * 37 + 13) % 256) as u8).collect();
        let a_f32: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32) / (m * k) as f32) * 2.0 - 1.0)
            .collect();
        let weight_scale: Vec<f32> = (0..n)
            .map(|i| 0.01 + (i as f32) * 0.0001)
            .collect();
        let bias_data: Vec<f32> = vec![0.1; n];

        let a_shape = vec![m, k];
        let ws_shape = vec![n];
        let bias_shape = vec![n];

        let a_view = lele::tensor::TensorView::new(&a_f32, &a_shape);
        let ws_view = lele::tensor::TensorView::new(&weight_scale, &ws_shape);
        let bias_view = lele::tensor::TensorView::new(&bias_data, &bias_shape);

        // Prepare weights (one-time cost, cached in real model)
        let pw = lele::kernels::quantization::prepare_weights(&b_u8, k, n);

        // Warmup
        for _ in 0..warmup {
            let mut out = Vec::new();
            let _ = lele::kernels::quantization::fused_dq_gemm_prepared_x86(
                &a_view,
                &pw,
                Some(128),
                &ws_view,
                Some(&bias_view),
                apply_relu,
                &mut out,
            );
        }

        // Benchmark single call
        let start = Instant::now();
        for _ in 0..iterations {
            let mut out = Vec::new();
            let _ = lele::kernels::quantization::fused_dq_gemm_prepared_x86(
                &a_view,
                &pw,
                Some(128),
                &ws_view,
                Some(&bias_view),
                apply_relu,
                &mut out,
            );
        }
        let single_time = start.elapsed().as_secs_f64() / iterations as f64;

        let total_for_calls = single_time * num_calls as f64;
        total_time_all_calls += total_for_calls;

        // Integer ops: M * K * N * 2 (multiply + accumulate)
        let ops = 2.0 * m as f64 * k as f64 * n as f64;
        let gops = ops / single_time / 1e9;
        println!("  Single call: {:.3} ms", single_time * 1000.0);
        println!(
            "  All {} calls: {:.3} ms",
            num_calls,
            total_for_calls * 1000.0
        );
        println!("  Int8 GOPS:   {:.2}", gops);
    }

    println!("\n  ──────────────────────────────────────");
    println!(
        "  TOTAL estimated int8 GEMM time: {:.1} ms (out of ~570ms inference)",
        total_time_all_calls * 1000.0
    );
    println!(
        "  Target (ORT-level): ~{:.0} ms",
        total_time_all_calls * 1000.0 * 272.0 / 570.0
    );

    Ok(())
}

fn benchmark_dynamic_quantize() -> Result<()> {
    println!("\n\n=== Dynamic Quantize Linear (f32 → u8) ===");

    // SenseVoice: input is [93, K] where K ∈ {512, 560, 2048}
    let test_cases = vec![
        (93 * 512, "SenseVoice (93×512 = 47.6K)"),
        (93 * 2048, "SenseVoice FFN (93×2048 = 190.5K)"),
        (256 * 1024, "Large (256×1024 = 256K)"),
    ];

    let warmup = 10;
    let iterations = 100;

    for (size, name) in test_cases {
        println!("\n{}", name);

        let data: Vec<f32> = (0..size)
            .map(|i| ((i as f32) / size as f32) * 4.0 - 2.0)
            .collect();
        let shape = vec![size];
        let input = lele::tensor::TensorView::new(&data, &shape);

        let mut out_y = Vec::new();
        let mut out_s = Vec::new();
        let mut out_z = Vec::new();

        // Warmup
        for _ in 0..warmup {
            out_y.clear();
            out_s.clear();
            out_z.clear();
            let _ = lele::kernels::quantization::dynamic_quantize_linear(
                &input, &mut out_y, &mut out_s, &mut out_z,
            );
        }

        // Benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            out_y.clear();
            out_s.clear();
            out_z.clear();
            let _ = lele::kernels::quantization::dynamic_quantize_linear(
                &input, &mut out_y, &mut out_s, &mut out_z,
            );
        }
        let time = start.elapsed().as_secs_f64() / iterations as f64;

        let bytes = size as f64 * 4.0 + size as f64; // read f32 + write u8(as f32)
        let gbps = bytes / time / 1e9;
        println!("  Time: {:.3} ms ({:.1} µs)", time * 1000.0, time * 1e6);
        println!("  Throughput: {:.2} GB/s", gbps);
    }

    Ok(())
}

fn benchmark_transpose() -> Result<()> {
    println!("\n\n=== Transpose (heavily used in attention) ===");

    // SenseVoice attention: lots of transposes with various shapes
    let test_cases = vec![
        (vec![1, 93, 8, 64], vec![0, 2, 1, 3], "Attn reshape (1,93,8,64)→(1,8,93,64)"),
        (vec![1, 8, 93, 64], vec![0, 1, 3, 2], "Attn K^T (1,8,93,64)→(1,8,64,93)"),
        (vec![1, 8, 93, 93], vec![0, 2, 1, 3], "Attn out (1,8,93,93)→(1,93,8,93)"),
        (vec![93, 512], vec![1, 0], "2D transpose (93,512)→(512,93)"),
        (vec![93, 1536], vec![1, 0], "2D transpose (93,1536)→(1536,93)"),
    ];

    let warmup = 10;
    let iterations = 100;

    for (shape, perm, name) in &test_cases {
        println!("\n{}", name);

        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size)
            .map(|i| (i as f32) * 0.01)
            .collect();

        let input = lele::tensor::TensorView::new(&data, shape);

        // Warmup
        for _ in 0..warmup {
            let mut out = Vec::new();
            let _ = lele::kernels::manipulation::transpose(&input, perm, &mut out);
        }

        // Benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            let mut out = Vec::new();
            let _ = lele::kernels::manipulation::transpose(&input, perm, &mut out);
        }
        let time = start.elapsed().as_secs_f64() / iterations as f64;

        let bytes = size as f64 * 4.0 * 2.0; // read + write
        let gbps = bytes / time / 1e9;
        println!("  Time: {:.3} ms ({:.1} µs)", time * 1000.0, time * 1e6);
        println!("  Throughput: {:.2} GB/s", gbps);
    }

    Ok(())
}

fn benchmark_prepare_weights() -> Result<()> {
    println!("\n\n=== Prepare Weights (one-time weight packing) ===");

    let test_cases = vec![
        (512, 512, "K=512×N=512"),
        (512, 1536, "K=512×N=1536"),
        (512, 2048, "K=512×N=2048"),
        (2048, 512, "K=2048×N=512"),
        (512, 25055, "K=512×N=25055 (logits)"),
    ];

    let warmup = 5;
    let iterations = 50;

    for (k, n, name) in test_cases {
        println!("\n{}", name);

        let b_u8: Vec<u8> = (0..k * n).map(|i| ((i * 37 + 13) % 256) as u8).collect();

        // Warmup
        for _ in 0..warmup {
            let _ = lele::kernels::quantization::prepare_weights(&b_u8, k, n);
        }

        // Benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = lele::kernels::quantization::prepare_weights(&b_u8, k, n);
        }
        let time = start.elapsed().as_secs_f64() / iterations as f64;

        let bytes = (k * n) as f64 * 2.0; // read + write
        let gbps = bytes / time / 1e9;
        println!("  Time: {:.3} ms ({:.1} µs)", time * 1000.0, time * 1e6);
        println!("  Throughput: {:.2} GB/s", gbps);
    }

    Ok(())
}

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          Lele vs Candle Performance Benchmark               ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    #[cfg(target_arch = "aarch64")]
    println!("\nPlatform: ARM64 (Apple Silicon or ARM)");
    #[cfg(target_arch = "x86_64")]
    println!("\nPlatform: x86_64");

    // === SenseVoice critical path benchmarks ===
    benchmark_int8_gemm()?;
    benchmark_dynamic_quantize()?;
    benchmark_prepare_weights()?;
    benchmark_transpose()?;

    // === General operator benchmarks ===
    benchmark_matmul()?;
    benchmark_mul()?;
    benchmark_silu()?;
    benchmark_softmax()?;
    benchmark_layer_norm()?;
    benchmark_erf()?;
    // benchmark_conv1d()?;  // Skipped: Candle's conv1d has capacity issues
    // benchmark_conv2d()?;  // Skipped: Too slow even with minimal data
    // benchmark_lstm()?;     // Skipped: Too slow even with minimal data

    println!("\n\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    Benchmark Complete                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    Ok(())
}
