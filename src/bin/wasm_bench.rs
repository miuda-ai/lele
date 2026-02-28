//! WASM performance benchmark for key kernels.
//! Build with: cargo build --release --bin wasm_bench --target wasm32-wasip1
//! Run with:   wasmtime --wasm simd target/wasm32-wasip1/release/wasm_bench.wasm

use lele::kernels::fused_quantized_linear;
use lele::kernels::quantization::{dynamic_quantize_linear, mat_mul_integer_with_scale_bias};
use lele::kernels::{gemm, math, norm};
use lele::tensor::TensorView;
use std::borrow::Cow;

/// Simple timer
fn now_ns() -> u64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static START: OnceLock<Instant> = OnceLock::new();
    let start = START.get_or_init(Instant::now);
    start.elapsed().as_nanos() as u64
}

fn elapsed_ms(start: u64, end: u64) -> f64 {
    (end - start) as f64 / 1_000_000.0
}

/// Generate deterministic data
fn gen_data(n: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; n];
    for i in 0..n {
        v[i] = ((i as f32 * 0.001) % 2.0) - 1.0; // values in [-1, 1]
    }
    v
}

fn bench_matmul_nn(m: usize, k: usize, n: usize, iters: usize) -> f64 {
    let a_data = gen_data(m * k);
    let b_data = gen_data(k * n);
    let a = TensorView {
        data: Cow::Borrowed(&a_data),
        shape: Cow::Owned(vec![m, k]),
    };
    let b = TensorView {
        data: Cow::Borrowed(&b_data),
        shape: Cow::Owned(vec![k, n]),
    };
    let mut out = Vec::new();

    // Warmup
    let _ = gemm::matmul(&a, &b, &mut out);

    let start = now_ns();
    for _ in 0..iters {
        let _ = gemm::matmul(&a, &b, &mut out);
    }
    let end = now_ns();
    elapsed_ms(start, end) / iters as f64
}

fn bench_gemm_transb(m: usize, k: usize, n: usize, iters: usize) -> f64 {
    // B is stored as [n, k] (transposed)
    let a_data = gen_data(m * k);
    let b_data = gen_data(n * k); // B^T shape [n, k]
    let a = TensorView {
        data: Cow::Borrowed(&a_data),
        shape: Cow::Owned(vec![m, k]),
    };
    let b = TensorView {
        data: Cow::Borrowed(&b_data),
        shape: Cow::Owned(vec![n, k]),
    };
    let mut out = Vec::new();

    // Warmup
    let _ = gemm::gemm(&a, &b, None, 1.0, 0.0, false, true, &mut out);

    let start = now_ns();
    for _ in 0..iters {
        let _ = gemm::gemm(&a, &b, None, 1.0, 0.0, false, true, &mut out);
    }
    let end = now_ns();
    elapsed_ms(start, end) / iters as f64
}

fn bench_add(n: usize, iters: usize) -> f64 {
    let a_data = gen_data(n);
    let b_data = gen_data(n);
    let a = TensorView {
        data: Cow::Borrowed(&a_data),
        shape: Cow::Owned(vec![n]),
    };
    let b = TensorView {
        data: Cow::Borrowed(&b_data),
        shape: Cow::Owned(vec![n]),
    };
    let mut out = vec![0.0f32; n];

    // Warmup
    let _ = math::add(&a, &b, &mut out);

    let start = now_ns();
    for _ in 0..iters {
        let _ = math::add(&a, &b, &mut out);
    }
    let end = now_ns();
    elapsed_ms(start, end) / iters as f64
}

fn bench_mul(n: usize, iters: usize) -> f64 {
    let a_data = gen_data(n);
    let b_data = gen_data(n);
    let a = TensorView {
        data: Cow::Borrowed(&a_data),
        shape: Cow::Owned(vec![n]),
    };
    let b = TensorView {
        data: Cow::Borrowed(&b_data),
        shape: Cow::Owned(vec![n]),
    };
    let mut out = vec![0.0f32; n];

    let _ = math::mul(&a, &b, &mut out);

    let start = now_ns();
    for _ in 0..iters {
        let _ = math::mul(&a, &b, &mut out);
    }
    let end = now_ns();
    elapsed_ms(start, end) / iters as f64
}

fn bench_sub(n: usize, iters: usize) -> f64 {
    let a_data = gen_data(n);
    let b_data = gen_data(n);
    let a = TensorView {
        data: Cow::Borrowed(&a_data),
        shape: Cow::Owned(vec![n]),
    };
    let b = TensorView {
        data: Cow::Borrowed(&b_data),
        shape: Cow::Owned(vec![n]),
    };
    let mut out = vec![0.0f32; n];

    let _ = math::sub(&a, &b, &mut out);

    let start = now_ns();
    for _ in 0..iters {
        let _ = math::sub(&a, &b, &mut out);
    }
    let end = now_ns();
    elapsed_ms(start, end) / iters as f64
}

fn bench_gelu(n: usize, iters: usize) -> f64 {
    let data = gen_data(n);
    let input = TensorView {
        data: Cow::Borrowed(&data),
        shape: Cow::Owned(vec![n]),
    };
    let mut out = vec![0.0f32; n];

    let _ = math::gelu(&input, &mut out);

    let start = now_ns();
    for _ in 0..iters {
        let _ = math::gelu(&input, &mut out);
    }
    let end = now_ns();
    elapsed_ms(start, end) / iters as f64
}

fn bench_fast_gelu(n: usize, iters: usize) -> f64 {
    let data = gen_data(n);
    let input = TensorView {
        data: Cow::Borrowed(&data),
        shape: Cow::Owned(vec![n]),
    };
    let mut out = vec![0.0f32; n];

    let _ = math::fast_gelu(&input, &mut out);

    let start = now_ns();
    for _ in 0..iters {
        let _ = math::fast_gelu(&input, &mut out);
    }
    let end = now_ns();
    elapsed_ms(start, end) / iters as f64
}

fn bench_erf(n: usize, iters: usize) -> f64 {
    let data = gen_data(n);
    let input = TensorView {
        data: Cow::Borrowed(&data),
        shape: Cow::Owned(vec![n]),
    };
    let mut out = vec![0.0f32; n];

    let _ = math::erf(&input, &mut out);

    let start = now_ns();
    for _ in 0..iters {
        let _ = math::erf(&input, &mut out);
    }
    let end = now_ns();
    elapsed_ms(start, end) / iters as f64
}

fn bench_sigmoid(n: usize, iters: usize) -> f64 {
    let data = gen_data(n);
    let input = TensorView {
        data: Cow::Borrowed(&data),
        shape: Cow::Owned(vec![n]),
    };
    let mut out = vec![0.0f32; n];

    let _ = math::sigmoid(&input, &mut out);

    let start = now_ns();
    for _ in 0..iters {
        let _ = math::sigmoid(&input, &mut out);
    }
    let end = now_ns();
    elapsed_ms(start, end) / iters as f64
}

/// Benchmark fused_quantized_linear (dynamic quantize + INT8 matmul + scale/bias)
/// Reflects the dominant cost in SenseVoice encoder layers.
fn bench_int8_linear(m: usize, k: usize, n: usize, iters: usize) -> f64 {
    // Input: f32 activations
    let input_data: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32 * 0.01) % 2.0) - 1.0)
        .collect();
    // Weights: u8-range values as f32 (as returned by weight_u8)
    let weight_data: Vec<f32> = (0..k * n).map(|i| (i % 256) as f32).collect();
    let scale_data: Vec<f32> = vec![0.003921f32; n]; // 1/255
    let zero_data: Vec<f32> = vec![128.0f32];
    let bias_data: Vec<f32> = vec![0.0f32; n];

    let input = TensorView {
        data: Cow::Borrowed(&input_data),
        shape: Cow::Owned(vec![m, k]),
    };
    let weight = TensorView {
        data: Cow::Borrowed(&weight_data),
        shape: Cow::Owned(vec![k, n]),
    };
    let scale = TensorView {
        data: Cow::Borrowed(&scale_data),
        shape: Cow::Owned(vec![n]),
    };
    let zero = TensorView {
        data: Cow::Borrowed(&zero_data),
        shape: Cow::Owned(vec![1]),
    };
    let bias = TensorView {
        data: Cow::Borrowed(&bias_data),
        shape: Cow::Owned(vec![n]),
    };
    let mut out = Vec::new();

    // Warmup
    let _ = fused_quantized_linear(&input, &weight, &scale, &zero, &bias, false, &mut out);

    let start = now_ns();
    for _ in 0..iters {
        let _ = fused_quantized_linear(&input, &weight, &scale, &zero, &bias, false, &mut out);
    }
    let end = now_ns();
    elapsed_ms(start, end) / iters as f64
}

/// Scalar reference implementation for INT8 matmul: C = (A - zp_a) * (B - zp_b)
fn ref_mat_mul_integer(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    zp_a: f32,
    zp_b: f32,
) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += (a[i * k + l] - zp_a) * (b[l * n + j] - zp_b);
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Test mat_mul_integer_with_scale_bias correctness against scalar reference
fn test_matmul_integer_correctness() -> bool {
    println!("--- INT8 MatMul Correctness Tests ---");
    let mut all_pass = true;

    // Test various shapes: M, K, N — especially non-multiples of 4 and 16
    let test_shapes = [
        (1, 16, 16),     // tiny
        (3, 32, 17),     // N not multiple of 16
        (5, 64, 15),     // N = 15 (remainder only)
        (7, 128, 33),    // M=7 (4+3 remainder), N=33
        (93, 512, 1536), // SenseVoice QKV
        (93, 512, 512),  // SenseVoice out_proj
        (93, 512, 2048), // SenseVoice FFN1
        (93, 2048, 512), // SenseVoice FFN2
        (10, 192, 192),  // YOLO26-like
        (100, 384, 384), // YOLO26-like
        (25, 768, 768),  // YOLO26-like
        (4, 128, 93),    // Attention K*Q^T shape (small M)
        (1, 64, 300),    // YOLO26 head-like
        (300, 192, 80),  // YOLO26 class head (N=80, not multiple of 16)
        (300, 192, 4),   // YOLO26 box head (N=4!)
        (300, 64, 1),    // N=1 edge case
        (1, 1, 1),       // minimal
        (2, 3, 5),       // all primes
    ];

    let zp_a = 128.0f32;
    let zp_b = 127.0f32;

    for &(m, k, n) in &test_shapes {
        // Generate deterministic u8-range data
        let mut a_data = vec![0.0f32; m * k];
        let mut b_data = vec![0.0f32; k * n];
        for i in 0..a_data.len() {
            a_data[i] = ((i * 7 + 13) % 256) as f32;
        }
        for i in 0..b_data.len() {
            b_data[i] = ((i * 11 + 3) % 256) as f32;
        }

        // Reference result
        let ref_c = ref_mat_mul_integer(&a_data, &b_data, m, k, n, zp_a, zp_b);

        // WASM SIMD result via mat_mul_integer_with_scale_bias (no scale/bias)
        let a_tv = TensorView {
            data: Cow::Borrowed(&a_data),
            shape: Cow::Owned(vec![m, k]),
        };
        let b_tv = TensorView {
            data: Cow::Borrowed(&b_data),
            shape: Cow::Owned(vec![k, n]),
        };
        let zp_a_tv = TensorView {
            data: Cow::Owned(vec![zp_a]),
            shape: Cow::Owned(vec![1]),
        };
        let zp_b_tv = TensorView {
            data: Cow::Owned(vec![zp_b]),
            shape: Cow::Owned(vec![1]),
        };
        let mut out = Vec::new();
        let result = mat_mul_integer_with_scale_bias(
            &a_tv,
            &b_tv,
            Some(&zp_a_tv),
            Some(&zp_b_tv),
            None,
            None,
            &mut out,
        );

        // Compare
        let mut max_err: f32 = 0.0;
        let mut max_rel_err: f32 = 0.0;
        let mut fail_idx = 0;
        for i in 0..ref_c.len() {
            let diff = (result.data[i] - ref_c[i]).abs();
            let rel = if ref_c[i].abs() > 1.0 {
                diff / ref_c[i].abs()
            } else {
                diff
            };
            if diff > max_err {
                max_err = diff;
                fail_idx = i;
            }
            if rel > max_rel_err {
                max_rel_err = rel;
            }
        }

        // Allow small FP rounding (accumulation order differs)
        let tol = k as f32 * 0.01; // tolerance scales with K
        let pass = max_err < tol && !result.data.iter().any(|v| v.is_nan());
        if pass {
            println!("  ✓ [{m}x{k}]*[{k}x{n}] max_err={max_err:.4}");
        } else {
            println!(
                "  ✗ [{m}x{k}]*[{k}x{n}] FAIL max_err={max_err:.2} at idx={fail_idx} ref={:.4} got={:.4}",
                ref_c[fail_idx], result.data[fail_idx]
            );
            if result.data.iter().any(|v| v.is_nan()) {
                println!("    NaN detected in output!");
            }
            all_pass = false;
        }
    }
    all_pass
}

/// Test conv_integer correctness (verifies conv2d faer_matmul call is NOT skipped on WASM)
fn test_conv_integer_correctness() -> bool {
    use lele::kernels::conv2d::conv_integer;

    println!("--- Conv Integer (conv2d) Correctness Tests ---");
    let mut all_pass = true;

    // Test 1: 1x1 conv (exercises the fast 1x1 path in conv2d_with_zero_points)
    // Input: [1, 3, 4, 4], Weight: [8, 3, 1, 1]
    {
        let in_c = 3usize;
        let out_c = 8usize;
        let h = 4usize;
        let w = 4usize;
        let spatial = h * w;

        // Create input data (quantized-like f32 values)
        let mut input_data = vec![0.0f32; 1 * in_c * h * w];
        for i in 0..input_data.len() {
            input_data[i] = ((i % 256) as f32) - 128.0; // int8-like values
        }
        let input = TensorView {
            data: Cow::Borrowed(&input_data),
            shape: Cow::Owned(vec![1, in_c, h, w]),
        };

        // Create weight data
        let mut weight_data = vec![0.0f32; out_c * in_c * 1 * 1];
        for i in 0..weight_data.len() {
            weight_data[i] = ((i % 256) as f32) - 128.0;
        }
        let weights = TensorView {
            data: Cow::Borrowed(&weight_data),
            shape: Cow::Owned(vec![out_c, in_c, 1, 1]),
        };

        let mut out = Vec::new();
        let result = conv_integer(
            &input,
            &weights,
            None,
            None,
            &[1, 1],
            1,
            &[0, 0, 0, 0],
            &[1, 1],
            &mut out,
        );

        // Verify output shape
        assert!(
            result.shape.as_ref() == &[1, out_c, h, w],
            "conv 1x1 shape mismatch"
        );

        // Compute reference: for 1x1 conv, it's just matmul: W[out_c, in_c] * X[in_c, spatial]
        let mut ref_out = vec![0.0f32; out_c * spatial];
        for oc in 0..out_c {
            for s in 0..spatial {
                let mut acc = 0.0f32;
                for ic in 0..in_c {
                    acc += weight_data[oc * in_c + ic] * input_data[ic * spatial + s];
                }
                ref_out[oc * spatial + s] = acc;
            }
        }

        let max_err = ref_out
            .iter()
            .zip(result.data.iter())
            .map(|(r, a)| (r - a).abs())
            .fold(0.0f32, f32::max);

        let pass = max_err < 1e-3;
        if pass {
            println!(
                "  ✓ conv 1x1 [1,{},{},{}] w:[{},{},1,1] max_err={:.4}",
                in_c, h, w, out_c, in_c, max_err
            );
        } else {
            println!(
                "  ✗ conv 1x1 [1,{},{},{}] w:[{},{},1,1] max_err={:.4} FAIL",
                in_c, h, w, out_c, in_c, max_err
            );
            // Print first few values for debugging
            println!("    ref[0..4]: {:?}", &ref_out[..4.min(ref_out.len())]);
            println!(
                "    out[0..4]: {:?}",
                &result.data[..4.min(result.data.len())]
            );
            all_pass = false;
        }
    }

    // Test 2: 3x3 conv (exercises the im2col path)
    // Input: [1, 3, 8, 8], Weight: [16, 3, 3, 3], stride=1, pad=1
    {
        let in_c = 3usize;
        let out_c = 16usize;
        let h = 8usize;
        let w = 8usize;

        let mut input_data = vec![0.0f32; 1 * in_c * h * w];
        for i in 0..input_data.len() {
            input_data[i] = ((i * 7 + 3) % 256) as f32 - 128.0;
        }
        let input = TensorView {
            data: Cow::Borrowed(&input_data),
            shape: Cow::Owned(vec![1, in_c, h, w]),
        };

        let mut weight_data = vec![0.0f32; out_c * in_c * 3 * 3];
        for i in 0..weight_data.len() {
            weight_data[i] = ((i * 13 + 5) % 256) as f32 - 128.0;
        }
        let weights = TensorView {
            data: Cow::Borrowed(&weight_data),
            shape: Cow::Owned(vec![out_c, in_c, 3, 3]),
        };

        let mut out = Vec::new();
        let result = conv_integer(
            &input,
            &weights,
            None,
            None,
            &[1, 1],
            1,
            &[1, 1, 1, 1], // padding=1 => same spatial
            &[1, 1],
            &mut out,
        );

        assert!(
            result.shape.as_ref() == &[1, out_c, h, w],
            "conv 3x3 shape mismatch"
        );

        // Check output is not all zeros (the old bug left output uninitialized/zero)
        let sum: f32 = result.data.iter().map(|x| x.abs()).sum();
        let non_zero_count = result.data.iter().filter(|&&x| x != 0.0).count();
        let total = result.data.len();
        let pass = sum > 0.0 && non_zero_count > total / 2;
        if pass {
            println!(
                "  ✓ conv 3x3 [1,{},{},{}] w:[{},{},3,3] pad=1 sum={:.1} nonzero={}/{}",
                in_c, h, w, out_c, in_c, sum, non_zero_count, total
            );
        } else {
            println!(
                "  ✗ conv 3x3 [1,{},{},{}] w:[{},{},3,3] pad=1 sum={:.1} nonzero={}/{} FAIL",
                in_c, h, w, out_c, in_c, sum, non_zero_count, total
            );
            println!(
                "    out[0..8]: {:?}",
                &result.data[..8.min(result.data.len())]
            );
            all_pass = false;
        }
    }

    // Test 3: 1x1 conv with zero points (x_zp != 0)
    {
        let in_c = 4usize;
        let out_c = 8usize;
        let h = 4usize;
        let w = 4usize;

        let mut input_data = vec![0.0f32; 1 * in_c * h * w];
        for i in 0..input_data.len() {
            input_data[i] = (i % 256) as f32; // unsigned-like [0..255]
        }
        let input = TensorView {
            data: Cow::Borrowed(&input_data),
            shape: Cow::Owned(vec![1, in_c, h, w]),
        };

        let mut weight_data = vec![0.0f32; out_c * in_c];
        for i in 0..weight_data.len() {
            weight_data[i] = ((i * 11 + 7) % 256) as f32 - 128.0;
        }
        let weights = TensorView {
            data: Cow::Borrowed(&weight_data),
            shape: Cow::Owned(vec![out_c, in_c, 1, 1]),
        };

        let x_zp_data = vec![128.0f32];
        let x_zp = TensorView {
            data: Cow::Borrowed(&x_zp_data),
            shape: Cow::Owned(vec![1]),
        };

        let mut out = Vec::new();
        let result = conv_integer(
            &input,
            &weights,
            Some(&x_zp),
            None,
            &[1, 1],
            1,
            &[0, 0, 0, 0],
            &[1, 1],
            &mut out,
        );

        assert!(
            result.shape.as_ref() == &[1, out_c, h, w],
            "conv 1x1 zp shape mismatch"
        );

        let sum: f32 = result.data.iter().map(|x| x.abs()).sum();
        let non_zero_count = result.data.iter().filter(|&&x| x != 0.0).count();
        let total = result.data.len();
        let pass = sum > 0.0 && non_zero_count > total / 2;
        if pass {
            println!(
                "  ✓ conv 1x1 zp=128 [1,{},{},{}] w:[{},{},1,1] sum={:.1} nonzero={}/{}",
                in_c, h, w, out_c, in_c, sum, non_zero_count, total
            );
        } else {
            println!(
                "  ✗ conv 1x1 zp=128 [1,{},{},{}] w:[{},{},1,1] sum={:.1} nonzero={}/{} FAIL",
                in_c, h, w, out_c, in_c, sum, non_zero_count, total
            );
            all_pass = false;
        }
    }

    all_pass
}

fn main() {
    println!("=== Lele WASM Kernel Benchmark ===");
    println!();

    // Correctness tests FIRST
    let pass = test_matmul_integer_correctness();
    println!();
    if !pass {
        println!("!!! CORRECTNESS TESTS FAILED - aborting benchmarks !!!");
        return;
    }

    let pass2 = test_conv_integer_correctness();
    println!();
    if !pass2 {
        println!("!!! CONV INTEGER TESTS FAILED - aborting benchmarks !!!");
        return;
    }

    // GEMM benchmarks - the biggest bottleneck
    println!("--- GEMM (MatMul) ---");
    let ms = bench_matmul_nn(128, 256, 128, 20);
    println!("  MatMul [128x256] * [256x128] (no trans): {:.3} ms", ms);

    let ms = bench_matmul_nn(256, 512, 256, 10);
    println!("  MatMul [256x512] * [512x256] (no trans): {:.3} ms", ms);

    let ms = bench_gemm_transb(128, 256, 128, 20);
    println!("  Gemm   [128x256] * [128x256]^T (transB): {:.3} ms", ms);

    let ms = bench_gemm_transb(256, 512, 256, 10);
    println!("  Gemm   [256x512] * [256x512]^T (transB): {:.3} ms", ms);

    let ms = bench_gemm_transb(64, 512, 1024, 10);
    println!("  Gemm   [64x512]  * [1024x512]^T (transB): {:.3} ms", ms);

    // Elementwise benchmarks
    println!();
    println!("--- Elementwise (n=65536) ---");
    let n = 65536;
    let iters = 200;

    let ms = bench_add(n, iters);
    println!(
        "  add:      {:.4} ms ({:.1} MFLOPS)",
        ms,
        n as f64 / ms / 1e3
    );

    let ms = bench_mul(n, iters);
    println!(
        "  mul:      {:.4} ms ({:.1} MFLOPS)",
        ms,
        n as f64 / ms / 1e3
    );

    let ms = bench_sub(n, iters);
    println!(
        "  sub:      {:.4} ms ({:.1} MFLOPS)",
        ms,
        n as f64 / ms / 1e3
    );

    // Activation benchmarks
    println!();
    println!("--- Activations (n=65536) ---");

    let ms = bench_gelu(n, iters);
    println!(
        "  gelu:     {:.4} ms ({:.1} MFLOPS)",
        ms,
        n as f64 / ms / 1e3
    );

    let ms = bench_fast_gelu(n, iters);
    println!(
        "  fast_gelu:{:.4} ms ({:.1} MFLOPS)",
        ms,
        n as f64 / ms / 1e3
    );

    let ms = bench_erf(n, iters);
    println!(
        "  erf:      {:.4} ms ({:.1} MFLOPS)",
        ms,
        n as f64 / ms / 1e3
    );

    let ms = bench_sigmoid(n, iters);
    println!(
        "  sigmoid:  {:.4} ms ({:.1} MFLOPS)",
        ms,
        n as f64 / ms / 1e3
    );

    // INT8 matmul benchmarks (SenseVoice encoder shapes)
    println!();
    println!("--- INT8 Linear / fused_quantized_linear ---");

    let ms = bench_int8_linear(93, 560, 1536, 5);
    let flops = 93.0 * 560.0 * 1536.0 * 2.0;
    println!(
        "  INT8 [93x560]*[560x1536] (QKV): {:.2} ms ({:.0} MFLOPS)",
        ms,
        flops / ms / 1e3
    );

    let ms = bench_int8_linear(93, 512, 2048, 5);
    let flops = 93.0 * 512.0 * 2048.0 * 2.0;
    println!(
        "  INT8 [93x512]*[512x2048] (FFN): {:.2} ms ({:.0} MFLOPS)",
        ms,
        flops / ms / 1e3
    );

    let ms = bench_int8_linear(93, 560, 512, 5);
    let flops = 93.0 * 560.0 * 512.0 * 2.0;
    println!(
        "  INT8 [93x560]*[560x512]  (out): {:.2} ms ({:.0} MFLOPS)",
        ms,
        flops / ms / 1e3
    );

    // SenseVoice encoder layer simulation (T=93, d=512, heads=4, head_dim=128)
    println!();
    println!("--- INT8 Phase Breakdown (93x512 -> 1536) ---");
    {
        let m = 93;
        let k = 512;
        let n = 1536;
        let input_data: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32 * 0.01) % 2.0) - 1.0)
            .collect();
        let weight_data: Vec<f32> = (0..k * n).map(|i| (i % 256) as f32).collect();
        let scale_data = vec![0.003921f32; n];
        let zero_data = vec![128.0f32];
        let bias_data = vec![0.0f32; n];

        let input = TensorView {
            data: Cow::Borrowed(&input_data),
            shape: Cow::Owned(vec![m, k]),
        };
        let weight = TensorView {
            data: Cow::Borrowed(&weight_data),
            shape: Cow::Owned(vec![k, n]),
        };
        let w_scale = TensorView {
            data: Cow::Borrowed(&scale_data),
            shape: Cow::Owned(vec![n]),
        };
        let w_zero = TensorView {
            data: Cow::Borrowed(&zero_data),
            shape: Cow::Owned(vec![1]),
        };
        let bias = TensorView {
            data: Cow::Borrowed(&bias_data),
            shape: Cow::Owned(vec![n]),
        };

        let iters = 5;

        // Phase 1: dynamic_quantize_linear
        let mut buf_q = Vec::new();
        let mut buf_s = Vec::new();
        let mut buf_z = Vec::new();
        let _ = dynamic_quantize_linear(&input, &mut buf_q, &mut buf_s, &mut buf_z);
        let start = now_ns();
        for _ in 0..iters {
            let _ = dynamic_quantize_linear(&input, &mut buf_q, &mut buf_s, &mut buf_z);
        }
        let dq_ms = elapsed_ms(start, now_ns()) / iters as f64;
        println!("  dynamic_quantize_linear [{}x{}]: {:.3} ms", m, k, dq_ms);

        // Phase 2: combined_scale = mul(s, w_scale)
        let (q, s, z) = dynamic_quantize_linear(&input, &mut buf_q, &mut buf_s, &mut buf_z);
        let mut buf_sm = Vec::new();
        let cs = math::mul(&s, &w_scale, &mut buf_sm);
        let _ = cs;
        let start = now_ns();
        for _ in 0..iters {
            let mut tmp = Vec::new();
            let _ = math::mul(&s, &w_scale, &mut tmp);
        }
        let mul_ms = elapsed_ms(start, now_ns()) / iters as f64;
        println!("  mul(scale) [{}]:                 {:.3} ms", n, mul_ms);

        // Phase 3: mat_mul_integer_with_scale_bias
        let combined_scale = math::mul(&s, &w_scale, &mut buf_sm);
        let mut out = Vec::new();
        let _ = mat_mul_integer_with_scale_bias(
            &q,
            &weight,
            Some(&z),
            Some(&w_zero),
            Some(&combined_scale),
            Some(&bias),
            &mut out,
        );
        let start = now_ns();
        for _ in 0..iters {
            let _ = mat_mul_integer_with_scale_bias(
                &q,
                &weight,
                Some(&z),
                Some(&w_zero),
                Some(&combined_scale),
                Some(&bias),
                &mut out,
            );
        }
        let mm_ms = elapsed_ms(start, now_ns()) / iters as f64;
        let flops = m as f64 * k as f64 * n as f64 * 2.0;
        println!(
            "  mat_mul_integer [{}x{}]*[{}x{}]: {:.3} ms ({:.0} MFLOPS)",
            m,
            k,
            k,
            n,
            mm_ms,
            flops / mm_ms / 1e3
        );

        println!(
            "  total:                          {:.3} ms",
            dq_ms + mul_ms + mm_ms
        );

        // Check output for NaN
        let has_nan = out.iter().any(|v| v.is_nan());
        let has_inf = out.iter().any(|v| v.is_infinite());
        println!(
            "  output: len={}, has_nan={}, has_inf={}",
            out.len(),
            has_nan,
            has_inf
        );
        if out.len() > 0 {
            println!(
                "  output[0..3]: [{:.4}, {:.4}, {:.4}]",
                out[0], out[1], out[2]
            );
        }
    }

    println!();
    println!("--- SenseVoice Encoder Layer Simulation (T=93, d=512) ---");
    bench_encoder_layer(93, 512, 4);

    println!();
    println!("=== Done ===");
}

/// Benchmark all operations of a single SenseVoice encoder layer.
/// T = sequence length, d = hidden dim, h = num_heads.
fn bench_encoder_layer(t: usize, d: usize, h: usize) {
    let head_dim = d / h;
    let ffn_dim = d * 4; // 2048
    let iters = 5;

    let mut total_ms = 0.0;

    // ---- 1. LayerNorm (norm1): [T, d] -> [T, d] ----
    {
        let data = gen_data(t * d);
        let scale_data = vec![1.0f32; d];
        let bias_data = vec![0.0f32; d];
        let input = TensorView {
            data: Cow::Borrowed(&data),
            shape: Cow::Owned(vec![t, d]),
        };
        let scale = TensorView {
            data: Cow::Borrowed(&scale_data),
            shape: Cow::Owned(vec![d]),
        };
        let bias = TensorView {
            data: Cow::Borrowed(&bias_data),
            shape: Cow::Owned(vec![d]),
        };
        let mut out = Vec::new();
        let _ = norm::layer_norm(&input, &scale, &bias, -1, 1e-5, &mut out);
        let start = now_ns();
        for _ in 0..iters {
            let _ = norm::layer_norm(&input, &scale, &bias, -1, 1e-5, &mut out);
        }
        let ms = elapsed_ms(start, now_ns()) / iters as f64;
        println!("  layer_norm [{}x{}]:            {:.3} ms", t, d, ms);
        total_ms += ms;
    }

    // ---- 2. INT8 QKV linear: [T, d] -> [T, 3*d] ----
    {
        let ms = bench_int8_linear(t, d, d * 3, iters);
        println!(
            "  INT8 QKV [{}x{}]*[{}x{}]:     {:.3} ms",
            t,
            d,
            d,
            d * 3,
            ms
        );
        total_ms += ms;
    }

    // ---- 3. Attention: Q*K^T and attn*V ----
    // Q*K^T: [h, T, head_dim] * [h, head_dim, T] -> [h, T, T]  (batched matmul)
    {
        let q_data = gen_data(h * t * head_dim);
        let kt_data = gen_data(h * head_dim * t);
        let q = TensorView {
            data: Cow::Borrowed(&q_data),
            shape: Cow::Owned(vec![h, t, head_dim]),
        };
        let kt = TensorView {
            data: Cow::Borrowed(&kt_data),
            shape: Cow::Owned(vec![h, head_dim, t]),
        };
        let mut out = Vec::new();
        let _ = gemm::matmul(&q, &kt, &mut out);
        let start = now_ns();
        for _ in 0..iters {
            let _ = gemm::matmul(&q, &kt, &mut out);
        }
        let ms = elapsed_ms(start, now_ns()) / iters as f64;
        let flops = h as f64 * t as f64 * head_dim as f64 * t as f64 * 2.0;
        println!(
            "  Q*K^T [{}x{}x{}]*[{}x{}x{}]: {:.3} ms ({:.0} MFLOPS)",
            h,
            t,
            head_dim,
            h,
            head_dim,
            t,
            ms,
            flops / ms / 1e3
        );
        total_ms += ms;
    }

    // ---- 4. Softmax: [h, T, T] ----
    {
        let data = gen_data(h * t * t);
        let input = TensorView {
            data: Cow::Borrowed(&data),
            shape: Cow::Owned(vec![h, t, t]),
        };
        let mut out = Vec::new();
        let _ = norm::softmax(&input, -1, &mut out);
        let start = now_ns();
        for _ in 0..iters {
            let _ = norm::softmax(&input, -1, &mut out);
        }
        let ms = elapsed_ms(start, now_ns()) / iters as f64;
        println!("  softmax [{}x{}x{}]:          {:.3} ms", h, t, t, ms);
        total_ms += ms;
    }

    // ---- 5. Attn * V: [h, T, T] * [h, T, head_dim] -> [h, T, head_dim] ----
    {
        let a_data = gen_data(h * t * t);
        let v_data = gen_data(h * t * head_dim);
        let a = TensorView {
            data: Cow::Borrowed(&a_data),
            shape: Cow::Owned(vec![h, t, t]),
        };
        let v = TensorView {
            data: Cow::Borrowed(&v_data),
            shape: Cow::Owned(vec![h, t, head_dim]),
        };
        let mut out = Vec::new();
        let _ = gemm::matmul(&a, &v, &mut out);
        let start = now_ns();
        for _ in 0..iters {
            let _ = gemm::matmul(&a, &v, &mut out);
        }
        let ms = elapsed_ms(start, now_ns()) / iters as f64;
        let flops = h as f64 * t as f64 * t as f64 * head_dim as f64 * 2.0;
        println!(
            "  attn*V [{}x{}x{}]*[{}x{}x{}]: {:.3} ms ({:.0} MFLOPS)",
            h,
            t,
            t,
            h,
            t,
            head_dim,
            ms,
            flops / ms / 1e3
        );
        total_ms += ms;
    }

    // ---- 6. INT8 output projection: [T, d] -> [T, d] ----
    {
        let ms = bench_int8_linear(t, d, d, iters);
        println!("  INT8 out_proj [{}x{}]*[{}x{}]: {:.3} ms", t, d, d, d, ms);
        total_ms += ms;
    }

    // ---- 7. LayerNorm (norm2): [T, d] -> [T, d] ----
    {
        let data = gen_data(t * d);
        let scale_data = vec![1.0f32; d];
        let bias_data = vec![0.0f32; d];
        let input = TensorView {
            data: Cow::Borrowed(&data),
            shape: Cow::Owned(vec![t, d]),
        };
        let scale = TensorView {
            data: Cow::Borrowed(&scale_data),
            shape: Cow::Owned(vec![d]),
        };
        let bias = TensorView {
            data: Cow::Borrowed(&bias_data),
            shape: Cow::Owned(vec![d]),
        };
        let mut out = Vec::new();
        let _ = norm::layer_norm(&input, &scale, &bias, -1, 1e-5, &mut out);
        let start = now_ns();
        for _ in 0..iters {
            let _ = norm::layer_norm(&input, &scale, &bias, -1, 1e-5, &mut out);
        }
        let ms = elapsed_ms(start, now_ns()) / iters as f64;
        println!("  layer_norm2 [{}x{}]:           {:.3} ms", t, d, ms);
        total_ms += ms;
    }

    // ---- 8. INT8 FFN w1 (+ ReLU): [T, d] -> [T, 4d] ----
    {
        let ms = bench_int8_linear(t, d, ffn_dim, iters);
        println!(
            "  INT8 FFN1 [{}x{}]*[{}x{}]:   {:.3} ms",
            t, d, d, ffn_dim, ms
        );
        total_ms += ms;
    }

    // ---- 9. INT8 FFN w2: [T, 4d] -> [T, d] ----
    {
        let ms = bench_int8_linear(t, ffn_dim, d, iters);
        println!(
            "  INT8 FFN2 [{}x{}]*[{}x{}]: {:.3} ms",
            t, ffn_dim, ffn_dim, d, ms
        );
        total_ms += ms;
    }

    // ---- 10. Elementwise ops: ~9 x add/mul on [T, d] ----
    {
        let data = gen_data(t * d);
        let a = TensorView {
            data: Cow::Borrowed(&data),
            shape: Cow::Owned(vec![t, d]),
        };
        let b = TensorView {
            data: Cow::Borrowed(&data),
            shape: Cow::Owned(vec![t, d]),
        };
        let mut out = vec![0.0f32; t * d];
        let _ = math::add(&a, &b, &mut out);
        let start = now_ns();
        for _ in 0..(iters * 9) {
            let _ = math::add(&a, &b, &mut out);
        }
        let ms = elapsed_ms(start, now_ns()) / iters as f64;
        println!("  9x add/mul [{}x{}]:            {:.3} ms", t, d, ms);
        total_ms += ms;
    }

    println!("  ─────────────────────────────────────");
    println!("  TOTAL per layer:                {:.3} ms", total_ms);
    println!(
        "  x70 layers:                     {:.1} ms",
        total_ms * 70.0
    );
    println!(
        "  Estimated full model:           {:.1} ms (5.6s audio → RTF {:.3})",
        total_ms * 70.0,
        total_ms * 70.0 / 5600.0
    );
}
