#!/usr/bin/env python3
"""
ORT baseline benchmark for comparison with lele
Run with: python3 benches/bench_ort.py
"""
import numpy as np
import onnxruntime as ort
import time
import json
import onnx
from onnx import helper, TensorProto

# Warmup iterations
WARMUP = 10
# Benchmark iterations
ITERATIONS = 100

def bench_matmul():
    """MatMul benchmark"""
    results = {}
    sizes = [
        (4, 512, 512),
        (8, 512, 512),
        (16, 256, 256),
        (1, 512, 2048),
        (128, 128, 128),
    ]

    for m, k, n in sizes:
        a_info = helper.make_tensor_value_info("a", TensorProto.FLOAT, [m, k])
        b_info = helper.make_tensor_value_info("b", TensorProto.FLOAT, [k, n])
        c_info = helper.make_tensor_value_info("c", TensorProto.FLOAT, [m, n])
        node = helper.make_node("MatMul", ["a", "b"], ["c"])
        graph = helper.make_graph([node], "matmul", [a_info, b_info], [c_info])
        model = helper.make_model(graph)
        model.ir_version = 8
        model.opset_import[0].version = 12

        sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])

        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)

        # Warmup
        for _ in range(WARMUP):
            sess.run(None, {"a": a, "b": b})

        # Benchmark
        times = []
        for _ in range(ITERATIONS):
            start = time.perf_counter()
            sess.run(None, {"a": a, "b": b})
            times.append(time.perf_counter() - start)

        times = np.array(times)
        key = f"matmul_{m}x{k}x{n}"
        results[key] = {
            "mean_us": float(np.mean(times) * 1e6),
            "min_us": float(np.min(times) * 1e6),
            "median_us": float(np.median(times) * 1e6),
        }
        print(f"matmul {m}x{k}x{n}: {results[key]['median_us']:.2f} us")

    return results

def bench_softmax():
    """Softmax benchmark"""
    results = {}
    sizes = [
        (1, 512),
        (4, 512),
        (16, 256),
        (64, 128),
    ]

    for batch, size in sizes:
        x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [batch, size])
        y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [batch, size])
        node = helper.make_node("Softmax", ["x"], ["y"], axis=-1)
        graph = helper.make_graph([node], "softmax", [x_info], [y_info])
        model = helper.make_model(graph)
        model.ir_version = 8
        model.opset_import[0].version = 12

        sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])

        x = np.random.randn(batch, size).astype(np.float32)

        # Warmup
        for _ in range(WARMUP):
            sess.run(None, {"x": x})

        # Benchmark
        times = []
        for _ in range(ITERATIONS):
            start = time.perf_counter()
            sess.run(None, {"x": x})
            times.append(time.perf_counter() - start)

        times = np.array(times)
        key = f"softmax_{batch}x{size}"
        results[key] = {
            "mean_us": float(np.mean(times) * 1e6),
            "min_us": float(np.min(times) * 1e6),
            "median_us": float(np.median(times) * 1e6),
        }
        print(f"softmax {batch}x{size}: {results[key]['median_us']:.2f} us")

    return results

def bench_layernorm():
    """LayerNorm benchmark"""
    results = {}
    sizes = [
        (1, 512),
        (4, 512),
        (16, 256),
        (128, 512),
    ]

    for batch, size in sizes:
        x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [batch, size])
        scale_info = helper.make_tensor_value_info("scale", TensorProto.FLOAT, [size])
        bias_info = helper.make_tensor_value_info("bias", TensorProto.FLOAT, [size])
        y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [batch, size])
        node = helper.make_node(
            "LayerNormalization",
            ["x", "scale", "bias"],
            ["y"],
            axis=-1,
            epsilon=1e-5
        )
        graph = helper.make_graph([node], "layernorm", [x_info, scale_info, bias_info], [y_info])
        model = helper.make_model(graph)
        model.ir_version = 8
        model.opset_import[0].version = 12

        sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])

        x = np.random.randn(batch, size).astype(np.float32)
        scale = np.ones(size, dtype=np.float32)
        bias = np.zeros(size, dtype=np.float32)

        # Warmup
        for _ in range(WARMUP):
            sess.run(None, {"x": x, "scale": scale, "bias": bias})

        # Benchmark
        times = []
        for _ in range(ITERATIONS):
            start = time.perf_counter()
            sess.run(None, {"x": x, "scale": scale, "bias": bias})
            times.append(time.perf_counter() - start)

        times = np.array(times)
        key = f"layernorm_{batch}x{size}"
        results[key] = {
            "mean_us": float(np.mean(times) * 1e6),
            "min_us": float(np.min(times) * 1e6),
            "median_us": float(np.median(times) * 1e6),
        }
        print(f"layernorm {batch}x{size}: {results[key]['median_us']:.2f} us")

    return results

def bench_add_mul():
    """Elementwise add/mul benchmark"""
    results = {}
    sizes = [512, 1024, 2048, 4096]

    for size in sizes:
        for op in ["Add", "Mul"]:
            a_info = helper.make_tensor_value_info("a", TensorProto.FLOAT, [size])
            b_info = helper.make_tensor_value_info("b", TensorProto.FLOAT, [size])
            c_info = helper.make_tensor_value_info("c", TensorProto.FLOAT, [size])
            node = helper.make_node(op, ["a", "b"], ["c"])
            graph = helper.make_graph([node], f"{op.lower()}", [a_info, b_info], [c_info])
            model = helper.make_model(graph)
            model.ir_version = 8
            model.opset_import[0].version = 12

            sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])

            a = np.random.randn(size).astype(np.float32)
            b = np.random.randn(size).astype(np.float32)

            # Warmup
            for _ in range(WARMUP):
                sess.run(None, {"a": a, "b": b})

            # Benchmark
            times = []
            for _ in range(ITERATIONS):
                start = time.perf_counter()
                sess.run(None, {"a": a, "b": b})
                times.append(time.perf_counter() - start)

            times = np.array(times)
            key = f"{op.lower()}_{size}"
            results[key] = {
                "mean_us": float(np.mean(times) * 1e6),
                "min_us": float(np.min(times) * 1e6),
                "median_us": float(np.median(times) * 1e6),
            }
            print(f"{op.lower()} {size}: {results[key]['median_us']:.2f} us")

    return results

def bench_relu():
    """ReLU benchmark"""
    results = {}
    sizes = [512, 1024, 2048, 4096]

    for size in sizes:
        x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [size])
        y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [size])
        node = helper.make_node("Relu", ["x"], ["y"])
        graph = helper.make_graph([node], "relu", [x_info], [y_info])
        model = helper.make_model(graph)
        model.ir_version = 8
        model.opset_import[0].version = 12

        sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])

        x = np.random.randn(size).astype(np.float32)

        # Warmup
        for _ in range(WARMUP):
            sess.run(None, {"x": x})

        # Benchmark
        times = []
        for _ in range(ITERATIONS):
            start = time.perf_counter()
            sess.run(None, {"x": x})
            times.append(time.perf_counter() - start)

        times = np.array(times)
        key = f"relu_{size}"
        results[key] = {
            "mean_us": float(np.mean(times) * 1e6),
            "min_us": float(np.min(times) * 1e6),
            "median_us": float(np.median(times) * 1e6),
        }
        print(f"relu {size}: {results[key]['median_us']:.2f} us")

    return results

def main():
    print("=" * 60)
    print("ORT Baseline Benchmark")
    print("=" * 60)

    all_results = {}

    print("\n--- MatMul ---")
    all_results["matmul"] = bench_matmul()

    print("\n--- Softmax ---")
    all_results["softmax"] = bench_softmax()

    print("\n--- LayerNorm ---")
    all_results["layernorm"] = bench_layernorm()

    print("\n--- Elementwise ---")
    all_results["elementwise"] = bench_add_mul()

    print("\n--- ReLU ---")
    all_results["relu"] = bench_relu()

    # Save results
    with open("ort_benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to ort_benchmark_results.json")

if __name__ == "__main__":
    main()
