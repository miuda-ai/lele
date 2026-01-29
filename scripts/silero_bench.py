import onnxruntime as ort
import numpy as np
import time
import sys

def main():
    model_path = 'examples/silero/silero.onnx'
    # Use CPU provider
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    
    try:
        sess = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    input_data = np.zeros((1, 512), dtype=np.float32)
    state_data = np.zeros((2, 1, 128), dtype=np.float32)
    sr_data = np.array([16000], dtype=np.int64)

    print("Warmup (100 iterations)...")
    for _ in range(100):
        _ = sess.run(None, {'input': input_data, 'state': state_data, 'sr': sr_data})

    print("Benchmarking (1000 iterations)...")
    iterations = 1000
    start = time.perf_counter()
    for _ in range(iterations):
        outputs = sess.run(None, {'input': input_data, 'state': state_data, 'sr': sr_data})
        # Update state for next iteration (matches lele bench behavior)
        state_data = outputs[1] 
    end = time.perf_counter()

    avg_ms = (end - start) * 1000 / iterations
    rtf = avg_ms / 32.0

    print("\n=== ORT Results ===")
    print(f"Avg Latency: {avg_ms:.4f} ms")
    print(f"RTF: {rtf:.6f}")

if __name__ == "__main__":
    main()
