"""ORT benchmark and accuracy comparison for YOLO26n-seg (instance segmentation).

Run from workspace root:
    python3 scripts/yolo26n_seg_ort.py [image_path]

Compare against lele:
    cargo run --release -p yolo26n-seg-example --bin yolo26n-seg -- fixtures/bus.jpg
"""
import sys
import time
import numpy as np
import onnxruntime as ort

MODEL_PATH = "examples/yolo26n-seg/yolo26n-seg.onnx"
FRAME_MS = 1000.0 / 30.0  # 33.33ms per frame at 30fps

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush",
]
NUM_CLASSES = len(COCO_CLASSES)


def preprocess(img_path: str) -> tuple[np.ndarray, int, int]:
    """Load and preprocess image to NCHW float32 [0,1], matching Rust nearest-neighbor resize."""
    try:
        from PIL import Image as PilImage
        img = PilImage.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img_resized = img.resize((640, 640), PilImage.NEAREST)
        arr = np.asarray(img_resized, dtype=np.float32) / 255.0  # HWC
    except ImportError:
        orig_w, orig_h = 640, 640
        arr = np.full((640, 640, 3), 0.5, dtype=np.float32)

    chw = arr.transpose(2, 0, 1)  # CHW
    return chw[np.newaxis], orig_w, orig_h  # NCHW


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))


def postprocess_seg(output0: np.ndarray, output1: np.ndarray,
                    img_w: int, img_h: int, threshold: float = 0.3):
    """
    Post-process YOLO26n-seg outputs:
      output0: [1, 300, 38]  - [x1,y1,x2,y2, score, class_id, mask_coeffs(32)]
      output1: [1, 32, 160, 160] - mask prototype features
    bbox coordinates are in 640x640 pixel space.
    """
    preds = output0[0]    # [300, 38]
    mask_protos = output1[0]  # [32, 160, 160]

    scale_x = img_w / 640.0
    scale_y = img_h / 640.0

    detections = []
    for i in range(300):
        row = preds[i]
        score = float(row[4])
        if score < threshold:
            continue

        x1_raw, y1_raw, x2_raw, y2_raw = row[0], row[1], row[2], row[3]
        if x2_raw <= x1_raw or y2_raw <= y1_raw:
            continue

        class_id = int(row[5])
        class_id = max(0, min(class_id, NUM_CLASSES - 1))
        mask_coeffs = row[6:38]  # 32 values

        x1 = max(x1_raw * scale_x, 0.0)
        y1 = max(y1_raw * scale_y, 0.0)
        x2 = min(x2_raw * scale_x, img_w)
        y2 = min(y2_raw * scale_y, img_h)

        detections.append({
            "class": COCO_CLASSES[class_id],
            "score": score,
            "bbox": [x1, y1, x2, y2],
            "mask_coeffs": mask_coeffs,
        })

    detections.sort(key=lambda d: -d["score"])
    return detections


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else "fixtures/bus.jpg"
    threshold = 0.3
    n_warmup = 3
    n_runs = 10

    # Load model (single-threaded for fair comparison with lele)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(MODEL_PATH, opts, providers=["CPUExecutionProvider"])

    print(f"=== ORT YOLO26n-Seg Benchmark ===")
    print(f"Model: {MODEL_PATH}")

    # Preprocess
    inp, orig_w, orig_h = preprocess(img_path)
    print(f"Image: {img_path}  ({orig_w}x{orig_h})")

    # Warmup
    print(f"Warming up ({n_warmup} runs)...")
    for _ in range(n_warmup):
        outs = sess.run(None, {"images": inp})

    # Benchmark
    print(f"Benchmarking ({n_runs} runs)...")
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        outs = sess.run(None, {"images": inp})
        times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    min_t = times[0]
    max_t = times[-1]
    avg_t = sum(times) / len(times)
    med_t = times[len(times) // 2]

    print(f"\n=== Results (ORT) ===")
    print(f"  Min:    {min_t:.2f}ms  RTF={min_t/FRAME_MS:.4f}")
    print(f"  Max:    {max_t:.2f}ms")
    print(f"  Avg:    {avg_t:.2f}ms  RTF={avg_t/FRAME_MS:.4f}")
    print(f"  Median: {med_t:.2f}ms")
    print(f"  (RTF@30fps < 1.0 = real-time capable)")
    for i, t in enumerate(times):
        print(f"  run {i+1}: {t:.2f}ms")

    # Accuracy: show detections
    output0, output1 = outs
    dets = postprocess_seg(output0, output1, orig_w, orig_h, threshold)
    print(f"\n=== Detections (threshold={threshold}) ===")
    for i, d in enumerate(dets):
        b = d["bbox"]
        print(f"  {i+1}: {d['class']:15s}  score={d['score']:.3f}  "
              f"bbox=[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]")
    if not dets:
        print("  (none)")

    # Summary for comparison
    print(f"\n=== RTF Summary ===")
    print(f"  ORT avg: {avg_t:.2f}ms  RTF={avg_t/FRAME_MS:.4f}")
    print(f"  ORT min: {min_t:.2f}ms  RTF={min_t/FRAME_MS:.4f}")
    print(f"  Run lele equivalent:")
    print(f"    cargo run --release -p yolo26n-seg-example --bin yolo26n-seg -- {img_path}")


if __name__ == "__main__":
    main()
