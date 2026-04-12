"""ORT benchmark and accuracy comparison for YOLO26 (detection).

Run from workspace root:
    python3 scripts/yolo26_ort.py [image_path]

Compare against lele:
    cargo run --release -p yolo26-example --bin yolo26 -- fixtures/bus.jpg
"""
import sys
import time
import numpy as np
import onnxruntime as ort

MODEL_PATH = "examples/yolo26/yolo26.onnx"
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


def preprocess(img_path: str) -> tuple[np.ndarray, int, int]:
    """Load and preprocess image to NCHW float32 [0,1], matching Rust nearest-neighbor resize."""
    try:
        from PIL import Image as PilImage
        img = PilImage.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        # Rust uses: src = floor((dst + 0.5) * src_size / 640), matching PIL center-pixel NEAREST
        img_resized = img.resize((640, 640), PilImage.NEAREST)
        arr = np.asarray(img_resized, dtype=np.float32) / 255.0  # HWC
    except ImportError:
        import struct, pathlib
        # Fallback: create dummy input
        orig_w, orig_h = 640, 640
        arr = np.full((640, 640, 3), 0.5, dtype=np.float32)

    chw = arr.transpose(2, 0, 1)  # CHW
    return chw[np.newaxis], orig_w, orig_h  # NCHW


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def postprocess(logits: np.ndarray, pred_boxes: np.ndarray,
                img_w: int, img_h: int, threshold: float = 0.3):
    """DETR-style post-processing: logits [1,300,80], pred_boxes [1,300,4]."""
    logits = logits[0]       # [300, 80]
    pred_boxes = pred_boxes[0]  # [300, 4]

    scores = sigmoid(logits)   # [300, 80]
    max_scores = scores.max(axis=1)   # [300]
    class_ids = scores.argmax(axis=1)  # [300]

    mask = max_scores >= threshold
    detections = []
    for i in np.where(mask)[0]:
        cx, cy, w, h = pred_boxes[i]
        x1 = max((cx - w / 2) * img_w, 0)
        y1 = max((cy - h / 2) * img_h, 0)
        x2 = min((cx + w / 2) * img_w, img_w)
        y2 = min((cy + h / 2) * img_h, img_h)
        detections.append({
            "class": COCO_CLASSES[class_ids[i]],
            "score": float(max_scores[i]),
            "bbox": [x1, y1, x2, y2],
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

    print(f"=== ORT YOLO26 Benchmark ===")
    print(f"Model: {MODEL_PATH}")

    # Preprocess
    inp, orig_w, orig_h = preprocess(img_path)
    print(f"Image: {img_path}  ({orig_w}x{orig_h})")

    # Warmup
    print(f"Warming up ({n_warmup} runs)...")
    for _ in range(n_warmup):
        outs = sess.run(None, {"pixel_values": inp})

    # Benchmark
    print(f"Benchmarking ({n_runs} runs)...")
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        outs = sess.run(None, {"pixel_values": inp})
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
    logits, pred_boxes = outs
    dets = postprocess(logits, pred_boxes, orig_w, orig_h, threshold)
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
    print(f"    cargo run --release -p yolo26-example --bin yolo26 -- {img_path}")


if __name__ == "__main__":
    main()
