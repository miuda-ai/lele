#!/bin/sh

# Build and run YOLO26n-Seg example
# Weights are generated into examples/yolo26n-seg/src/yolo26seg_weights.bin at build time.
#
# Compare against ORT baseline:
#   python3 scripts/yolo26n_seg_ort.py fixtures/bus.jpg
#
# Run benchmark only:
#   cargo run --release -p yolo26n-seg-example --bin yolo26n-seg-bench

cargo run --release -p yolo26n-seg-example --bin yolo26n-seg -- fixtures/bus.jpg
