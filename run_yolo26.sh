#!/bin/sh

# Build and run YOLO26 detection example
# Weights are generated into examples/yolo26/src/yolo26_weights.bin at build time.
#
# Compare against ORT baseline:
#   python3 scripts/yolo26_ort.py fixtures/bus.jpg
#
# Run benchmark only:
#   cargo run --release -p yolo26-example --bin yolo26-bench

cargo run --release -p yolo26-example --bin yolo26 -- fixtures/bus.jpg
