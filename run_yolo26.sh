#!/bin/sh

# Build and run YOLO26 example
# The build.rs will automatically download the model and generate code

cargo run --release -p yolo26-example --bin yolo26 -- fixtures/bus.jpg
