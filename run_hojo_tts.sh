#!/bin/sh
# Build and run the Hojo-TTS-Light example.
# build.rs downloads the ONNX models + tokenizer from Hugging Face and AOT
# compiles them to Rust on first build. Default prompt is assets/zh1.wav.
#
# Usage:
#   ./run_hojo_tts.sh "<target text>"
#   ./run_hojo_tts.sh "Hello world." --prompt-wav assets/zh1.wav --prompt-text "..."

cargo run --release -p hojo-tts-example --bin hojo_tts -- "$@"
