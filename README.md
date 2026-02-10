# lele: Bare-Metal Rust AI Framework

**lele** is a standalone, dependency-free inference engine for intelligence, built from scratch in pure Rust.

It rejects the "general-purpose runtime" approach (wrapping C++ libs like ORT or using heavy Torch ports) in favor of **hand-crafted, domain-specific kernels**.

## Overview

`lele` is designed to run deep learning models (specifically speech-related ones like SenseVoice, Silero VAD, and TTS, even `yolo26` ) with minimal overhead. 

## Performance Benchmarks (2026-02-10)

In-depth comparison between **lele** and **ONNX Runtime (CPU)** on macOS (Apple Silicon). All benchmarks run with single-thread affinity for fair comparison.

| Model | ORT RTF (CPU) | lele RTF | Speedup |
| :--- | :--- | :--- | :--- |
| **Silero VAD** | 0.0031 | 0.0018 | 1.72x |
| **SenseVoice** | **0.032** | 0.093 | 0.34x |
| **Supertonic** | **0.122** | 0.178| 0.68x |
| **Yolo26** | **759.19** | 1050.56ms | 0.72x |

*Note: RTF (Real-Time Factor) is defined as (Inference Time / Audio Duration). Lower is better.*

## Key Features

- **Zero Runtime Dependencies**: Generated models are pure Rust.
- **AOT Compilation**: Converts ONNX models to specialized Rust code for maximum performance.
- **SIMD Optimized**: Hand-written kernels using Apple Silicon (NEON) and x86_64 (AVX/SSE) intrinsics.
- **Memory Efficient**: Static buffer allocation and zero-copy weight loading.
- **Speech Optimized**: Built-in feature extraction for audio (FFT, Mel-spectrogram, LFR, CMVN).

## Supported Models

- **SenseVoiceSmall**: High-accuracy multi-lingual ASR.
- **Silero VAD**: Reliable Voice Activity Detection.
- **Supertonic**: Fast and high-quality Text-to-Speech.
- **Yolo26**: Real-time object detection.

## Getting Started

### Prerequisites

- Rust (Latest stable)
- `cargo`

### Compilation & Generation

To compile an ONNX model into Rust code:

```bash
cargo run --release --bin lele_gen -- <model.onnx> <output_path.rs>
```

### Running Examples

```bash
# SenseVoice ASR
./run_sensevoice.sh

# Supertonic TTS
./run_supertonic.sh

# Silero VAD
./run_silero.sh

# Yolo26 Object Detection
./run_yolo26.sh
```


## Roadmap

1. Performance optimizations (SIMD, multi-threading, etc.), better than ONNX Runtime.
2. Support for more audio models (e.g., Whisper, CosyVoice, etc.)
3. GPU acceleration backend (wgpu); Quantization (INT8/FP16)
4. Advanced attention mechanisms (FlashAttention, PagedAttention)
5. Voice API server (RESTful service), including ASR/TTS/Denoise endpoints.

## License

MIT
