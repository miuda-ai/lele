# lele: Bare-Metal Rust Audio AI Framework

**lele** is a standalone, dependency-free inference engine for audio intelligence, built from scratch in pure Rust.

It rejects the "general-purpose runtime" approach (wrapping C++ libs like ORT or using heavy Torch ports) in favor of **hand-crafted, domain-specific kernels**.

## Overview

`lele` is designed to run deep learning models (specifically speech-related ones like SenseVoice, Silero VAD, and TTS) with minimal overhead. It avoids heavy runtimes like ONNX Runtime or `burn` by compiling ONNX graphs directly into optimized Rust source code.

## Performance Benchmarks (2026-01-29)

In-depth comparison between **lele** and **ONNX Runtime (CPU)** on macOS (Apple Silicon). All benchmarks run with single-thread affinity for fair comparison.

| Model | ORT RTF (CPU) | lele RTF | Speedup |
| :--- | :--- | :--- | :--- |
| **Silero VAD** | 0.0031 | 0.0031 | - |
| **SenseVoice** | **0.0318** | 0.1348 | 0.24x |
| **Supertonic** | **0.1225** | 0.2335 | 0.52x |

*Note: RTF (Real-Time Factor) is defined as (Inference Time / Audio Duration). Lower is better. `lele` optimizations currently focus on transformer/convolution patterns found in large-scale ASR models.*


**Why Not ORT/Burn?**
*   **Generic Overhead:** General runtimes carry massive baggage (graph optimization, dynamic shapes, thousands of unused ops) that slows down specific, small-batch audio models.
*   **FFI Penalties:** Binding layers introduce latency and inhibit compiler inlining.
*   **Black Box Memory:** We need absolute control over every byte of allocation for embedded/real-time constraints.

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
```


## Roadmap

1. Performance optimizations (SIMD, multi-threading, etc.), better than ONNX Runtime.
2. Support for more audio models (e.g., Whisper, CosyVoice, etc.)
3. GPU acceleration backend (wgpu); Quantization (INT8/FP16)
4. Advanced attention mechanisms (FlashAttention, PagedAttention)
5. Voice API server (RESTful service), including ASR/TTS/Denoise endpoints.

## License

MIT
