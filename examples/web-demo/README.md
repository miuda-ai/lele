# Lele Web Demo

WebAssembly demo proving that **lele** can compile to WASM and run ML inference directly in a web browser — no server needed.

## Features

| Tab | Model | Task |
|-----|-------|------|
| 🎤 Speech Recognition | SenseVoice (INT8) | Upload WAV → ASR transcription |
| 🔊 Text-to-Speech | Supertonic (4 models) | Text input → synthesized speech |
| 📷 Object Detection | YOLO26 | Upload image → bounding boxes |

## Prerequisites

```bash
# Rust nightly (already configured via rust-toolchain.toml)
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

## Build

```bash
# From this directory:
./build_wasm.sh

# Or manually:
wasm-pack build --target web --out-dir web/pkg --release
```

The build script will:
1. Compile the WASM module with `wasm-pack`
2. Copy model weight files to `web/models/`

> **Note:** The first build downloads ONNX models from HuggingFace Hub and
> generates Rust inference code. Subsequent builds are cached.

## Run

```bash
cd web
python3 -m http.server 8080
# Open http://localhost:8080
```

## Architecture

```
┌─────────────────────────────────────────┐
│           Browser (JavaScript)           │
│  ┌──────────┬───────────┬────────────┐  │
│  │ Audio UI │ Text UI   │ Image UI   │  │
│  └────┬─────┴─────┬─────┴──────┬─────┘  │
│       │           │            │         │
│  ┌────▼───────────▼────────────▼─────┐  │
│  │         WebAssembly Module        │  │
│  │  ┌─────────┐ ┌──────┐ ┌───────┐  │  │
│  │  │SenseVoice│ │Super-│ │YOLO26 │  │  │
│  │  │  ASR    │ │tonic │ │ Det.  │  │  │
│  │  └────┬────┘ │ TTS  │ └───┬───┘  │  │
│  │       │      └──┬───┘     │      │  │
│  │  ┌────▼─────────▼─────────▼───┐  │  │
│  │  │    lele (Rust kernels)     │  │  │
│  │  │  matmul · conv · softmax   │  │  │
│  │  └────────────────────────────┘  │  │
│  └──────────────────────────────────┘  │
│                                         │
│  Model weights loaded via fetch()       │
└─────────────────────────────────────────┘
```

## How It Works

1. **Build-time**: ONNX models are compiled to pure Rust code by `lele-build`
2. **Build-time**: `wasm-pack` compiles the Rust code to `.wasm`
3. **Runtime**: Browser loads the `.wasm` module + model weights via `fetch()`
4. **Runtime**: All inference runs in the browser, no server required

### WASM Compatibility

lele achieves WASM compatibility through:

- **Target-gated dependencies**: `faer` (BLAS) is only used on native targets;
  WASM uses custom SIMD128-optimized matmul with tiling (MC=64, NC=256, KC=64)
- **Architecture dispatch**: NEON/AVX kernels are behind `#[cfg(target_arch)]`
  guards; WASM SIMD128 paths added for wasm32 target
- **No filesystem**: Model weights are passed as byte slices from JavaScript
- **WASM SIMD128**: Intrinsics from `std::arch::wasm32` for 4x f32 vector ops

### Performance Optimizations (2026-02-10)

#### WASM SIMD128 Kernels

| Kernel | Scalar (baseline) | WASM SIMD128 (optimized) | Speedup |
|--------|-------------------|--------------------------|----------|
| **MatMul** | Naive O(m×n×k) triple loop | Tiled micro-kernel with 4×4 blocking + `f32x4_mul`/`f32x4_add` | ~4x |
| **tanh** | `f32::tanh()` per element | Polynomial exp approximation with SIMD | ~3-4x |
| **sigmoid** | `1/(1+exp(-x))` scalar | SIMD exp + division | ~3-4x |
| **relu** | `max(0, x)` scalar | `f32x4_max(v, zero)` | ~3-4x |
| **silu** | `x/(1+exp(-x))` scalar | SIMD combined operation | ~3-4x |
| **softmax** | Scalar max/exp/sum | SIMD reduction + vectorized normalize | ~2-3x |
| **layer_norm** | Scalar mean/variance | SIMD mean/var + vectorized normalize | ~2-3x |
| **conv1d bias+relu** | Scalar post-process | SIMD bias broadcast + relu | ~3-4x |

#### Build Optimization Settings

```toml
# .cargo/config.toml
[target.wasm32-unknown-unknown]
rustflags = ["-C", "target-feature=+simd128"]

# Environment overrides in build script
CARGO_PROFILE_RELEASE_OPT_LEVEL=3
CARGO_PROFILE_RELEASE_LTO=true
CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1
CARGO_PROFILE_RELEASE_PANIC=abort
```

#### Binary Size Comparison

| Module | Dev Build (unoptimized) | Release + LTO + SIMD | Reduction |
|--------|-------------------------|----------------------|----------|
| sensevoice | 2.9M | **1.7M** | 41% |
| yolo26 | 767K | **508K** | 34% |
| supertonic | 5.1M | **1.7M** | 67% |

*All release builds include `wasm-opt -O3` post-processing.*

#### Expected Runtime Performance

- **Release mode baseline**: 10-50x faster than dev builds (no optimization, inline, assertions)
- **SIMD128 kernels**: Additional 2-4x speedup on hot paths (matmul, activations, norms)
- **Combined effect**: **20-100x** total speedup over unoptimized scalar WASM

> **Note**: Actual speedup depends on model characteristics. Models with heavy matmul/conv 
> workloads benefit most from SIMD optimizations.

## File Structure

```
examples/web-demo/
├── Cargo.toml          # Crate config (wasm-bindgen, lele, etc.)
├── build.rs            # Generates model code for all models
├── build_wasm.sh       # Build + deploy script
├── README.md           # This file
├── src/
│   ├── lib.rs          # WASM entry point (wasm-bindgen exports)
│   ├── audio.rs        # WAV decode/encode (no filesystem)
│   ├── image.rs        # Image preprocessing for YOLO26
│   ├── tokenizer.rs    # Token decoder for SenseVoice
│   ├── processor.rs    # Text processor for Supertonic
│   ├── config.rs       # Config types for Supertonic
│   └── gen/            # Generated model code (build-time)
└── web/
    ├── index.html      # Demo page
    ├── style.css       # Styles
    ├── app.js          # JavaScript application
    ├── pkg/            # wasm-pack output (generated)
    └── models/         # Model weight files (copied by build script)
```
