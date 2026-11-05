# lele

**Bare-metal ONNX inference engine in pure Rust.**

lele compiles ONNX graphs ahead of time into specialized Rust code and a single
weights blob, then runs them with tuned kernels and no runtime dependencies: no
interpreter, no ONNX parser at runtime, no C++.

- **Ahead-of-time**: graph traversal, fusion, buffer allocation and dispatch are
  resolved at build time. Each node becomes a direct function call.
- **One SIMD source**: the `fearless_simd` kernels compile to NEON (including
  int8 dot products), AVX2/FMA, AVX-512 and WASM SIMD128, with scalar fallbacks
  tested on every target.
- **Quantization**: static QDQ, dynamic int8, weight-only int8 and f16 weights.
- **Small deployment**: with `default-features = false` you ship only the
  generated code and the weights; the ONNX parser is left out.

> Upgrading from 0.1.x: 0.2.0 moves the generated-code runtime into
> `lele::runtime` and changes parts of the compiler API. Regenerate your models
> with the matching compiler version.

## Models

| Domain | Model | Task |
|--------|-------|------|
| ASR | SenseVoice Small | Multilingual speech recognition (int8) |
| ASR | Whisper (base, int8) | Multilingual ASR, encoder/decoder with KV cache |
| ASR | T-one | Streaming Russian telephony ASR (8 kHz, CTC) |
| VAD | Silero VAD, FireRedVAD | Streaming voice activity detection |
| TTS | Supertonic 2 / 3 | 5 / 31 languages, expression tags |
| TTS | Hojo-TTS-Light | Bilingual ZH/EN voice cloning (0.08B token LM) |
| TTS | MOSS-TTS-Nano | End-to-end neural TTS with int8 GEMV |
| Vision | Yolo26, Yolo26n-Seg | Object detection, instance segmentation |
| OCR | PP-OCRv6_tiny | Text detection + recognition |

All of the above also run in the browser via WebAssembly.

## Performance

Reference measurements, single thread, Apple M2 Pro, against ONNX Runtime (CPU,
`intra_op_num_threads=1`, `inter_op_num_threads=1`). Speech models use
steady-state RTF (warmup plus multi-run average); lower is better.

| Model | ORT | lele | Speedup |
| :--- | :--- | :--- | :--- |
| Silero VAD | RTF 0.002882 | RTF 0.0022 | 1.31x |
| SenseVoice | RTF 0.0294 | RTF 0.0256 | 1.15x |
| Supertonic 2 | RTF 0.1667 | RTF 0.0550 | 3.03x |
| Yolo26 | 704.50 ms | 534.97 ms | 1.32x |
| Yolo26n-Seg | 126.51 ms | 64.82 ms | 1.95x |
| Hojo-TTS-Light | RTF 8.75 | RTF 6.69 | 1.31x |
| MOSS-TTS-Nano | RTF 0.376 | RTF 0.293 | 1.28x |
| PP-OCRv6 DET | 47.78 ms | 46.83 ms | 1.02x |
| PP-OCRv6 REC | 2.94 ms/region | 2.33 ms/region | 1.26x |

Kernel-level benchmarks: `cargo bench --bench kernels`.

## Quick start

```toml
# Cargo.toml
[dependencies]
lele = "0.2"
```

### Generate from a build script (recommended)

```toml
# model.toml
[model]
source = "hf-hub"
repo = "your-org/your-model"
files = [{ file = "model.onnx" }]

[codegen]
class_name = "MyModel"
```

```rust
// build.rs
fn main() -> lele_build::Result {
    // Downloads the model and writes src/mymodel.rs and
    // src/mymodel_weights.bin unless they are up to date.
    lele_build::build_model("model.toml", "src")
}
```

```rust
let weights = std::fs::read("src/mymodel_weights.bin")?;
let model = MyModel::new(&weights);
let output = model.forward(input);
```

`build_models` handles several models from one `model.toml`. `HF_ENDPOINT`
selects a mirror; `LELE_SKIP_MODEL_GEN=1` skips generation and
`LELE_FORCE_REGEN=1` forces it. A download or codegen error fails the build.

### Or use the CLI

```bash
cargo install lele
lele_gen model.onnx src/ MyModel
```

This writes `src/mymodel.rs` and `src/mymodel_weights.bin`. `--shape
input=1,512`, `--dim batch_size=1` and `--value sr=16000` compile the model for
fixed inputs (see below).

## Compiler

The compiler builds a typed IR and runs its passes on it: constant and shape
folding, dead-code and common-subexpression elimination, operator fusion and
buffer allocation. Fusions include activations into convolutions and products,
SwiGLU feed-forwards, RMSNorm, ScaleNorm pairs, rotary embeddings, SimAM, Snake,
QDQ pairs around integer matmuls, and fused quantized linears.

Fixed inputs let it fold more and size the workspace up front:

```toml
[codegen.shapes]   # input shapes by name
input = [1, 512]
state = [2, 1, 128]
[codegen.dims]     # named symbolic dimensions
batch_size = 1
[codegen.values]   # inputs compiled in as constants
sr = 16000
```

The same settings exist on the `Compiler` builder (`with_input_shape`,
`with_dim`, `with_input_value`). When every shape is known,
`<Model>Workspace::BYTES` is the exact workspace size.

Models that share weights (e.g. a prefill and a decode step of the same
transformer) can share one weights file with `Compiler::with_shared_weights`.

### Extending it

```rust
use lele::compiler::{Compiler, Optimizations};

let compiler = Compiler::new()
    .with_name("MyModel")
    .with_default_optimizations()
    .with_pass_before_fusion(|graph| { /* rewrite the IR */ })
    .with_override("MyCustomOp", |codegen, w| {
        // Emit Rust code for this operator.
        Ok(())
    });

// Pick optimizations individually:
let compiler = Compiler::new().with_optimizations(Optimizations {
    operator_fusion: false,
    ..Optimizations::all()
});
```

## Examples

```bash
./run_sensevoice.sh                 # SenseVoice ASR
./run_silero.sh                     # Silero VAD
./run_supertonic.sh                 # Supertonic 2 TTS
./run_supertonic3.sh                # Supertonic 3 TTS
./run_yolo26.sh                     # Yolo26 detection
./run_yolo26n_seg.sh                # Yolo26n-Seg segmentation
./run_hojo_tts.sh "text"            # Hojo-TTS-Light voice cloning

cargo run --release -p ppocr-example -- image.png
cargo run --release -p fireredvad-example -- in.wav
cargo run --release -p tone-example --bin tone -- in.wav
cargo run --release -p whisper-example --bin whisper -- in.wav --lang en
```

The browser demo (ASR, TTS, detection, segmentation) lives in
`examples/web-demo`; see its README.

## Supported ONNX operators

**70+ operators**, including:

- **Math**: Add, Sub, Mul, Div, Mod, Pow, Sqrt, Neg, Exp, Log, Sin, Cos, Erf,
  Clip, Round, CumSum, Max, Range, Einsum (partial)
- **NN**: Conv, ConvTranspose, ConvInteger, Gemm, MatMul, MatMulInteger, LSTM,
  GRU, BatchNormalization, InstanceNormalization, LayerNormalization, MaxPool,
  AveragePool, GlobalAveragePool, Resize
- **Activation**: Relu, LeakyRelu, Elu, Sigmoid, Tanh, HardSigmoid, PRelu, Softmax,
  LogSoftmax, ArgMax, IsNaN
- **Tensor**: Reshape, Transpose, Concat, Split, Slice, Gather, GatherElements,
  ScatterND, ScatterElements, Pad, Expand, Tile, Where, TopK, Trilu, Flatten,
  Squeeze, Unsqueeze
- **Reduction**: ReduceSum, ReduceMean, ReduceMax, ReduceL2
- **Comparison / logic**: Equal, Less, Greater, LessOrEqual, GreaterOrEqual, And,
  Or, Xor, Not
- **Quantization**: DynamicQuantizeLinear, QuantizeLinear, DequantizeLinear,
  QLinearMatMul, FakeQuantize
- **Control flow / signal**: If, STFT
- **Other**: Shape, Size, Cast, Constant, ConstantOfShape, Identity

`Loop` is not supported. SiLU is fused into its sigmoid and multiply.

## Architecture

```
ONNX model ──▶ lele-build ──▶ Compiler ──▶ <model>.rs + <model>_weights.bin
                 (download,    (typed IR,        (pure Rust, no runtime deps)
                  cache)        folding,              │
                                fusion,      ┌────────┼────────┐
                                allocation)  ▼        ▼        ▼
                                          Native     WASM    Embedded
                                        (NEON/AVX) (SIMD128)
```

## License

MIT
