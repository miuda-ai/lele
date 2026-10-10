# Where the example models spend their time

Op shapes, call counts and time shares of the example models (plus three outside models:
VoxBlink2, an int8 BERT and Smart Turn v3.2),
measured to decide which kernels are worth work. Numbers are from one machine and one
input per model; they are for ranking, not for quoting as benchmarks.

- **Machine:** AMD Ryzen 7 5800X (Zen 3, AVX2 + FMA, no AVX-512 or VNNI), single thread,
  Linux 7.2. Peak f32 FMA throughput is about 150 GFLOP/s per core.
- **Code:** branch `pr_simd_gemm` at `c45f040`, so f32 matmul goes through
  `kernels::matmul` (fearless_simd), not faer. Models whose generated code needs kernels
  from `main` (T-one, VoxBlink2, the int8 BERT, Smart Turn, and the examples generated
  fresh for this document) ran on that branch with `main` merged in.
- **Inputs:** each example's defaults (`fixtures/bus.jpg`, `fixtures/ocr_test.png`,
  `fixtures/zh.wav`, the default TTS sentence). T-one: 30 chunks of a synthetic 8 kHz
  signal. VoxBlink2: a synthetic 500-frame fbank (5 s). BERT: two Russian texts of 11
  and 107 tokens. Smart Turn: a synthetic 8 s log-mel (its fixed input size). Shapes depend only on lengths, so synthetic inputs are fine.

## Method

Two runs per model:

1. **perf** (`perf record -F 4000`, whole process, no instrumentation): self time per
   symbol, grouped into categories. Includes model loading and image/audio I/O, which are
   small everywhere except where noted.
2. **Shape log:** an uncommitted patch timed every call of `kernels::matmul` (with its
   caller, via `#[track_caller]`), `conv1d`, `conv2d`, `conv_transpose`, `MatMul`,
   `MatMulAdd`, `Gemm`, the int8 matmuls (`MatMulInteger`, `fused_quantized_linear`,
   `qmatmul_i8`, `DynamicQuantizeLinear`), `Slice`, `Concat`, `Pad` and `Transpose`, and
   logged the shapes. Op-level times include the GEMMs inside them.

GEMM shapes below are `m x k x n` for `C[m,n] = A[m,k] * B[k,n]`, as `kernels::matmul`
sees them, after convolutions are lowered: a pointwise conv1d is
`C_out x C_in x L`, and an im2col conv2d is `C_out x (C_in*kh*kw) x (H_out*W_out)`.
All f32 GEMMs in these models had row-major A, B and C (except one transposed `Gemm` in
VoxBlink2). GFLOP/s is `2mkn / time`.

## Summary

Share of whole-process samples:

| model | task | run | top category | next |
|---|---|---|---|---|
| Supertonic3 | TTS | 1.2 s for 4.3 s of audio | f32 GEMM 80% | data movement 9%, elementwise 7% |
| Supertonic 2 | TTS | 0.39 s for 4.3 s of audio | f32 GEMM 81% | data movement 10% |
| MOSS-TTS-Nano | TTS (LLM) | 4.0 s for 4.6 s of audio | f32 GEMM 64% (GEMV) | libc + OS 21%, data movement 7% |
| Hojo-TTS-Light | TTS (LLM) | 30.5 s for 2.2 s of audio | **f16 weight decode 72%** | app `matvec` 9%, conv_transpose 4% |
| SenseVoice (int8) | ASR | 0.32 s for 5.6 s of audio | **int8 GEMM 83%** | libc 4%, f32 GEMM 3% |
| T-one | streaming ASR | 193 ms per 300 ms chunk | **f16 weight decode 89%** | f32 GEMM 8% |
| Smart Turn v3.2 (QDQ int8) | turn detection | 61 ms per 8 s window | int8 GEMM 57% | f32 attention 20%, fake-quantize 6% |
| Silero VAD | VAD | ~60 µs per 32 ms chunk | conv1d 35% | GEMV (LSTM) 23%, data movement 20% |
| VoxBlink2 SimAM-ResNet34 | speaker embedding | 1.09 s per 5 s of audio | f32 GEMM (3x3 conv) 80% | `reduce_sum` 7%, im2col 7% |
| rubert-tiny-style BERT (int8) | text embedding | 1.3 ms (11 tok), 5.2 ms (107 tok) | int8 GEMM 54% | f32 GEMM 17%, weight-cache lookup 7% |
| YOLO26n | detection | 5 forward passes, 0.55 s | f32 GEMM 45% | **`Slice` 25%**, conv 18% |
| YOLO26n-seg | segmentation | 5 forward passes, 2.0 s | **conv_transpose 30%** | f32 GEMM 22%, app postprocess 30% |
| PP-OCR det + rec | OCR | 0.3 s | conv (direct) 51% | f32 GEMM 20%, non-inlined intrinsics 19% |

"Data movement" is `Slice`, `Concat`, `Transpose`, `Pad`, `Gather` and the like.

## Findings, by expected payoff

1. **f16 models decode their weights on every forward (T-one 89%, Hojo 72%).** When an
   ONNX model stores f16 weights, the generated code calls `weight_f16` inside `forward`:
   230 times per T-one chunk, and 346/208/78 times per Hojo encoder/LLM/decoder call, so
   `TensorView::from_bytes_f16` re-converts the whole model each time (about 170 of 193 ms
   per T-one chunk; 22 of Hojo's 30 s). Converting once at load, or keeping f16 and
   widening inside the kernels, would make T-one about 8x and Hojo about 4x faster.
2. **`conv_transpose` is slow everywhere it appears.**
   - YOLO26n-seg's mask upsampler `in[1,64,80,80] w[64,64,2,2] s2` takes 122 ms per
     call, more than all its conv2d calls together. As a GEMM it is
     `256x64x6400` (0.2 GFLOP), which should take ~2 ms.
   - Hojo's depthwise `k12 s2` upsamplers (`[1,32..128, 22k..89k]`) take 26 ms per call,
     and its ISTFT-like `in[1,1920,112] w[1920,1,1920] s480` takes 270 ms per call.
   - PP-OCR's 2x2 stride-2 ones take 11–15 ms per call.
3. **`Slice` along the channel axis is slow (YOLO26: 126 ms, 25% of the run).** Splitting
   `[1, 64, 80, 80]` into two halves along axis 1 takes 2.1 ms per half. That half is one
   contiguous 0.8 MB block, so it should take tens of microseconds as a single `memcpy`.
   Every C2f/ELAN block in YOLO does this.
4. **PP-OCR's depthwise 5x5 conv2d is the slowest op per FLOP.**
   `in[1,64,120,160] w[64,1,5,5] g64` takes 39 ms per call, about 1.6 GFLOP/s. It runs
   through code that calls `_mm256_fmadd_ps` / `_mm256_loadu_ps` as out-of-line
   functions: the intrinsics are not inlined, because the caller has no
   `target_feature`. That accounts for 19% of PP-OCR on its own.
5. **int8 matmul runs at about f32 speed (SenseVoice 83%, Smart Turn 57%, BERT 54%).**
   Without VNNI, the `MatMulInteger` / `fused_quantized_linear` path
   (`avx::quantization::gemm_2rows_avx2`) does `97x512x2048` in 1.33 ms, about
   150 GOP/s, the same as the f32 GEMM's FLOP rate. The QDQ path Smart Turn takes
   (`qmatmul_i8` → `avx::qgemm::qgemm_u8s8_f32_avx2`) does `400x384x1536` in 2.6 ms,
   about 180 GOP/s. On AVX2, `vpmaddubsw`-based int8 should reach about twice the f32
   rate, and the two int8 paths could share the faster kernel.
   SenseVoice's int8 `[97,512] x [512,25055]` CTC head is 16 ms per call. In BERT, a
   per-call `HashMap` lookup of the cached quantized weights is 7% of the run, and an
   11-token forward still costs 1.3 ms against 5.2 ms for 107 tokens.
6. **MOSS decode is GEMV, and DRAM-bound.** 68% of its GEMM time is `m = 1`:
   `1x3072x768`, `1x768x3072`, `1x768x2304`, `1x768x768`, and the `1x768x16384` LM head.
   These run at 13–20 GFLOP/s, which is 35–40 GB/s of weights, about what DRAM delivers.
   The decode-step weight file alone is 440 MB of f32. Kernel tuning cannot help these
   much; f16 or int8 weights would (half or a quarter of the bytes). The libc share (15%)
   is `memcpy`/`memset` (likely the KV-cache `Concat`s and buffer setup; not attributed
   further), and the OS share (7%) is page faults.
7. **VoxBlink2 is 3x3-conv compute, ~88 GFLOP per 5 s of audio.** Its ResNet34 convs go
   through im2col GEMMs (`256x2304x2500`, `64x576x40000`, `128x1152x10000`,
   `512x4608x630`) at 80–124 GFLOP/s. The GEMM is near peak for the larger ones; only a
   different algorithm (Winograd F(2,3) cuts multiplies 2.25x) or the
   `64x576x40000` case (80 GFLOP/s, the im2col matrix is 92 MB) leaves room. SimAM's
   `reduce_sum` is another 7%.
8. **Supertonic (2 and 3) is pointwise conv1d, i.e. tall GEMMs with a short N.** About 80%
   of each run is `C x 4C x L` and `4C x C x L` (C = 256 or 512), with `L` the latent
   length (62 for the default sentence) times batch 2. These run at 126–131 GFLOP/s, near
   this machine's practical peak. Next are `Pad` (edge mode, `[2,512,62]`, 0.23 ms per
   call) and last-axis `Slice` of attention heads (`[8,2,62,64]` → half, 0.33 ms per
   call). Both look 10x slower than a copy should be.
9. **im2col GEMMs with few output channels are memory-bound.** In YOLO and PP-OCR,
   shapes like `8x144x25600`, `16x576x19200` and `16x27x76800` run at 30–55 GFLOP/s: an
   im2col matrix of tens of MB is streamed for only 8–16 rows of output. A direct
   convolution (or im2col in tiles that stay in cache) would beat any GEMM tuning here.
10. **Small-M linear layers (T-one).** `10x384x1536`, `10x1536x384`, `5x384x1536` and
    `5x1536x384` are 67% of T-one's GEMM time, at 72–93 GFLOP/s. The grouped conv
    `384→1536 k3 g384` becomes 11 520 GEMMs of `4x3x5` and takes 0.35 ms per call: small
    in total, but the wrong lowering.
11. **Silero is call overhead.** A 32 ms chunk takes ~60 µs: the STFT conv
    (`1→258 k256 s128`), four tiny `k3` convs, an LSTM with two `512x128x1` GEMVs, and
    `Slice`s (16%) and libc (8%) around them. No single kernel dominates.

### Codegen problems found along the way

- **VoxBlink2 does not compile.** The fused `conv1d_relu` takes its input by value, and
  the same tensor (`_model_pooling_Reshape_output_0`, the attentive-pooling input) is used
  again by the following `Mul`. Profiled with a `.clone()` patched in.
- **Hojo-TTS-Light's current upstream model (v2, 2026-08-21) does not compile.** The
  decoder gets `kernels::sub(f32, i64)`. The example was profiled with the June model
  (revision `4c8706688a`, the one with `onnx/`), which works; `model.toml` still points
  at the removed `onnx/` URLs.
- The checked-in `examples/supertonic/text_encoder.onnx` and `vocoder.onnx` are corrupt
  (protobuf decode error; the vocoder is 41 MB against 101 MB upstream). Re-downloaded
  from `Supertone/supertonic-2`.
- `examples/yolo26n-seg/src/yolo26seg_weights.bin` is empty. The model was re-exported
  with ultralytics (`yolo26n-seg.pt`, imgsz 640); its output head does not match the
  example's postprocessing (scores > 1, 235 detections), so the 30% of time in
  `yolo26n_seg::main` is postprocessing garbage and should be ignored. The backbone and
  mask head shapes are valid.

## Per-model detail

### Supertonic3

Text encoder, duration predictor, vector estimator (batch 2, latent length 62, 10 steps),
vocoder (length 372). GEMM total 796 ms out of 1.2 s.

| m x k x n | from | calls | ms | GFLOP/s |
|---|---|---:|---:|---:|
| 512x2048x62 | conv1d k1 | 280 | 287 | 127 |
| 2048x512x62 | conv1d k1 | 280 | 286 | 127 |
| 512x2048x372 | conv1d k1 (vocoder) | 11 | 68 | 126 |
| 2048x512x372 | conv1d k1 (vocoder) | 10 | 60 | 131 |
| 62x512x512 | MatMulAdd | 80 | 26 | 98 |
| 2048x1536x372 | conv1d k3 | 1 | 18 | 129 |
| 62x256x512 | MatMul | 80 | 14 | 96 |
| 50x256x256 | MatMul | 80 | 5 | 103 |
| 62x64x62, 62x62x64 | attention | 336 each | 2 | 80–93 |

Other ops: depthwise conv1d `512ch k5` (dilations 1/2/4/8) 1.5 ms total. `Pad` 45 ms,
`Slice` 31 ms, `Transpose` 16 ms. Elementwise ops: `Mul` 4.5%, `gelu_erf` 1.6%.

### Supertonic 2

Same architecture at half the width in the vector estimator: `1024x512x62` and
`512x1024x62` (140 calls each, 72 ms each, 126 GFLOP/s), and the same vocoder as
Supertonic3 (`512x2048x372`, `2048x512x372`). GEMM total 341 ms of 0.39 s; `Pad` 27 ms.

### MOSS-TTS-Nano

Prefill (202 tokens) 420 ms, then 58 frames × (local 13 ms + decode 27 ms), then the
codec 600 ms. GEMM total 1.82 s.

| m x k x n | from | calls | ms | GFLOP/s |
|---|---|---:|---:|---:|
| 1x3072x768 | MLP down (decode) | 1699 | 430 | 19 |
| 1x768x3072 | MLP up (decode) | 1699 | 397 | 20 |
| 1x768x2304 | QKV (decode) | 1699 | 297 | 20 |
| 1x768x16384 | LM head | 59 | 118 | 13 |
| 1x768x768 | attention out (decode) | 1699 | 106 | 19 |
| 202x768x3072, 202x3072x768 | prefill MLP | 12 each | 105, 101 | 108–114 |
| 1x768x1024 | local head | 944 | 96 | 16 |
| 202x768x2304 | prefill QKV | 12 | 77 | 112 |
| 1856x64x1856, 1856x1856x64 | codec attention | 16 each | 64, 56 | 111–127 |
| 1856x256x1024 etc. | codec MLP | 4 each | 22–30 | 130 |

Data movement: `Concat` 98 ms (KV cache, and RoPE `[...,32,1]` pairs), `Transpose` 92 ms,
`Slice` 69 ms (RoPE even/odd with step 2: `[1,1,12,64]`, 3398 calls of 4 µs each).

### Hojo-TTS-Light (June model)

Encoder + LLM (112 audio codes) + decoder, 30.5 s in total, of which ~22 s is f16 weight
decode. Decoder 747 ms. f32 GEMM total 805 ms.

| op | shape | calls | ms per call |
|---|---|---:|---:|
| conv_transpose | `in[1,1920,112] w[1920,1,1920] s480` | 2 | 270 |
| conv_transpose | `in[1,32..128,22k..89k] w[C,1,12] gC s2` (depthwise) | 21 | 26 |
| Pad | `in[1,128,44736] edge` | 7 | 21 |

| m x k x n | from | calls | ms | GFLOP/s |
|---|---|---:|---:|---:|
| 128x896x22368 | conv1d k7 | 3 | 153 | 101 |
| 256x1792x5592 | conv1d k7 | 3 | 133 | 116 |
| 512x3584x1398 | conv1d k7 | 3 | 121 | 128 |
| 64x448x44736 | conv1d k7 | 3 | 91 | 84 |
| 32x224x89472 | conv1d k7 | 3 | 58 | 66 |
| 112x2304x768, 112x768x2304 | LLM prefill | 8 each | 25 | 126–129 |

### SenseVoice (int8)

One 5.6 s utterance (97 frames after LFR), 0.32 s per run (the example runs it 11 times).
int8 `fused_quantized_linear` is 3.2 s of the 4 s process.

| op | shape | calls | ms per call |
|---|---|---:|---:|
| int8 linear | `[97,512] x [512,2048]` | 770 | 1.33 |
| int8 linear | `[97,2048] x [2048,512]` | 770 | 1.28 |
| int8 linear | `[97,512] x [512,1536]` | 748 | 1.00 |
| int8 linear | `[97,512] x [512,512]` | 759 | 0.33 |
| int8 linear (CTC head) | `[97,512] x [512,25055]` | 11 | 16.3 |
| f32 attention | `97x128x97`, `97x97x128` | 3080 each | 0.02 |
| depthwise conv1d | `512ch k11` | 770 | 0.02 |

### T-one

Per 300 ms chunk: 15 conformer layers at length 10 (or 5 after the stride), plus a
Conv2d front end. GEMM total 479 ms for 30 chunks (16 ms per chunk); the rest of the
193 ms is weight decode (finding 1).

| m x k x n | from | calls | ms | GFLOP/s |
|---|---|---:|---:|---:|
| 10x384x1536 | FFN up | 960 | 156 | 72 |
| 10x1536x384 | FFN down | 480 | 75 | 75 |
| 5x384x1536 | FFN up | 960 | 72 | 79 |
| 64x3872x340 | conv2d `32→64 k11x11 s[3,1]` | 30 | 53 | 96 |
| 5x1536x384 | FFN down | 480 | 32 | 88 |
| 10x384x384 | attention projections | 540 | 22 | 73 |
| 768x384x10, 768x384x5 | pointwise conv1d | 240 each | 18, 11 | 66–80 |
| 4x3x5 | grouped conv1d `384→1536 k3 g384` | 11520 | 3 | — |

Other ops: depthwise conv1d `384ch k31` at length 35/40, about 12 µs per call. `Slice`
54 ms, mostly the state tensor `[1,219729]` and the left context `[1,384,35..40]` →
last 30.

### Smart Turn v3.2

`pipecat-ai/smart-turn-v3`, `smart-turn-v3.2-cpu.onnx`: Whisper-tiny encoder (4 layers,
384 wide, 6 heads, 400 frames after the stride-2 conv) and a small classifier, static
QDQ int8. Input is a fixed 8 s log-mel `[1, 80, 800]`; 61 ms per forward on one thread.
QuantizeLinear/DequantizeLinear pairs on activations become `fake_quantize_linear`
(78 calls per forward, 6% of the run); weight matmuls become `qmatmul_i8`.

| op | shape | calls per forward | ms per call | GOP/s |
|---|---|---:|---:|---:|
| int8 matmul (FFN up) | `[400,384] x [384,1536]` | 4 | 2.6 | 181 |
| int8 matmul (Q/K/V/out) | `[400,384] x [384,384]` | 16 | 0.64 | 183 |
| int8 matmul (FFN down) | `[400,1536] x [1536,384]` | 4 | 2.4 | 199 |
| f32 attention | `6 x (400x64x400)`, `6 x (400x400x64)` | 4 each | 1.0 | 119–129 |
| conv1d | `in[1,384,800] w[384,384,3] p1 s2` | 1 | 3.0 | 132 |
| conv1d | `in[1,80,800] w[384,80,3] p1` | 1 | 1.3 | |

### Silero VAD

175 chunks of 512 samples (+64 context), ~60 µs each.

| op | shape | µs per call |
|---|---|---:|
| conv1d (STFT) | `in[1,1,576] w[258,1,256] s128` | 6.3 |
| conv1d | `in[1,129,3] w[128,129,3] p1` | 6.6 |
| conv1d | `in[1,128,3] w[64,128,3] p1 s2` | 5.3 |
| conv1d | `in[1,64,2] w[64,64,3] p1 s2`, `in[1,64,1] w[128,64,3] p1` | 2–3 |
| LSTM GEMV | `512x128x1` (x2) | 4.2 each |

### VoxBlink2 SimAM-ResNet34

`/home/obj/voxblink2_samresnet34.onnx`: fbank `[1, 500, 80]` → 256-dim embedding,
1.09 s per forward. conv2d total 2.88 s for 3 forwards.

| m x k x n | conv | calls | ms | GFLOP/s |
|---|---|---:|---:|---:|
| 256x2304x2500 | `256→256 3x3` on 20x125 | 33 | 827 | 118 |
| 64x576x40000 | `64→64 3x3` on 80x500 | 18 | 663 | 80 |
| 128x1152x10000 | `128→128 3x3` on 40x250 | 21 | 635 | 98 |
| 512x4608x630 | `512→512 3x3` on 10x63 | 15 | 360 | 124 |
| strided 3x3 and 1x1 downsamples | | 18 | 154 | |
| 5120x128x63, 128x5120x63 | attentive pooling | 3 each | 2 | 96–119 |

### Int8 BERT (`/home/obj/scr/model.sim.int8.onnx`)

312-hidden, 3-layer BERT (rubert-tiny style), dynamically quantized: `MatMulInteger`
for Q/K/V and the output projection, `fused_quantized_linear` for the FFN, f32 attention.

| op | shape (107 tokens) | ms per call |
|---|---|---:|
| MatMulInteger | `[107,312] x [312,312]` | 0.135 |
| int8 linear | `[107,312] x [312,600]`, `[107,600] x [600,312]` | 0.26–0.27 |
| f32 attention | `12 x (107x26x107)`, `12 x (107x107x26)` | 0.09 |
| f32 attention scores + mask (`MatMulAdd`) | `12 x (107x26x107)` | 0.27 |

The fused bias add of `MatMulAdd` (`gemm::matmul_fused_add`, 10% of the run) costs twice
the matmul it is fused to: the mask is broadcast over the 12 heads. At 11 tokens the same ops are 10–20 µs each, and the forward is 1.3 ms: fixed per-op
cost (weight-cache lookup, quantizing activations, buffer setup) dominates short texts.

### YOLO26n

5 forward passes of 640x640. conv2d total 330 ms (GEMM part 242 ms); `Slice` 126 ms.

| m x k x n | from | calls | ms | GFLOP/s |
|---|---|---:|---:|---:|
| 64x576x6400 | conv 3x3 s2 | 5 | 27 | 88 |
| 128x1152x1600 | conv 3x3 s2 | 5 | 19 | 123 |
| 32x144x25600 | conv 3x3 s2 | 5 | 17 | 71 |
| 32x288x1600 | conv 3x3 | 60 | 17 | 107 |
| 128x192x1600 | conv 1x1 | 20 | 13 | 123 |
| 8x144x25600 | conv 3x3 | 5 | 10 | 29 |
| 16x27x102400 | first conv (3 ch) | 5 | 9 | 48 |
| 256x384x400 | conv 1x1 | 15 | 9 | 131 |

GEMM time: 68% has both m and n ≥ 64, 32% has m < 64. The 3x3 convs go through im2col
(3.3% of the run), and `max_pool2d` is 4%. Slowest `Slice`s: `[1,64,80,80]` → 32
channels (2.1 ms), `[1,32,160,160]` → 16 channels (4.2 ms), `[1,128,40,40]` → 64
channels (1.0 ms).

### YOLO26n-seg

The YOLO26n backbone plus a mask head at 160x160. 5 forward passes, 2.0 s.
`conv_transpose` `in[1,64,80,80] w[64,64,2,2] s2` is 613 ms (122 ms per call); the mask
head's `64→64 3x3` convs on 160x160 (`64x576x25600`, 85 GFLOP/s) are 111 ms.

### PP-OCR (det + rec)

Detection on 480x640 takes 129 ms; recognition of 6 regions takes 15 ms.

| op | shape | calls | ms per call |
|---|---|---:|---:|
| depthwise conv2d | `in[1,64,120,160] w[64,1,5,5] g64 p2` | 2 | 39 |
| conv_transpose | `in[1,16,120,160] w[16,16,2,2] s2` | 2 | 15 |
| conv2d 3x3 | `in[1,64,120,160] w[16,64,3,3]` | 2 | 14 |
| conv_transpose | `in[1,16,240,320] w[16,1,2,2] s2` | 2 | 11 |
| conv2d 3x3 s2 | `in[1,32,240,320] w[16,32,3,3]` | 2 | 9 |
| depthwise conv2d | `in[1,64,60,80] w[64,1,5,5] g64 p2` | 2 | 9 |

GEMMs are mostly a short M with a long N: `16x576x19200` (39 GFLOP/s),
`16x288x19200` (38), `64x32x19200` (67), `8x64x76800` (34).

The recognized text on `fixtures/ocr_test.png` is wrong (confidence 0.12–0.34) on both
this branch and `main`, and YOLO26n finds only the bus and one person in
`fixtures/bus.jpg`; both are pre-existing.
