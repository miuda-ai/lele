import onnxruntime as ort
import numpy as np
import soundfile as sf
import time
import os
import torch
from torchaudio.compliance import kaldi

WARMUP_RUNS = 5
MEASURE_RUNS = 20

def load_tokens(path):
    tokens = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().rsplit(' ', 1)
            if len(parts) == 2:
                tokens[int(parts[1])] = parts[0].replace('▁', ' ')
    return tokens

def greedy_decode(logits, tokens):
    # logits shape: [1, T, Vocab]
    indices = np.argmax(logits[0], axis=-1)
    res = []
    prev = -1
    for idx in indices:
        if idx != 0 and idx != prev:
            res.append(tokens.get(idx, f"<{idx}>"))
        prev = idx
    return "".join(res)

def load_am_mvn(path):
    """FunASR am.mvn (Kaldi NNet text format): AddShift holds -mean and
    Rescale holds 1/std, so the normalization is (x + addshift) * rescale."""
    addshift = rescale = None
    section = None
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if '<AddShift>' in line:
                section = 'add'
            elif '<Rescale>' in line:
                section = 'rescale'
            elif '[' in line:
                vals = np.array(line.split('[')[1].split(']')[0].split(), dtype=np.float32)
                if section == 'add':
                    addshift = vals
                elif section == 'rescale':
                    rescale = vals
    return addshift, rescale

def main():
    wav_path = 'fixtures/zh.wav'
    model_path = 'examples/sensevoice/model_quant.onnx'
    tokens_path = 'examples/sensevoice/sensevoice.int8.tokens.txt'
    mvn_path = 'examples/sensevoice/am.mvn'

    if not os.path.exists(wav_path):
        print(f"Error: {wav_path} not found")
        return

    # 1. Load Audio
    audio, sr = sf.read(wav_path)
    duration = len(audio) / sr
    print(f"Audio file: {wav_path}")
    print(f"Audio duration: {duration:.2f}s")

    # 2. Feature Extraction (Kaldi fbank + LFR + global CMVN from am.mvn —
    #    the preprocessing SenseVoice was trained with)
    print("Extracting features (Kaldi fbank + LFR + am.mvn)...")
    start_feat = time.perf_counter()
    waveform = torch.from_numpy(audio).float().unsqueeze(0)
    fbank = kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25.0,
        frame_shift=10.0,
        dither=0.0,
        sample_frequency=16000,
        window_type='povey',
    ).numpy()  # [T, 80]

    def compute_lfr(features, m=7, n=6):
        T, D = features.shape
        lfr_features = []
        for i in range(0, T, n):
            start = i - (m // 2)
            frames = []
            for j in range(start, start + m):
                frames.append(features[max(0, min(T - 1, j))])
            lfr_features.append(np.concatenate(frames))
        return np.array(lfr_features)

    lfr_feats = compute_lfr(fbank)  # [T_lfr, 560]
    addshift, rescale = load_am_mvn(mvn_path)
    lfr_feats = (lfr_feats + addshift) * rescale
    feat_time = (time.perf_counter() - start_feat) * 1000
    print(f"✓ Features extracted: {lfr_feats.shape}, took {feat_time:.2f}ms")

    # 3. Model Inference (ORT)
    print("Initializing ONNX Runtime...")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1

    sess = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])

    T_lfr = lfr_feats.shape[0]
    inputs = {
        'speech': lfr_feats[np.newaxis, ...].astype(np.float32),
        'speech_lengths': np.array([T_lfr], dtype=np.int32),
        'language': np.array([3], dtype=np.int32),   # zh (auto=0, zh=3, en=4, yue=7, ja=11, ko=12)
        'textnorm': np.array([15], dtype=np.int32),  # woitn (withitn=14: ITN + punctuation)
    }

    print("Running model inference...")

    # Cold-start inference (first run, includes one-time kernel/setup overhead)
    cold_start = time.perf_counter()
    outputs = sess.run(None, inputs)
    cold_time = (time.perf_counter() - cold_start) * 1000

    # Additional warmup runs before steady-state measurement
    for _ in range(WARMUP_RUNS):
        _ = sess.run(None, inputs)

    # Steady-state measurement
    times_ms = []
    for _ in range(MEASURE_RUNS):
        t0 = time.perf_counter()
        outputs = sess.run(None, inputs)
        times_ms.append((time.perf_counter() - t0) * 1000)

    inf_time = float(np.mean(times_ms))
    p50_time = float(np.percentile(times_ms, 50))
    min_time = float(np.min(times_ms))
    max_time = float(np.max(times_ms))
    std_time = float(np.std(times_ms))

    # 4. Decoding
    tokens = load_tokens(tokens_path)
    text = greedy_decode(outputs[0], tokens)

    print(f"\n=== SenseVoice ORT Results ===")
    print(f"Result: {text}")
    print(f"Cold inference: {cold_time:.2f} ms")
    print(
        f"Steady inference (avg/p50/min/max/std, {MEASURE_RUNS} runs): "
        f"{inf_time:.2f} / {p50_time:.2f} / {min_time:.2f} / {max_time:.2f} / {std_time:.2f} ms"
    )
    print(f"Cold Model RTF: {cold_time/1000 / duration:.4f}")
    print(f"Steady Model RTF (avg): {inf_time/1000 / duration:.4f}")

    total_time = feat_time + inf_time
    print(f"Total RTF: {(total_time/1000) / duration:.4f}")

if __name__ == "__main__":
    main()
