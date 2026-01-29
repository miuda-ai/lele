import onnxruntime as ort
import numpy as np
import soundfile as sf
import time
import os
import torchaudio
import torch

def load_tokens(path):
    tokens = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                token = parts[0]
                idx = int(parts[1])
                # Replace the special marker ▁ with space for display
                token = token.replace('▁', ' ')
                tokens[idx] = token
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

def main():
    wav_path = 'fixtures/zh.wav'
    model_path = 'examples/sensevoice/sensevoice.int8.onnx'
    tokens_path = 'examples/sensevoice/sensevoice.int8.tokens.txt'

    if not os.path.exists(wav_path):
        print(f"Error: {wav_path} not found")
        return

    # 1. Load Audio
    audio, sr = sf.read(wav_path)
    if sr != 16000:
        # Simple resample to 16k if needed or warn
        pass
    
    duration = len(audio) / sr
    print(f"Audio file: {wav_path}")
    print(f"Audio duration: {duration:.2f}s")

    # 2. Feature Extraction
    print("Extracting features (Mel + LFR)...")
    start_feat = time.perf_counter()
    
    waveform = torch.from_numpy(audio).float().unsqueeze(0)
    # Match lele's FeatureConfig: n_mels=80, window=400 (25ms), shift=160 (10ms)
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=80,
        center=False # lele doesn't seem to center by default
    )(waveform)
    
    mel = torch.log(mel_spec + 1e-6)
    mel = mel.squeeze(0).transpose(0, 1).numpy() # [T, 80]
    
    # LFR: stack 7, shift 6
    def compute_lfr(features, m=7, n=6):
        T, D = features.shape
        lfr_features = []
        for i in range(0, T, n):
            start = i - (m // 2)
            frames = []
            for j in range(start, start + m):
                if j < 0:
                    frames.append(features[0])
                elif j >= T:
                    frames.append(features[T-1])
                else:
                    frames.append(features[j])
            lfr_features.append(np.concatenate(frames))
        return np.array(lfr_features)

    lfr_feats = compute_lfr(mel) # [T_lfr, 560]
    
    # CMVN (Mean/Std normalization)
    lfr_feats = (lfr_feats - np.mean(lfr_feats, axis=0)) / (np.std(lfr_feats, axis=0) + 1e-6)
    
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
        'x': lfr_feats[np.newaxis, ...].astype(np.float32),
        'x_length': np.array([T_lfr], dtype=np.int32),
        'language': np.array([3], dtype=np.int32), # Chinese
        'text_norm': np.array([0], dtype=np.int32)
    }

    print("Running model inference...")
    # Warmup
    _ = sess.run(None, inputs)
    
    start_inf = time.perf_counter()
    outputs = sess.run(None, inputs)
    inf_time = (time.perf_counter() - start_inf) * 1000
    
    # 4. Decoding
    tokens = load_tokens(tokens_path)
    text = greedy_decode(outputs[0], tokens)
    
    print(f"\n=== SenseVoice ORT Results ===")
    print(f"Result: {text}")
    print(f"Inference: {inf_time:.2f} ms")
    print(f"Model RTF: {inf_time/1000 / duration:.4f}")
    
    total_time = feat_time + inf_time
    print(f"Total RTF: {(total_time/1000) / duration:.4f}")

if __name__ == "__main__":
    main()
