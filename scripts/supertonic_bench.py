
import os
import json
import time
import numpy as np
import onnxruntime as ort

# Configuration
CONFIG_PATH = "examples/supertonic/models/onnx/tts.json"
MODELS_DIR = "examples/supertonic/models/onnx"
INDEXER_PATH = "examples/supertonic/models/onnx/unicode_indexer.json"
STYLE_PATH = "examples/supertonic/models/voice_styles/M1.json"

def get_input_names(session):
    return [i.name for i in session.get_inputs()]

def load_models():
    # Use single thread for fair comparison with lele
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    
    providers = ["CPUExecutionProvider"]
    
    dp = ort.InferenceSession(os.path.join(MODELS_DIR, "duration_predictor.onnx"), sess_options=opts, providers=providers)
    te = ort.InferenceSession(os.path.join(MODELS_DIR, "text_encoder.onnx"), sess_options=opts, providers=providers)
    ve = ort.InferenceSession(os.path.join(MODELS_DIR, "vector_estimator.onnx"), sess_options=opts, providers=providers)
    voc = ort.InferenceSession(os.path.join(MODELS_DIR, "vocoder.onnx"), sess_options=opts, providers=providers)
    
    return dp, te, ve, voc

import re

def process_text(text, indexer, lang="en"):
    # Match Rust preprocessing
    text = text.strip()
    if text and not re.search(r'[.!?;:,\'"\u201C\u201D\u2018\u2019)\]}…。」』】〉》›»]$', text):
        text += "."
    
    text = f"<{lang}>{text}</{lang}>"
    tokens = []
    for char in text:
        code = ord(char)
        if code < len(indexer):
            token = indexer[code]
            if token != -1:
                tokens.append(token)
            else:
                tokens.append(0)
        else:
            tokens.append(0)
    return np.array(tokens, dtype=np.int64).reshape(1, -1)

def main():
    print("Loading models and assets...")
    with open(INDEXER_PATH, "r") as f:
        indexer = json.load(f)
    
    with open(STYLE_PATH, "r") as f:
        style_data = json.load(f)
        style_dp = np.array(style_data["style_dp"]["data"], dtype=np.float32)
        style_ttl = np.array(style_data["style_ttl"]["data"], dtype=np.float32)
    
    print(f"style_dp shape: {style_dp.shape}")
    print(f"style_ttl shape: {style_ttl.shape}")
    
    dp_sess, te_sess, ve_sess, voc_sess = load_models()
    
    text = "This is getting complex for a 1-shot implementation."
    tokens = process_text(text, indexer, lang="en")
    
    print(f"Text: {text}")
    print(f"Tokens shape: {tokens.shape}")
    
    # 1. Duration Predictor
    start_time = time.time()
    
    dp_in = get_input_names(dp_sess)
    # ['text_ids', 'style_dp', 'text_mask']
    # If rank 3 expected, try [1, 1, T] or [1, T, 1]
    # In Transformers it's usually [B, 1, T] for attention masks
    text_mask = np.ones((1, 1, tokens.shape[1]), dtype=np.float32)
    dp_outputs = dp_sess.run(None, {
        "text_ids": tokens,
        "style_dp": style_dp,
        "text_mask": text_mask
    })
    durations = dp_outputs[0]
    
    # 2. Text Encoder
    te_in = get_input_names(te_sess)
    # ['text_ids', 'style_ttl', 'text_mask']
    te_outputs = te_sess.run(None, {
        "text_ids": tokens,
        "style_ttl": style_ttl,
        "text_mask": text_mask
    })
    text_encoding = te_outputs[0]
    
    # Calculate latent length
    # Config values from tts.json
    sample_rate = 44100
    base_chunk_size = 512
    chunk_compress_factor = 6
    chunk_size = base_chunk_size * chunk_compress_factor
    
    total_duration = np.sum(durations)
    wav_len = int(total_duration * sample_rate)
    latent_len = (wav_len + chunk_size - 1) // chunk_size
    if latent_len == 0: latent_len = 1
    
    # Initialize latent (gaussian noise)
    # Channels = 24 * 6 = 144
    latent = np.random.normal(size=(1, 144, latent_len)).astype(np.float32)
    mask = np.ones((1, 1, latent_len), dtype=np.float32)
    
    # 3. Vector Estimator (Iterative)
    num_steps = 5
    ve_time = 0
    curr_latent = latent.astype(np.float32)
    ve_in = get_input_names(ve_sess)
    # ['noisy_latent', 'text_emb', 'style_ttl', 'latent_mask', 'text_mask', 'current_step', 'total_step']
    
    total_steps_const = np.array([num_steps], dtype=np.float32)
    for i in range(num_steps):
        step_start = time.time()
        curr_step_const = np.array([i], dtype=np.float32)
        input_dict = {
            "noisy_latent": curr_latent,
            "text_emb": text_encoding,
            "style_ttl": style_ttl,
            "latent_mask": mask,
            "text_mask": text_mask,
            "current_step": curr_step_const,
            "total_step": total_steps_const
        }
        ve_out = ve_sess.run(None, input_dict)
        curr_latent = ve_out[0]
        ve_time += (time.time() - step_start)
    
    # Apply latent mask (matching Rust logic)
    curr_latent = curr_latent * mask
    
    # 4. Vocoder
    voc_start = time.time()
    # Normalize
    curr_latent_norm = curr_latent * 0.25
    
    voc_in = get_input_names(voc_sess)
    # ['latent']
    audio_out = voc_sess.run(None, {
        "latent": curr_latent_norm
    })
    audio = audio_out[0].flatten()
    
    # Truncate audio to match predicted duration
    audio = audio[:wav_len]
    
    voc_time = time.time() - voc_start
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Sample rate from config
    audio_duration = audio.shape[-1] / float(sample_rate)
    rtf = total_time / audio_duration
    
    print("\nPerformance Results (ORT):")
    print(f"Total Inference Time: {total_time:.4f}s")
    print(f"Audio Duration: {audio_duration:.4f}s")
    print(f"RTF: {rtf:.4f}")
    print(f"  DP + TE: {voc_start - start_time - ve_time:.4f}s")
    print(f"  VE (5 steps): {ve_time:.4f}s")
    print(f"  Vocoder: {voc_time:.4f}s")

if __name__ == "__main__":
    main()
