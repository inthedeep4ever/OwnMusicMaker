# ...existing code...
import torch
import torchaudio
import os
import numpy as np
from model import SamplePackMixer
from dataset import load_mono, SR
import threading

# Define CHECKPOINT path
CHECKPOINT = "checkpoints/model.pth"  # adjust to your actual checkpoint location

# limit CPU threads
torch.set_num_threads(1)

# Load model once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SamplePackMixer().to(device)
if os.path.exists(CHECKPOINT):
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()

# ...existing code...

# limit CPU threads to avoid oversubscription (sneller single-job latency)
torch.set_num_threads(1)

# Load model once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SamplePackMixer().to(device)
if os.path.exists(CHECKPOINT):
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()

# add cache and helper
_sample_cache = {}
_cache_lock = threading.Lock()

def load_cached_mono(path, sr=SR):
    key = f"{os.path.abspath(path)}@{sr}"
    with _cache_lock:
        cached = _sample_cache.get(key)
    if cached is not None:
        return cached.clone()
    wav = load_mono(path, sr=sr)
    with _cache_lock:
        _sample_cache[key] = wav.clone()
    return wav.clone()

def pad_to_len(wav, target_len):
    # ...existing code...
    c,t = wav.shape
    if t >= target_len:
        return wav[:, :target_len]
    pad = torch.zeros((c, target_len-t))
    return torch.cat([wav, pad], dim=1)

def overlap_add(segments, hop):
    # ...existing code...
    # unchanged
    N = len(segments)
    seg_len = segments[0].shape[-1]
    out_len = hop*(N-1)+seg_len
    out = np.zeros(out_len, dtype=np.float32)
    win = np.hanning(seg_len)
    for i,s in enumerate(segments):
        start = i*hop
        out[start:start+seg_len] += s*win
    out = out / (np.max(np.abs(out))+1e-9)
    return out

def generate_mix(sample_paths, out_path="generated/generated_mix.wav", duration_sec=60):
    target_len = SR * min(duration_sec, 600)
    max_sample_len = SR * 5
    waves = []
    for p in sample_paths:
        wav = load_cached_mono(p)              # <-- use cached loader
        wav = pad_to_len(wav, max_sample_len)
        waves.append(wav)
    waves = torch.stack(waves, dim=0).unsqueeze(0).to(device)

    # larger chunks -> fewer forward passes (adjust to memory)
    chunk_frames = 8 * SR
    n_segments = int(np.ceil(target_len / (chunk_frames//2)))
    outs = []
    with torch.no_grad():
        for _ in range(n_segments):
            out_chunk = model(waves, target_frames=chunk_frames)
            outs.append(out_chunk.squeeze().cpu().numpy())
    output = overlap_add(outs, chunk_frames//2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torchaudio.save(out_path, torch.from_numpy(output).unsqueeze(0), SR)
    return out_path
# ...existing code...