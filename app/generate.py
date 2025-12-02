# app/generate.py
import torch
import torchaudio
import os
import numpy as np
from model import SamplePackMixer
from dataset import load_mono, SR

CHECKPOINT = "checkpoints/mixer_epoch_last.pth"

# Load model once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SamplePackMixer().to(device)
if os.path.exists(CHECKPOINT):
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()

def pad_to_len(wav, target_len):
    c,t = wav.shape
    if t >= target_len:
        return wav[:, :target_len]
    pad = torch.zeros((c, target_len-t))
    return torch.cat([wav, pad], dim=1)

def overlap_add(segments, hop):
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
        wav = load_mono(p)
        wav = pad_to_len(wav, max_sample_len)
        waves.append(wav)
    waves = torch.stack(waves, dim=0).unsqueeze(0).to(device)
    chunk_frames = 4*SR
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

# ________________________________________-

# filepath: [generate.py](http://_vscodecontentref_/8)
# ...existing code...
import threading
# add cache and helper
_sample_cache = {}
_cache_lock = threading.Lock()

def load_cached_mono(path, sr=SR):
    # check cache first (key = absolute path + sr)
    key = f"{os.path.abspath(path)}@{sr}"
    with _cache_lock:
        cached = _sample_cache.get(key)
    if cached is not None:
        return cached.clone()  # return copy to avoid accidental in-place ops
    wav = load_mono(path, sr=sr)  # uses dataset.load_mono
    with _cache_lock:
        _sample_cache[key] = wav.clone()
    return wav.clone()

# replace inside generate_mix: use load_cached_mono instead of load_mono / pad_to_len
# ...existing code...
def generate_mix(sample_paths, out_path="generated/generated_mix.wav", duration_sec=60):
    target_len = SR * min(duration_sec, 600)
    max_sample_len = SR * 5
    waves = []
    for p in sample_paths:
        wav = load_cached_mono(p)              # <-- cached loader
        wav = pad_to_len(wav, max_sample_len)
        waves.append(wav)
    # ...existing code...
    # consider increasing chunk_frames to reduce number of model calls:
    chunk_frames = 8 * SR  # larger chunks -> fewer forward passes
    n_segments = int(np.ceil(target_len / (chunk_frames//2)))
    # ...existing code...
# ...existing code...