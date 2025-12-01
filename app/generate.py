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
