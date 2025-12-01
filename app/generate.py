# app/generate.py
import torch
import torchaudio
import os
from model import SamplePackMixer
from dataset import load_mono, SR
import math
import numpy as np

CHECKPOINT = "checkpoints/mixer_epoch_last.pth"

def pad_to_len(wav, target_len):
    c,t = wav.shape
    if t >= target_len:
        return wav[:, :target_len]
    pad = torch.zeros((c, target_len - t))
    return torch.cat([wav, pad], dim=1)

def overlap_add(segments, hop):
    """segments: list of (1, T) numpy arrays; hop: hop length"""
    # compute output length
    N = len(segments)
    seg_len = segments[0].shape[-1]
    out_len = hop*(N-1) + seg_len
    out = np.zeros(out_len, dtype=np.float32)
    win = np.hanning(seg_len)
    for i,s in enumerate(segments):
        start = i*hop
        out[start:start+seg_len] += s * win
    # normalize
    maxv = np.max(np.abs(out)) + 1e-9
    out = out / maxv
    return out

def generate_mix(sample_paths, out_path="generated/generated_mix.wav", duration_sec=60, sr=SR, device='cpu'):
    device = torch.device(device)
    model = SamplePackMixer().to(device)
    if os.path.exists(CHECKPOINT):
        model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    # Prepare each sample: load, mono, resample
    target_len = sr * min(duration_sec, 600)  # cap 10 minutes
    # Strategy: create sliding windows from latent decoder to cover duration via overlap-add
    # For simplicity we will tile the same mixed latent into segments and overlap-add
    # Create batch for all samples: pad/truncate to sample_max_len
    max_sample_len = sr * 5
    waves = []
    for p in sample_paths:
        wav = load_mono(p, sr=sr)
        wav = pad_to_len(wav, max_sample_len)
        waves.append(wav)
    waves = torch.stack(waves, dim=0)  # (N,1,T)
    waves = waves.unsqueeze(0)  # (1,N,1,T)

    # Get model output for a coarse chunk length (e.g., chunk_seconds=4)
    chunk_seconds = 4
    chunk_frames = chunk_seconds * sr

    outs = []
    # produce segments to cover duration
    n_segments = math.ceil(target_len / (chunk_frames // 2))  # 50% overlap
    for i in range(n_segments):
        # optionally you could modify sample_weights per segment to vary arrangement
        with torch.no_grad():
            out_chunk = model(waves.to(device), target_frames=chunk_frames)  # (1,1,chunk_frames)
        out_np = out_chunk.squeeze().cpu().numpy()
        outs.append(out_np)

    # overlap-add
    hop = chunk_frames//2
    output = overlap_add(outs, hop)
    output = (output * 0.95).astype(np.float32)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torchaudio.save(out_path, torch.from_numpy(output).unsqueeze(0), sr)
    return out_path

if __name__ == "__main__":
    # voorbeeldgebruik
    paths = ["packs/pack1/loops/loop1.wav", "packs/pack1/claps/clap1.wav"]
    generate_mix(paths, duration_sec=120)