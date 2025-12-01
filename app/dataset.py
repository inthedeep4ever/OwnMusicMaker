# app/dataset.py
import os
import torchaudio
import random
import torch
from torch.utils.data import Dataset

SR = 44100
MAX_SAMPLE_LEN = SR * 10  # 10s default for samples, but dataset supports longer tracks

def load_mono(path, sr=SR):
    wav, orig_sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    return wav

class SamplePackCatalog:
    """
    Indexes directory structure:
    /packs/<packname>/<category>/*.wav
    category can be 'claps','kicks','loops','tracks',...
    """
    def __init__(self, root_packs):
        self.root = root_packs
        self.catalog = {}  # pack -> category -> [paths]
        self._build()

    def _build(self):
        if not os.path.exists(self.root):
            return
        for pack in os.listdir(self.root):
            pack_dir = os.path.join(self.root, pack)
            if not os.path.isdir(pack_dir):
                continue
            self.catalog[pack] = {}
            for category in os.listdir(pack_dir):
                cat_dir = os.path.join(pack_dir, category)
                if not os.path.isdir(cat_dir):
                    continue
                wavs = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.lower().endswith('.wav')]
                if wavs:
                    self.catalog[pack][category] = wavs

    def list_packs(self):
        return self.catalog.keys()

    def list_categories(self, pack):
        return list(self.catalog.get(pack, {}).keys())

    def list_samples(self, pack, category):
        return self.catalog.get(pack, {}).get(category, [])

class MixingDataset(Dataset):
    """
    Returns batches of N samples padded to same length.
    items: choose N random samples from catalog (or include user track)
    """
    def __init__(self, catalog: SamplePackCatalog, n_samples=6, sample_max_len=MAX_SAMPLE_LEN, sr=SR):
        self.catalog = catalog
        self.n_samples = n_samples
        self.sample_max_len = sample_max_len
        self.sr = sr
        # flatten sample list for random sampling
        self.all_samples = []
        for pack, cats in self.catalog.catalog.items():
            for cat, files in cats.items():
                self.all_samples.extend(files)

    def __len__(self):
        return max(1000, len(self.all_samples))

    def pad_or_trim(self, wav, target_len):
        c, t = wav.shape
        if t > target_len:
            start = random.randint(0, t - target_len)
            return wav[:, start:start+target_len]
        elif t < target_len:
            pad = torch.zeros((c, target_len - t))
            return torch.cat([wav, pad], dim=1)
        else:
            return wav

    def __getitem__(self, idx):
        selection = random.sample(self.all_samples, k=self.n_samples)
        waves = []
        for p in selection:
            wav = load_mono(p, sr=self.sr)
            wav = self.pad_or_trim(wav, self.sample_max_len)
            waves.append(wav)
        # stack -> shape (N, 1, T)
        waves = torch.stack(waves, dim=0)
        return waves  # (N,1,T)