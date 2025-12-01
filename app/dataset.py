import torchaudio
from torch.utils.data import Dataset
import os


class TechnoDataset(Dataset):
def __init__(self, folder):
self.files = [folder + '/' + f for f in os.listdir(folder) if f.endswith('.wav')]


def __len__(self):
return len(self.files)


def __getitem__(self, idx):
wav, sr = torchaudio.load(self.files[idx])
wav = wav.mean(dim=0, keepdim=True) # mono
wav = wav[:, :44100 * 10] # limit 10 sec
return wav

class TechnoDataset(Dataset):
    def __init__(self, folder):
        # alleen .wav bestanden laden
        self.files = [f"{folder}/{f}" for f in os.listdir(folder) if f.endswith(".wav")]
