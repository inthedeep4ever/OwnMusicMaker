import torch
import torch.nn as nn


class TechnoAutoencoder(nn.Module):
def __init__(self):
super().__init__()
self.encoder = nn.Sequential(
nn.Conv1d(1, 32, 4, stride=2), nn.ReLU(),
nn.Conv1d(32, 64, 4, stride=2), nn.ReLU(),
nn.Conv1d(64, 128, 4, stride=2), nn.ReLU(),
)
self.decoder = nn.Sequential(
nn.ConvTranspose1d(128, 64, 4, stride=2), nn.ReLU(),
nn.ConvTranspose1d(64, 32, 4, stride=2), nn.ReLU(),
nn.ConvTranspose1d(32, 1, 4, stride=2), nn.Tanh(),
)


def forward(self, x):
z = self.encoder(x)
out = self.decoder(z)
return out