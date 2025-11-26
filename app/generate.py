import torch
from model import TechnoAutoencoder
import torchaudio


model = TechnoAutoencoder()
model.load_state_dict(torch.load("checkpoints/techno.pth"))
model.eval()


z = torch.randn(1, 128, 100) # random latent code
out = model.decoder(z).detach()


torchaudio.save("generated/output.wav", out, 44100)