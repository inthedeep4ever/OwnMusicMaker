import torch
from model import TechnoAutoencoder
import torchaudio


model = TechnoAutoencoder()
model.load_state_dict(torch.load("checkpoints/techno.pth"))
model.eval()


z = torch.randn(1, 128, 100) # random latent code
out = model.decoder(z).detach()

duration_sec = 180
out = model.decoder(torch.randn(1, 128, duration_sec * 44100 // model_stride))


torchaudio.save("generated/output.wav", out, 44100)