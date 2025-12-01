# app/train.py
import torch
from torch.utils.data import DataLoader
from model import SamplePackMixer
from dataset import SamplePackCatalog, MixingDataset, SR
import os

def train_loop(root_packs="packs", epochs=50, batch_size=4, lr=1e-4, checkpoint_dir="checkpoints"):
    catalog = SamplePackCatalog(root_packs)
    ds = MixingDataset(catalog, n_samples=6, sample_max_len=SR*5)  # 5s chunks for training
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SamplePackMixer().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(loader):
            # batch: (B, N, 1, T)
            batch = batch.to(device)
            # target: naive mix (sum) normalized
            target = batch.sum(dim=1)  # (B,1,T)
            target = target / (target.abs().max(dim=2, keepdim=True)[0] + 1e-9)
            out = model(batch, target_frames=target.shape[-1]).clamp(-1,1)
            loss = criterion(out, target)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item()
            if (i+1) % 50 == 0:
                print(f"Epoch {epoch} iter {i} loss {running_loss/50:.4f}")
                running_loss = 0.0

        # checkpoint
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"mixer_epoch{epoch}.pth"))

if __name__ == "__main__":
    train_loop()