# app/model.py
import torch
import torch.nn as nn

class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=9, stride=1):
        super().__init__()
        padding = kernel // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.LeakyReLU(0.2)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class UpsampleBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, scale=2):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=scale*2, stride=scale, padding=scale//2, output_padding=scale%2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.LeakyReLU(0.2)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SampleEncoder(nn.Module):
    def __init__(self, in_ch=1, base_channels=32, n_layers=4):
        super().__init__()
        ch = base_channels
        layers = [ConvBlock1D(in_ch, ch, stride=2)]
        for _ in range(1, n_layers):
            layers.append(ConvBlock1D(ch, ch*2, stride=2))
            ch *= 2
        self.net = nn.Sequential(*layers)
        self.out_dim = ch
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_ch=256, n_layers=5):
        super().__init__()
        ch = latent_ch
        layers = []
        for _ in range(n_layers):
            layers.append(UpsampleBlock1D(ch, ch//2))
            ch = ch//2
        layers.append(nn.Conv1d(ch, 1, kernel_size=9, padding=4))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class SamplePackMixer(nn.Module):
    def __init__(self, base_channels=32, encoder_layers=4, transformer_dim=256, transformer_heads=4, transformer_layers=3, decoder_layers=5):
        super().__init__()
        self.encoder = SampleEncoder(in_ch=1, base_channels=base_channels, n_layers=encoder_layers)
        self.pool_proj = nn.Linear(self.encoder.out_dim, transformer_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=transformer_heads, dim_feedforward=transformer_dim*4, batch_first=True)
        self.mixer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.seq_proj = nn.Linear(transformer_dim, transformer_dim)
        self.decoder = Decoder(latent_ch=transformer_dim, n_layers=decoder_layers)

    def forward(self, x, target_frames=None):
        B, N, C, T = x.shape
        x = x.view(B*N, C, T)
        z = self.encoder(x)
        z = z.mean(dim=2)
        z = self.pool_proj(z).view(B, N, -1)
        mixed = self.mixer(z)
        mixed = mixed.mean(dim=1)
        if target_frames is None:
            target_frames = T * 32  # heuristic
        L = max(16, target_frames // 1024)
        seq = self.seq_proj(mixed).unsqueeze(-1).repeat(1, 1, L)
        out = self.decoder(seq)
        return out
