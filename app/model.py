# app/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=9, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class UpsampleBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=9, scale=2):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=scale*2, stride=scale, padding=scale//2, output_padding=scale%2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SampleEncoder(nn.Module):
    """Encode a single sample (mono wav) into a compact embedding sequence."""
    def __init__(self, in_ch=1, base_channels=32, n_layers=4):
        super().__init__()
        ch = base_channels
        layers = [ConvBlock1D(in_ch, ch, kernel=9, stride=2)]
        for i in range(1, n_layers):
            layers.append(ConvBlock1D(ch, ch*2, kernel=9, stride=2))
            ch *= 2
        self.net = nn.Sequential(*layers)
        self.out_dim = ch

    def forward(self, x):
        # x: (B, 1, T)
        return self.net(x)  # (B, C, T_down)

class CrossMixer(nn.Module):
    """Mixes multiple sample-embeddings using attention"""
    def __init__(self, dim, nhead=4, num_layers=3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, dim_feedforward=dim*4, activation='relu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        # x: (B, N, D) where N = number of sample-embeddings (pooled)
        return self.transformer(x)  # (B, N, D)

class Decoder(nn.Module):
    """Decode mixed latent sequence back to waveform frames"""
    def __init__(self, latent_ch, base_channels=256, n_layers=4):
        super().__init__()
        layers = []
        ch = latent_ch
        for i in range(n_layers):
            layers.append(UpsampleBlock1D(ch, ch//2, scale=2))
            ch = ch//2
        # final conv to 1 channel
        layers.append(nn.Conv1d(ch, 1, kernel_size=9, padding=4))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # (B, 1, T_up)

class SamplePackMixer(nn.Module):
    """
    Overall model:
    - per-sample encoder -> embeddings
    - pool embeddings -> seq -> CrossMixer
    - expand/mask -> Decoder outputs waveform (mono)
    """
    def __init__(self, base_channels=32, encoder_layers=4, transformer_dim=256, transformer_heads=4, transformer_layers=3, decoder_layers=5):
        super().__init__()
        self.encoder = SampleEncoder(in_ch=1, base_channels=base_channels, n_layers=encoder_layers)
        # pooling to a fixed dimension per-sample (mean over time)
        self.pool_proj = nn.Linear(self.encoder.out_dim, transformer_dim)
        self.mixer = CrossMixer(transformer_dim, nhead=transformer_heads, num_layers=transformer_layers)
        # projector to latent seq (expand to temporal sequence)
        self.seq_proj = nn.Linear(transformer_dim, transformer_dim)  # will be used to create time-frames
        self.decoder = Decoder(latent_ch=transformer_dim, base_channels=transformer_dim, n_layers=decoder_layers)

    def forward(self, samples_waveforms, target_frames=None):
        """
        samples_waveforms: list or tensor (B, N, 1, T_var) -> we expect a tensor (B, N, 1, Tmax) padded
        target_frames: desired length in frames after upsampling (optional)
        """
        # pack dims
        B, N, C, T = samples_waveforms.shape
        x = samples_waveforms.view(B*N, C, T)  # (B*N,1,T)
        z = self.encoder(x)  # (B*N, C_e, T_e)
        # pool temporal dimension to get per-sample embedding
        z_pooled = z.mean(dim=2)  # (B*N, C_e)
        z_proj = self.pool_proj(z_pooled)  # (B*N, D)
        z_proj = z_proj.view(B, N, -1)  # (B, N, D)
        # mix across samples
        mixed = self.mixer(z_proj)  # (B, N, D)
        # collapse N dimension (mean or weighted sum) -> (B, D)
        mixed = mixed.mean(dim=1)  # (B, D)
        # expand to latent time-frames. We generate a coarse temporal sequence
        # target_frames default: infer from input T (upsample factor)
        if target_frames is None:
            # get encoder time reduction factor to estimate output length
            # simple heuristic: output upsample factor = 2**(encoder_layers) * 2**(decoder_layers)
            decoder_layers = 5
            target_frames = max(1, T * (2 ** decoder_layers))
        # create a temporal latent of shape (B, D, L)
        # simple repeated expansion:
        L = max(16, target_frames // 1024)  # coarse length
        seq = self.seq_proj(mixed).unsqueeze(-1).repeat(1, 1, L)  # (B, D, L)
        out = self.decoder(seq)  # (B, 1, T_out)
        return out