"""A small time-conditioned U-Net noise predictor -- the network DDPM
trains to predict the noise added to an image at a given diffusion
timestep. Fully implemented: U-Nets and sinusoidal time embeddings predate
this paper (used in earlier conditional-generation work); DDPM's actual
contribution is the *training objective* and the *sampling process* built
around this network, which is train.py's placeholder, not this file's.
"""

import math

import torch
from torch import nn


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: (B,) integer timesteps. Returns (B, dim) -- same sin/cos
    construction as the transformer models' positional encoding, just
    embedding a timestep instead of a sequence position."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(torch.nn.functional.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, base_channels: int = 64, time_dim: int = 256):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 2
        self.in_conv = nn.Conv2d(3, c1, 3, padding=1)

        self.down1 = ResBlock(c1, c1, time_dim)
        self.downsample1 = nn.Conv2d(c1, c1, 4, stride=2, padding=1)  # 32x32 -> 16x16
        self.down2 = ResBlock(c1, c2, time_dim)
        self.downsample2 = nn.Conv2d(c2, c2, 4, stride=2, padding=1)  # 16x16 -> 8x8

        self.mid = ResBlock(c2, c3, time_dim)

        self.upsample2 = nn.ConvTranspose2d(c3, c3, 4, stride=2, padding=1)  # 8x8 -> 16x16
        self.up2 = ResBlock(c3 + c2, c2, time_dim)
        self.upsample1 = nn.ConvTranspose2d(c2, c2, 4, stride=2, padding=1)  # 16x16 -> 32x32
        self.up1 = ResBlock(c2 + c1, c1, time_dim)

        self.out_norm = nn.GroupNorm(8, c1)
        self.out_conv = nn.Conv2d(c1, 3, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 32, 32) noisy images, t: (B,) integer timesteps.
        Returns predicted noise, same shape as x."""
        t_emb = self.time_mlp(sinusoidal_time_embedding(t, self.time_dim))

        h0 = self.in_conv(x)
        h1 = self.down1(h0, t_emb)
        h2 = self.down2(self.downsample1(h1), t_emb)
        h_mid = self.mid(self.downsample2(h2), t_emb)

        u2 = self.up2(torch.cat([self.upsample2(h_mid), h2], dim=1), t_emb)
        u1 = self.up1(torch.cat([self.upsample1(u2), h1], dim=1), t_emb)

        return self.out_conv(torch.nn.functional.silu(self.out_norm(u1)))
