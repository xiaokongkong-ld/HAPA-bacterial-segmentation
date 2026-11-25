"""Compact residual U-Net for the small HAPA bacterial segmentation set."""

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1,
                               padding_mode="replicate", bias=False)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1,
                               padding_mode="replicate", bias=False)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.activation = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.skip = (nn.Identity() if in_channels == out_channels else
                     nn.Conv2d(in_channels, out_channels, 1, bias=False))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.norm1(self.conv1(inputs)))
        x = self.dropout(self.norm2(self.conv2(x)))
        return self.activation(x + self.skip(inputs))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        hidden = max(8, skip_channels // 2)
        self.skip_projection = nn.Conv2d(skip_channels, hidden, 1, bias=False)
        self.gate_projection = nn.Conv2d(out_channels, hidden, 1, bias=False)
        self.attention_norm = nn.GroupNorm(8, hidden)
        self.attention_mask = nn.Conv2d(hidden, 1, 1)
        self.block = ResidualBlock(out_channels + skip_channels, out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear",
                          align_corners=False)
        x = self.reduce(x)
        attention = self.attention_mask(F.silu(self.attention_norm(
            self.skip_projection(skip) + self.gate_projection(x))))
        gated_skip = skip * torch.sigmoid(attention)
        return self.block(torch.cat((x, gated_skip), dim=1))


class U_Net(nn.Module):
    """Four-level attention-gated residual U-Net with 16-bit input scaling."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        channels = (16, 32, 64, 128, 256)
        self.pool = nn.MaxPool2d(2)
        self.enc0 = ResidualBlock(in_channels, channels[0])
        self.enc1 = ResidualBlock(channels[0], channels[1])
        self.enc2 = ResidualBlock(channels[1], channels[2])
        self.enc3 = ResidualBlock(channels[2], channels[3], dropout=0.05)
        self.bottleneck = ResidualBlock(channels[3], channels[4], dropout=0.1)
        self.dec3 = DecoderBlock(channels[4], channels[3], channels[3])
        self.dec2 = DecoderBlock(channels[3], channels[2], channels[2])
        self.dec1 = DecoderBlock(channels[2], channels[1], channels[1])
        self.dec0 = DecoderBlock(channels[1], channels[0], channels[0])
        self.output = nn.Conv2d(channels[0], out_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.float() / 65535.0
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        x = self.dec3(b, e3)
        x = self.dec2(x, e2)
        x = self.dec1(x, e1)
        x = self.dec0(x, e0)
        return self.output(x)
