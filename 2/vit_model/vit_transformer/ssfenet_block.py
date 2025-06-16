# vit_model/vit_transformer/ssfenet_block.py
import torch
import torch.nn as nn
from vit_model.vit_transformer.frft_fdconv import FrFDConv, FrFBM

class BandWeightGenerator(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        weights = self.fc(y).view(b, c, 1, 1)
        return x * weights


class SpectralSelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4, stride=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels * 4, in_channels)
        )
        self.conv_reduce = FrFDConv(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + self.norm1(x)
        x = self.mlp(x)
        x = x + self.norm2(x)
        x = x.permute(0, 2, 1).view(b, c, h, w)
        x = self.conv_reduce(x)
        x = self.bn(x)
        return self.relu(x)


class SpectralNoiseSuppression(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.frfbm = FrFBM(in_channels, n_bands=4, alpha=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        weights = self.frfbm(x)
        return x * weights


class SSFENetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.band_weight_generator = BandWeightGenerator(in_channels)
        self.spectral_self_attention = SpectralSelfAttention(in_channels, out_channels, stride=stride)
        self.noise_suppression = SpectralNoiseSuppression(out_channels)
        self.downsample = nn.Sequential(
            FrFDConv(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if stride != 1 or in_channels != out_channels else nn.Identity()
        self.output_fix = FrFDConv(out_channels, 64, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x)
        x = self.band_weight_generator(x)
        x = self.spectral_self_attention(x)
        x = self.noise_suppression(x)
        if x.shape != identity.shape:
            identity = torch.nn.functional.interpolate(identity, size=x.shape[2:], mode='bilinear', align_corners=False)
        x += identity
        x = self.output_fix(x)
        return self.relu(x)
