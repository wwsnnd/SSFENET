import torch
import torch.nn as nn
from .band_weight import BandWeightGenerator
from .spectral_attention import SpectralSelfAttention, SpatioSpectralAttention
from .noise_suppression import SpectralNoiseSuppression
from .multiscale import SpatioSpectralMultiScale
from vit_model.layers.frft_fdconv import FrFDConv

class SSFENetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SSFENetBlock, self).__init__()
        self.band_weight_generator = BandWeightGenerator(in_channels)
        self.spectral_self_attention = SpectralSelfAttention(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.spatio_spectral_attention = SpatioSpectralAttention(out_channels)
        self.noise_suppression = SpectralNoiseSuppression(out_channels)
        self.multi_scale = SpatioSpectralMultiScale(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            FrFDConv(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if stride != 1 or in_channels != out_channels else nn.Identity()
        self.output_fix = FrFDConv(out_channels, 64, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        identity = self.downsample(x)
        x = self.band_weight_generator(x)
        x = self.spectral_self_attention(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.spatio_spectral_attention(x)
        x = self.noise_suppression(x)
        x = self.multi_scale(x)
        x = self.bn2(x)
        x = x + identity
        x = self.output_fix(x)
        return self.relu(x)