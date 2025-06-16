import torch
import torch.nn as nn
from vit_model.layers.frft_fdconv import FrFDConv

class SpatioSpectralMultiScale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatioSpectralMultiScale, self).__init__()
        self.input_fix = FrFDConv(in_channels, 64, kernel_size=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(64, 21, kernel_size=3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(64, 21, kernel_size=5, padding=2, bias=False)
        self.conv7 = nn.Conv2d(64, 22, kernel_size=7, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.spectral_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.fusion_conv = FrFDConv(out_channels, out_channels, kernel_size=1, bias=False)
        self.output_fix = FrFDConv(out_channels, 64, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x = self.input_fix(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        x = torch.cat([x3, x5, x7], dim=1)
        x = self.bn(x)
        x = self.relu(x)
        x = x * self.spectral_attention(x)
        x = self.fusion_conv(x)
        return self.output_fix(x)