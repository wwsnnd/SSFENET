import torch
import torch.nn as nn
from vit_model.layers.frft_fdconv import FrFDConv

class SpectralSelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4, stride=1):
        super(SpectralSelfAttention, self).__init__()
        self.stride = stride
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(in_channels)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels * 4, in_channels)
        )
        self.norm2 = nn.LayerNorm(in_channels)
        self.conv_reduce = FrFDConv(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.channel_fix = FrFDConv(out_channels, 64, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, H * W).permute(0, 2, 1)
        x = self.norm1(x)
        x = self.attn(x, x, x)[0] + x
        x = self.mlp(x) + self.norm2(x)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.conv_reduce(x)
        x = self.channel_fix(x)
        x = self.bn(x)
        return self.relu(x)

class SpatioSpectralAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8, dropout_rate=0.1):
        super(SpatioSpectralAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(in_channels)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.GELU(),
            nn.Linear(in_channels * 4, in_channels),
            nn.Dropout(dropout_rate)
        )
        self.norm2 = nn.LayerNorm(in_channels)

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, H * W).permute(0, 2, 1)
        x = self.norm1(x)
        x = self.attn(x, x, x)[0] + x
        x = self.mlp(x) + self.norm2(x)
        return x.permute(0, 2, 1).view(B, C, H, W)
