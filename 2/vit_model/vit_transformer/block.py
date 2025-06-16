# vit_model/transformer/block.py
import torch.nn as nn
from .attention import Attention
from .mlp import Mlp

class Block(nn.Module):
    def __init__(self, config, vis=False, mode="sa"):
        super().__init__()
        self.mode = mode  # "sa" or "mba"

        self.attn_norm_rgb = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn_norm_hsi = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm_rgb = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm_hsi = nn.LayerNorm(config.hidden_size, eps=1e-6)

        self.attn = Attention(config, vis=vis, mode=mode)
        self.ffn_rgb = Mlp(config)
        self.ffn_hsi = Mlp(config)

    def forward(self, rgb, hsi):
        # Attention
        rgb_residual = rgb
        hsi_residual = hsi
        rgb = self.attn_norm_rgb(rgb)
        hsi = self.attn_norm_hsi(hsi)

        rgb, hsi, weights = self.attn(rgb, hsi)

        rgb = rgb + rgb_residual
        hsi = hsi + hsi_residual

        # MLP
        rgb_residual = rgb
        hsi_residual = hsi
        rgb = self.ffn_norm_rgb(rgb)
        hsi = self.ffn_norm_hsi(hsi)
        rgb = self.ffn_rgb(rgb) + rgb_residual
        hsi = self.ffn_hsi(hsi) + hsi_residual

        return rgb, hsi, weights
