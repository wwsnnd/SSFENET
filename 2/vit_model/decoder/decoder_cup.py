# vit_model/decoder/decoder_cup.py
import torch
import torch.nn as nn
import numpy as np
from vit_model.decoder.decoder_block import DecoderBlock, Conv2dGELU

class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 64
        self.conv_more = Conv2dGELU(config.hidden_size, head_channels, kernel_size=3, padding=1, use_batchnorm=True)
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        skip_channels = config.skip_channels
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch)
            for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            skip = features[i] if (features is not None and i < len(features)) else None
            x = decoder_block(x, skip=skip)
        return x
