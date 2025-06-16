import torch
import torch.nn as nn
from vit_model.layers.frft_fdconv import FrFBM

class SpectralNoiseSuppression(nn.Module):
    def __init__(self, in_channels):
        super(SpectralNoiseSuppression, self).__init__()
        self.frfbm = FrFBM(in_channels, n_bands=4, alpha=0.5)

    def forward(self, x):
        return x * self.frfbm(x)