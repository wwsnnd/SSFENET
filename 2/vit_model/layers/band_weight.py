import torch
import torch.nn as nn

class BandWeightGenerator(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(BandWeightGenerator, self).__init__()
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