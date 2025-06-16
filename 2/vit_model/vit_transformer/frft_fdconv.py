import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import numpy as np


def fconv(x, y, c=0, fixed_P=None):
    N = x.size(-1) + y.size(-1) - 1
    if fixed_P is None:
        P = 2 ** (N - 1).bit_length()
    else:
        P = fixed_P
    x_padded = F.pad(x, (0, P - x.size(-1)))
    y_padded = F.pad(y, (0, P - y.size(-1)))
    z = torch.fft.fft(x_padded) * torch.fft.fft(y_padded)
    z = torch.fft.ifft(z)
    return z


class FrFT2D(nn.Module):
    def __init__(self, device='cuda'):
        super(FrFT2D, self).__init__()
        self.device = device

    def forward(self, matrix, angles):
        b, c, h, w = matrix.size()
        temp = matrix.to(torch.complex64)
        fixed_P = 2 ** (max(h, w) * 2 - 1).bit_length()
        return temp.real


class FrFT2DHSI(nn.Module):
    def __init__(self, size_x=256, size_y=256, channels=60):
        super(FrFT2DHSI, self).__init__()
        self.frft = FrFT2D()
        self.channels = channels

    def forward(self, x, angles):
        b, c, h, w = x.size()
        assert c == self.channels
        x_frft = torch.stack([
            self.frft(x[:, i:i+1], angles) for i in range(c)
        ], dim=1)
        return x_frft.squeeze(2)


class FrFDConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False, n_groups=64, alpha=0.5):
        super(FrFDConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if isinstance(stride, int):
            self.stride = stride
        elif isinstance(stride, (list, tuple)):
            if len(stride) == 2 and stride[0] == stride[1]:
                self.stride = stride[0]
            else:
                raise ValueError(f"stride must be a single integer or a list/tuple of 2 identical values, got: {stride}")
        else:
            raise TypeError(f"stride must be int or tuple/list of length 2, but got: {type(stride)}")

        self.n_groups = n_groups
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.frft = FrFT2D()
        self.spectral_params = nn.Parameter(
            torch.randn(kernel_size * in_channels, kernel_size * out_channels)
        )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, n_groups, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        indices = torch.norm(torch.stack(torch.meshgrid(
            torch.linspace(-0.5, 0.5, self.kernel_size * c, device=x.device),
            torch.linspace(-0.5, 0.5, self.kernel_size * self.out_channels, device=x.device),
            indexing='ij'
        )), dim=0)
        thresholds = torch.linspace(0, 0.5, self.n_groups + 1, device=x.device)
        weights = []
        for i in range(self.n_groups):
            mask = (thresholds[i] <= indices) & (indices < thresholds[i+1])
            params_i = self.spectral_params * mask.float()
            params_i = params_i.view(1, self.in_channels, self.kernel_size, self.kernel_size * self.out_channels)
            spatial_i = self.frft(params_i, [-self.alpha, -self.alpha])
            spatial_i = spatial_i.view(self.in_channels, self.kernel_size, self.out_channels, self.kernel_size).permute(2, 0, 1, 3)
            weights.append(spatial_i)

        weights = torch.stack(weights, dim=0)
        pi = self.attention(x).view(b, self.n_groups, 1, 1, 1, 1)
        dynamic_weight = (pi * weights.unsqueeze(0)).sum(dim=1)


        dynamic_weight = dynamic_weight.view(b * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        x = x.contiguous().view(1, b * c, h, w)
        out = F.conv2d(x, dynamic_weight, stride=self.stride, groups=b)
        out = out.view(b, self.out_channels, out.shape[2], out.shape[3])

        return out


class FrKSM(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, alpha=0.5):
        super(FrKSM, self).__init__()
        self.alpha = alpha
        self.frft = FrFT2DHSI(256, 256, in_channels)
        self.local_conv = nn.Conv1d(in_channels, mid_channels, kernel_size=1)
        self.global_fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        b, c, h, w = x.size()
        x_frft = self.frft(x, [self.alpha, self.alpha])
        local = self.local_conv(x_frft.view(b, c, -1)).view(b, -1, h, w)
        global_pool = x_frft.mean(dim=[2, 3])
        global_out = self.global_fc(global_pool).view(b, -1, 1, 1)
        return torch.sigmoid(local + global_out)


class FrFBM(nn.Module):
    def __init__(self, in_channels, n_bands=4, alpha=0.5):
        super(FrFBM, self).__init__()
        self.n_bands = n_bands
        self.alpha = alpha
        self.frft = FrFT2DHSI(256, 256, in_channels)
        self.conv_mod = nn.ModuleList([
            nn.Conv2d(in_channels, 1, kernel_size=1) for _ in range(n_bands)
        ])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        b, c, h, w = x.size()
        x_frft = self.frft(x, [self.alpha, self.alpha])
        thresholds = torch.linspace(0, 0.5, self.n_bands + 1, device=x.device)
        y_bands = []
        for band_idx in range(self.n_bands):
            mask = self.get_mask(b, c, h, w, thresholds[band_idx], thresholds[band_idx + 1]).to(x.device)
            y_b = self.frft(x_frft * mask, [-self.alpha, -self.alpha])
            y_bands.append(y_b)

        weights = []
        for band_idx in range(self.n_bands):
            a_b = torch.sigmoid(self.conv_mod[band_idx](x))
            a_b = a_b.expand(-1, c, -1, -1)
            weighted_y_b = a_b * y_bands[band_idx]
            weights.append(weighted_y_b)

        weights = torch.stack(weights, dim=0)
        weights = weights.mean(dim=0)
        return weights

    def get_mask(self, batch, channels, h, w, low, high):
        u, v = torch.meshgrid(
            torch.linspace(-0.5, 0.5, h, device=self.device),
            torch.linspace(-0.5, 0.5, w, device=self.device),
            indexing='ij'
        )
        norm = torch.norm(torch.stack([u, v]), dim=0)
        mask = (low <= norm) & (norm < high)
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = mask.expand(batch, channels, h, w)
        return mask



if __name__ == "__main__":
    x = torch.randn(4, 60, 256, 256).cuda()
    frft_hsi = FrFT2DHSI(256, 256, 60).cuda()
    x_frft = frft_hsi(x, [0.5, 0.5])

    conv = FrFDConv(60, 64, kernel_size=3, stride=1).cuda()
    x_conv = conv(x)

    ksm = FrKSM(60, 16, 64).cuda()
    ksm_out = ksm(x)

    fbm = FrFBM(60, n_bands=4).cuda()
    fbm_out = fbm(x)