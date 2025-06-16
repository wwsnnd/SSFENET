import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.utils import _pair

class Embeddings(nn.Module):
    def __init__(self, config, img_size, in_channels_rgb=3, in_channels_hsi=60):
        super().__init__()
        self.config = config
        self.use_ssfenet = getattr(config, "use_ssfenet", False)
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            self.hybrid = True
            grid_size = _pair(config.patches["grid"])
            patch_size = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])
        else:
            self.hybrid = False
            patch_size = _pair(config.patches["size"])

        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        if self.hybrid:
            from vit_model.vit_transformer.hybrid_resnet import FuseResNetV2
            self.hybrid_model = FuseResNetV2(
                block_units=config.resnet["num_layers"],
                width_factor=config.resnet["width_factor"],
                num_classes=config.n_classes,
                in_channels_hsi=64 if self.use_ssfenet else in_channels_hsi
            )
            if self.use_ssfenet:
                from vit_model.vit_transformer.ssfenet_block import SSFENetBlock
                self.my_spectral_module = SSFENetBlock(in_channels_hsi, out_channels=64, stride=4)
            in_channels_common = 256
        else:
            in_channels_common = in_channels_rgb

        self.patch_embeddings_rgb = nn.Conv2d(in_channels_common, config.hidden_size, kernel_size=patch_size, stride=patch_size)
        self.patch_embeddings_hsi = nn.Conv2d(in_channels_common, config.hidden_size, kernel_size=patch_size, stride=patch_size)

        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches, config.hidden_size))
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x_rgb, x_hsi):
        if self.hybrid:
            if self.use_ssfenet:
                x_hsi = self.my_spectral_module(x_hsi)
            _, reg_loss, features = self.hybrid_model(x_rgb, x_hsi)
            x_rgb = features[0]
            x_hsi = features[0]
        else:
            features = None
            reg_loss = None

        x_rgb = self.patch_embeddings_rgb(x_rgb).flatten(2).transpose(1, 2)
        x_hsi = self.patch_embeddings_hsi(x_hsi).flatten(2).transpose(1, 2)

        x_rgb = self.dropout(x_rgb + self.position_embeddings)
        x_hsi = self.dropout(x_hsi + self.position_embeddings)

        return x_rgb, x_hsi, features, reg_loss
