# vit_model/transformer/transformer.py
import torch.nn as nn
from .embeddings import Embeddings
from .encoder import Encoder

class Transformer(nn.Module):
    def __init__(self, config, img_size, vis=False):
        super().__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.encoder(x)
        return x