import torch
import torch.nn as nn
from vit_model.vit_transformer.transformer import Transformer
from vit_model.decoder.decoder_cup import DecoderCup
from vit_model.decoder.segmentation_head import SegmentationHead

class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=config.n_classes,
            kernel_size=3,
            upsampling=2
        )

    def forward(self, x_rgb, x_hsi):
        embeddings_rgb, embeddings_hsi, features, reg_loss = self.transformer.embeddings(x_rgb, x_hsi)
        encoded_rgb, encoded_hsi, attn_weights = self.transformer.encoder(embeddings_rgb, embeddings_hsi)
        fused_features = encoded_rgb + encoded_hsi
        decoded = self.decoder(fused_features, features)
        logits = self.segmentation_head(decoded)
        return logits, reg_loss, attn_weights, features