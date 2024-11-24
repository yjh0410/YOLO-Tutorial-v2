import torch.nn as nn

from .modules import AttentionPoolingClassifier
from .vit     import ImageEncoderViT


class ViTForImageClassification(nn.Module):
    def __init__(self,
                 image_encoder :ImageEncoderViT,
                 num_classes   :int   = 1000,
                 qkv_bias      :bool  = True,
                 ):
        super().__init__()
        # -------- Model parameters --------
        self.encoder    = image_encoder
        self.classifier = AttentionPoolingClassifier(
            image_encoder.patch_embed_dim, num_classes, image_encoder.num_heads, qkv_bias, num_queries=1)

    def forward(self, x):
        """
        Inputs:
            x: (torch.Tensor) -> [B, C, H, W]. Input image.
        """
        x = self.encoder(x)
        x, x_cls = self.classifier(x)

        return x

