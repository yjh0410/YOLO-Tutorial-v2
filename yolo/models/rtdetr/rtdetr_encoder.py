import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_modules.backbone import build_backbone
from .basic_modules.fpn      import build_fpn


# ----------------- Image Encoder -----------------
class ImageEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # ---------------- Basic settings ----------------
        ## Basic parameters
        self.cfg = cfg
        ## Network parameters
        self.strides    = cfg.out_stride
        self.hidden_dim = cfg.hidden_dim
        self.num_levels = len(self.strides)
        
        # ---------------- Network settings ----------------
        ## Backbone Network
        self.backbone = build_backbone(cfg, pretrained=cfg.pretrained)
        self.fpn_feat_dims = self.backbone.feat_dims[-3:]

        ## Feature Pyramid Network
        self.fpn = build_fpn(cfg, self.fpn_feat_dims)
        self.fpn_dims = self.fpn.out_dims
        
    def forward(self, x):
        pyramid_feats = self.backbone(x)
        pyramid_feats = self.fpn(pyramid_feats)

        return pyramid_feats
