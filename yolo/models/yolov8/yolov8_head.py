import torch
import torch.nn as nn
from typing import List

try:
    from .modules import ConvModule
except:
    from  modules import ConvModule


# -------------------- Detection Head --------------------
## Single-level Detection Head
class DetHead(nn.Module):
    def __init__(self,
                 in_dim       :int  = 256,
                 cls_head_dim :int  = 256,
                 reg_head_dim :int  = 256,
                 num_cls_head :int  = 2,
                 num_reg_head :int  = 2,
                 ):
        super().__init__()
        # --------- Basic Parameters ----------
        self.in_dim = in_dim
        self.num_cls_head = num_cls_head
        self.num_reg_head = num_reg_head
        
        # --------- Network Parameters ----------
        ## classification head
        cls_feats = []
        self.cls_head_dim = cls_head_dim
        for i in range(num_cls_head):
            if i == 0:
                cls_feats.append(ConvModule(in_dim, self.cls_head_dim, kernel_size=3, padding=1, stride=1))
            else:
                cls_feats.append(ConvModule(self.cls_head_dim, self.cls_head_dim, kernel_size=3, padding=1, stride=1))
        
        ## bbox regression head
        reg_feats = []
        self.reg_head_dim = reg_head_dim
        for i in range(num_reg_head):
            if i == 0:
                reg_feats.append(ConvModule(in_dim, self.reg_head_dim, kernel_size=3, padding=1, stride=1))
            else:
                reg_feats.append(ConvModule(self.reg_head_dim, self.reg_head_dim, kernel_size=3, padding=1, stride=1))
        
        self.cls_feats = nn.Sequential(*cls_feats)
        self.reg_feats = nn.Sequential(*reg_feats)

        self.init_weights()
        
    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.reset_parameters()

    def forward(self, x):
        """
            in_feats: (Tensor) [B, C, H, W]
        """
        cls_feats = self.cls_feats(x)
        reg_feats = self.reg_feats(x)

        return cls_feats, reg_feats
    
## Multi-level Detection Head
class Yolov8DetHead(nn.Module):
    def __init__(self, cfg, in_dims: List = [256, 512, 1024]):
        super().__init__()
        self.num_levels = len(cfg.out_stride)
        ## ----------- Network Parameters -----------
        self.multi_level_heads = nn.ModuleList(
            [DetHead(in_dim       = in_dims[level],
                     cls_head_dim = max(in_dims[0], min(cfg.num_classes, 128)),
                     reg_head_dim = max(in_dims[0]//4, 16, 4*cfg.reg_max),
                     num_cls_head = cfg.num_cls_head,
                     num_reg_head = cfg.num_reg_head,
                     ) for level in range(self.num_levels)])
        # --------- Basic Parameters ----------
        self.in_dims = in_dims
        self.cls_head_dim = self.multi_level_heads[0].cls_head_dim
        self.reg_head_dim = self.multi_level_heads[0].reg_head_dim

    def forward(self, feats):
        """
            feats: List[(Tensor)] [[B, C, H, W], ...]
        """
        cls_feats = []
        reg_feats = []
        for feat, head in zip(feats, self.multi_level_heads):
            # ---------------- Pred ----------------
            cls_feat, reg_feat = head(feat)

            cls_feats.append(cls_feat)
            reg_feats.append(reg_feat)

        return cls_feats, reg_feats


if __name__=='__main__':
    import time
    from thop import profile
    
    # YOLOv8-Base config
    class Yolov8BaseConfig(object):
        def __init__(self) -> None:
            # ---------------- Model config ----------------
            self.width    = 0.25
            self.depth    = 0.34
            self.ratio    = 2.0
            self.reg_max  = 16
            self.out_stride = [8, 16, 32]
            self.max_stride = 32
            self.num_levels = 3
            ## Head
            self.num_cls_head = 2
            self.num_reg_head = 2

    cfg = Yolov8BaseConfig()
    cfg.num_classes = 80

    # Build a head
    fpn_dims = [64, 128, 256]
    pyramid_feats = [torch.randn(1, fpn_dims[0], 80, 80),
                     torch.randn(1, fpn_dims[1], 40, 40),
                     torch.randn(1, fpn_dims[2], 20, 20)]
    head = Yolov8DetHead(cfg, fpn_dims)


    # Inference
    t0 = time.time()
    cls_feats, reg_feats = head(pyramid_feats)
    t1 = time.time()
    print('Time: ', t1 - t0)
    print("====== Yolov8 Head output ======")
    for level, (cls_f, reg_f) in enumerate(zip(cls_feats, reg_feats)):
        print("- Level-{} : ".format(level), cls_f.shape, reg_f.shape)

    flops, params = profile(head, inputs=(pyramid_feats, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
    