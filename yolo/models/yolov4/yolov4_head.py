import torch
import torch.nn as nn
from typing import List

try:
    from .modules import ConvModule
except:
    from  modules import ConvModule


class DecoupledHead(nn.Module):
    def __init__(self, cfg, in_dim: int = 256):
        super().__init__()
        self.in_dim = in_dim
        self.cls_head_dim = cfg.head_dim
        self.reg_head_dim = cfg.head_dim
        self.num_cls_head = cfg.num_cls_head
        self.num_reg_head = cfg.num_reg_head

        # classification feature head
        cls_feats = []
        for i in range(self.num_cls_head):
            if i == 0:
                cls_feats.append(ConvModule(in_dim, self.cls_head_dim, kernel_size=3, stride=1))
            else:
                cls_feats.append(ConvModule(self.cls_head_dim, self.cls_head_dim, kernel_size=3, stride=1))
                
        # box regression feature head
        reg_feats = []
        for i in range(self.num_reg_head):
            if i == 0:
                reg_feats.append(ConvModule(in_dim, self.reg_head_dim, kernel_size=3, stride=1))
            else:
                reg_feats.append(ConvModule(self.reg_head_dim, self.reg_head_dim, kernel_size=3, stride=1))

        self.cls_feats = nn.Sequential(*cls_feats)
        self.reg_feats = nn.Sequential(*reg_feats)

    def forward(self, x):
        """
            in_feats: (Tensor) [B, C, H, W]
        """
        cls_feats = self.cls_feats(x)
        reg_feats = self.reg_feats(x)

        return cls_feats, reg_feats


if __name__=='__main__':
    from thop import profile
    
    # YOLOv2 configuration
    class Yolov4BaseConfig(object):
        def __init__(self) -> None:
            # ---------------- Model config ----------------
            self.head_dim  = 256
            self.num_cls_head = 2
            self.num_reg_head = 2
    cfg = Yolov4BaseConfig()

    # Build a head
    model = DecoupledHead(cfg, in_dim= 256)

    # Randomly generate a input data
    x = torch.randn(2, 256, 20, 20)

    # Inference
    cls_feats, reg_feats = model(x)
    print(' - the shape of input :  ', x.shape)
    print(' - the shape of cls feats : ', cls_feats.shape)
    print(' - the shape of reg feats : ', reg_feats.shape)

    x = torch.randn(1, 256, 20, 20)
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('============== FLOPs & Params ================')
    print(' - FLOPs  : {:.2f} G'.format(flops / 1e9 * 2))
    print(' - Params : {:.2f} M'.format(params / 1e6))