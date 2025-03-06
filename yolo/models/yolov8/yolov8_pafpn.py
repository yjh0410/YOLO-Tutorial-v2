import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

try:
    from .modules import ConvModule, C2fBlock
except:
    from  modules import ConvModule, C2fBlock


# YOLOv8's PaFPN
class Yolov8PaFPN(nn.Module):
    def __init__(self, cfg, in_dims :List = [256, 512, 1024]) -> None:
        super(Yolov8PaFPN, self).__init__()
        # --------------------------- Basic Parameters ---------------------------
        self.in_dims = in_dims[::-1]
        self.out_dims = [round(256*cfg.width),
                         round(512*cfg.width),
                         round(512*cfg.width*cfg.ratio)]

        # ----------------------------- Yolov8's Top-down FPN -----------------------------
        ## P5 -> P4
        self.top_down_layer_1 = C2fBlock(in_dim     = self.in_dims[0] + self.in_dims[1],
                                         out_dim    = round(512*cfg.width),
                                         expansion  = 0.5,
                                         num_blocks = round(3 * cfg.depth),
                                         shortcut   = False,
                                         )
        ## P4 -> P3
        self.top_down_layer_2 = C2fBlock(in_dim     = self.in_dims[2] + round(512*cfg.width),
                                         out_dim    = round(256*cfg.width),
                                         expansion  = 0.5,
                                         num_blocks = round(3 * cfg.depth),
                                         shortcut   = False,
                                         )
        # ----------------------------- Yolov8's Bottom-up PAN -----------------------------
        ## P3 -> P4
        self.dowmsample_layer_1 = ConvModule(round(256*cfg.width), round(256*cfg.width), kernel_size=3, padding=1, stride=2)
        self.bottom_up_layer_1 = C2fBlock(in_dim     = round(256*cfg.width) + round(512*cfg.width),
                                          out_dim    = round(512*cfg.width),
                                          expansion  = 0.5,
                                          num_blocks = round(3 * cfg.depth),
                                          shortcut   = False,
                                          )
        ## P4 -> P5
        self.dowmsample_layer_2 = ConvModule(round(512*cfg.width), round(512*cfg.width), kernel_size=3, padding=1, stride=2)
        self.bottom_up_layer_2 = C2fBlock(in_dim     = round(512*cfg.width) + self.in_dims[0],
                                          out_dim    = round(512*cfg.width*cfg.ratio),
                                          expansion  = 0.5,
                                          num_blocks = round(3 * cfg.depth),
                                          shortcut   = False,
                                          )

        self.init_weights()
        
    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.reset_parameters()

    def forward(self, features):
        c3, c4, c5 = features

        # ------------------ Top down FPN ------------------
        ## P5 -> P4
        p5_up = F.interpolate(c5, scale_factor=2.0)
        p4 = self.top_down_layer_1(torch.cat([p5_up, c4], dim=1))

        ## P4 -> P3
        p4_up = F.interpolate(p4, scale_factor=2.0)
        p3 = self.top_down_layer_2(torch.cat([p4_up, c3], dim=1))

        # ------------------ Bottom up FPN ------------------
        ## p3 -> P4
        p3_ds = self.dowmsample_layer_1(p3)
        p4 = self.bottom_up_layer_1(torch.cat([p3_ds, p4], dim=1))

        ## P4 -> 5
        p4_ds = self.dowmsample_layer_2(p4)
        p5 = self.bottom_up_layer_2(torch.cat([p4_ds, c5], dim=1))

        out_feats = [p3, p4, p5] # [P3, P4, P5]
                
        return out_feats
    

if __name__=='__main__':
    import time
    from thop import profile
    # Model config
    
    # YOLOv8-Base config
    class Yolov8BaseConfig(object):
        def __init__(self) -> None:
            # ---------------- Model config ----------------
            self.width    = 0.25
            self.depth    = 0.34
            self.ratio    = 2.0
            self.out_stride = [8, 16, 32]
            self.max_stride = 32
            self.num_levels = 3
            ## Head
            self.head_dim = 256

    cfg = Yolov8BaseConfig()
    # Build a head
    in_dims  = [64, 128, 256]
    fpn = Yolov8PaFPN(cfg, in_dims)

    # Inference
    x = [torch.randn(1, in_dims[0], 80, 80),
         torch.randn(1, in_dims[1], 40, 40),
         torch.randn(1, in_dims[2], 20, 20)]
    t0 = time.time()
    output = fpn(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
    print('====== FPN output ====== ')
    for level, feat in enumerate(output):
        print("- Level-{} : ".format(level), feat.shape)

    flops, params = profile(fpn, inputs=(x, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
