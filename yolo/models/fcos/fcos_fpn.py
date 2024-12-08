import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ------------------ Basic Feature Pyramid Network ------------------
class FcosFPN(nn.Module):
    def __init__(self, cfg, in_dims: List = [512, 1024, 2048]):
        super().__init__()
        # ------------------ Basic parameters -------------------
        self.out_dim = cfg.head_dim

        # ------------------ Network parameters -------------------
        self.input_proj_1 = nn.Conv2d(in_dims[0], self.out_dim, kernel_size=1)
        self.input_proj_2 = nn.Conv2d(in_dims[1], self.out_dim, kernel_size=1)
        self.input_proj_3 = nn.Conv2d(in_dims[2], self.out_dim, kernel_size=1)

        self.smooth_layer_1 = nn.Conv2d(self.out_dim, self.out_dim, kernel_size=3, padding=1, stride=1)
        self.smooth_layer_2 = nn.Conv2d(self.out_dim, self.out_dim, kernel_size=3, padding=1, stride=1)
        self.smooth_layer_3 = nn.Conv2d(self.out_dim, self.out_dim, kernel_size=3, padding=1, stride=1)

        self.p6_conv = nn.Conv2d(self.out_dim, self.out_dim, kernel_size=3, stride=2, padding=1)

    def forward(self, feats):
        """
            feats: (List of Tensor) [C3, C4, C5]
        """
        c3, c4, c5 = feats

        # -------- Input projection --------
        p3 = self.input_proj_1(c3)
        p4 = self.input_proj_2(c4)
        p5 = self.input_proj_3(c5)
        
        # -------- Feature fusion --------
        outputs = [self.smooth_layer_3(p5)]
        # P5 -> P4
        p4 = p4 + F.interpolate(p5, size=p4.shape[2:], mode='nearest')
        outputs.insert(0, self.smooth_layer_2(p4))

        # P4 -> P3
        p3 = p3 + F.interpolate(p4, size=p3.shape[2:], mode='nearest')
        outputs.insert(0, self.smooth_layer_1(p3))

        # P5 -> P6
        outputs.append(self.p6_conv(outputs[-1]))

        # [P3, P4, P5, P6]
        return outputs


if __name__=='__main__':
    import time
    from thop import profile
    # Model config
    
    # YOLOv2-Base config
    class FcosBaseConfig(object):
        def __init__(self) -> None:
            # ---------------- Model config ----------------
            self.width    = 0.50
            self.depth    = 0.34
            self.out_stride = [8, 16, 32, 64]
            ## Head
            self.head_dim = 256

    cfg = FcosBaseConfig()
    # Build a head
    in_dims  = [128, 256, 512]
    fpn = FcosFPN(cfg, in_dims)

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
