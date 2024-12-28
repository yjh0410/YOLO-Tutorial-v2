import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

try:
    from .modules import ConvModule, CSPBlock
except:
    from  modules import ConvModule, CSPBlock


# PaFPN-CSP
class Yolov4PaFPN(nn.Module):
    def __init__(self, 
                 in_dims: List = [256, 512, 1024],
                 head_dim: int = 256,
                 ):
        super(Yolov4PaFPN, self).__init__()
        self.in_dims = in_dims
        self.head_dim = head_dim
        self.fpn_out_dims = [head_dim] * 3
        c3, c4, c5 = in_dims

        # top down
        ## P5 -> P4
        self.reduce_layer_1   = ConvModule(c5, 512, kernel_size=1)
        self.top_down_layer_1 = CSPBlock(in_dim = c4 + 512,
                                         out_dim = 512,
                                         expand_ratio = 0.5,
                                         num_blocks = 3,
                                         shortcut = False,
                                         )

        ## P4 -> P3
        self.reduce_layer_2   = ConvModule(512, 256, kernel_size=1)
        self.top_down_layer_2 = CSPBlock(in_dim = c3 + 256, 
                                         out_dim = 256,
                                         expand_ratio = 0.5,
                                         num_blocks = 3,
                                         shortcut = False,
                                         )

        # bottom up
        ## P3 -> P4
        self.reduce_layer_3    = ConvModule(256, 256, kernel_size=3, stride=2)
        self.bottom_up_layer_1 = CSPBlock(in_dim = 256 + 256,
                                          out_dim = 512,
                                          expand_ratio = 0.5,
                                          num_blocks = 3,
                                          shortcut = False,
                                          )

        ## P4 -> P5
        self.reduce_layer_4    = ConvModule(512, 512, kernel_size=3, stride=2)
        self.bottom_up_layer_2 = CSPBlock(in_dim = 512 + 512,
                                          out_dim = 1024,
                                          expand_ratio = 0.5,
                                          num_blocks = 3,
                                          shortcut = False,
                                          )

        # output proj layers
        self.out_layers = nn.ModuleList([ConvModule(in_dim, head_dim, kernel_size=1)
                                         for in_dim in [256, 512, 1024]
                                         ])

    def forward(self, features):
        c3, c4, c5 = features

        c6 = self.reduce_layer_1(c5)
        c7 = F.interpolate(c6, scale_factor=2.0)   # s32->s16
        c8 = torch.cat([c7, c4], dim=1)
        c9 = self.top_down_layer_1(c8)
        # P3/8
        c10 = self.reduce_layer_2(c9)
        c11 = F.interpolate(c10, scale_factor=2.0)   # s16->s8
        c12 = torch.cat([c11, c3], dim=1)
        c13 = self.top_down_layer_2(c12)  # to det
        # p4/16
        c14 = self.reduce_layer_3(c13)
        c15 = torch.cat([c14, c10], dim=1)
        c16 = self.bottom_up_layer_1(c15)  # to det
        # p5/32
        c17 = self.reduce_layer_4(c16)
        c18 = torch.cat([c17, c6], dim=1)
        c19 = self.bottom_up_layer_2(c18)  # to det

        out_feats = [c13, c16, c19] # [P3, P4, P5]

        # output proj layers
        out_feats_proj = []
        for feat, layer in zip(out_feats, self.out_layers):
            out_feats_proj.append(layer(feat))
        return out_feats_proj


if __name__=='__main__':
    import time
    from thop import profile
    # Model config
    
    # Build a head
    in_dims  = [128, 256, 512]
    fpn = Yolov4PaFPN(in_dims, head_dim=256)

    # Randomly generate a input data
    x = [torch.randn(1, in_dims[0], 80, 80),
         torch.randn(1, in_dims[1], 40, 40),
         torch.randn(1, in_dims[2], 20, 20)]
    
    # Inference
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
