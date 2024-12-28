import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

try:
    from .modules import ConvModule, ConvBlocks
except:
    from  modules import ConvModule, ConvBlocks


# Yolov3FPN
class Yolov3FPN(nn.Module):
    def __init__(self,
                 in_dims: List = [256, 512, 1024],
                 head_dim: int = 256,
                 ):
        super(Yolov3FPN, self).__init__()
        self.in_dims = in_dims
        self.head_dim = head_dim
        self.fpn_out_dims = [head_dim] * 3
        c3, c4, c5 = in_dims

        # P5 -> P4
        self.top_down_layer_1 = ConvBlocks(c5, 512)
        self.reduce_layer_1   = ConvModule(512, 256, kernel_size=1)

        # P4 -> P3
        self.top_down_layer_2 = ConvBlocks(c4 + 256, 256)
        self.reduce_layer_2   = ConvModule(256, 128, kernel_size=1)

        # P3
        self.top_down_layer_3 = ConvBlocks(c3 + 128, 128)

        # output proj layers
        self.out_layers = nn.ModuleList([ConvModule(in_dim, head_dim, kernel_size=1)
                                         for in_dim in [128, 256, 512]
                                         ])

    def forward(self, features):
        c3, c4, c5 = features
        
        # p5/32
        p5 = self.top_down_layer_1(c5)

        # p4/16
        p5_up = F.interpolate(self.reduce_layer_1(p5), scale_factor=2.0)
        p4 = self.top_down_layer_2(torch.cat([c4, p5_up], dim=1))

        # P3/8
        p4_up = F.interpolate(self.reduce_layer_2(p4), scale_factor=2.0)
        p3 = self.top_down_layer_3(torch.cat([c3, p4_up], dim=1))

        out_feats = [p3, p4, p5]

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
    fpn = Yolov3FPN(in_dims, head_dim=256)

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
