from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .yolov3_basic import BasicConv, ResBlock
except:
    from  yolov3_basic import BasicConv, ResBlock


# Yolov3FPN
class Yolov3FPN(nn.Module):
    def __init__(self, cfg, in_dims: List = [256, 512, 1024],
                 ):
        super(Yolov3FPN, self).__init__()
        self.in_dims = in_dims
        c3, c4, c5 = in_dims

        # ---------------------- Yolov3's Top down FPN ----------------------
        ## P5 -> P4
        self.top_down_layer_1 = ResBlock(in_dim     = c5,
                                         out_dim    = round(512*cfg.width),
                                         num_blocks = round(3*cfg.depth),
                                         expansion  = 0.5,
                                         shortcut   = False,
                                         act_type   = cfg.fpn_act,
                                         norm_type  = cfg.fpn_norm,
                                         depthwise  = cfg.fpn_depthwise)
        self.reduce_layer_1   = BasicConv(round(512*cfg.width), round(256*cfg.width), kernel_size=1, act_type=cfg.fpn_act, norm_type=cfg.fpn_norm)

        ## P4 -> P3
        self.top_down_layer_2 = ResBlock(in_dim     = c4 + round(256*cfg.width),
                                         out_dim    = round(256*cfg.width),
                                         num_blocks = round(3*cfg.depth),
                                         expansion  = 0.5,
                                         shortcut   = False,
                                         act_type   = cfg.fpn_act,
                                         norm_type  = cfg.fpn_norm,
                                         depthwise  = cfg.fpn_depthwise)
        self.reduce_layer_2   = BasicConv(round(256*cfg.width), round(128*cfg.width), kernel_size=1, act_type=cfg.fpn_act, norm_type=cfg.fpn_norm)
        
        ## P3
        self.top_down_layer_3 = ResBlock(in_dim     = c3 + round(128*cfg.width),
                                         out_dim    = round(128*cfg.width),
                                         num_blocks = round(3*cfg.depth),
                                         expansion  = 0.5,
                                         shortcut   = False,
                                         act_type   = cfg.fpn_act,
                                         norm_type  = cfg.fpn_norm,
                                         depthwise  = cfg.fpn_depthwise)

        # ---------------------- Yolov3's output projection ----------------------
        self.out_layers = nn.ModuleList([
            BasicConv(in_dim, round(cfg.head_dim*cfg.width), kernel_size=1,
                      act_type=cfg.fpn_act, norm_type=cfg.fpn_norm)
                      for in_dim in [round(128*cfg.width), round(256*cfg.width), round(512*cfg.width)]
                      ])
        self.out_dims = [round(cfg.head_dim*cfg.width)] * 3

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
    
    # YOLOv2-Base config
    class Yolov3BaseConfig(object):
        def __init__(self) -> None:
            # ---------------- Model config ----------------
            self.width    = 0.50
            self.depth    = 0.34
            self.out_stride = [8, 16, 32]
            self.max_stride = 32
            self.num_levels = 3
            ## FPN
            self.fpn_act  = 'silu'
            self.fpn_norm = 'BN'
            self.fpn_depthwise = False
            ## Head
            self.head_dim = 256

    cfg = Yolov3BaseConfig()
    # Build a head
    in_dims  = [128, 256, 512]
    fpn = Yolov3FPN(cfg, in_dims)

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
