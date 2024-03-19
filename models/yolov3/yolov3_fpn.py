from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolov3_basic import BasicConv, ResBlock


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
