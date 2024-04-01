from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolox2_basic import BasicConv, CSPBlock


# Yolov5FPN
class Yolov5PaFPN(nn.Module):
    def __init__(self, cfg, in_dims: List = [256, 512, 1024],
                 ):
        super(Yolov5PaFPN, self).__init__()
        self.in_dims = in_dims
        c3, c4, c5 = in_dims

        # ---------------------- Yolox's Top down FPN ----------------------
        ## P5 -> P4
        self.reduce_layer_1   = BasicConv(c5, round(512*cfg.width),
                                          kernel_size=1, padding=0, stride=1,
                                          act_type=cfg.fpn_act, norm_type=cfg.fpn_norm)
        self.top_down_layer_1 = CSPBlock(in_dim     = c4 + round(512*cfg.width),
                                         out_dim    = round(512*cfg.width),
                                         num_blocks = round(3*cfg.depth),
                                         expansion  = 0.5,
                                         shortcut   = False,
                                         act_type   = cfg.fpn_act,
                                         norm_type  = cfg.fpn_norm,
                                         depthwise  = cfg.fpn_depthwise)

        ## P4 -> P3
        self.reduce_layer_2   = BasicConv(round(512*cfg.width), round(256*cfg.width),
                                          kernel_size=1, padding=0, stride=1,
                                          act_type=cfg.fpn_act, norm_type=cfg.fpn_norm)
        self.top_down_layer_2 = CSPBlock(in_dim     = c3 + round(256*cfg.width),
                                         out_dim    = round(256*cfg.width),
                                         num_blocks = round(3*cfg.depth),
                                         expansion  = 0.5,
                                         shortcut   = False,
                                         act_type   = cfg.fpn_act,
                                         norm_type  = cfg.fpn_norm,
                                         depthwise  = cfg.fpn_depthwise)
        
        # ---------------------- Yolox's Bottom up PAN ----------------------
        ## P3 -> P4
        self.downsample_layer_1 = BasicConv(round(256*cfg.width), round(256*cfg.width),
                                            kernel_size=3, padding=1, stride=2,
                                            act_type=cfg.fpn_act, norm_type=cfg.fpn_norm, depthwise=cfg.fpn_depthwise)
        self.bottom_up_layer_1  = CSPBlock(in_dim     = round(256*cfg.width) + round(256*cfg.width),
                                           out_dim    = round(512*cfg.width),
                                           num_blocks = round(3*cfg.depth),
                                           expansion  = 0.5,
                                           shortcut   = False,
                                           act_type   = cfg.fpn_act,
                                           norm_type  = cfg.fpn_norm,
                                           depthwise  = cfg.fpn_depthwise)
        ## P4 -> P5
        self.downsample_layer_2 = BasicConv(round(512*cfg.width), round(512*cfg.width),
                                            kernel_size=3, padding=1, stride=2,
                                            act_type=cfg.fpn_act, norm_type=cfg.fpn_norm, depthwise=cfg.fpn_depthwise)
        self.bottom_up_layer_2  = CSPBlock(in_dim     = round(512*cfg.width) + round(512*cfg.width),
                                           out_dim    = round(1024*cfg.width),
                                           num_blocks = round(3*cfg.depth),
                                           expansion  = 0.5,
                                           shortcut   = False,
                                           act_type   = cfg.fpn_act,
                                           norm_type  = cfg.fpn_norm,
                                           depthwise  = cfg.fpn_depthwise)

        # ---------------------- Yolox's output projection ----------------------
        self.out_layers = nn.ModuleList([
            BasicConv(in_dim, round(cfg.head_dim*cfg.width), kernel_size=1,
                      act_type=cfg.fpn_act, norm_type=cfg.fpn_norm)
                      for in_dim in [round(256*cfg.width), round(512*cfg.width), round(1024*cfg.width)]
                      ])
        self.out_dims = [round(cfg.head_dim*cfg.width)] * 3

    def forward(self, features):
        c3, c4, c5 = features
        
        # ------------------ Top down FPN ------------------
        ## P5 -> P4
        p5 = self.reduce_layer_1(c5)
        p5_up = F.interpolate(p5, scale_factor=2.0)
        p4 = self.top_down_layer_1(torch.cat([c4, p5_up], dim=1))

        ## P4 -> P3
        p4 = self.reduce_layer_2(p4)
        p4_up = F.interpolate(p4, scale_factor=2.0)
        p3 = self.top_down_layer_2(torch.cat([c3, p4_up], dim=1))

        # ------------------ Bottom up PAN ------------------
        ## P3 -> P4
        p3_ds = self.downsample_layer_1(p3)
        p4 = self.bottom_up_layer_1(torch.cat([p4, p3_ds], dim=1))

        ## P4 -> P5
        p4_ds = self.downsample_layer_2(p4)
        p5 = self.bottom_up_layer_2(torch.cat([p5, p4_ds], dim=1))

        out_feats = [p3, p4, p5]

        # output proj layers
        out_feats_proj = []
        for feat, layer in zip(out_feats, self.out_layers):
            out_feats_proj.append(layer(feat))
            
        return out_feats_proj
