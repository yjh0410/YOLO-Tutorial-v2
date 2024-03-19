from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolov7_basic import BasicConv, ELANLayer, MDown


# PaFPN-ELAN (YOLOv7's)
class Yolov7PaFPN(nn.Module):
    def __init__(self, cfg, in_dims: List = [512, 1024, 512]):
        super(Yolov7PaFPN, self).__init__()
        # ----------------------------- Basic parameters -----------------------------
        self.in_dims = in_dims
        c3, c4, c5 = in_dims

        # ----------------------------- Top-down FPN -----------------------------
        ## P5 -> P4
        self.reduce_layer_1 = BasicConv(c5, round(256*cfg.width),
                                        kernel_size=1, act_type=cfg.fpn_act, norm_type=cfg.fpn_norm)
        self.reduce_layer_2 = BasicConv(c4, round(256*cfg.width),
                                        kernel_size=1, act_type=cfg.fpn_act, norm_type=cfg.fpn_norm)
        self.top_down_layer_1 = ELANLayer(in_dim     = round(256*cfg.width) + round(256*cfg.width),
                                          out_dim    = round(256*cfg.width),
                                          expansion  = 0.5,
                                          num_blocks = round(3*cfg.depth),
                                          act_type   = cfg.fpn_act,
                                          norm_type  = cfg.fpn_norm,
                                          depthwise  = cfg.fpn_depthwise,
                                          )
        ## P4 -> P3
        self.reduce_layer_3 = BasicConv(round(256*cfg.width), round(128*cfg.width),
                                        kernel_size=1, act_type=cfg.fpn_act, norm_type=cfg.fpn_norm)
        self.reduce_layer_4 = BasicConv(c3, round(128*cfg.width),
                                        kernel_size=1, act_type=cfg.fpn_act, norm_type=cfg.fpn_norm)
        self.top_down_layer_2 = ELANLayer(in_dim     = round(128*cfg.width) + round(128*cfg.width),
                                          out_dim    = round(128*cfg.width),
                                          expansion  = 0.5,
                                          num_blocks = round(3*cfg.depth),
                                          act_type   = cfg.fpn_act,
                                          norm_type  = cfg.fpn_norm,
                                          depthwise  = cfg.fpn_depthwise,
                                          )
        # ----------------------------- Bottom-up FPN -----------------------------
        ## P3 -> P4
        self.downsample_layer_1 = MDown(round(128*cfg.width), round(256*cfg.width),
                                        act_type=cfg.fpn_act, norm_type=cfg.fpn_norm)
        self.bottom_up_layer_1 = ELANLayer(in_dim     = round(256*cfg.width) + round(256*cfg.width),
                                           out_dim    = round(256*cfg.width),
                                           expansion  = 0.5,
                                           num_blocks = round(3*cfg.depth),
                                           act_type   = cfg.fpn_act,
                                           norm_type  = cfg.fpn_norm,
                                           depthwise  = cfg.fpn_depthwise,
                                           )
        ## P4 -> P5
        self.downsample_layer_2 = MDown(round(256*cfg.width), round(512*cfg.width),
                                        act_type=cfg.fpn_act, norm_type=cfg.fpn_norm)
        self.bottom_up_layer_2 = ELANLayer(in_dim     = round(512*cfg.width) + c5,
                                           out_dim    = round(512*cfg.width),
                                           expansion  = 0.5,
                                           num_blocks = round(3*cfg.depth),
                                           act_type   = cfg.fpn_act,
                                           norm_type  = cfg.fpn_norm,
                                           depthwise  = cfg.fpn_depthwise,
                                           )

        # ----------------------------- Head conv layers -----------------------------
        ## Head convs
        self.head_conv_1 = BasicConv(round(128*cfg.width), round(256*cfg.width),
                                     kernel_size=3, padding=1, stride=1,
                                     act_type=cfg.fpn_act, norm_type=cfg.fpn_norm, depthwise=cfg.fpn_depthwise)
        self.head_conv_2 = BasicConv(round(256*cfg.width), round(512*cfg.width),
                                     kernel_size=3, padding=1, stride=1,
                                     act_type=cfg.fpn_act, norm_type=cfg.fpn_norm, depthwise=cfg.fpn_depthwise)
        self.head_conv_3 = BasicConv(round(512*cfg.width), round(1024*cfg.width),
                                     kernel_size=3, padding=1, stride=1,
                                     act_type=cfg.fpn_act, norm_type=cfg.fpn_norm, depthwise=cfg.fpn_depthwise)

        # ---------------------- Yolov5's output projection ----------------------
        self.out_layers = nn.ModuleList([
            BasicConv(in_dim, round(cfg.head_dim*cfg.width), kernel_size=1,
                      act_type=cfg.fpn_act, norm_type=cfg.fpn_norm)
                      for in_dim in [round(256*cfg.width), round(512*cfg.width), round(1024*cfg.width)]
                      ])
        self.out_dims = [round(cfg.head_dim*cfg.width)] * 3


    def forward(self, features):
        c3, c4, c5 = features

        ## P5 -> P4
        p5 = self.reduce_layer_1(c5)
        p5_up = F.interpolate(p5, scale_factor=2.0)
        p4 = self.reduce_layer_2(c4)
        p4 = self.top_down_layer_1(torch.cat([p5_up, p4], dim=1))

        ## P4 -> P3
        p4_in = self.reduce_layer_3(p4)
        p4_up = F.interpolate(p4_in, scale_factor=2.0)
        p3 = self.reduce_layer_4(c3)
        p3 = self.top_down_layer_2(torch.cat([p4_up, p3], dim=1))

        ## P3 -> P4
        p3_ds = self.downsample_layer_1(p3)
        p4 = torch.cat([p3_ds, p4], dim=1)
        p4 = self.bottom_up_layer_1(p4)

        ## P4 -> P5
        p4_ds = self.downsample_layer_2(p4)
        p5 = torch.cat([p4_ds, c5], dim=1)
        p5 = self.bottom_up_layer_2(p5)

        out_feats = [self.head_conv_1(p3), self.head_conv_2(p4), self.head_conv_3(p5)]

        # output proj layers
        out_feats_proj = []
        for feat, layer in zip(out_feats, self.out_layers):
            out_feats_proj.append(layer(feat))
            
        return out_feats_proj
