from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolov7_af_basic import BasicConv, ELANLayerFPN, MDown


# PaFPN-ELAN (YOLOv7's)
class Yolov7PaFPN(nn.Module):
    def __init__(self, cfg, in_dims: List = [512, 1024, 512]):
        super(Yolov7PaFPN, self).__init__()
        # ----------------------------- Basic parameters -----------------------------
        self.in_dims = in_dims
        self.out_dims = [round(256*cfg.width), round(512*cfg.width), round(1024*cfg.width)]
        c3, c4, c5 = in_dims

        # ----------------------------- Yolov7's Top-down FPN -----------------------------
        ## P5 -> P4
        self.reduce_layer_1 = BasicConv(c5, round(256*cfg.width),
                                        kernel_size=1, act_type=cfg.fpn_act, norm_type=cfg.fpn_norm)
        self.reduce_layer_2 = BasicConv(c4, round(256*cfg.width),
                                        kernel_size=1, act_type=cfg.fpn_act, norm_type=cfg.fpn_norm)
        self.top_down_layer_1 = ELANLayerFPN(in_dim     = round(256*cfg.width) + round(256*cfg.width),
                                             out_dim    = round(256*cfg.width),
                                             expansions   = cfg.fpn_expansions,
                                             branch_width = cfg.fpn_block_bw,
                                             branch_depth = cfg.fpn_block_dw,
                                             act_type   = cfg.fpn_act,
                                             norm_type  = cfg.fpn_norm,
                                             depthwise  = cfg.fpn_depthwise,
                                             )
        ## P4 -> P3
        self.reduce_layer_3 = BasicConv(round(256*cfg.width), round(128*cfg.width),
                                        kernel_size=1, act_type=cfg.fpn_act, norm_type=cfg.fpn_norm)
        self.reduce_layer_4 = BasicConv(c3, round(128*cfg.width),
                                        kernel_size=1, act_type=cfg.fpn_act, norm_type=cfg.fpn_norm)
        self.top_down_layer_2 = ELANLayerFPN(in_dim     = round(128*cfg.width) + round(128*cfg.width),
                                             out_dim    = round(128*cfg.width),
                                             expansions   = cfg.fpn_expansions,
                                             branch_width = cfg.fpn_block_bw,
                                             branch_depth = cfg.fpn_block_dw,
                                             act_type   = cfg.fpn_act,
                                             norm_type  = cfg.fpn_norm,
                                             depthwise  = cfg.fpn_depthwise,
                                             )
        # ----------------------------- Yolov7's Bottom-up PAN -----------------------------
        ## P3 -> P4
        self.downsample_layer_1 = MDown(round(128*cfg.width), round(256*cfg.width),
                                        act_type=cfg.fpn_act, norm_type=cfg.fpn_norm)
        self.bottom_up_layer_1 = ELANLayerFPN(in_dim     = round(256*cfg.width) + round(256*cfg.width),
                                              out_dim    = round(256*cfg.width),
                                              expansions   = cfg.fpn_expansions,
                                              branch_width = cfg.fpn_block_bw,
                                              branch_depth = cfg.fpn_block_dw,
                                              act_type     = cfg.fpn_act,
                                              norm_type    = cfg.fpn_norm,
                                              depthwise    = cfg.fpn_depthwise,
                                              )
        ## P4 -> P5
        self.downsample_layer_2 = MDown(round(256*cfg.width), round(512*cfg.width),
                                        act_type=cfg.fpn_act, norm_type=cfg.fpn_norm)
        self.bottom_up_layer_2 = ELANLayerFPN(in_dim     = round(512*cfg.width) + c5,
                                              out_dim    = round(512*cfg.width),
                                              expansions   = cfg.fpn_expansions,
                                              branch_width = cfg.fpn_block_bw,
                                              branch_depth = cfg.fpn_block_dw,
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

    def forward(self, features):
        c3, c4, c5 = features

        # ------------------ Top down FPN ------------------
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

        # ------------------ Bottom up PAN ------------------
        ## P3 -> P4
        p3_ds = self.downsample_layer_1(p3)
        p4 = torch.cat([p3_ds, p4], dim=1)
        p4 = self.bottom_up_layer_1(p4)

        ## P4 -> P5
        p4_ds = self.downsample_layer_2(p4)
        p5 = torch.cat([p4_ds, c5], dim=1)
        p5 = self.bottom_up_layer_2(p5)

        out_feats = [self.head_conv_1(p3), self.head_conv_2(p4), self.head_conv_3(p5)]
            
        return out_feats
