import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .yolov8_basic import BasicConv, ELANLayer


# Modified YOLOv8's PaFPN
class Yolov8PaFPN(nn.Module):
    def __init__(self,
                 cfg,
                 in_dims :List = [256, 512, 1024],
                 ) -> None:
        super(Yolov8PaFPN, self).__init__()
        print('==============================')
        print('FPN: {}'.format("Yolo PaFPN"))
        # --------------------------- Basic Parameters ---------------------------
        self.in_dims = in_dims[::-1]
        self.out_dims = [round(cfg.head_dim * cfg.width)] * 3

        # ---------------- Top dwon ----------------
        ## P5 -> P4
        self.top_down_layer_1 = ELANLayer(in_dim     = self.in_dims[0] + self.in_dims[1],
                                          out_dim    = round(512*cfg.width),
                                          expansion  = 0.5,
                                          num_blocks = round(3 * cfg.depth),
                                          shortcut   = False,
                                          act_type   = cfg.fpn_act,
                                          norm_type  = cfg.fpn_norm,
                                          depthwise  = cfg.fpn_depthwise,
                                          )
        ## P4 -> P3
        self.top_down_layer_2 = ELANLayer(in_dim     = self.in_dims[2] + round(512*cfg.width),
                                          out_dim    = round(256*cfg.width),
                                          expansion  = 0.5,
                                          num_blocks = round(3 * cfg.depth),
                                          shortcut   = False,
                                          act_type   = cfg.fpn_act,
                                          norm_type  = cfg.fpn_norm,
                                          depthwise  = cfg.fpn_depthwise,
                                          )
        # ---------------- Bottom up ----------------
        ## P3 -> P4
        self.dowmsample_layer_1 = BasicConv(round(256*cfg.width), round(256*cfg.width),
                                            kernel_size=3, padding=1, stride=2,
                                            act_type=cfg.fpn_act, norm_type=cfg.fpn_norm, depthwise=cfg.fpn_depthwise)
        self.bottom_up_layer_1 = ELANLayer(in_dim     = round(256*cfg.width) + round(512*cfg.width),
                                           out_dim    = round(512*cfg.width),
                                           expansion  = 0.5,
                                           num_blocks = round(3 * cfg.depth),
                                           shortcut   = False,
                                           act_type   = cfg.fpn_act,
                                           norm_type  = cfg.fpn_norm,
                                           depthwise  = cfg.fpn_depthwise,
                                           )
        ## P4 -> P5
        self.dowmsample_layer_2 = BasicConv(round(512*cfg.width), round(512*cfg.width),
                                            kernel_size=3, padding=1, stride=2,
                                            act_type=cfg.fpn_act, norm_type=cfg.fpn_norm, depthwise=cfg.fpn_depthwise)
        self.bottom_up_layer_2 = ELANLayer(in_dim     = round(512*cfg.width) + self.in_dims[0],
                                           out_dim    = round(512*cfg.width*cfg.ratio),
                                           expansion  = 0.5,
                                           num_blocks = round(3 * cfg.depth),
                                           shortcut   = False,
                                           act_type   = cfg.fpn_act,
                                           norm_type  = cfg.fpn_norm,
                                           depthwise  = cfg.fpn_depthwise,
                                           )
        self.out_layers = nn.ModuleList([
            BasicConv(feat_dim, self.out_dims[i], kernel_size=1, act_type=cfg.fpn_act, norm_type=cfg.fpn_norm)
            for i, feat_dim in enumerate([round(256*cfg.width), round(512*cfg.width), round(512*cfg.width*cfg.ratio)])
            ])

        self.init_weights()
        
    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
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
        
        # output proj layers
        out_feats_proj = []
        for feat, layer in zip(out_feats, self.out_layers):
            out_feats_proj.append(layer(feat))
        
        return out_feats_proj
