import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .rtcdet_basic import ELANLayerFPN, MDown, BasicConv


# Modified YOLOv8's PaFPN
class RTCPaFPN(nn.Module):
    def __init__(self,
                 cfg,
                 in_dims :List = [256, 512, 1024],
                 ) -> None:
        super(RTCPaFPN, self).__init__()
        print('==============================')
        print('FPN: {}'.format("RTC-PaFPN"))
        # --------------------------- Basic Parameters ---------------------------
        self.in_dims = in_dims[::-1]
        self.out_dims = [round(256*cfg.channel_width), round(512*cfg.channel_width), round(1024*cfg.channel_width)]

        # ----------------------------- Yolov8's Top-down FPN -----------------------------
        ## P5 -> P4
        self.top_down_layer_1 = ELANLayerFPN(in_dim     = self.in_dims[0] + self.in_dims[1],
                                             out_dim    = round(512*cfg.channel_width),
                                             expansion  = 0.5,
                                             num_blocks = cfg.fpn_num_blocks,
                                             act_type   = cfg.fpn_act,
                                             norm_type  = cfg.fpn_norm,
                                             depthwise  = cfg.fpn_depthwise,
                                             )
        ## P4 -> P3
        self.top_down_layer_2 = ELANLayerFPN(in_dim     = self.in_dims[2] + round(512*cfg.channel_width),
                                             out_dim    = round(256*cfg.channel_width),
                                             expansion  = 0.5,
                                             num_blocks = cfg.fpn_num_blocks,
                                             act_type   = cfg.fpn_act,
                                             norm_type  = cfg.fpn_norm,
                                             depthwise  = cfg.fpn_depthwise,
                                             )
        # ----------------------------- Yolov8's Bottom-up PAN -----------------------------
        ## P3 -> P4
        self.dowmsample_layer_1 = MDown(round(256*cfg.channel_width), round(256*cfg.channel_width),
                                        act_type=cfg.fpn_act, norm_type=cfg.fpn_norm, depthwise=cfg.fpn_depthwise)
        self.bottom_up_layer_1  = ELANLayerFPN(in_dim     = round(256*cfg.channel_width) + round(512*cfg.channel_width),
                                               out_dim    = round(512*cfg.channel_width),
                                               expansion  = 0.5,
                                               num_blocks = cfg.fpn_num_blocks,
                                               act_type   = cfg.fpn_act,
                                               norm_type  = cfg.fpn_norm,
                                               depthwise  = cfg.fpn_depthwise,
                                               )
        ## P4 -> P5
        self.dowmsample_layer_2 = MDown(round(512*cfg.channel_width), round(512*cfg.channel_width),
                                        act_type=cfg.fpn_act, norm_type=cfg.fpn_norm, depthwise=cfg.fpn_depthwise)
        self.bottom_up_layer_2  = ELANLayerFPN(in_dim     = round(512*cfg.channel_width) + self.in_dims[0],
                                               out_dim    = round(1024*cfg.channel_width),
                                               expansion  = 0.5,
                                               num_blocks = cfg.fpn_num_blocks,
                                               act_type   = cfg.fpn_act,
                                               norm_type  = cfg.fpn_norm,
                                               depthwise  = cfg.fpn_depthwise,
                                               )

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
                
        return out_feats
