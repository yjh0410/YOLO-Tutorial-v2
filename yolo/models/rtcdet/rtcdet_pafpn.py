import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

try:
    from .rtcdet_basic import BasicConv, DWConv, ElanLayer, MDown, ADown
except:
    from  rtcdet_basic import BasicConv, DWConv, ElanLayer, MDown, ADown


# -------------- Feature pyramid network --------------
class RTCPaFPN(nn.Module):
    def __init__(self,
                 cfg,
                 in_dims :List = [256, 512, 1024],
                 ) -> None:
        super(RTCPaFPN, self).__init__()
        print('==============================')
        print('FPN: {}'.format("RTC-PaFPN"))
        # ----------- Basic Parameters -----------
        self.in_dims = in_dims[::-1]

        # ----------- Yolov8's Top-down FPN -----------
        ## P5 -> P4
        self.top_down_layer_1 = self.make_fpn_block(cfg, self.in_dims[0] + self.in_dims[1], round(512*cfg.width), round(3 * cfg.depth))
        ## P4 -> P3
        self.top_down_layer_2 = self.make_fpn_block(cfg, self.in_dims[2] + round(512*cfg.width), round(256*cfg.width), round(3 * cfg.depth))

        # ----------- Yolov8's Bottom-up PAN -----------
        ## P3 -> P4
        self.dowmsample_layer_1 = self.make_downsample_block(cfg, round(256*cfg.width), round(256*cfg.width))
        self.bottom_up_layer_1  = self.make_fpn_block(cfg, round(256*cfg.width) + round(512*cfg.width), round(512*cfg.width), round(3 * cfg.depth))
        ## P4 -> P5
        self.dowmsample_layer_2 = self.make_downsample_block(cfg, round(512*cfg.width), round(512*cfg.width))
        self.bottom_up_layer_2  = self.make_fpn_block(cfg, round(512*cfg.width) + self.in_dims[0], round(512*cfg.width*cfg.ratio), round(3 * cfg.depth))

        # ----------- Output projection -----------
        self.out_layers = nn.ModuleList([
            BasicConv(in_dim, round(256*cfg.width), kernel_size=1, act_type=cfg.fpn_act, norm_type=cfg.fpn_norm)
                      for in_dim in [round(256*cfg.width), round(512*cfg.width), round(512*cfg.width*cfg.ratio)]])
        self.out_dims = [round(256*cfg.width)] * 3

        self.init_weights()
        
    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.reset_parameters()

    def make_downsample_block(self, cfg, in_dim, out_dim):
        if cfg.fpn_ds_block == "conv":
            return BasicConv(in_dim, out_dim, kernel_size=3, padding=1, stride=2,
                             act_type=cfg.fpn_act, norm_type=cfg.fpn_norm, depthwise=cfg.fpn_depthwise)
        if cfg.fpn_ds_block == "dw_conv":
            return DWConv(in_dim, out_dim, kernel_size=3, padding=1, stride=2,
                             act_type=cfg.fpn_act, norm_type=cfg.fpn_norm)
        if cfg.fpn_ds_block == "mdown":
            return MDown(in_dim, out_dim, cfg.bk_act, cfg.bk_norm, cfg.bk_depthwise)
        if cfg.fpn_ds_block == "adown":
            return ADown(in_dim, out_dim, cfg.bk_act, cfg.bk_norm, cfg.bk_depthwise)
        else:
            raise NotImplementedError("Unknown fpn downsample block: {}".format(cfg.fpn_ds_block))
        
    def make_fpn_block(self, cfg, in_dim, out_dim, block_depth):
        if cfg.fpn_block == "elan_layer":
            return ElanLayer(in_dim     = in_dim,
                             out_dim    = out_dim,
                             num_blocks = block_depth,
                             expansion  = 0.5,
                             shortcut   = False,
                             act_type   = cfg.fpn_act,
                             norm_type  = cfg.fpn_norm,
                             depthwise  = cfg.fpn_depthwise)
        else:
            raise NotImplementedError("Unknown stage block: {}".format(cfg.bk_block))
        
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
                
        # ------------------ Output projection ------------------
        out_feats_proj = []
        for feat, layer in zip(out_feats, self.out_layers):
            out_feats_proj.append(layer(feat))
            
        return out_feats_proj
    