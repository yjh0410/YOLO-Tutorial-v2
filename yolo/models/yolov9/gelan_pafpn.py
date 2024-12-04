import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

try:
    from .gelan_basic import RepGElanLayer, ADown
except:
    from  gelan_basic import RepGElanLayer, ADown


# PaFPN-ELAN
class GElanPaFPN(nn.Module):
    def __init__(self,
                 cfg,
                 in_dims :List = [256, 512, 256],
                 ) -> None:
        super(GElanPaFPN, self).__init__()
        print('==============================')
        print('FPN: {}'.format("GELAN PaFPN"))
        # --------------------------- Basic Parameters ---------------------------
        self.in_dims = in_dims[::-1]
        self.out_dims = [cfg.fpn_feats_td["p3"][1], cfg.fpn_feats_bu["p4"][1], cfg.fpn_feats_bu["p5"][1]]

        # ---------------- Top dwon ----------------
        ## P5 -> P4
        self.top_down_layer_1 = RepGElanLayer(in_dim     = self.in_dims[0] + self.in_dims[1],
                                              inter_dims = cfg.fpn_feats_td["p4"][0],
                                              out_dim    = cfg.fpn_feats_td["p4"][1],
                                              num_blocks = cfg.fpn_depth,
                                              shortcut   = False,
                                              act_type   = cfg.fpn_act,
                                              norm_type  = cfg.fpn_norm,
                                              depthwise  = cfg.fpn_depthwise,
                                              )
        ## P4 -> P3
        self.top_down_layer_2 = RepGElanLayer(in_dim     = cfg.fpn_feats_td["p4"][1] + self.in_dims[2],
                                              inter_dims = cfg.fpn_feats_td["p3"][0],
                                              out_dim    = cfg.fpn_feats_td["p3"][1],
                                              num_blocks = cfg.fpn_depth,
                                              shortcut   = False,
                                              act_type   = cfg.fpn_act,
                                              norm_type  = cfg.fpn_norm,
                                              depthwise  = cfg.fpn_depthwise,
                                              )
        # ---------------- Bottom up ----------------
        ## P3 -> P4
        self.dowmsample_layer_1 = ADown(cfg.fpn_feats_td["p3"][1], cfg.fpn_feats_td["p3"][1],
                                        act_type=cfg.fpn_act, norm_type=cfg.fpn_norm, depthwise=cfg.fpn_depthwise)
        self.bottom_up_layer_1  = RepGElanLayer(in_dim     = cfg.fpn_feats_td["p3"][1] + cfg.fpn_feats_td["p4"][1],
                                                inter_dims = cfg.fpn_feats_bu["p4"][0],
                                                out_dim    = cfg.fpn_feats_bu["p4"][1],
                                                num_blocks = cfg.fpn_depth,
                                                shortcut   = False,
                                                act_type   = cfg.fpn_act,
                                                norm_type  = cfg.fpn_norm,
                                                depthwise  = cfg.fpn_depthwise,
                                                )
        ## P4 -> P5
        self.dowmsample_layer_2 = ADown(cfg.fpn_feats_bu["p4"][1], cfg.fpn_feats_bu["p4"][1],
                                        act_type=cfg.fpn_act, norm_type=cfg.fpn_norm, depthwise=cfg.fpn_depthwise)
        self.bottom_up_layer_2  = RepGElanLayer(in_dim     = cfg.fpn_feats_td["p4"][1] + self.in_dims[0],
                                                inter_dims = cfg.fpn_feats_bu["p5"][0],
                                                out_dim    = cfg.fpn_feats_bu["p5"][1],
                                                num_blocks = cfg.fpn_depth,
                                                shortcut   = False,
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


if __name__=='__main__':
    import time
    from thop import profile
    # Model config
    
    # GElan-Base config
    class GElanBaseConfig(object):
        def __init__(self) -> None:
            # ---------------- Model config ----------------
            self.width    = 0.50
            self.depth    = 0.34
            self.ratio    = 2.0
            self.out_stride = [8, 16, 32]
            self.max_stride = 32
            self.num_levels = 3
            ## FPN
            self.fpn      = 'gelan_pafpn'
            self.fpn_act  = 'silu'
            self.fpn_norm = 'BN'
            self.fpn_depthwise = False
            self.fpn_depth    = 3
            self.fpn_feats_td = {
                "p4": [[256, 128], 256],
                "p3": [[128, 64],  128],
            }
            self.fpn_feats_bu = {
                "p4": [[256, 128], 256],
                "p5": [[256, 128], 256],
            }

    cfg = GElanBaseConfig()
    # Build a head
    in_dims  = [128, 256, 256]
    fpn = GElanPaFPN(cfg, in_dims)

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
