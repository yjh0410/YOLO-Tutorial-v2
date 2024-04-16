import torch
import torch.nn as nn

try:
    from .yolof_basic import BasicConv
except:
    from  yolof_basic import BasicConv


class YolofUpsampler(nn.Module):
    def __init__(self, cfg, in_dims, out_dim):
        super(YolofUpsampler, self).__init__()
        # ----------- Model parameters -----------
        self.input_proj_1 = BasicConv(in_dims[-1], out_dim, kernel_size=1, act_type=cfg.neck_act, norm_type=cfg.neck_norm)
        self.input_proj_2 = BasicConv(in_dims[-2], out_dim, kernel_size=1, act_type=cfg.neck_act, norm_type=cfg.neck_norm)
        self.output_proj  = nn.Sequential(
            BasicConv(out_dim * 2, out_dim, kernel_size=1, act_type=cfg.neck_act, norm_type=cfg.neck_norm),
            BasicConv(out_dim, out_dim, kernel_size=3, padding=1, stride=1, act_type=cfg.neck_act, norm_type=cfg.neck_norm),
        )

    def forward(self, pyramid_feats):
        x1 = self.input_proj_1(pyramid_feats[-1])
        x2 = self.input_proj_2(pyramid_feats[-2])
        
        x1_up = nn.functional.interpolate(x1, scale_factor=2.0)

        x3 = torch.cat([x2, x1_up], dim=1)
        out = self.output_proj(x3)
        
        return out
