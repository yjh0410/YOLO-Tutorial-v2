import torch
import torch.nn as nn

try:
    from .yolof_basic import BasicConv
except:
    from  yolof_basic import BasicConv


class YolofUpsampler(nn.Module):
    def __init__(self, cfg, in_dim, out_dim):
        super(YolofUpsampler, self).__init__()
        # ----------- Basic parameters -----------
        self.upscale_factor = cfg.upscale_factor
        inter_dim = self.upscale_factor ** 2 * in_dim
        # ----------- Model parameters -----------
        self.input_proj = BasicConv(in_dim, inter_dim, kernel_size=1, act_type=cfg.neck_act, norm_type=cfg.neck_norm)
        self.output_proj = BasicConv(in_dim, out_dim, kernel_size=1, act_type=cfg.neck_act, norm_type=cfg.neck_norm)

    def forward(self, x):
        # [B, C, H, W] -> [B, 4*C, H, W]
        x = self.input_proj(x)

        # [B, 4*C, H, W] -> [B, C, 2*H, 2*W]
        x = torch.pixel_shuffle(x, upscale_factor=self.upscale_factor)
        
        x = self.output_proj(x)
        
        return x
