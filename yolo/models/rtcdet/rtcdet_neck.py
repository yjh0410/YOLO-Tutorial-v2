import torch
import torch.nn as nn

from .rtcdet_basic import BasicConv


# -------------- Neck network --------------
class SPPF(nn.Module):
    """
        This code referenced to https://github.com/ultralytics/yolov5
    """
    def __init__(self, cfg, in_dim, out_dim):
        super().__init__()
        ## ----------- Basic Parameters -----------
        inter_dim = round(in_dim * cfg.neck_expand_ratio)
        self.out_dim = out_dim
        ## ----------- Network Parameters -----------
        self.input_proj  = BasicConv(in_dim, inter_dim, kernel_size=1,
                                     act_type=cfg.neck_act, norm_type=cfg.neck_norm)
        self.output_proj = BasicConv(inter_dim * 4, out_dim, kernel_size=1,
                                     act_type=cfg.neck_act, norm_type=cfg.neck_norm)
        self.module = nn.MaxPool2d(cfg.spp_pooling_size, stride=1, padding=cfg.spp_pooling_size//2)

        # Initialize all layers
        self.init_weights()
                
    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.reset_parameters()

    def forward(self, x):
        x = self.input_proj(x)
        y1 = self.module(x)
        y2 = self.module(y1)

        return self.output_proj(torch.cat((x, y1, y2, self.module(y2)), 1))
    