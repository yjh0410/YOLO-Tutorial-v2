import torch
import torch.nn as nn

from .gelan_basic import BasicConv


# SPPF (from yolov5)
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
        self.cv1 = BasicConv(in_dim, inter_dim,
                             kernel_size=1, padding=0, stride=1,
                             act_type=cfg.neck_act, norm_type=cfg.neck_norm)
        self.cv2 = BasicConv(inter_dim * 4, out_dim,
                             kernel_size=1, padding=0, stride=1,
                             act_type=cfg.neck_act, norm_type=cfg.neck_norm)
        self.m = nn.MaxPool2d(kernel_size=cfg.spp_pooling_size,
                              stride=1,
                              padding=cfg.spp_pooling_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)

        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

# SPP-ELAN (from yolov9)
class SPPElan(nn.Module):
    def __init__(self, cfg, in_dim):
        """SPPElan looks like the SPPF."""
        super().__init__()
        ## ----------- Basic Parameters -----------
        self.in_dim = in_dim
        self.inter_dim = cfg.spp_inter_dim
        self.out_dim   = cfg.spp_out_dim
        ## ----------- Network Parameters -----------
        self.conv_layer_1 = BasicConv(in_dim, self.inter_dim, kernel_size=1, act_type=cfg.neck_act, norm_type=cfg.neck_norm)
        self.conv_layer_2 = BasicConv(self.inter_dim * 4, self.out_dim, kernel_size=1, act_type=cfg.neck_act, norm_type=cfg.neck_norm)
        self.pool_layer   = nn.MaxPool2d(kernel_size=cfg.spp_pooling_size, stride=1, padding=cfg.spp_pooling_size // 2)

    def forward(self, x):
        y = [self.conv_layer_1(x)]
        y.extend(self.pool_layer(y[-1]) for _ in range(3))
        
        return self.conv_layer_2(torch.cat(y, 1))
    