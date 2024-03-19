import torch
import torch.nn as nn
from .yolov7_basic import BasicConv


# Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
class SPPF(nn.Module):
    """
        This code referenced to https://github.com/ultralytics/yolov5
    """
    def __init__(self, cfg, in_dim, out_dim, expansion=0.5):
        super().__init__()
        ## ----------- Basic Parameters -----------
        inter_dim = round(in_dim * expansion)
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

# SPPF block with CSP module
class SPPFBlockCSP(nn.Module):
    """
        CSP Spatial Pyramid Pooling Block
    """
    def __init__(self, cfg, in_dim, out_dim):
        super(SPPFBlockCSP, self).__init__()
        inter_dim = int(in_dim * cfg.neck_expand_ratio)
        self.out_dim = out_dim
        self.cv1 = BasicConv(in_dim, inter_dim, kernel_size=1, act_type=cfg.neck_act, norm_type=cfg.neck_norm)
        self.cv2 = BasicConv(in_dim, inter_dim, kernel_size=1, act_type=cfg.neck_act, norm_type=cfg.neck_norm)
        self.module = nn.Sequential(
            BasicConv(inter_dim, inter_dim, kernel_size=3, padding=1, 
                      act_type=cfg.neck_act, norm_type=cfg.neck_norm, depthwise=cfg.neck_depthwise),
            SPPF(cfg, inter_dim, inter_dim, expansion=1.0),
            BasicConv(inter_dim, inter_dim, kernel_size=3, padding=1, 
                      act_type=cfg.neck_act, norm_type=cfg.neck_norm, depthwise=cfg.neck_depthwise),
                      )
        self.cv3 = BasicConv(inter_dim * 2, self.out_dim, kernel_size=1, act_type=cfg.neck_act, norm_type=cfg.neck_norm)

        
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.module(self.cv2(x))
        y = self.cv3(torch.cat([x1, x2], dim=1))

        return y
