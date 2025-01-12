import torch
import torch.nn as nn

try:
    from .modules import ConvModule
except:
    from  modules import ConvModule
    

# Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
class SPPF(nn.Module):
    """
        This code referenced to https://github.com/ultralytics/yolov5
    """
    def __init__(self, cfg, in_dim, out_dim):
        super().__init__()
        ## ----------- Basic Parameters -----------
        inter_dim = in_dim // 2
        self.out_dim = out_dim
        ## ----------- Network Parameters -----------
        self.cv1 = ConvModule(in_dim, inter_dim, kernel_size=1, padding=0, stride=1, act_type="silu")
        self.cv2 = ConvModule(inter_dim * 4, out_dim, kernel_size=1, padding=0, stride=1, act_type="silu")
        self.m = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)

        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
