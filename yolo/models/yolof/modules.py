import torch
import torch.nn as nn
from typing import List


# --------------------- Basic modules ---------------------
class ConvModule(nn.Module):
    def __init__(self, 
                 in_dim: int,           # in channels
                 out_dim: int,          # out channels 
                 kernel_size: int = 1,  # kernel size 
                 padding: int = 0,      # padding
                 stride: int = 1,       # padding
                 dilation: int = 1,     # dilation
                 use_act: bool = False,
                ):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.norm = nn.BatchNorm2d(out_dim)
        self.act  = nn.ReLU(inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
