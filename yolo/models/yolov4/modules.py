import torch
import torch.nn as nn


# --------------------- Basic modules ---------------------
class ConvModule(nn.Module):
    def __init__(self, 
                 in_dim: int,          # in channels
                 out_dim: int,         # out channels 
                 kernel_size: int = 1, # kernel size 
                 stride:int = 1,       # padding
                 ):
        super(ConvModule, self).__init__()
        convs = []
        convs.append(nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=kernel_size//2, stride=stride, bias=False))
        convs.append(nn.BatchNorm2d(out_dim))
        convs.append(nn.SiLU(inplace=True))
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        return self.convs(x)

class Bottleneck(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 expand_ratio: float = 0.5,
                 shortcut: bool = False,
                 ):
        super(Bottleneck, self).__init__()
        inter_dim = int(out_dim * expand_ratio)  # hidden channels            
        self.cv1 = ConvModule(in_dim, inter_dim, kernel_size=1)
        self.cv2 = ConvModule(inter_dim, out_dim, kernel_size=3, stride=1)
        self.shortcut = shortcut and in_dim == out_dim

    def forward(self, x):
        h = self.cv2(self.cv1(x))

        return x + h if self.shortcut else h

class CSPBlock(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 expand_ratio: float = 0.5,
                 num_blocks: int = 1,
                 shortcut: bool = False,
                 ):
        super(CSPBlock, self).__init__()
        inter_dim = int(out_dim * expand_ratio)
        self.cv1 = ConvModule(in_dim, inter_dim, kernel_size=1)
        self.cv2 = ConvModule(in_dim, inter_dim, kernel_size=1)
        self.cv3 = ConvModule(2 * inter_dim, out_dim, kernel_size=1)
        self.m = nn.Sequential(*[
            Bottleneck(inter_dim, inter_dim, expand_ratio=1.0, shortcut=shortcut)
                       for _ in range(num_blocks)
                       ])

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m(x1)
        out = self.cv3(torch.cat([x3, x2], dim=1))

        return out
    