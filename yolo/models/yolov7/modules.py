import numpy as np
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

class ELANBlock(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 expansion: float = 0.5,
                 branch_depth: int = 2,
                 ):
        super(ELANBlock, self).__init__()
        inter_dim = int(in_dim * expansion)
        self.cv1 = ConvModule(in_dim, inter_dim, kernel_size=1)
        self.cv2 = ConvModule(in_dim, inter_dim, kernel_size=1)
        self.cv3 = nn.Sequential(*[ConvModule(inter_dim, inter_dim, kernel_size=3)
                                   for _ in range(round(branch_depth))
                                   ])
        self.cv4 = nn.Sequential(*[ConvModule(inter_dim, inter_dim, kernel_size=3)
                                   for _ in range(round(branch_depth))
                                   ])
        self.out = ConvModule(inter_dim*4, out_dim, kernel_size=1)



    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)
        out = self.out(torch.cat([x1, x2, x3, x4], dim=1))

        return out

class ELANBlockFPN(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 expansion: float = 0.5,
                 branch_width: int = 4,
                 branch_depth: int = 1,
                 ):
        super(ELANBlockFPN, self).__init__()
        # Basic parameters
        inter_dim = int(in_dim * expansion)
        inter_dim2 = int(inter_dim * expansion) 
        # Network structure
        self.cv1 = ConvModule(in_dim, inter_dim, kernel_size=1)
        self.cv2 = ConvModule(in_dim, inter_dim, kernel_size=1)
        self.cv3 = nn.ModuleList()
        for idx in range(round(branch_width)):
            if idx == 0:
                cvs = [ConvModule(inter_dim, inter_dim2, kernel_size=3)]
            else:
                cvs = [ConvModule(inter_dim2, inter_dim2, kernel_size=3)]
            # deeper
            if round(branch_depth) > 1:
                for _ in range(1, round(branch_depth)):
                    cvs.append(ConvModule(inter_dim2, inter_dim2, kernel_size=3))
                self.cv3.append(nn.Sequential(*cvs))
            else:
                self.cv3.append(cvs[0])

        self.out = ConvModule(inter_dim*2 + inter_dim2*len(self.cv3), out_dim, kernel_size=1)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        inter_outs = [x1, x2]
        for m in self.cv3:
            y1 = inter_outs[-1]
            y2 = m(y1)
            inter_outs.append(y2)
        out = self.out(torch.cat(inter_outs, dim=1))

        return out

class DownSample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        inter_dim = out_dim // 2
        self.mp = nn.MaxPool2d((2, 2), 2)
        self.cv1 = ConvModule(in_dim, inter_dim, kernel_size=1)
        self.cv2 = nn.Sequential(
            ConvModule(in_dim, inter_dim, kernel_size=1),
            ConvModule(inter_dim, inter_dim, kernel_size=3, stride=2)
        )

    def forward(self, x):
        x1 = self.cv1(self.mp(x))
        x2 = self.cv2(x)
        out = torch.cat([x1, x2], dim=1)

        return out
