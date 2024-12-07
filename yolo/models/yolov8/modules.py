import torch
import torch.nn as nn
from typing import List


# --------------------- Basic modules ---------------------
class ConvModule(nn.Module):
    def __init__(self, 
                 in_dim,        # in channels
                 out_dim,       # out channels 
                 kernel_size=1, # kernel size 
                 padding=0,     # padding
                 stride=1,      # padding
                 dilation=1,    # dilation
                ):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.norm = nn.BatchNorm2d(out_dim)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class YoloBottleneck(nn.Module):
    def __init__(self,
                 in_dim      :int,
                 out_dim     :int,
                 kernel_size :List  = [1, 3],
                 expansion   :float = 0.5,
                 shortcut    :bool  = False,
                 ):
        super(YoloBottleneck, self).__init__()
        inter_dim = int(out_dim * expansion)
        # ----------------- Network setting -----------------
        self.conv_layer1 = ConvModule(in_dim, inter_dim, kernel_size=kernel_size[0], padding=kernel_size[0]//2, stride=1)
        self.conv_layer2 = ConvModule(inter_dim, out_dim, kernel_size=kernel_size[1], padding=kernel_size[1]//2, stride=1)
        self.shortcut = shortcut and in_dim == out_dim

    def forward(self, x):
        h = self.conv_layer2(self.conv_layer1(x))

        return x + h if self.shortcut else h

class C2fBlock(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 expansion : float = 0.5,
                 num_blocks : int = 1,
                 shortcut  : bool = False,
                 ):
        super(C2fBlock, self).__init__()
        inter_dim = round(out_dim * expansion)
        self.input_proj  = ConvModule(in_dim, inter_dim * 2, kernel_size=1)
        self.output_proj = ConvModule((2 + num_blocks) * inter_dim, out_dim, kernel_size=1)
        self.module = nn.ModuleList([
            YoloBottleneck(in_dim = inter_dim,
                           out_dim = inter_dim,
                           kernel_size = [3, 3],
                           expansion = 1.0,
                           shortcut = shortcut,
                           ) for _ in range(num_blocks)])

    def forward(self, x):
        # Input proj
        x1, x2 = torch.chunk(self.input_proj(x), 2, dim=1)
        out = list([x1, x2])

        # Bottlenecl
        out.extend(m(out[-1]) for m in self.module)

        # Output proj
        out = self.output_proj(torch.cat(out, dim=1))

        return out
