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


# ---------------------------- Basic Modules ----------------------------
class MDown(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, ):
        super().__init__()
        inter_dim = out_dim // 2
        self.downsample_1 = nn.Sequential(
            nn.MaxPool2d((2, 2), stride=2),
            ConvModule(in_dim, inter_dim, kernel_size=1)
        )
        self.downsample_2 = nn.Sequential(
            ConvModule(in_dim, inter_dim, kernel_size=1),
            ConvModule(inter_dim, inter_dim, kernel_size=3, padding=1, stride=2)
        )
        if in_dim == out_dim:
            self.output_proj = nn.Identity()
        else:
            self.output_proj = ConvModule(inter_dim * 2, out_dim, kernel_size=1)

    def forward(self, x):
        x1 = self.downsample_1(x)
        x2 = self.downsample_2(x)

        out = self.output_proj(torch.cat([x1, x2], dim=1))

        return out

class ELANLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 expansion  :float = 0.5,
                 num_blocks :int   = 1,
                 ) -> None:
        super(ELANLayer, self).__init__()
        self.inter_dim = round(in_dim * expansion)
        self.conv_layer_1 = ConvModule(in_dim, self.inter_dim, kernel_size=1)
        self.conv_layer_2 = ConvModule(in_dim, self.inter_dim, kernel_size=1)
        self.conv_layer_3 = ConvModule(self.inter_dim * 4, out_dim, kernel_size=1)
        self.elan_layer_1 = nn.Sequential(*[ConvModule(self.inter_dim, self.inter_dim, kernel_size=3, padding=1)
                                           for _ in range(num_blocks)])
        self.elan_layer_2 = nn.Sequential(*[ConvModule(self.inter_dim, self.inter_dim, kernel_size=3, padding=1)
                                           for _ in range(num_blocks)])

    def forward(self, x):
        # Input proj
        x1 = self.conv_layer_1(x)
        x2 = self.conv_layer_2(x)
        x3 = self.elan_layer_1(x2)
        x4 = self.elan_layer_2(x3)
    
        out = self.conv_layer_3(torch.cat([x1, x2, x3, x4], dim=1))

        return out

class ELANLayerFPN(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 expansions   :List = [0.5, 0.5],
                 branch_width :int  = 4,
                 branch_depth :int  = 1,
                 ):
        super(ELANLayerFPN, self).__init__()
        # Basic parameters
        inter_dim  = round(in_dim * expansions[0])
        inter_dim2 = round(inter_dim * expansions[1]) 
        # Network structure
        self.cv1 = ConvModule(in_dim, inter_dim, kernel_size=1)
        self.cv2 = ConvModule(in_dim, inter_dim, kernel_size=1)
        self.cv3 = nn.ModuleList()
        for idx in range(round(branch_width)):
            if idx == 0:
                cvs = [ConvModule(inter_dim, inter_dim2, kernel_size=3, padding=1)]
            else:
                cvs = [ConvModule(inter_dim2, inter_dim2, kernel_size=3, padding=1)]
            # deeper
            if round(branch_depth) > 1:
                for _ in range(1, round(branch_depth)):
                    cvs.append(ConvModule(inter_dim2, inter_dim2, kernel_size=3, padding=1))
                self.cv3.append(nn.Sequential(*cvs))
            else:
                self.cv3.append(cvs[0])

        self.output_proj = ConvModule(inter_dim*2+inter_dim2*len(self.cv3), out_dim, kernel_size=1)


    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        inter_outs = [x1, x2]
        for m in self.cv3:
            y1 = inter_outs[-1]
            y2 = m(y1)
            inter_outs.append(y2)
        out = self.output_proj(torch.cat(inter_outs, dim=1))

        return out
