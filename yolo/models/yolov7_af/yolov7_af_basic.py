import torch
import torch.nn as nn
from typing import List


# --------------------- Basic modules ---------------------
def get_conv2d(c1, c2, k, p, s, d, g, bias=False):
    conv = nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=bias)

    return conv

def get_activation(act_type=None):
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_type == 'mish':
        return nn.Mish(inplace=True)
    elif act_type == 'silu':
        return nn.SiLU(inplace=True)
    elif act_type is None:
        return nn.Identity()
    else:
        raise NotImplementedError

def get_norm(norm_type, dim):
    if norm_type == 'BN':
        return nn.BatchNorm2d(dim)
    elif norm_type == 'GN':
        return nn.GroupNorm(num_groups=32, num_channels=dim)
    elif norm_type is None:
        return nn.Identity()
    else:
        raise NotImplementedError

class BasicConv(nn.Module):
    def __init__(self, 
                 in_dim,                   # in channels
                 out_dim,                  # out channels 
                 kernel_size=1,            # kernel size 
                 padding=0,                # padding
                 stride=1,                 # padding
                 dilation=1,               # dilation
                 act_type  :str = 'lrelu', # activation
                 norm_type :str = 'BN',    # normalization
                 depthwise :bool = False
                ):
        super(BasicConv, self).__init__()
        self.depthwise = depthwise
        use_bias = False if norm_type is not None else True
        if not depthwise:
            self.conv = get_conv2d(in_dim, out_dim, k=kernel_size, p=padding, s=stride, d=dilation, g=1, bias=use_bias)
            self.norm = get_norm(norm_type, out_dim)
        else:
            self.conv1 = get_conv2d(in_dim, in_dim, k=kernel_size, p=padding, s=stride, d=dilation, g=in_dim, bias=use_bias)
            self.norm1 = get_norm(norm_type, in_dim)
            self.conv2 = get_conv2d(in_dim, out_dim, k=1, p=0, s=1, d=1, g=1)
            self.norm2 = get_norm(norm_type, out_dim)
        self.act  = get_activation(act_type)

    def forward(self, x):
        if not self.depthwise:
            return self.act(self.norm(self.conv(x)))
        else:
            # Depthwise conv
            x = self.norm1(self.conv1(x))
            # Pointwise conv
            x = self.norm2(self.conv2(x))
            return x


# ---------------------------- Basic Modules ----------------------------
class MDown(nn.Module):
    def __init__(self,
                 in_dim    :int,
                 out_dim   :int,
                 act_type  :str   = 'silu',
                 norm_type :str   = 'BN',
                 depthwise :bool  = False,
                 ) -> None:
        super().__init__()
        inter_dim = out_dim // 2
        self.downsample_1 = nn.Sequential(
            nn.MaxPool2d((2, 2), stride=2),
            BasicConv(in_dim, inter_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        )
        self.downsample_2 = nn.Sequential(
            BasicConv(in_dim, inter_dim, kernel_size=1, act_type=act_type, norm_type=norm_type),
            BasicConv(inter_dim, inter_dim,
                      kernel_size=3, padding=1, stride=2,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        if in_dim == out_dim:
            self.output_proj = nn.Identity()
        else:
            self.output_proj = BasicConv(inter_dim * 2, out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)

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
                 act_type   :str   = 'silu',
                 norm_type  :str   = 'BN',
                 depthwise  :bool  = False,
                 ) -> None:
        super(ELANLayer, self).__init__()
        self.inter_dim = round(in_dim * expansion)
        self.conv_layer_1 = BasicConv(in_dim, self.inter_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.conv_layer_2 = BasicConv(in_dim, self.inter_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.conv_layer_3 = BasicConv(self.inter_dim * 4, out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.elan_layer_1 = nn.Sequential(*[BasicConv(self.inter_dim, self.inter_dim,
                                                      kernel_size=3, padding=1,
                                                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
                                                      for _ in range(num_blocks)])
        self.elan_layer_2 = nn.Sequential(*[BasicConv(self.inter_dim, self.inter_dim,
                                                      kernel_size=3, padding=1,
                                                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
                                                      for _ in range(num_blocks)])

    def forward(self, x):
        # Input proj
        x1 = self.conv_layer_1(x)
        x2 = self.conv_layer_2(x)
        x3 = self.elan_layer_1(x2)
        x4 = self.elan_layer_2(x3)
    
        out = self.conv_layer_3(torch.cat([x1, x2, x3, x4], dim=1))

        return out

## PaFPN's ELAN-Block proposed by YOLOv7
class ELANLayerFPN(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 expansions   :List = [0.5, 0.5],
                 branch_width :int  = 4,
                 branch_depth :int  = 1,
                 act_type     :str  = 'silu',
                 norm_type    :str  = 'BN',
                 depthwise=False):
        super(ELANLayerFPN, self).__init__()
        # Basic parameters
        inter_dim  = round(in_dim * expansions[0])
        inter_dim2 = round(inter_dim * expansions[1]) 
        # Network structure
        self.cv1 = BasicConv(in_dim, inter_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = BasicConv(in_dim, inter_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.cv3 = nn.ModuleList()
        for idx in range(round(branch_width)):
            if idx == 0:
                cvs = [BasicConv(inter_dim, inter_dim2,
                                 kernel_size=3, padding=1,
                                 act_type=act_type, norm_type=norm_type, depthwise=depthwise)]
            else:
                cvs = [BasicConv(inter_dim2, inter_dim2,
                                 kernel_size=3, padding=1,
                                 act_type=act_type, norm_type=norm_type, depthwise=depthwise)]
            # deeper
            if round(branch_depth) > 1:
                for _ in range(1, round(branch_depth)):
                    cvs.append(BasicConv(inter_dim2, inter_dim2, kernel_size=3, padding=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise))
                self.cv3.append(nn.Sequential(*cvs))
            else:
                self.cv3.append(cvs[0])

        self.output_proj = BasicConv(inter_dim*2+inter_dim2*len(self.cv3), out_dim,
                                     kernel_size=1, act_type=act_type, norm_type=norm_type)


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
