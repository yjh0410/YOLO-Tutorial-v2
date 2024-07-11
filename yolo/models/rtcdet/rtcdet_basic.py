import torch
import torch.nn as nn
from typing import List


# --------------------- Basic modules ---------------------
def get_conv2d(c1, c2, k, p, s, d=1, g=1, bias=False):
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
    if norm_type == 'bn':
        return nn.BatchNorm2d(dim)
    elif norm_type == 'gn':
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
                 groups=1,                 # group
                 act_type  :str = 'lrelu', # activation
                 norm_type :str = 'bn',    # normalization
                 depthwise :bool = False
                ):
        super(BasicConv, self).__init__()
        self.depthwise = depthwise
        use_bias = False if norm_type is not None else True
        if not depthwise:
            self.conv = get_conv2d(in_dim, out_dim, k=kernel_size, p=padding, s=stride, d=dilation, g=groups, bias=use_bias)
            self.norm = get_norm(norm_type, out_dim)
        else:
            self.conv1 = get_conv2d(in_dim, in_dim, k=kernel_size, p=padding, s=stride, d=dilation, g=in_dim, bias=use_bias)
            self.norm1 = get_norm(norm_type, in_dim)
            self.conv2 = get_conv2d(in_dim, out_dim, k=1, p=0, s=1, d=1, g=1, bias=use_bias)
            self.norm2 = get_norm(norm_type, out_dim)
        self.act  = get_activation(act_type)

    def forward(self, x):
        if not self.depthwise:
            return self.act(self.norm(self.conv(x)))
        else:
            # Depthwise conv
            x = self.act(self.norm1(self.conv1(x)))
            # Pointwise conv
            x = self.act(self.norm2(self.conv2(x)))
            return x

class DWConv(nn.Module):
    def __init__(self, 
                 in_dim      :int,           # in channels
                 out_dim     :int,           # out channels 
                 kernel_size :int = 1,       # kernel size 
                 padding     :int = 0,       # padding
                 stride      :int = 1,       # padding
                 dilation    :int = 1,       # dilation
                 act_type    :str = 'lrelu', # activation
                 norm_type   :str = 'BN',    # normalization
                ):
        super(DWConv, self).__init__()
        assert in_dim == out_dim
        use_bias = False if norm_type is not None else True
        self.conv = get_conv2d(in_dim, out_dim, k=kernel_size, p=padding, s=stride, d=dilation, g=out_dim, bias=use_bias)
        self.norm = get_norm(norm_type, out_dim)
        self.act  = get_activation(act_type)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# --------------------- Downsample modules ---------------------
class ADown(nn.Module):
    def __init__(self,
                 in_dim    :int,
                 out_dim   :int,
                 act_type  :str  = "silu",
                 norm_type :str  = "bn",
                 depthwise :bool = False):
        super().__init__()
        inter_dim = out_dim // 2
        self.conv_layer_1 = BasicConv(in_dim // 2, inter_dim, kernel_size=3, padding=1, stride=2,
                                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.conv_layer_2 = BasicConv(in_dim // 2, inter_dim, kernel_size=1,
                                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
    def forward(self, x):
        # Split
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1,x2 = x.chunk(2, 1)

        # Downsample branch - 1
        x1 = self.conv_layer_1(x1)

        # Downsample branch - 2
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.conv_layer_2(x2)

        return torch.cat([x1, x2], dim=1)

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

    def forward(self, x):
        x1 = self.downsample_1(x)
        x2 = self.downsample_2(x)

        return torch.cat([x1, x2], dim=1)


# --------------------- Feature processing modules ---------------------
class MBottleneck(nn.Module):
    def __init__(self,
                 in_dim    :int,
                 out_dim   :int,
                 expansion :float = 0.5,
                 shortcut  :bool  = False,
                 act_type  :str   = 'silu',
                 norm_type :str   = 'bn',
                 depthwise :bool  = False,
                 ) -> None:
        super(MBottleneck, self).__init__()
        inter_dim = int(out_dim * expansion)
        # ----------------- Network setting -----------------
        self.conv_layer = nn.Sequential(
            # 3x3 conv + bn + silu
            BasicConv(in_dim, inter_dim, kernel_size=3, padding=1, stride=1,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            # 5x5 dw conv
            DWConv(inter_dim, inter_dim, kernel_size=5, padding=2, stride=1,
                   act_type=None, norm_type=norm_type),
            # 3x3 conv + bn + silu
            BasicConv(inter_dim, out_dim, kernel_size=3, padding=1, stride=1,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise),
        )
        self.shortcut = shortcut and in_dim == out_dim

    def forward(self, x):
        h = self.conv_layer(x)

        return x + h if self.shortcut else h

class CSPLayer(nn.Module):
    # CSP Bottleneck
    def __init__(self,
                 in_dim      :int,
                 out_dim     :int,
                 num_blocks  :int   = 1,
                 expansion   :float = 0.5,
                 shortcut    :bool  = True,
                 act_type    :str   = 'silu',
                 norm_type   :str   = 'bn',
                 depthwise   :bool  = False,
                 ) -> None:
        super().__init__()
        inter_dim = round(out_dim * expansion)
        self.input_proj = BasicConv(in_dim, out_dim, kernel_size=1, act_type=None, norm_type=norm_type, depthwise=depthwise)
        self.module = nn.Sequential(*[MBottleneck(inter_dim,
                                                  inter_dim,
                                                  expansion   = 1.0,
                                                  shortcut    = shortcut,
                                                  act_type    = act_type,
                                                  norm_type   = norm_type,
                                                  depthwise   = depthwise,
                                                  ) for _ in range(num_blocks)])

    def forward(self, x):
        # Split
        x1, x2 = torch.chunk(self.input_proj(x), chunks=2, dim=1)

        # Branch
        x2 = self.module(x2)

        # Output proj
        out = torch.cat([x1, x2], dim=1)

        return out

class ElanLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 expansion  :float = 0.5,
                 num_blocks :int   = 1,
                 shortcut   :bool  = False,
                 act_type   :str   = 'silu',
                 norm_type  :str   = 'bn',
                 depthwise  :bool  = False,
                 ) -> None:
        super(ElanLayer, self).__init__()
        inter_dim = round(out_dim * expansion)
        self.input_proj  = BasicConv(in_dim, inter_dim * 2, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.output_proj = BasicConv((2 + num_blocks) * inter_dim, out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.module      = nn.ModuleList([MBottleneck(inter_dim,
                                                      inter_dim,
                                                      expansion   = 1.0,
                                                      shortcut    = shortcut,
                                                      act_type    = act_type,
                                                      norm_type   = norm_type,
                                                      depthwise   = depthwise)
                                                      for _ in range(num_blocks)])

    def forward(self, x):
        # Input proj
        x1, x2 = torch.chunk(self.input_proj(x), 2, dim=1)
        out = list([x1, x2])

        # Bottleneck
        out.extend(m(out[-1]) for m in self.module)

        # Output proj
        out = self.output_proj(torch.cat(out, dim=1))

        return out
    
class GElanLayer(nn.Module):
    """Modified YOLOv9's GELAN module"""
    def __init__(self,
                 in_dim     :int,
                 inter_dims :List,
                 out_dim    :int,
                 num_blocks :int   = 1,
                 shortcut   :bool  = False,
                 act_type   :str   = 'silu',
                 norm_type  :str   = 'bn',
                 depthwise  :bool  = False,
                 ) -> None:
        super(GElanLayer, self).__init__()
        # ----------- Basic parameters -----------
        self.in_dim = in_dim
        self.inter_dims = inter_dims
        self.out_dim = out_dim

        # ----------- Network parameters -----------
        self.conv_layer_1  = BasicConv(in_dim, inter_dims[0], kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.elan_module_1 = nn.Sequential(
             CSPLayer(inter_dims[0]//2,
                      inter_dims[1],
                      num_blocks  = num_blocks,
                      shortcut    = shortcut,
                      expansion   = 0.5,
                      act_type    = act_type,
                      norm_type   = norm_type,
                      depthwise   = depthwise),
            BasicConv(inter_dims[1], inter_dims[1], kernel_size=3, padding=1,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        self.elan_module_2 = nn.Sequential(
             CSPLayer(inter_dims[1],
                      inter_dims[1],
                      num_blocks  = num_blocks,
                      shortcut    = shortcut,
                      expansion   = 0.5,
                      act_type    = act_type,
                      norm_type   = norm_type,
                      depthwise   = depthwise),
            BasicConv(inter_dims[1], inter_dims[1], kernel_size=3, padding=1,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        self.conv_layer_2 = BasicConv(inter_dims[0] + 2*self.inter_dims[1], out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)

    def forward(self, x):
        # Input proj
        x1, x2 = torch.chunk(self.conv_layer_1(x), 2, dim=1)
        out = list([x1, x2])

        # ELAN module
        out.append(self.elan_module_1(out[-1]))
        out.append(self.elan_module_2(out[-1]))

        # Output proj
        out = self.conv_layer_2(torch.cat(out, dim=1))

        return out
