import torch
import torch.nn as nn
from   typing import List

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
        if not depthwise:
            self.conv = get_conv2d(in_dim, out_dim, k=kernel_size, p=padding, s=stride, d=dilation, g=1)
            self.norm = get_norm(norm_type, out_dim)
        else:
            self.conv1 = get_conv2d(in_dim, in_dim, k=kernel_size, p=padding, s=stride, d=dilation, g=in_dim)
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
            x = self.act(self.norm2(self.conv2(x)))
            return x


# ---------------------------- Basic Modules ----------------------------
class YoloBottleneck(nn.Module):
    def __init__(self,
                 in_dim       :int,
                 out_dim      :int,
                 kernel_size  :List  = [1, 3],
                 expand_ratio :float = 0.5,
                 shortcut     :bool  = False,
                 act_type     :str   = 'silu',
                 norm_type    :str   = 'BN',
                 depthwise    :bool  = False,
                 ) -> None:
        super(YoloBottleneck, self).__init__()
        inter_dim = int(out_dim * expand_ratio)
        # ----------------- Network setting -----------------
        self.conv_layer1 = BasicConv(in_dim, inter_dim,
                                     kernel_size=kernel_size[0], padding=kernel_size[0]//2, stride=1,
                                     act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.conv_layer2 = BasicConv(inter_dim, out_dim,
                                     kernel_size=kernel_size[1], padding=kernel_size[1]//2, stride=1,
                                     act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.shortcut = shortcut and in_dim == out_dim

    def forward(self, x):
        h = self.conv_layer2(self.conv_layer1(x))

        return x + h if self.shortcut else h

class CSPBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_blocks   :int   = 1,
                 expand_ratio :float = 0.5,
                 shortcut     :bool  = False,
                 act_type     :str   = 'silu',
                 norm_type    :str   = 'BN',
                 depthwise    :bool  = False,
                 ):
        super(CSPBlock, self).__init__()
        # ---------- Basic parameters ----------
        self.num_blocks = num_blocks
        self.expand_ratio = expand_ratio
        self.shortcut = shortcut
        inter_dim = round(out_dim * expand_ratio)
        # ---------- Model parameters ----------
        self.conv_layer_1 = BasicConv(in_dim, inter_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.conv_layer_2 = BasicConv(in_dim, inter_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.conv_layer_3 = BasicConv(inter_dim * 2, out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.module       = nn.Sequential(*[YoloBottleneck(inter_dim,
                                                           inter_dim,
                                                           kernel_size  = [1, 3],
                                                           expand_ratio = 1.0,
                                                           shortcut     = shortcut,
                                                           act_type     = act_type,
                                                           norm_type    = norm_type,
                                                           depthwise    = depthwise)
                                                           for _ in range(num_blocks)
                                                           ])

    def forward(self, x):
        x1 = self.conv_layer_1(x)
        x2 = self.module(self.conv_layer_2(x))
        out = self.conv_layer_3(torch.cat([x1, x2], dim=1))

        return out
    