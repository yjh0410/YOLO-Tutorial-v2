import torch
import torch.nn as nn


# ----------------- Basic CNN Ops -----------------
def get_conv2d(c1, c2, k, p, s, g, bias=False):
    conv = nn.Conv2d(c1, c2, k, stride=s, padding=p, groups=g, bias=bias)

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
    elif act_type == 'gelu':
        return nn.GELU()
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

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# ----------------- CNN Modules -----------------
class BasicConv(nn.Module):
    def __init__(self, 
                 in_dim,                   # in channels
                 out_dim,                  # out channels 
                 kernel_size=1,            # kernel size 
                 padding=0,                # padding
                 stride=1,                 # padding
                 act_type  :str = 'lrelu', # activation
                 norm_type :str = 'BN',    # normalization
                 depthwise :bool = False
                ):
        super(BasicConv, self).__init__()
        add_bias = False if norm_type else True
        self.depthwise = depthwise
        if not depthwise:
            self.conv = get_conv2d(in_dim, out_dim, k=kernel_size, p=padding, s=stride, g=1, bias=add_bias)
            self.norm = get_norm(norm_type, out_dim)
        else:
            self.conv1 = get_conv2d(in_dim, in_dim, k=kernel_size, p=padding, s=stride, g=1, bias=add_bias)
            self.norm1 = get_norm(norm_type, in_dim)
            self.conv2 = get_conv2d(in_dim, out_dim, k=kernel_size, p=padding, s=stride, g=1, bias=add_bias)
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

class Bottleneck(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio = 0.5,
                 kernel_sizes = [3, 3],
                 shortcut     = True,
                 act_type     = 'silu',
                 norm_type    = 'BN',
                 depthwise    = False,):
        super(Bottleneck, self).__init__()
        inter_dim = int(out_dim * expand_ratio)
        paddings = [k // 2 for k in kernel_sizes]
        self.cv1 = BasicConv(in_dim, inter_dim,
                             kernel_size=kernel_sizes[0], padding=paddings[0],
                             act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.cv2 = BasicConv(inter_dim, out_dim,
                             kernel_size=kernel_sizes[1], padding=paddings[1],
                             act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.shortcut = shortcut and in_dim == out_dim

    def forward(self, x):
        h = self.cv2(self.cv1(x))

        return x + h if self.shortcut else h

class ELANLayer(nn.Module):
    def __init__(self,
                 in_dim       :int,
                 out_dim      :int,
                 num_blocks   :int   = 1,
                 expand_ratio :float = 0.5,
                 shortcut     :bool  = False,
                 act_type     :str   = 'silu',
                 norm_type    :str   = 'BN',
                 depthwise    :bool  = False,):
        super(ELANLayer, self).__init__()
        self.inter_dim = round(out_dim * expand_ratio)
        self.conv1 = BasicConv(in_dim, self.inter_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.conv2 = BasicConv(in_dim, self.inter_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.cmodules = nn.ModuleList([Bottleneck(self.inter_dim, self.inter_dim,
                                                   1.0, [3, 3], shortcut,
                                                   act_type, norm_type, depthwise)
                                                   for _ in range(num_blocks)])
        self.conv3 = BasicConv(self.inter_dim * (2 + num_blocks), out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)

    def forward(self, x):
        x1, x2 = self.conv1(x), self.conv2(x)
        out = [x1, x2]
        for m in self.cmodules:
            x2 = m(x2)
            out.append(x2)

        return self.conv3(torch.cat(out, dim=1))
    