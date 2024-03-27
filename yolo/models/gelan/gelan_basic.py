import numpy as np
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
                 group=1,                  # group
                 act_type  :str = 'lrelu', # activation
                 norm_type :str = 'BN',    # normalization
                 depthwise :bool = False
                ):
        super(BasicConv, self).__init__()
        self.depthwise = depthwise
        if not depthwise:
            self.conv = get_conv2d(in_dim, out_dim, k=kernel_size, p=padding, s=stride, d=dilation, g=group)
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
            x = self.norm2(self.conv2(x))
            return x


# --------------------- GELAN modules (from yolov9) ---------------------
class ADown(nn.Module):
    def __init__(self, in_dim, out_dim, act_type="silu", norm_type="BN", depthwise=False, use_pooling=True):
        super().__init__()
        inter_dim = out_dim // 2
        self.use_pooling = use_pooling
        if use_pooling:
            self.conv_layer_1 = BasicConv(in_dim // 2, inter_dim,
                                        kernel_size=3, padding=1, stride=2,
                                        act_type=act_type, norm_type=norm_type, depthwise=depthwise)
            self.conv_layer_2 = BasicConv(in_dim // 2, inter_dim, kernel_size=1,
                                        act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        else:
            self.conv_layer = BasicConv(in_dim, out_dim, kernel_size=3, padding=1, stride=2,
                                        act_type=act_type, norm_type=norm_type, depthwise=depthwise)
    def forward(self, x):
        if self.use_pooling:
            x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
            x1,x2 = x.chunk(2, 1)
            x1 = self.conv_layer_1(x1)
            x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
            x2 = self.conv_layer_2(x2)

            return torch.cat((x1, x2), 1)
        else:
            return self.conv_layer(x)

class RepConvN(nn.Module):
    """RepConv is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    def __init__(self, in_dim, out_dim, k=3, s=1, p=1, g=1, act_type='silu', norm_type='BN', depthwise=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.act = get_activation(act_type)

        self.bn = None
        self.conv1 = BasicConv(in_dim, out_dim,
                               kernel_size=k, padding=p, stride=s, group=g,
                               act_type=None, norm_type=norm_type, depthwise=depthwise)
        self.conv2 = BasicConv(in_dim, out_dim,
                               kernel_size=1, padding=(p - k // 2), stride=s, group=g,
                               act_type=None, norm_type=norm_type, depthwise=depthwise)

    def forward(self, x):
        """Forward process"""
        if hasattr(self, 'conv'):
            return self.forward_fuse(x)
        else:
            id_out = 0 if self.bn is None else self.bn(x)
            return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def forward_fuse(self, x):
        """Forward process"""
        return self.act(self.conv(x))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.in_dim
        groups = self.g
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
        return k

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, BasicConv):
            kernel       = branch.conv.weight
            running_mean = branch.norm.running_mean
            running_var  = branch.norm.running_var
            gamma        = branch.norm.weight
            beta         = branch.norm.bias
            eps          = branch.norm.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_dim // self.g
                kernel_value = np.zeros((self.in_dim, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_dim):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel       = self.id_tensor
            running_mean = branch.running_mean
            running_var  = branch.running_var
            gamma        = branch.weight
            beta         = branch.bias
            eps          = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels  = self.conv1.conv.in_channels,
                              out_channels = self.conv1.conv.out_channels,
                              kernel_size  = self.conv1.conv.kernel_size,
                              stride       = self.conv1.conv.stride,
                              padding      = self.conv1.conv.padding,
                              dilation     = self.conv1.conv.dilation,
                              groups       = self.conv1.conv.groups,
                              bias         = True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')

class RepNBottleneck(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 shortcut=True,
                 kernel_size=(3, 3),
                 expansion=0.5,
                 act_type='silu',
                 norm_type='BN',
                 depthwise=False
                 ):
        super().__init__()
        inter_dim = round(out_dim * expansion)
        self.conv_layer_1 = RepConvN(in_dim, inter_dim, kernel_size[0], p=kernel_size[0]//2, s=1, act_type=act_type, norm_type=norm_type)
        self.conv_layer_2 = BasicConv(inter_dim, out_dim, kernel_size[1], padding=kernel_size[1]//2, stride=1, act_type=act_type, norm_type=norm_type)
        self.add = shortcut and in_dim == out_dim

    def forward(self, x):
        h = self.conv_layer_2(self.conv_layer_1(x))
        return x + h if self.add else h

class RepNCSP(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_blocks=1,
                 shortcut=True,
                 expansion=0.5,
                 act_type='silu',
                 norm_type='BN',
                 depthwise=False
                 ):
        super().__init__()
        inter_dim = int(out_dim * expansion)
        self.conv_layer_1 = BasicConv(in_dim, inter_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.conv_layer_2 = BasicConv(in_dim, inter_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.conv_layer_3 = BasicConv(2 * inter_dim, out_dim, kernel_size=1)
        self.module       = nn.Sequential(*(RepNBottleneck(inter_dim,
                                                           inter_dim,
                                                           kernel_size = [3, 3],
                                                           shortcut    = shortcut,
                                                           expansion   = 1.0,
                                                           act_type    = act_type,
                                                           norm_type   = norm_type,
                                                           depthwise   = depthwise)
                                                           for _ in range(num_blocks)))

    def forward(self, x):
        x1 = self.conv_layer_1(x)
        x2 = self.module(self.conv_layer_2(x))

        return self.conv_layer_3(torch.cat([x1, x2], dim=1))

class RepGElanLayer(nn.Module):
    """YOLOv9's GELAN module"""
    def __init__(self,
                 in_dim     :int,
                 inter_dims :List,
                 out_dim    :int,
                 num_blocks :int   = 1,
                 shortcut   :bool  = False,
                 act_type   :str   = 'silu',
                 norm_type  :str   = 'BN',
                 depthwise  :bool  = False,
                 ) -> None:
        super(RepGElanLayer, self).__init__()
        # ----------- Basic parameters -----------
        self.in_dim = in_dim
        self.inter_dims = inter_dims
        self.out_dim = out_dim

        # ----------- Network parameters -----------
        self.conv_layer_1  = BasicConv(in_dim, inter_dims[0], kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.elan_module_1 = nn.Sequential(
             RepNCSP(inter_dims[0]//2,
                     inter_dims[1],
                     num_blocks  = num_blocks,
                     shortcut    = shortcut,
                     expansion   = 0.5,
                     act_type    = act_type,
                     norm_type   = norm_type,
                     depthwise   = depthwise),
            BasicConv(inter_dims[1], inter_dims[1],
                      kernel_size=3, padding=1,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        self.elan_module_2 = nn.Sequential(
             RepNCSP(inter_dims[1],
                     inter_dims[1],
                     num_blocks  = num_blocks,
                     shortcut    = shortcut,
                     expansion   = 0.5,
                     act_type    = act_type,
                     norm_type   = norm_type,
                     depthwise   = depthwise),
            BasicConv(inter_dims[1], inter_dims[1],
                      kernel_size=3, padding=1,
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
    