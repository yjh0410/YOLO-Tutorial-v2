import torch
import torch.nn as nn


def get_activation(act_type=None):
    if   act_type == 'sigmoid':
        return nn.Sigmoid()
    elif act_type == 'relu':
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
    if   norm_type == 'bn':
        return nn.BatchNorm2d(dim)
    elif norm_type == 'ln':
        return LayerNorm2d(dim)
    elif norm_type == 'gn':
        return nn.GroupNorm(num_groups=32, num_channels=dim)
    elif norm_type is None:
        return nn.Identity()
    else:
        raise NotImplementedError

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        
        return x
    

# Basic convolutional module
class ConvModule(nn.Module):
    def __init__(self,
                 in_dim      :int,
                 out_dim     :int,
                 kernel_size :int  = 1,
                 padding     :int  = 0,
                 stride      :int  = 1,
                 act_type    :str  = "relu",
                 norm_type   :str  = "bn",
                 depthwise   :bool = False) -> None:
        super().__init__()
        use_bias = False if norm_type is not None else True
        self.depthwise = depthwise
        if not depthwise:
            self.conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim,
                                kernel_size=kernel_size, padding=padding, stride=stride,
                                bias=use_bias)
            self.norm  = get_norm(norm_type, out_dim)
        else:
            self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,
                                   kernel_size=kernel_size, padding=padding, stride=stride, groups=in_dim,
                                   bias=use_bias)
            self.norm1 = get_norm(norm_type, in_dim)
            self.conv2 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim,
                                   kernel_size=1, padding=0, stride=1,
                                   bias=use_bias)
            self.norm2 = get_norm(norm_type, out_dim)
        self.act   = get_activation(act_type)

    def forward(self, x):
        if self.depthwise:
            x = self.norm1(self.conv1(x))
            x = self.act(self.norm2(self.conv2(x)))
        else:
            x = self.act(self.norm(self.conv(x)))

        return x
