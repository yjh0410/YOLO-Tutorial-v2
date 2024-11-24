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
        return nn.BatchNorm1d(dim)
    elif norm_type == 'ln':
        return nn.LayerNorm(dim)
    elif norm_type == 'gn':
        return nn.GroupNorm(num_groups=32, num_channels=dim)
    elif norm_type is None:
        return nn.Identity()
    else:
        raise NotImplementedError


# Single Layer Perceptron
class SLP(nn.Module):
    def __init__(self,
                 in_dim    :int,
                 out_dim   :int,
                 act_type  :str = "sigmoid",
                 norm_type :str = "bn") -> None:
        super().__init__()
        use_bias = False if norm_type is not None else True
        self.layer = nn.Linear(in_features=in_dim, out_features=out_dim, bias=use_bias)
        self.norm  = get_norm(norm_type, out_dim)
        self.act   = get_activation(act_type)

    def forward(self, x):
        return self.act(self.norm(self.layer(x)))
