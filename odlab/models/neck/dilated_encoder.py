import torch.nn as nn
from utils import weight_init

from ..basic.conv import ConvModule


# BottleNeck
class Bottleneck(nn.Module):
    def __init__(self, in_dim, dilation, expand_ratio, act_type='relu', norm_type='BN'):
        super(Bottleneck, self).__init__()
        # ------------------ Basic parameters -------------------
        self.in_dim = in_dim
        self.dilation = dilation
        self.expand_ratio = expand_ratio
        inter_dim = round(in_dim * expand_ratio)
        # ------------------ Network parameters -------------------
        self.branch = nn.Sequential(
            ConvModule(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            ConvModule(inter_dim, inter_dim, k=3, p=dilation, d=dilation, act_type=act_type, norm_type=norm_type),
            ConvModule(inter_dim, in_dim, k=1, act_type=act_type, norm_type=norm_type)
        )

    def forward(self, x):
        return x + self.branch(x)

# Dilated Encoder
class DilatedEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, expand_ratio, dilations=[2, 4, 6, 8], act_type='relu', norm_type='BN'):
        super(DilatedEncoder, self).__init__()
        # ------------------ Basic parameters -------------------
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.expand_ratio = expand_ratio
        self.dilations = dilations
        # ------------------ Network parameters -------------------
        ## proj layer
        self.projector = nn.Sequential(
            ConvModule(in_dim, out_dim, k=1, act_type=None, norm_type=norm_type),
            ConvModule(out_dim, out_dim, k=3, p=1, act_type=None, norm_type=norm_type)
        )
        ## encoder layers
        self.encoders = nn.Sequential(
            *[Bottleneck(out_dim, d, expand_ratio, act_type, norm_type) for d in dilations])

        self._init_weight()


    def _init_weight(self):
        for m in self.projector:
            if isinstance(m, nn.Conv2d):
                weight_init.c2_xavier_fill(m)
                weight_init.c2_xavier_fill(m)
            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.encoders.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.projector(x)
        x = self.encoders(x)

        return x
