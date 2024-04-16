import torch
import torch.nn as nn

try:
    from .yolof_basic import BasicConv
except:
    from  yolof_basic import BasicConv


# BottleNeck
class Bottleneck(nn.Module):
    def __init__(self,
                 in_dim       :int,
                 out_dim      :int,
                 dilation     :int,
                 expand_ratio :float = 0.5,
                 shortcut     :bool  = False,
                 act_type     :str   = 'relu',
                 norm_type    :str   = 'BN',
                 depthwise    :bool  = False,):
        super(Bottleneck, self).__init__()
        # ------------------ Basic parameters -------------------
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dilation = dilation
        self.expand_ratio = expand_ratio
        self.shortcut = shortcut and in_dim == out_dim
        inter_dim = round(in_dim * expand_ratio)
        # ------------------ Network parameters -------------------
        self.branch = nn.Sequential(
            BasicConv(in_dim, inter_dim,
                      kernel_size=1, padding=0, stride=1,
                      act_type=act_type, norm_type=norm_type),
            BasicConv(inter_dim, inter_dim,
                      kernel_size=3, padding=dilation, dilation=dilation, stride=1,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            BasicConv(inter_dim, in_dim,
                      kernel_size=1, padding=0, stride=1,
                      act_type=act_type, norm_type=norm_type)
        )

    def forward(self, x):
        h = self.branch(x)

        return x + self.branch(x) if self.shortcut else h

# CSP-style Dilated Encoder
class YolofEncoder(nn.Module):
    def __init__(self, cfg, in_dim, out_dim):
        super(YolofEncoder, self).__init__()
        # ------------------ Basic parameters -------------------
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.expand_ratio = cfg.neck_expand_ratio
        self.dilations    = cfg.neck_dilations
        # ------------------ Network parameters -------------------
        ## proj layer
        self.projector = nn.Sequential(
            BasicConv(in_dim, out_dim, kernel_size=1, act_type=None, norm_type=cfg.neck_norm),
            BasicConv(out_dim, out_dim, kernel_size=3, padding=1, act_type=None, norm_type=cfg.neck_norm)
        )
        ## encoder layers
        self.encoders = nn.Sequential(*[Bottleneck(in_dim      = out_dim,
                                                   out_dim     = out_dim,
                                                   dilation    = d,
                                                   expand_ratio = self.expand_ratio,
                                                   shortcut     = True,
                                                   act_type     = cfg.neck_act,
                                                   norm_type    = cfg.neck_norm,
                                                   depthwise    = cfg.neck_depthwise,
                                                   ) for d in self.dilations])

        # Initialize all layers
        self.init_weights()

    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()

    def forward(self, x):
        x = self.projector(x)
        x = self.encoders(x)

        return x
