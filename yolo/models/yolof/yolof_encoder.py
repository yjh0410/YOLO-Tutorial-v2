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

# ELAN-style Dilated Encoder
class YolofEncoder(nn.Module):
    def __init__(self, cfg, in_dim, out_dim):
        super(YolofEncoder, self).__init__()
        # ------------------ Basic parameters -------------------
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.expand_ratio = cfg.neck_expand_ratio
        self.dilations    = cfg.neck_dilations
        # ------------------ Network parameters -------------------
        ## input layer
        self.input_proj = BasicConv(in_dim, out_dim, kernel_size=1, act_type=cfg.neck_act, norm_type=cfg.neck_norm)
        ## dilated layers
        self.module = nn.ModuleList([Bottleneck(in_dim       = out_dim,
                                                out_dim      = out_dim,
                                                dilation     = dilation,
                                                expand_ratio = self.expand_ratio,
                                                shortcut     = True,
                                                act_type     = cfg.neck_act,
                                                norm_type    = cfg.neck_norm,
                                                depthwise    = cfg.neck_depthwise,
                                                ) for dilation in self.dilations])
        ## output layer
        self.output_proj = BasicConv(out_dim * (len(self.dilations) + 1), out_dim,
                                     kernel_size=1, padding=0, stride=1,
                                     act_type=cfg.neck_act, norm_type=cfg.neck_norm)

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
        x = self.input_proj(x)

        out = [x]
        for m in self.module:
            x = m(x)
            out.append(x)

        out = self.output_proj(torch.cat(out, dim=1))

        return out
