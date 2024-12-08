import torch
import torch.nn as nn

try:
    from .modules import ConvModule
except:
    from  modules import ConvModule


# BottleNeck
class Bottleneck(nn.Module):
    def __init__(self, in_dim: int, dilation: int = 1, expansion: float = 0.5):
        super(Bottleneck, self).__init__()
        # ------------------ Basic parameters -------------------
        self.in_dim = in_dim
        self.dilation = dilation
        self.expansion = expansion
        inter_dim = round(in_dim * expansion)
        # ------------------ Network parameters -------------------
        self.branch = nn.Sequential(
            ConvModule(in_dim, inter_dim, kernel_size=1),
            ConvModule(inter_dim, inter_dim, kernel_size=3, padding=dilation, dilation=dilation),
            ConvModule(inter_dim, in_dim, kernel_size=1)
        )

    def forward(self, x):
        return x + self.branch(x)

# Dilated Encoder
class DilatedEncoder(nn.Module):
    def __init__(self, cfg, in_dim, out_dim):
        super(DilatedEncoder, self).__init__()
        # ------------------ Basic parameters -------------------
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.expand_ratio = cfg.neck_expand_ratio
        self.dilations    = cfg.neck_dilations
        # ------------------ Network parameters -------------------
        ## proj layer
        self.projector = nn.Sequential(
            ConvModule(in_dim,  out_dim, kernel_size=1, use_act=False),
            ConvModule(out_dim, out_dim, kernel_size=3, padding=1, use_act=False)
        )
        ## encoder layers
        self.encoders = nn.Sequential(
            *[Bottleneck(in_dim = out_dim,
                         dilation = d,
                         expansion = self.expand_ratio,
                         ) for d in self.dilations])

    def forward(self, x):
        x = self.projector(x)
        x = self.encoders(x)

        return x


if __name__=='__main__':
    from thop import profile

    # YOLOv1 configuration
    class YolofBaseConfig(object):
        def __init__(self) -> None:
            # ---------------- Model config ----------------
            self.out_stride = 32
            ## Backbone
            self.backbone = 'resnet18'
            self.use_pretrained = True

            self.neck_expand_ratio = 0.25
            self.neck_dilations = [2, 4, 6, 8]

    cfg = YolofBaseConfig()

    # Randomly generate a input data
    x = torch.randn(2, 512, 20, 20)

    # Build backbone
    model = DilatedEncoder(cfg, in_dim=512, out_dim=512)

    # Inference
    output = model(x)
    print(' - the shape of input :  ', x.shape)
    print(' - the shape of output : ', output.shape)

    x = torch.randn(1, 512, 20, 20)
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('============== FLOPs & Params ================')
    print(' - FLOPs  : {:.2f} G'.format(flops / 1e9 * 2))
    print(' - Params : {:.2f} M'.format(params / 1e6))
