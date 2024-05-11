import torch
import torch.nn as nn

try:
    from .modules import BasicConv, ELANLayer
except:
    from  modules import BasicConv, ELANLayer
   

## ELAN-based DarkNet
class ELANDarkNet(nn.Module):
    def __init__(self, img_dim=3, width=1.0, depth=1.0, ratio=1.0, num_classes=1000, act_type='silu', norm_type='BN', depthwise=False):
        super(ELANDarkNet, self).__init__()
        # ---------------- Basic parameters ----------------
        self.width_factor = width
        self.depth_factor = depth
        self.last_stage_factor = ratio
        self.feat_dims = [round(64 * width),
                          round(128 * width),
                          round(256 * width),
                          round(512 * width),
                          round(512 * width * ratio)
                          ]
        # ---------------- Network parameters ----------------
        ## P1/2
        self.layer_1 = BasicConv(img_dim, self.feat_dims[0],
                                 kernel_size=3, padding=1, stride=2,
                                 act_type=act_type, norm_type=norm_type)
        ## P2/4
        self.layer_2 = nn.Sequential(
            BasicConv(self.feat_dims[0], self.feat_dims[1],
                      kernel_size=3, padding=1, stride=2,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            ELANLayer(in_dim     = self.feat_dims[1],
                      out_dim    = self.feat_dims[1],
                      num_blocks = round(3*depth),
                      shortcut   = True,
                      act_type   = act_type,
                      norm_type  = norm_type,
                      depthwise  = depthwise,
                      )
        )
        ## P3/8
        self.layer_3 = nn.Sequential(
            BasicConv(self.feat_dims[1], self.feat_dims[2],
                      kernel_size=3, padding=1, stride=2,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            ELANLayer(in_dim     = self.feat_dims[2],
                      out_dim    = self.feat_dims[2],
                      num_blocks = round(6*depth),
                      shortcut   = True,
                      act_type   = act_type,
                      norm_type  = norm_type,
                      depthwise  = depthwise,
                      )
        )
        ## P4/16
        self.layer_4 = nn.Sequential(
            BasicConv(self.feat_dims[2], self.feat_dims[3],
                      kernel_size=3, padding=1, stride=2,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            ELANLayer(in_dim     = self.feat_dims[3],
                      out_dim    = self.feat_dims[3],
                      num_blocks = round(6*depth),
                      shortcut   = True,
                      act_type   = act_type,
                      norm_type  = norm_type,
                      depthwise  = depthwise,
                      )
        )
        ## P5/32
        self.layer_5 = nn.Sequential(
            BasicConv(self.feat_dims[3], self.feat_dims[4],
                      kernel_size=3, padding=1, stride=2,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            ELANLayer(in_dim     = self.feat_dims[4],
                      out_dim    = self.feat_dims[4],
                      num_blocks = round(3*depth),
                      shortcut   = True,
                      act_type   = act_type,
                      norm_type  = norm_type,
                      depthwise  = depthwise,
                      )
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.feat_dims[4], num_classes)


    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        c5 = self.avgpool(c5)
        c5 = torch.flatten(c5, 1)
        c5 = self.fc(c5)

        return c5


# ------------------------ Model Functions ------------------------
def elandarknet_n(img_dim=3, num_classes=1000) -> ELANDarkNet:
    return ELANDarkNet(img_dim=img_dim,
                       width=0.25,
                       depth=0.34,
                       ratio=2.0,
                       act_type='silu',
                       norm_type='BN',
                       depthwise=False,
                       num_classes=num_classes
                       )

def elandarknet_s(img_dim=3, num_classes=1000) -> ELANDarkNet:
    return ELANDarkNet(img_dim=img_dim,
                       width=0.50,
                       depth=0.34,
                       ratio=2.0,
                       act_type='silu',
                       norm_type='BN',
                       depthwise=False,
                       num_classes=num_classes
                       )

def elandarknet_m(img_dim=3, num_classes=1000) -> ELANDarkNet:
    return ELANDarkNet(img_dim=img_dim,
                       width=0.75,
                       depth=0.67,
                       ratio=1.5,
                       act_type='silu',
                       norm_type='BN',
                       depthwise=False,
                       num_classes=num_classes
                       )

def elandarknet_l(img_dim=3, num_classes=1000) -> ELANDarkNet:
    return ELANDarkNet(img_dim=img_dim,
                       width=1.0,
                       depth=1.0,
                       ratio=1.0,
                       act_type='silu',
                       norm_type='BN',
                       depthwise=False,
                       num_classes=num_classes
                       )

def elandarknet_x(img_dim=3, num_classes=1000) -> ELANDarkNet:
    return ELANDarkNet(img_dim=img_dim,
                       width=1.25,
                       depth=1.34,
                       ratio=1.0,
                       act_type='silu',
                       norm_type='BN',
                       depthwise=False,
                       num_classes=num_classes
                       )


if __name__ == '__main__':
    import torch
    from thop import profile

    # build model
    model = elandarknet_s()

    x = torch.randn(1, 3, 224, 224)
    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
