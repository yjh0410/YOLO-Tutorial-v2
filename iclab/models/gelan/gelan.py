import torch
import torch.nn as nn

try:
    from .modules import BasicConv, RepGElanLayer, ADown
except:
    from  modules import BasicConv, RepGElanLayer, ADown


# ---------------------------- GELAN Backbone ----------------------------
class GElanCBackbone(nn.Module):
    def __init__(self, img_dim=3, num_classes=1000, act_type='silu', norm_type='BN', depthwise=False):
        super(GElanCBackbone, self).__init__()
        # ------------------ Basic setting ------------------
        self.feat_dims = {
            "c1": [64],
            "c2": [128, [128, 64],  256],
            "c3": [256, [256, 128], 512],
            "c4": [512, [512, 256], 512],
            "c5": [512, [512, 256], 512],
        }
        
        # ------------------ Network setting ------------------
        ## P1/2
        self.layer_1 = BasicConv(img_dim, self.feat_dims["c1"][0],
                                 kernel_size=3, padding=1, stride=2,
                                 act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        # P2/4
        self.layer_2 = nn.Sequential(
            BasicConv(self.feat_dims["c1"][0], self.feat_dims["c2"][0],
                      kernel_size=3, padding=1, stride=2,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            RepGElanLayer(in_dim     = self.feat_dims["c2"][0],
                          inter_dims = self.feat_dims["c2"][1],
                          out_dim    = self.feat_dims["c2"][2],
                          num_blocks = 1,
                          shortcut   = True,
                          act_type   = act_type,
                          norm_type  = norm_type,
                          depthwise  = depthwise)
        )
        # P3/8
        self.layer_3 = nn.Sequential(
            ADown(self.feat_dims["c2"][2], self.feat_dims["c3"][0],
                  act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            RepGElanLayer(in_dim     = self.feat_dims["c3"][0],
                          inter_dims = self.feat_dims["c3"][1],
                          out_dim    = self.feat_dims["c3"][2],
                          num_blocks = 1,
                          shortcut   = True,
                          act_type   = act_type,
                          norm_type  = norm_type,
                          depthwise  = depthwise)
        )
        # P4/16
        self.layer_4 = nn.Sequential(
            ADown(self.feat_dims["c3"][2], self.feat_dims["c4"][0],
                  act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            RepGElanLayer(in_dim     = self.feat_dims["c4"][0],
                          inter_dims = self.feat_dims["c4"][1],
                          out_dim    = self.feat_dims["c4"][2],
                          num_blocks = 1,
                          shortcut   = True,
                          act_type   = act_type,
                          norm_type  = norm_type,
                          depthwise  = depthwise)
        )
        # P5/32
        self.layer_5 = nn.Sequential(
            ADown(self.feat_dims["c4"][2], self.feat_dims["c5"][0],
                  act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            RepGElanLayer(in_dim     = self.feat_dims["c5"][0],
                          inter_dims = self.feat_dims["c5"][1],
                          out_dim    = self.feat_dims["c5"][2],
                          num_blocks = 1,
                          shortcut   = True,
                          act_type   = act_type,
                          norm_type  = norm_type,
                          depthwise  = depthwise)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.feat_dims["c5"][2], num_classes)

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
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        c5 = self.avgpool(c5)
        c5 = torch.flatten(c5, 1)
        c5 = self.fc(c5)

        return c5

class GElanSBackbone(nn.Module):
    def __init__(self, img_dim=3, num_classes=1000, act_type='silu', norm_type='BN', depthwise=False):
        super(GElanSBackbone, self).__init__()
        # ------------------ Basic setting ------------------
        self.feat_dims = {
            "c1": [32],
            "c2": [64,  [64, 32],   64],
            "c3": [64,  [64, 32],   128],
            "c4": [128, [128, 64],  256],
            "c5": [256, [256, 128], 256],
        }
        
        # ------------------ Network setting ------------------
        ## P1/2
        self.layer_1 = BasicConv(img_dim, self.feat_dims["c1"][0],
                                 kernel_size=3, padding=1, stride=2,
                                 act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        # P2/4
        self.layer_2 = nn.Sequential(
            BasicConv(self.feat_dims["c1"][0], self.feat_dims["c2"][0],
                      kernel_size=3, padding=1, stride=2,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            RepGElanLayer(in_dim     = self.feat_dims["c2"][0],
                          inter_dims = self.feat_dims["c2"][1],
                          out_dim    = self.feat_dims["c2"][2],
                          num_blocks = 3,
                          shortcut   = True,
                          act_type   = act_type,
                          norm_type  = norm_type,
                          depthwise  = depthwise)
        )
        # P3/8
        self.layer_3 = nn.Sequential(
            ADown(self.feat_dims["c2"][2], self.feat_dims["c3"][0],
                  act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            RepGElanLayer(in_dim     = self.feat_dims["c3"][0],
                          inter_dims = self.feat_dims["c3"][1],
                          out_dim    = self.feat_dims["c3"][2],
                          num_blocks = 3,
                          shortcut   = True,
                          act_type   = act_type,
                          norm_type  = norm_type,
                          depthwise  = depthwise)
        )
        # P4/16
        self.layer_4 = nn.Sequential(
            ADown(self.feat_dims["c3"][2], self.feat_dims["c4"][0],
                  act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            RepGElanLayer(in_dim     = self.feat_dims["c4"][0],
                          inter_dims = self.feat_dims["c4"][1],
                          out_dim    = self.feat_dims["c4"][2],
                          num_blocks = 3,
                          shortcut   = True,
                          act_type   = act_type,
                          norm_type  = norm_type,
                          depthwise  = depthwise)
        )
        # P5/32
        self.layer_5 = nn.Sequential(
            ADown(self.feat_dims["c4"][2], self.feat_dims["c5"][0],
                  act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            RepGElanLayer(in_dim     = self.feat_dims["c5"][0],
                          inter_dims = self.feat_dims["c5"][1],
                          out_dim    = self.feat_dims["c5"][2],
                          num_blocks = 3,
                          shortcut   = True,
                          act_type   = act_type,
                          norm_type  = norm_type,
                          depthwise  = depthwise)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.feat_dims["c5"][2], num_classes)

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
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        c5 = self.avgpool(c5)
        c5 = torch.flatten(c5, 1)
        c5 = self.fc(c5)

        return c5


# ---------------------------- Functions ----------------------------
def gelan_c(img_dim=3, num_classes=1000):
    return GElanCBackbone(img_dim,
                          num_classes=num_classes,
                          act_type='silu',
                          norm_type='BN',
                          depthwise=False)

def gelan_s(img_dim=3, num_classes=1000):
    return GElanSBackbone(img_dim,
                          num_classes=num_classes,
                          act_type='silu',
                          norm_type='BN',
                          depthwise=False)


if __name__ == '__main__':
    import torch
    from thop import profile

    # build model
    model = gelan_c()

    x = torch.randn(1, 3, 224, 224)
    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
