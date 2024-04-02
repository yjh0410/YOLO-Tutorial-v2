import torch
import torch.nn as nn

try:
    from .gelan_basic import BasicConv, RepGElanLayer, ADown
except:
    from  gelan_basic import BasicConv, RepGElanLayer, ADown


# ---------------------------- Basic functions ----------------------------
class GElanBackbone(nn.Module):
    def __init__(self, cfg):
        super(GElanBackbone, self).__init__()
        # ------------------ Basic setting ------------------
        self.feat_dims = [cfg.backbone_feats["c1"][-1],  # 64
                          cfg.backbone_feats["c2"][-1],  # 128
                          cfg.backbone_feats["c3"][-1],  # 256
                          cfg.backbone_feats["c4"][-1],  # 512
                          cfg.backbone_feats["c5"][-1],  # 512
                          ]
        
        # ------------------ Network setting ------------------
        ## P1/2
        self.layer_1 = BasicConv(3, cfg.backbone_feats["c1"][0],
                                 kernel_size=3, padding=1, stride=2,
                                 act_type=cfg.bk_act, norm_type=cfg.bk_norm, depthwise=cfg.bk_depthwise)
        # P2/4
        self.layer_2 = nn.Sequential(
            BasicConv(cfg.backbone_feats["c1"][0], cfg.backbone_feats["c2"][0],
                      kernel_size=3, padding=1, stride=2,
                      act_type=cfg.bk_act, norm_type=cfg.bk_norm, depthwise=cfg.bk_depthwise),
            RepGElanLayer(in_dim     = cfg.backbone_feats["c2"][0],
                          inter_dims = cfg.backbone_feats["c2"][1],
                          out_dim    = cfg.backbone_feats["c2"][2],
                          num_blocks = cfg.backbone_depth,
                          shortcut   = True,
                          act_type   = cfg.bk_act,
                          norm_type  = cfg.bk_norm,
                          depthwise  = cfg.bk_depthwise)
        )
        # P3/8
        self.layer_3 = nn.Sequential(
            ADown(cfg.backbone_feats["c2"][2], cfg.backbone_feats["c3"][0],
                  act_type=cfg.bk_act, norm_type=cfg.bk_norm, depthwise=cfg.bk_depthwise),
            RepGElanLayer(in_dim     = cfg.backbone_feats["c3"][0],
                          inter_dims = cfg.backbone_feats["c3"][1],
                          out_dim    = cfg.backbone_feats["c3"][2],
                          num_blocks = cfg.backbone_depth,
                          shortcut   = True,
                          act_type   = cfg.bk_act,
                          norm_type  = cfg.bk_norm,
                          depthwise  = cfg.bk_depthwise)
        )
        # P4/16
        self.layer_4 = nn.Sequential(
            ADown(cfg.backbone_feats["c3"][2], cfg.backbone_feats["c4"][0],
                  act_type=cfg.bk_act, norm_type=cfg.bk_norm, depthwise=cfg.bk_depthwise),
            RepGElanLayer(in_dim     = cfg.backbone_feats["c4"][0],
                          inter_dims = cfg.backbone_feats["c4"][1],
                          out_dim    = cfg.backbone_feats["c4"][2],
                          num_blocks = cfg.backbone_depth,
                          shortcut   = True,
                          act_type   = cfg.bk_act,
                          norm_type  = cfg.bk_norm,
                          depthwise  = cfg.bk_depthwise)
        )
        # P5/32
        self.layer_5 = nn.Sequential(
            ADown(cfg.backbone_feats["c4"][2], cfg.backbone_feats["c5"][0],
                  act_type=cfg.bk_act, norm_type=cfg.bk_norm, depthwise=cfg.bk_depthwise),
            RepGElanLayer(in_dim     = cfg.backbone_feats["c5"][0],
                          inter_dims = cfg.backbone_feats["c5"][1],
                          out_dim    = cfg.backbone_feats["c5"][2],
                          num_blocks = cfg.backbone_depth,
                          shortcut   = True,
                          act_type   = cfg.bk_act,
                          norm_type  = cfg.bk_norm,
                          depthwise  = cfg.bk_depthwise)
        )

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
        outputs = [c3, c4, c5]

        return outputs


# ---------------------------- Functions ----------------------------
## build Yolo's Backbone
def build_backbone(cfg): 
    # model
    backbone = GElanBackbone(cfg)
        
    return backbone


if __name__ == '__main__':
    import time
    from thop import profile
    base_config = {
        "bk_act": "silu",
        "bk_norm": "BN"
    }
    class BaseConfig(object):
        def __init__(self) -> None:
            self.bk_act = 'silu'
            self.bk_norm = 'BN'
            self.bk_depthwise = False
            self.backbone_feats = {
                "c1": [64],
                "c2": [128, [128, 64], 256],
                "c3": [256, [256, 128], 512],
                "c4": [512, [512, 256], 512],
                "c5": [512, [512, 256], 512],
            }
            self.backbone_depth = 1

    cfg = BaseConfig()
    model = build_backbone(cfg)
    x = torch.randn(1, 3, 640, 640)
    t0 = time.time()
    outputs = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
    for out in outputs:
        print(out.shape)

    x = torch.randn(1, 3, 640, 640)
    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
    