import torch
import torch.nn as nn

try:
    from .rtcdet_basic import BasicConv, ElanLayer, MDown, ADown
except:
    from  rtcdet_basic import BasicConv, ElanLayer, MDown, ADown


# ------------------ Basic functions ------------------
class RTCBackbone(nn.Module):
    def __init__(self, cfg):
        super(RTCBackbone, self).__init__()
        # ------------------ Basic setting ------------------
        self.stage_depth = [round(nb  * cfg.depth) for nb  in cfg.stage_depth]
        self.stage_dims  = [round(dim * cfg.width * cfg.ratio) if i == len(cfg.stage_dims) - 1
                            else round(dim * cfg.width) for i, dim in enumerate(cfg.stage_dims)]
        self.pyramid_feat_dims = self.stage_dims[-3:]
        
        # ------------------ Model setting ------------------
        ## P1/2
        self.layer_1 = BasicConv(3, self.stage_dims[0], kernel_size=6, padding=2, stride=2,
                                 act_type=cfg.bk_act, norm_type=cfg.bk_norm, depthwise=cfg.bk_depthwise)
        # P2/4
        self.layer_2 = nn.Sequential(
            self.make_downsample_block(cfg, self.stage_dims[0], self.stage_dims[1]),
            self.make_stage_block(cfg, self.stage_dims[1], self.stage_dims[1], self.stage_depth[0])
        )
        # P3/8
        self.layer_3 = nn.Sequential(
            self.make_downsample_block(cfg, self.stage_dims[1], self.stage_dims[2]),
            self.make_stage_block(cfg, self.stage_dims[2], self.stage_dims[2], self.stage_depth[1])
        )
        # P4/16
        self.layer_4 = nn.Sequential(
            self.make_downsample_block(cfg, self.stage_dims[2], self.stage_dims[3]),
            self.make_stage_block(cfg, self.stage_dims[3], self.stage_dims[3], self.stage_depth[2])
        )
        # P5/32
        self.layer_5 = nn.Sequential(
            self.make_downsample_block(cfg, self.stage_dims[3], self.stage_dims[4]),
            self.make_stage_block(cfg, self.stage_dims[4], self.stage_dims[4], self.stage_depth[3])
        )

        # Initialize all layers
        self.init_weights()
                        
    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.reset_parameters()

    def make_downsample_block(self, cfg, in_dim, out_dim):
        if cfg.bk_ds_block == "conv":
            return BasicConv(in_dim, out_dim, kernel_size=3, padding=1, stride=2,
                             act_type=cfg.bk_act, norm_type=cfg.bk_norm, depthwise=cfg.bk_depthwise)
        if cfg.bk_ds_block == "mdown":
            return MDown(in_dim, out_dim, cfg.bk_act, cfg.bk_norm, cfg.bk_depthwise)
        if cfg.bk_ds_block == "adown":
            return ADown(in_dim, out_dim, cfg.bk_act, cfg.bk_norm, cfg.bk_depthwise)
        if cfg.bk_ds_block == "maxpool":
            assert in_dim == out_dim
            return nn.MaxPool2d((2, 2), stride=2)
        else:
            raise NotImplementedError("Unknown fpn downsample block: {}".format(cfg.fpn_ds_block))
        
    def make_stage_block(self, cfg, in_dim, out_dim, stage_depth):
        if cfg.bk_block == "elan_layer":
            return ElanLayer(in_dim     = in_dim,
                             out_dim    = out_dim,
                             num_blocks = stage_depth,
                             expansion  = 0.5,
                             shortcut   = True,
                             act_type   = cfg.bk_act,
                             norm_type  = cfg.bk_norm,
                             depthwise  = cfg.bk_depthwise)
        else:
            raise NotImplementedError("Unknown stage block: {}".format(cfg.bk_block))
        
    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)
        outputs = [c3, c4, c5]

        return outputs


# ------------------ Functions ------------------
## build Yolo's Backbone
def build_backbone(cfg): 
    # model
    backbone = RTCBackbone(cfg)
        
    return backbone


if __name__ == '__main__':
    import time
    from thop import profile
    class BaseConfig(object):
        def __init__(self) -> None:
            self.stage_dims =  [64, 128, 256, 512, 512]
            self.stage_depth = [3, 6, 6, 3]
            self.bk_block = "elan_layer"
            self.bk_ds_block = "mdown"
            self.bk_act = 'silu'
            self.bk_norm = 'bn'
            self.bk_depthwise = False
            self.use_pretrained = False
            self.width = 0.5
            self.depth = 0.34
            self.ratio = 2.0

    cfg = BaseConfig()
    model = build_backbone(cfg).cuda()
    x = torch.randn(1, 3, 640, 640).cuda()

    for _ in range(5):
        t0 = time.time()
        outputs = model(x)
        t1 = time.time()
        print('Time: ', t1 - t0)
        
    for out in outputs:
        print(out.shape)

    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))