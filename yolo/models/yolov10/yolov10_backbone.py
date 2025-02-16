import torch
import torch.nn as nn

try:
    from .modules import ConvModule, C2fBlock, SCDown, SPPF, PSABlock
except:
    from  modules import ConvModule, C2fBlock, SCDown, SPPF, PSABlock


# ---------------------------- Basic functions ----------------------------
class Yolov10Backbone(nn.Module):
    def __init__(self, cfg):
        super(Yolov10Backbone, self).__init__()
        # ------------------ Basic setting ------------------
        self.model_scale = cfg.model_scale
        self.feat_dims = [round(64  * cfg.width),
                          round(128 * cfg.width),
                          round(256 * cfg.width),
                          round(512 * cfg.width),
                          round(512 * cfg.width * cfg.ratio)]
        
        # ------------------ Network setting ------------------
        ## P1/2
        self.layer_1 = ConvModule(3, self.feat_dims[0], kernel_size=3, stride=2)
        # P2/4
        self.layer_2 = nn.Sequential(
            ConvModule(self.feat_dims[0], self.feat_dims[1], kernel_size=3, stride=2),
            C2fBlock(in_dim     = self.feat_dims[1],
                     out_dim    = self.feat_dims[1],
                     num_blocks = round(3*cfg.depth),
                     expansion  = 0.5,
                     shortcut   = True,
                     use_cib    = False,
                     )
        )
        # P3/8
        self.layer_3 = nn.Sequential(
            ConvModule(self.feat_dims[1], self.feat_dims[2], kernel_size=3, stride=2),
            C2fBlock(in_dim     = self.feat_dims[2],
                     out_dim    = self.feat_dims[2],
                     num_blocks = round(6*cfg.depth),
                     expansion  = 0.5,
                     shortcut   = True,
                     use_cib    = False,
                     )
        )
        # P4/16
        self.layer_4 = nn.Sequential(
            SCDown(self.feat_dims[2], self.feat_dims[3], kernel_size=3, stride=2),
            C2fBlock(in_dim     = self.feat_dims[3],
                     out_dim    = self.feat_dims[3],
                     num_blocks = round(6*cfg.depth),
                     expansion  = 0.5,
                     shortcut   = True,
                     use_cib    = False,
                     )
        )
        # P5/32
        self.layer_5 = nn.Sequential(
            SCDown(self.feat_dims[3], self.feat_dims[4], kernel_size=3, stride=2),
            C2fBlock(in_dim     = self.feat_dims[4],
                     out_dim    = self.feat_dims[4],
                     num_blocks = round(3*cfg.depth),
                     expansion  = 0.5,
                     shortcut   = True,
                     use_cib    = True if self.model_scale in "smlx" else False,
                     )
        )

        # Extra module (no pretrained weight)
        self.layer_6 = SPPF(in_dim  = int(512 * cfg.width * cfg.ratio),
                            out_dim = int(512 * cfg.width * cfg.ratio),
                            )
        self.layer_7 = PSABlock(in_dim  = int(512 * cfg.width * cfg.ratio),
                                out_dim = int(512 * cfg.width * cfg.ratio),
                                expansion = 0.5,
                                )

        # Initialize all layers
        self.init_weights()
                
    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.reset_parameters()

    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        c5 = self.layer_6(c5)
        c5 = self.layer_7(c5)

        outputs = [c3, c4, c5]

        return outputs


if __name__ == '__main__':
    import time
    from thop import profile
    class BaseConfig(object):
        def __init__(self) -> None:
            self.width = 0.25
            self.depth = 0.34
            self.ratio = 2.0
            self.model_scale = "n"

            self.width = 0.50
            self.depth = 0.34
            self.ratio = 2.0
            self.model_scale = "s"

            self.width = 0.75
            self.depth = 0.67
            self.ratio = 1.5
            self.model_scale = "m"

            self.width = 1.0
            self.depth = 1.0
            self.ratio = 1.0
            self.model_scale = "l"

    cfg = BaseConfig()
    model = Yolov10Backbone(cfg)
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