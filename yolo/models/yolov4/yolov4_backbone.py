import torch
import torch.nn as nn

try:
    from .modules import ConvModule, CSPBlock
except:
    from  modules import ConvModule, CSPBlock

# IN1K pretrained weight
pretrained_urls = {
    'n': None,
    's': None,
    'm': None,
    'l': None,
    'x': None,
}

# --------------------- Yolov3's Backbone -----------------------
## Modified DarkNet
class Yolov4Backbone(nn.Module):
    def __init__(self, cfg):
        super(Yolov4Backbone, self).__init__()
        # ------------------ Basic setting ------------------
        self.model_scale = cfg.model_scale
        self.feat_dims = [round(64   * cfg.width),
                          round(128  * cfg.width),
                          round(256  * cfg.width),
                          round(512  * cfg.width),
                          round(1024 * cfg.width)]
        
        # ------------------ Network setting ------------------
        ## P1/2
        self.layer_1 = ConvModule(3, self.feat_dims[0], kernel_size=6, padding=2, stride=2)
        # P2/4
        self.layer_2 = nn.Sequential(
            ConvModule(self.feat_dims[0], self.feat_dims[1], kernel_size=3, padding=1, stride=2),
            CSPBlock(in_dim     = self.feat_dims[1],
                     out_dim    = self.feat_dims[1],
                     num_blocks = round(3*cfg.depth),
                     expansion  = 0.5,
                     shortcut   = True,
                     )
        )
        # P3/8
        self.layer_3 = nn.Sequential(
            ConvModule(self.feat_dims[1], self.feat_dims[2], kernel_size=3, padding=1, stride=2),
            CSPBlock(in_dim     = self.feat_dims[2],
                     out_dim    = self.feat_dims[2],
                     num_blocks = round(9*cfg.depth),
                     expansion  = 0.5,
                     shortcut   = True,
                     )
        )
        # P4/16
        self.layer_4 = nn.Sequential(
            ConvModule(self.feat_dims[2], self.feat_dims[3], kernel_size=3, padding=1, stride=2),
            CSPBlock(in_dim     = self.feat_dims[3],
                     out_dim    = self.feat_dims[3],
                     num_blocks = round(9*cfg.depth),
                     expansion  = 0.5,
                     shortcut   = True,
                     )
        )
        # P5/32
        self.layer_5 = nn.Sequential(
            ConvModule(self.feat_dims[3], self.feat_dims[4], kernel_size=3, padding=1, stride=2),
            CSPBlock(in_dim     = self.feat_dims[4],
                     out_dim    = self.feat_dims[4],
                     num_blocks = round(3*cfg.depth),
                     expansion  = 0.5,
                     shortcut   = True,
                     )
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
        outputs = [c3, c4, c5]

        return outputs


if __name__ == '__main__':
    import time
    from thop import profile
    class BaseConfig(object):
        def __init__(self) -> None:
            self.width = 0.5
            self.depth = 0.34
            self.model_scale = "s"
            self.use_pretrained = True

    cfg = BaseConfig()
    model = Yolov4Backbone(cfg)
    x = torch.randn(1, 3, 640, 640)
    t0 = time.time()
    outputs = model(x)
    print(model)
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
    