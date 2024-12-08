import torch
import torch.nn as nn

try:
    from .modules import ConvModule, C3k2fBlock
except:
    from  modules import ConvModule, C3k2fBlock


# ---------------------------- YOLO11 Backbone ----------------------------
class Yolo11Backbone(nn.Module):
    def __init__(self, cfg):
        super(Yolo11Backbone, self).__init__()
        # ------------------ Basic setting ------------------
        self.model_scale = cfg.model_scale
        self.feat_dims = [int(512 * cfg.width), int(512 * cfg.width), int(512 * cfg.width * cfg.ratio)]
        
        # ------------------ Network setting ------------------
        ## P1/2
        self.layer_1 = ConvModule(3, int(64 * cfg.width), kernel_size=3, stride=2)
        # P2/4
        self.layer_2 = nn.Sequential(
            ConvModule(int(64 * cfg.width), int(128 * cfg.width), kernel_size=3, stride=2),
            C3k2fBlock(in_dim     = int(128 * cfg.width),
                      out_dim    = int(256 * cfg.width),
                      num_blocks = round(2*cfg.depth),
                      shortcut   = True,
                      expansion  = 0.25,
                      use_c3k    = False if self.model_scale in "ns" else True,
                      )
        )
        # P3/8
        self.layer_3 = nn.Sequential(
            ConvModule(int(256 * cfg.width), int(256 * cfg.width), kernel_size=3, stride=2),
            C3k2fBlock(in_dim     = int(256 * cfg.width),
                      out_dim    = int(512 * cfg.width),
                      num_blocks = round(2*cfg.depth),
                      shortcut   = True,
                      expansion  = 0.25,
                      use_c3k    = False if self.model_scale in "ns" else True,
                      )
        )
        # P4/16
        self.layer_4 = nn.Sequential(
            ConvModule(int(512 * cfg.width), int(512 * cfg.width), kernel_size=3, stride=2),
            C3k2fBlock(in_dim     = int(512 * cfg.width),
                      out_dim    = int(512 * cfg.width),
                      num_blocks = round(2*cfg.depth),
                      shortcut   = True,
                      expansion  = 0.5,
                      use_c3k    = True,
                      )
        )
        # P5/32
        self.layer_5 = nn.Sequential(
            ConvModule(int(512 * cfg.width), int(512 * cfg.width * cfg.ratio), kernel_size=3, stride=2),
            C3k2fBlock(in_dim     = int(512 * cfg.width * cfg.ratio),
                      out_dim    = int(512 * cfg.width * cfg.ratio),
                      num_blocks = round(2*cfg.depth),
                      shortcut   = True,
                      expansion  = 0.5,
                      use_c3k    = True,
                      )
        )

        # Initialize all layers
        self.init_weights()
        
    def init_weights(self):
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
            self.width = 0.25
            self.depth = 0.34
            self.ratio = 2.0
            self.model_scale = "n"
            
    cfg = BaseConfig()
    model = Yolo11Backbone(cfg)
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
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
