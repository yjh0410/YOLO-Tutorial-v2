import torch
import torch.nn as nn

try:
    from .yolov6_basic import RepBlock, RepVGGBlock, RepCSPBlock
except:
    from  yolov6_basic import RepBlock, RepVGGBlock, RepCSPBlock


# --------------------- Yolov3's Backbone -----------------------
## Modified DarkNet
class Yolov6Backbone(nn.Module):
    def __init__(self, cfg):
        super(Yolov6Backbone, self).__init__()
        # ------------------ Basic setting ------------------
        self.model_scale = cfg.scale
        self.feat_dims = [round(64   * cfg.width),
                          round(128  * cfg.width),
                          round(256  * cfg.width),
                          round(512  * cfg.width),
                          round(1024 * cfg.width)]
        
        # ------------------ Network setting ------------------
        ## P1/2
        self.layer_1 = RepVGGBlock(3, self.feat_dims[0],
                                   kernel_size=3, padding=1, stride=2)
        # P2/4
        self.layer_2 = self.make_block(self.feat_dims[0], self.feat_dims[1], round(6*cfg.depth)) 
        # P3/8
        self.layer_3 = self.make_block(self.feat_dims[1], self.feat_dims[2], round(12*cfg.depth)) 
        # P4/16
        self.layer_4 = self.make_block(self.feat_dims[2], self.feat_dims[3], round(18*cfg.depth)) 
        # P5/32
        self.layer_5 = self.make_block(self.feat_dims[3], self.feat_dims[4], round(6*cfg.depth)) 

        # Initialize all layers
        self.init_weights()
    
    def make_block(self, in_dim, out_dim, num_blocks=1):
        if self.model_scale in ["s", "t", "n"]:
            block = nn.Sequential(
                RepVGGBlock(in_dim, out_dim,
                            kernel_size=3, padding=1, stride=2),
                RepBlock(in_channels  = out_dim,
                         out_channels = out_dim,
                         num_blocks   = num_blocks,
                         block        = RepVGGBlock)
                         )
        elif self.model_scale in ["m", "l", "x"]:
            block = nn.Sequential(
                RepVGGBlock(in_dim, out_dim,
                            kernel_size=3, padding=1, stride=2),
                RepCSPBlock(in_channels  = out_dim,
                            out_channels = out_dim,
                            num_blocks   = num_blocks,
                            expansion    = 0.5)
                            )
        else:
            raise NotImplementedError("Unknown model scale: {}".format(self.model_scale))
            
        return block

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


if __name__ == '__main__':
    import time
    from thop import profile
    class BaseConfig(object):
        def __init__(self) -> None:
            self.bk_depthwise = False
            self.width = 0.50
            self.depth = 0.34
            self.scale = "s"
            self.use_pretrained = True

    cfg = BaseConfig()
    model = Yolov6Backbone(cfg)
    x = torch.randn(1, 3, 640, 640)
    t0 = time.time()
    outputs = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
    for out in outputs:
        print(out.shape)
    
    for m in model.modules():
        if hasattr(m, "switch_to_deploy"):
            m.switch_to_deploy()

    x = torch.randn(1, 3, 640, 640)
    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))