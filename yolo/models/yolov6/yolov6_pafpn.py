from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .modules import ConvModule, RepBlock, RepCSPBlock
except:
    from  modules import ConvModule, RepBlock, RepCSPBlock


# Yolov6FPN
class Yolov6PaFPN(nn.Module):
    def __init__(self, cfg, in_dims: List = [256, 512, 1024]):
        super(Yolov6PaFPN, self).__init__()
        self.in_dims = in_dims
        self.model_scale = cfg.scale
        c3, c4, c5 = in_dims

        # ---------------------- Yolov6's Top down FPN ----------------------
        ## P5 -> P4
        self.reduce_layer_1   = ConvModule(c5, round(256*cfg.width),
                                          kernel_size=1, padding=0, stride=1,
                                          act_type=cfg.fpn_act, norm_type=cfg.fpn_norm)
        self.top_down_layer_1 = self.make_block(in_dim     = c4 + round(256*cfg.width),
                                                out_dim    = round(256*cfg.width),
                                                num_blocks = round(12*cfg.depth))

        ## P4 -> P3
        self.reduce_layer_2   = ConvModule(round(256*cfg.width), round(128*cfg.width),
                                          kernel_size=1, padding=0, stride=1,
                                          act_type=cfg.fpn_act, norm_type=cfg.fpn_norm)
        self.top_down_layer_2 = self.make_block(in_dim     = c3 + round(128*cfg.width),
                                                out_dim    = round(128*cfg.width),
                                                num_blocks = round(12*cfg.depth))
        
        # ---------------------- Yolov6's Bottom up PAN ----------------------
        ## P3 -> P4
        self.downsample_layer_1 = ConvModule(round(128*cfg.width), round(128*cfg.width),
                                            kernel_size=3, padding=1, stride=2,
                                            act_type=cfg.fpn_act, norm_type=cfg.fpn_norm, depthwise=cfg.fpn_depthwise)
        self.bottom_up_layer_1  = self.make_block(in_dim     = round(128*cfg.width) + round(128*cfg.width),
                                                  out_dim    = round(256*cfg.width),
                                                  num_blocks = round(12*cfg.depth))

        ## P4 -> P5
        self.downsample_layer_2 = ConvModule(round(256*cfg.width), round(256*cfg.width),
                                            kernel_size=3, padding=1, stride=2,
                                            act_type=cfg.fpn_act, norm_type=cfg.fpn_norm, depthwise=cfg.fpn_depthwise)
        self.bottom_up_layer_2  = self.make_block(in_dim     = round(256*cfg.width) + round(256*cfg.width),
                                                  out_dim    = round(512*cfg.width),
                                                  num_blocks = round(12*cfg.depth))

        # ---------------------- Yolov6's output projection ----------------------
        self.out_layers = nn.ModuleList([
            ConvModule(in_dim, in_dim, kernel_size=1,
                      act_type=cfg.fpn_act, norm_type=cfg.fpn_norm)
                      for in_dim in [round(128*cfg.width), round(256*cfg.width), round(512*cfg.width)]
                      ])
        self.out_dims = [round(128*cfg.width), round(256*cfg.width), round(512*cfg.width)]

    def make_block(self, in_dim, out_dim, num_blocks=1):
        if   self.model_scale in ["n", "s"]:
            block = RepBlock(in_channels  = in_dim,
                             out_channels = out_dim,
                             num_blocks   = num_blocks)
        elif self.model_scale in ["m", "l"]:
            block = RepCSPBlock(in_channels  = in_dim,
                                out_channels = out_dim,
                                num_blocks   = num_blocks,
                                expansion    = 0.5)
        else:
            raise NotImplementedError("Unknown model scale: {}".format(self.model_scale))
            
        return block        
    def forward(self, features):
        c3, c4, c5 = features
        
        # ------------------ Top down FPN ------------------
        ## P5 -> P4
        p5 = self.reduce_layer_1(c5)
        p5_up = F.interpolate(p5, scale_factor=2.0)
        p4 = self.top_down_layer_1(torch.cat([c4, p5_up], dim=1))

        ## P4 -> P3
        p4 = self.reduce_layer_2(p4)
        p4_up = F.interpolate(p4, scale_factor=2.0)
        p3 = self.top_down_layer_2(torch.cat([c3, p4_up], dim=1))

        # ------------------ Bottom up PAN ------------------
        ## P3 -> P4
        p3_ds = self.downsample_layer_1(p3)
        p4 = self.bottom_up_layer_1(torch.cat([p4, p3_ds], dim=1))

        ## P4 -> P5
        p4_ds = self.downsample_layer_2(p4)
        p5 = self.bottom_up_layer_2(torch.cat([p5, p4_ds], dim=1))

        out_feats = [p3, p4, p5]
            
        # output proj layers
        out_feats_proj = []
        for feat, layer in zip(out_feats, self.out_layers):
            out_feats_proj.append(layer(feat))
            
        return out_feats_proj


if __name__=='__main__':
    import time
    from thop import profile
    # Model config
    
    # YOLOv2-Base config
    class Yolov3BaseConfig(object):
        def __init__(self) -> None:
            # ---------------- Model config ----------------
            self.width    = 0.50
            self.depth    = 0.34
            self.out_stride = [8, 16, 32]
            self.max_stride = 32
            self.num_levels = 3
            ## FPN
            self.fpn_act  = 'silu'
            self.fpn_norm = 'BN'
            self.fpn_depthwise = False

    cfg = Yolov3BaseConfig()
    # Build a head
    in_dims  = [128, 256, 512]
    fpn = Yolov6PaFPN(cfg, in_dims)

    # Inference
    x = [torch.randn(1, in_dims[0], 80, 80),
         torch.randn(1, in_dims[1], 40, 40),
         torch.randn(1, in_dims[2], 20, 20)]
    t0 = time.time()
    output = fpn(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
    print('====== FPN output ====== ')
    for level, feat in enumerate(output):
        print("- Level-{} : ".format(level), feat.shape)

    flops, params = profile(fpn, inputs=(x, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
