from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .modules import ConvModule, ELANLayerFPN, MDown
except:
    from  modules import ConvModule, ELANLayerFPN, MDown


# Yolov7 af PaFPN
class Yolov7PaFPN(nn.Module):
    def __init__(self, cfg, in_dims: List = [512, 1024, 512]):
        super(Yolov7PaFPN, self).__init__()
        # ----------------------------- Basic parameters -----------------------------
        self.in_dims = in_dims
        c3, c4, c5 = in_dims

        # ----------------------------- Yolov7's Top-down FPN -----------------------------
        ## P5 -> P4
        self.reduce_layer_1 = ConvModule(c5, round(256*cfg.width), kernel_size=1)
        self.reduce_layer_2 = ConvModule(c4, round(256*cfg.width), kernel_size=1)
        self.top_down_layer_1 = ELANLayerFPN(in_dim       = round(256*cfg.width) + round(256*cfg.width),
                                             out_dim      = round(256*cfg.width),
                                             expansions   = cfg.fpn_expansions,
                                             branch_width = cfg.fpn_block_bw,
                                             branch_depth = cfg.fpn_block_dw,
                                             )
        ## P4 -> P3
        self.reduce_layer_3 = ConvModule(round(256*cfg.width), round(128*cfg.width), kernel_size=1)
        self.reduce_layer_4 = ConvModule(c3, round(128*cfg.width), kernel_size=1)
        self.top_down_layer_2 = ELANLayerFPN(in_dim       = round(128*cfg.width) + round(128*cfg.width),
                                             out_dim      = round(128*cfg.width),
                                             expansions   = cfg.fpn_expansions,
                                             branch_width = cfg.fpn_block_bw,
                                             branch_depth = cfg.fpn_block_dw,
                                             )
        # ----------------------------- Yolov7's Bottom-up PAN -----------------------------
        ## P3 -> P4
        self.downsample_layer_1 = MDown(round(128*cfg.width), round(256*cfg.width))
        self.bottom_up_layer_1 = ELANLayerFPN(in_dim       = round(256*cfg.width) + round(256*cfg.width),
                                              out_dim      = round(256*cfg.width),
                                              expansions   = cfg.fpn_expansions,
                                              branch_width = cfg.fpn_block_bw,
                                              branch_depth = cfg.fpn_block_dw,
                                              )
        ## P4 -> P5
        self.downsample_layer_2 = MDown(round(256*cfg.width), round(512*cfg.width))
        self.bottom_up_layer_2 = ELANLayerFPN(in_dim       = round(512*cfg.width) + c5,
                                              out_dim      = round(512*cfg.width),
                                              expansions   = cfg.fpn_expansions,
                                              branch_width = cfg.fpn_block_bw,
                                              branch_depth = cfg.fpn_block_dw,
                                              )

        # ----------------------------- Head conv layers -----------------------------
        ## Head convs
        self.head_conv_1 = ConvModule(round(128*cfg.width), round(256*cfg.width), kernel_size=3, padding=1, stride=1)
        self.head_conv_2 = ConvModule(round(256*cfg.width), round(512*cfg.width), kernel_size=3, padding=1, stride=1)
        self.head_conv_3 = ConvModule(round(512*cfg.width), round(1024*cfg.width), kernel_size=3, padding=1, stride=1)

        # ---------------------- Yolox's output projection ----------------------
        self.out_layers = nn.ModuleList([
            ConvModule(in_dim, round(cfg.head_dim*cfg.width), kernel_size=1)
                      for in_dim in [round(256*cfg.width), round(512*cfg.width), round(1024*cfg.width)]
                      ])
        self.out_dims = [round(cfg.head_dim*cfg.width)] * 3

        # Initialize all layers
        self.init_weights()

    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.reset_parameters()

    def forward(self, features):
        c3, c4, c5 = features

        # ------------------ Top down FPN ------------------
        ## P5 -> P4
        p5 = self.reduce_layer_1(c5)
        p5_up = F.interpolate(p5, scale_factor=2.0)
        p4 = self.reduce_layer_2(c4)
        p4 = self.top_down_layer_1(torch.cat([p5_up, p4], dim=1))

        ## P4 -> P3
        p4_in = self.reduce_layer_3(p4)
        p4_up = F.interpolate(p4_in, scale_factor=2.0)
        p3 = self.reduce_layer_4(c3)
        p3 = self.top_down_layer_2(torch.cat([p4_up, p3], dim=1))

        # ------------------ Bottom up PAN ------------------
        ## P3 -> P4
        p3_ds = self.downsample_layer_1(p3)
        p4 = torch.cat([p3_ds, p4], dim=1)
        p4 = self.bottom_up_layer_1(p4)

        ## P4 -> P5
        p4_ds = self.downsample_layer_2(p4)
        p5 = torch.cat([p4_ds, c5], dim=1)
        p5 = self.bottom_up_layer_2(p5)

        out_feats = [self.head_conv_1(p3), self.head_conv_2(p4), self.head_conv_3(p5)]
            
        # output proj layers
        out_feats_proj = []
        for feat, layer in zip(out_feats, self.out_layers):
            out_feats_proj.append(layer(feat))
            
        return out_feats_proj


if __name__=='__main__':
    import time
    from thop import profile
    # Model config
    
    # YOLOv7-Base config
    class Yolov7BaseConfig(object):
        def __init__(self) -> None:
            # ---------------- Model config ----------------
            self.width    = 0.50
            self.depth    = 0.34
            self.out_stride = [8, 16, 32]
            self.max_stride = 32
            self.num_levels = 3
            self.fpn_expansions = [0.5, 0.5]
            self.fpn_block_bw = 4
            self.fpn_block_dw = 1
            ## Head
            self.head_dim = 256

    cfg = Yolov7BaseConfig()
    # Build a head
    in_dims  = [128, 256, 512]
    fpn = Yolov7PaFPN(cfg, in_dims)

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