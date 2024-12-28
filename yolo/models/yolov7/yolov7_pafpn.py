import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .modules import ConvModule, ELANBlockFPN, DownSample
except:
    from  modules import ConvModule, ELANBlockFPN, DownSample


# PaFPN-ELAN (YOLOv7's)
class Yolov7PaFPN(nn.Module):
    def __init__(self, 
                 in_dims = [512, 1024, 512],
                 head_dim = 256,
                 ):
        super(Yolov7PaFPN, self).__init__()
        # ----------------------------- Basic parameters -----------------------------
        self.in_dims = in_dims
        self.head_dim = head_dim
        self.fpn_out_dims = [head_dim] * 3

        self.branch_width = 4
        self.branch_depth = 1

        c3, c4, c5 = self.in_dims

        # ----------------------------- Top-down FPN -----------------------------
        ## P5 -> P4
        self.reduce_layer_1 = ConvModule(c5, 256, kernel_size=1)
        self.reduce_layer_2 = ConvModule(c4, 256, kernel_size=1)
        self.top_down_layer_1 = ELANBlockFPN(in_dim = 256 + 256,
                                             out_dim = 256,
                                             expansion = 0.5,
                                             branch_width = self.branch_width,
                                             branch_depth = self.branch_depth,
                                             )
        ## P4 -> P3
        self.reduce_layer_3 = ConvModule(256, 128, kernel_size=1)
        self.reduce_layer_4 = ConvModule(c3, 128, kernel_size=1)
        self.top_down_layer_2 = ELANBlockFPN(in_dim = 128 + 128,
                                             out_dim = 128,
                                             expansion = 0.5,
                                             branch_width = self.branch_width,
                                             branch_depth = self.branch_depth,
                                             )
        # ----------------------------- Bottom-up FPN -----------------------------
        ## P3 -> P4
        self.downsample_layer_1 = DownSample(128, 256)
        self.bottom_up_layer_1 = ELANBlockFPN(in_dim = 256 + 256,
                                              out_dim = 256,
                                              expansion = 0.5,
                                              branch_width = self.branch_width,
                                              branch_depth = self.branch_depth,
                                              )
        ## P4 -> P5
        self.downsample_layer_2 = DownSample(256, 512)
        self.bottom_up_layer_2 = ELANBlockFPN(in_dim = 512 + c5,
                                              out_dim = 512,
                                              expansion = 0.5,
                                              branch_width = self.branch_width,
                                              branch_depth = self.branch_depth,
                                              )

        ## Head convs
        self.head_conv_1 = ConvModule(128, 256, kernel_size=3, stride=1)
        self.head_conv_2 = ConvModule(256, 512, kernel_size=3, stride=1)
        self.head_conv_3 = ConvModule(512, 1024, kernel_size=3, stride=1)

        ## Output projs
        self.out_layers = nn.ModuleList([ConvModule(in_dim, head_dim, kernel_size=1)
                                         for in_dim in [256, 512, 1024]
                                         ])

    def forward(self, features):
        c3, c4, c5 = features

        # Top down
        ## P5 -> P4
        c6 = self.reduce_layer_1(c5)
        c7 = F.interpolate(c6, scale_factor=2.0)
        c8 = torch.cat([c7, self.reduce_layer_2(c4)], dim=1)
        c9 = self.top_down_layer_1(c8)
        ## P4 -> P3
        c10 = self.reduce_layer_3(c9)
        c11 = F.interpolate(c10, scale_factor=2.0)
        c12 = torch.cat([c11, self.reduce_layer_4(c3)], dim=1)
        c13 = self.top_down_layer_2(c12)

        # Bottom up
        ## p3 -> P4
        c14 = self.downsample_layer_1(c13)
        c15 = torch.cat([c14, c9], dim=1)
        c16 = self.bottom_up_layer_1(c15)
        ## P4 -> P5
        c17 = self.downsample_layer_2(c16)
        c18 = torch.cat([c17, c5], dim=1)
        c19 = self.bottom_up_layer_2(c18)

        c20 = self.head_conv_1(c13)
        c21 = self.head_conv_2(c16)
        c22 = self.head_conv_3(c19)
        out_feats = [c20, c21, c22] # [P3, P4, P5]
        
        # output proj layers
        out_feats_proj = []
        for feat, layer in zip(out_feats, self.out_layers):
            out_feats_proj.append(layer(feat))
        return out_feats_proj


if __name__=='__main__':
    import time
    from thop import profile
    # Model config
    
    # Build a head
    in_dims  = [128, 256, 512]
    fpn = Yolov7PaFPN(in_dims, head_dim=256)

    # Randomly generate a input data
    x = [torch.randn(1, in_dims[0], 80, 80),
         torch.randn(1, in_dims[1], 40, 40),
         torch.randn(1, in_dims[2], 20, 20)]
    
    # Inference
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
