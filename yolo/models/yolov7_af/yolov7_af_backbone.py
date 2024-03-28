import torch
import torch.nn as nn

try:
    from .yolov7_af_basic import BasicConv, MDown, ELANLayer
except:
    from  yolov7_af_basic import BasicConv, MDown, ELANLayer


# ELANNet
class Yolov7TBackbone(nn.Module):
    def __init__(self, cfg):
        super(Yolov7TBackbone, self).__init__()
        # ---------------- Basic parameters ----------------
        self.model_scale = cfg.scale
        self.bk_act = cfg.bk_act
        self.bk_norm = cfg.bk_norm
        self.bk_depthwise = cfg.bk_depthwise
        self.elan_depth = 1
        self.feat_dims = [round(64  * cfg.width), round(128  * cfg.width),
                          round(256  * cfg.width), round(512 * cfg.width), round(1024 * cfg.width)]

        # ---------------- Model parameters ----------------
        self.layer_1 = self.make_stem(3, self.feat_dims[0])
        self.layer_2 = self.make_block(self.feat_dims[0], self.feat_dims[1], expansion=0.5)
        self.layer_3 = self.make_block(self.feat_dims[1], self.feat_dims[2], expansion=0.5)
        self.layer_4 = self.make_block(self.feat_dims[2], self.feat_dims[3], expansion=0.5)
        self.layer_5 = self.make_block(self.feat_dims[3], self.feat_dims[4], expansion=0.5)

        # Initialize all layers
        self.init_weights()
        
    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()

    def make_stem(self, in_dim, out_dim):
        stem = BasicConv(in_dim, out_dim, kernel_size=6, padding=2, stride=2,
                         act_type=self.bk_act, norm_type=self.bk_norm, depthwise=self.bk_depthwise)
        
        return stem

    def make_block(self, in_dim, out_dim, expansion=0.5):
        block = nn.Sequential(
            nn.MaxPool2d((2, 2), stride=2),             
            ELANLayer(in_dim, out_dim,
                    expansion=expansion, num_blocks=self.elan_depth,
                    act_type=self.bk_act, norm_type=self.bk_norm, depthwise=self.bk_depthwise),
                    )

        return block
    
    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)
        outputs = [c3, c4, c5]

        return outputs


class Yolov7LBackbone(nn.Module):
    def __init__(self, cfg):
        super(Yolov7LBackbone, self).__init__()
        # ---------------- Basic parameters ----------------
        self.model_scale = cfg.scale
        self.bk_act = cfg.bk_act
        self.bk_norm = cfg.bk_norm
        self.bk_depthwise = cfg.bk_depthwise
        self.elan_depth = 2
        self.feat_dims = [round(64  * cfg.width), round(128  * cfg.width), round(256  * cfg.width),
                          round(512  * cfg.width), round(1024 * cfg.width), round(1024 * cfg.width)]

        # ---------------- Model parameters ----------------
        self.layer_1 = self.make_stem(3, self.feat_dims[0])
        self.layer_2 = self.make_block(self.feat_dims[0], self.feat_dims[1], self.feat_dims[2], expansion=0.5, conv_downsample=True)
        self.layer_3 = self.make_block(self.feat_dims[2], self.feat_dims[2], self.feat_dims[3], expansion=0.5)
        self.layer_4 = self.make_block(self.feat_dims[3], self.feat_dims[3], self.feat_dims[4], expansion=0.5)
        self.layer_5 = self.make_block(self.feat_dims[4], self.feat_dims[4], self.feat_dims[5], expansion=0.25)

        # Initialize all layers
        self.init_weights()
        
    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()

    def make_stem(self, in_dim, out_dim):
        stem = nn.Sequential(
            BasicConv(in_dim, out_dim//2, kernel_size=3, padding=1, stride=1,
                        act_type=self.bk_act, norm_type=self.bk_norm, depthwise=self.bk_depthwise),
            BasicConv(out_dim//2, out_dim, kernel_size=3, padding=1, stride=2,
                        act_type=self.bk_act, norm_type=self.bk_norm, depthwise=self.bk_depthwise),
            BasicConv(out_dim, out_dim, kernel_size=3, padding=1, stride=1,
                        act_type=self.bk_act, norm_type=self.bk_norm, depthwise=self.bk_depthwise)

        )

        return stem

    def make_block(self, in_dim, out_dim_1, out_dim_2, expansion=0.5, conv_downsample=False):
        if conv_downsample:
            block = nn.Sequential(
                BasicConv(in_dim, out_dim_1, kernel_size=3, padding=1, stride=2,
                         act_type=self.bk_act, norm_type=self.bk_norm, depthwise=self.bk_depthwise),             
                ELANLayer(out_dim_1, out_dim_2,
                        expansion=expansion, num_blocks=self.elan_depth,
                        act_type=self.bk_act, norm_type=self.bk_norm, depthwise=self.bk_depthwise),
        )
        else:
            block = nn.Sequential(
                MDown(in_dim, out_dim_1,
                    act_type=self.bk_act, norm_type=self.bk_norm, depthwise=self.bk_depthwise),             
                ELANLayer(out_dim_1, out_dim_2,
                        expansion=expansion, num_blocks=self.elan_depth,
                        act_type=self.bk_act, norm_type=self.bk_norm, depthwise=self.bk_depthwise),
            )
        
        return block
    
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
            self.bk_act = 'silu'
            self.bk_norm = 'BN'
            self.bk_depthwise = False
            self.width = 1.0
            self.depth = 0.34
            self.scale = "l"

    cfg = BaseConfig()
    model = Yolov7LBackbone(cfg)
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