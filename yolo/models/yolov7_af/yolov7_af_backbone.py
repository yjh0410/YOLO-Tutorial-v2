import torch
import torch.nn as nn

try:
    from .yolov7_af_basic import BasicConv, MDown, ELANLayer
except:
    from  yolov7_af_basic import BasicConv, MDown, ELANLayer


# ELANNet
class Yolov7Backbone(nn.Module):
    def __init__(self, cfg):
        super(Yolov7Backbone, self).__init__()
        # ---------------- Basic parameters ----------------
        self.model_scale = cfg.scale
        if self.model_scale in ["l", "x"]:
            self.elan_depth = 2
            self.feat_dims = [round(64   * cfg.width), round(128  * cfg.width), round(256  * cfg.width),
                              round(512  * cfg.width), round(1024 * cfg.width), round(1024 * cfg.width)]
            self.last_stage_eratio = 0.25
        if self.model_scale in ["n", "s"]:
            self.elan_depth = 1
            self.feat_dims = [round(64   * cfg.width), round(64  * cfg.width), round(128  * cfg.width),
                              round(256  * cfg.width), round(512 * cfg.width), round(1024 * cfg.width)]
            self.last_stage_eratio = 0.5

        # ---------------- Model parameters ----------------
        
        # large backbone
        self.layer_1 = nn.Sequential(
            BasicConv(3, self.feat_dims[0]//2, kernel_size=3, padding=1, stride=1,
                      act_type=cfg.bk_act, norm_type=cfg.bk_norm, depthwise=cfg.bk_depthwise),
            BasicConv(self.feat_dims[0]//2, self.feat_dims[0], kernel_size=3, padding=1, stride=2,
                      act_type=cfg.bk_act, norm_type=cfg.bk_norm, depthwise=cfg.bk_depthwise),
            BasicConv(self.feat_dims[0], self.feat_dims[0], kernel_size=3, padding=1, stride=1,
                      act_type=cfg.bk_act, norm_type=cfg.bk_norm, depthwise=cfg.bk_depthwise)
        )
        self.layer_2 = nn.Sequential(   
            BasicConv(self.feat_dims[0], self.feat_dims[1],
                      kernel_size=3, padding=1, stride=2,
                      act_type=cfg.bk_act, norm_type=cfg.bk_norm, depthwise=cfg.bk_depthwise),      
            ELANLayer(self.feat_dims[1], self.feat_dims[2],
                      expansion=0.5, num_blocks=self.elan_depth,
                      act_type=cfg.bk_act, norm_type=cfg.bk_norm, depthwise=cfg.bk_depthwise),      
        )
        self.layer_3 = nn.Sequential(
            MDown(self.feat_dims[2], self.feat_dims[2],
                  act_type=cfg.bk_act, norm_type=cfg.bk_norm, depthwise=cfg.bk_depthwise),             
            ELANLayer(self.feat_dims[2], self.feat_dims[3],
                      expansion=0.5, num_blocks=self.elan_depth,
                      act_type=cfg.bk_act, norm_type=cfg.bk_norm, depthwise=cfg.bk_depthwise),      
        )
        self.layer_4 = nn.Sequential(
            MDown(self.feat_dims[3], self.feat_dims[3],
                  act_type=cfg.bk_act, norm_type=cfg.bk_norm, depthwise=cfg.bk_depthwise),             
            ELANLayer(self.feat_dims[3], self.feat_dims[4],
                      expansion=0.5, num_blocks=self.elan_depth,
                      act_type=cfg.bk_act, norm_type=cfg.bk_norm, depthwise=cfg.bk_depthwise),      
        )
        self.layer_5 = nn.Sequential(
            MDown(self.feat_dims[4], self.feat_dims[4],
                  act_type=cfg.bk_act, norm_type=cfg.bk_norm, depthwise=cfg.bk_depthwise),             
            ELANLayer(self.feat_dims[4], self.feat_dims[5],
                      expansion=self.last_stage_eratio, num_blocks=self.elan_depth,
                      act_type=cfg.bk_act, norm_type=cfg.bk_norm, depthwise=cfg.bk_depthwise),      
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


if __name__ == '__main__':
    import time
    from thop import profile
    class BaseConfig(object):
        def __init__(self) -> None:
            self.bk_act = 'silu'
            self.bk_norm = 'BN'
            self.bk_depthwise = False
            self.width = 0.5
            self.depth = 0.34
            self.scale = "s"

    cfg = BaseConfig()
    model = Yolov7Backbone(cfg)
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