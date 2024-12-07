import torch
import torch.nn as nn

try:
    from .modules import ConvModule, RepGElanLayer, ADown
except:
    from  modules import ConvModule, RepGElanLayer, ADown

# IN1K pretrained weight
pretrained_urls = {
    's': "https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/in1k_pretrained_weight/gelan_s_in1k_68.4.pth",
    'c': "https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/in1k_pretrained_weight/gelan_c_in1k_76.7.pth",
}

# ----------------- GELAN backbone proposed by YOLOv9 -----------------
class GElanBackbone(nn.Module):
    def __init__(self, cfg):
        super(GElanBackbone, self).__init__()
        # ---------- Basic setting ----------
        self.model_scale = cfg.model_scale
        self.feat_dims = [cfg.backbone_feats["c1"][-1],  # 64
                          cfg.backbone_feats["c2"][-1],  # 128
                          cfg.backbone_feats["c3"][-1],  # 256
                          cfg.backbone_feats["c4"][-1],  # 512
                          cfg.backbone_feats["c5"][-1],  # 512
                          ]
        
        # ---------- Network setting ----------
        ## P1/2
        self.layer_1 = ConvModule(3, cfg.backbone_feats["c1"][0], kernel_size=3, padding=1, stride=2)
        # P2/4
        self.layer_2 = nn.Sequential(
            ConvModule(cfg.backbone_feats["c1"][0], cfg.backbone_feats["c2"][0], kernel_size=3, padding=1, stride=2),
            RepGElanLayer(in_dim     = cfg.backbone_feats["c2"][0],
                          inter_dims = cfg.backbone_feats["c2"][1],
                          out_dim    = cfg.backbone_feats["c2"][2],
                          num_blocks = cfg.backbone_depth,
                          shortcut   = True,
                          )
        )
        # P3/8
        self.layer_3 = nn.Sequential(
            ADown(cfg.backbone_feats["c2"][2], cfg.backbone_feats["c3"][0]),
            RepGElanLayer(in_dim     = cfg.backbone_feats["c3"][0],
                          inter_dims = cfg.backbone_feats["c3"][1],
                          out_dim    = cfg.backbone_feats["c3"][2],
                          num_blocks = cfg.backbone_depth,
                          shortcut   = True,
                          )
        )
        # P4/16
        self.layer_4 = nn.Sequential(
            ADown(cfg.backbone_feats["c3"][2], cfg.backbone_feats["c4"][0]),
            RepGElanLayer(in_dim     = cfg.backbone_feats["c4"][0],
                          inter_dims = cfg.backbone_feats["c4"][1],
                          out_dim    = cfg.backbone_feats["c4"][2],
                          num_blocks = cfg.backbone_depth,
                          shortcut   = True,
                          )
        )
        # P5/32
        self.layer_5 = nn.Sequential(
            ADown(cfg.backbone_feats["c4"][2], cfg.backbone_feats["c5"][0]),
            RepGElanLayer(in_dim     = cfg.backbone_feats["c5"][0],
                          inter_dims = cfg.backbone_feats["c5"][1],
                          out_dim    = cfg.backbone_feats["c5"][2],
                          num_blocks = cfg.backbone_depth,
                          shortcut   = True,
                          )
        )

        # Initialize all layers
        self.init_weights()

        # Load imagenet pretrained weight
        if cfg.use_pretrained:
            self.load_pretrained()

    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.reset_parameters()

    def load_pretrained(self):
        url = pretrained_urls[self.model_scale]
        if url is not None:
            print('Loading backbone pretrained weight from : {}'.format(url))
            # checkpoint state dict
            checkpoint = torch.hub.load_state_dict_from_url(
                url=url, map_location="cpu", check_hash=True)
            checkpoint_state_dict = checkpoint.pop("model")
            # model state dict
            model_state_dict = self.state_dict()
            # check
            for k in list(checkpoint_state_dict.keys()):
                if k in model_state_dict:
                    shape_model = tuple(model_state_dict[k].shape)
                    shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                    if shape_model != shape_checkpoint:
                        checkpoint_state_dict.pop(k)
                else:
                    checkpoint_state_dict.pop(k)
                    print('Unused key: ', k)
            # load the weight
            self.load_state_dict(checkpoint_state_dict)
        else:
            print('No pretrained weight for model scale: {}.'.format(self.model_scale))

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
            self.use_pretrained = True
            # # Gelan-C scale
            # self.backbone_feats = {
            #     "c1": [64],
            #     "c2": [128, [128, 64], 256],
            #     "c3": [256, [256, 128], 512],
            #     "c4": [512, [512, 256], 512],
            #     "c5": [512, [512, 256], 512],
            # }
            # self.model_scale = "c"
            # self.backbone_depth = 1
            # Gelan-S scale
            self.backbone_feats = {
                "c1": [32],
                "c2": [64,  [64, 32],   64],
                "c3": [64,  [64, 32],   128],
                "c4": [128, [128, 64],  256],
                "c5": [256, [256, 128], 256],
            }
            self.model_scale = "s"
            self.backbone_depth = 3
    # 定义模型配置文件
    cfg = BaseConfig()

    # 构建GELAN主干网络
    model = GElanBackbone(cfg)

    # 随机生成输入数据
    x = torch.randn(1, 3, 640, 640)

    # 前向推理
    outputs = model(x)

    # 打印输出中的shape
    for out in outputs:
        print(out.shape)

    # 计算模型的参数量和理论计算量
    print('============ Params & FLOPs ============')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
    