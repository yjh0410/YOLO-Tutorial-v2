import torch
import torch.nn as nn

try:
    from .modules import ConvModule, C2fBlock
except:
    from  modules import ConvModule, C2fBlock

# IN1K pretrained weight
pretrained_urls = {
    'n': "https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/in1k_pretrained_weight/elandarknet_n_in1k_62.1.pth",
    's': "https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/in1k_pretrained_weight/elandarknet_s_in1k_71.3.pth",
    'm': "https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/in1k_pretrained_weight/elandarknet_m_in1k_75.7.pth",
    'l': "https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/in1k_pretrained_weight/elandarknet_l_in1k_77.3.pth",
    'x': None,
}

# ---------------------------- Basic functions ----------------------------
class Yolov8Backbone(nn.Module):
    def __init__(self, cfg):
        super(Yolov8Backbone, self).__init__()
        # ------------------ Basic setting ------------------
        self.model_scale = cfg.model_scale
        self.feat_dims = [round(64  * cfg.width),
                          round(128 * cfg.width),
                          round(256 * cfg.width),
                          round(512 * cfg.width),
                          round(512 * cfg.width * cfg.ratio)]
        
        # ------------------ Network setting ------------------
        ## P1/2
        self.layer_1 = ConvModule(3, self.feat_dims[0], kernel_size=3, padding=1, stride=2)
        # P2/4
        self.layer_2 = nn.Sequential(
            ConvModule(self.feat_dims[0], self.feat_dims[1], kernel_size=3, padding=1, stride=2),
            C2fBlock(in_dim     = self.feat_dims[1],
                     out_dim    = self.feat_dims[1],
                     num_blocks = round(3*cfg.depth),
                     expansion  = 0.5,
                     shortcut   = True,
                     )
        )
        # P3/8
        self.layer_3 = nn.Sequential(
            ConvModule(self.feat_dims[1], self.feat_dims[2], kernel_size=3, padding=1, stride=2),
            C2fBlock(in_dim     = self.feat_dims[2],
                     out_dim    = self.feat_dims[2],
                     num_blocks = round(6*cfg.depth),
                     expansion  = 0.5,
                     shortcut   = True,
                     )
        )
        # P4/16
        self.layer_4 = nn.Sequential(
            ConvModule(self.feat_dims[2], self.feat_dims[3], kernel_size=3, padding=1, stride=2),
            C2fBlock(in_dim     = self.feat_dims[3],
                     out_dim    = self.feat_dims[3],
                     num_blocks = round(6*cfg.depth),
                     expansion  = 0.5,
                     shortcut   = True,
                     )
        )
        # P5/32
        self.layer_5 = nn.Sequential(
            ConvModule(self.feat_dims[3], self.feat_dims[4], kernel_size=3, padding=1, stride=2),
            C2fBlock(in_dim     = self.feat_dims[4],
                     out_dim    = self.feat_dims[4],
                     num_blocks = round(3*cfg.depth),
                     expansion  = 0.5,
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

    # YOLOv8 config
    class BaseConfig(object):
        def __init__(self) -> None:
            self.use_pretrained = False
            self.width = 0.50
            self.depth = 0.34
            self.ratio = 2.00
            self.model_scale = "s"
    cfg = BaseConfig()

    # Build backbone
    model = Yolov8Backbone(cfg)

    # Randomly generate a input data
    x = torch.randn(2, 3, 640, 640)

    # Inference
    outputs = model(x)
    print(' - the shape of input :  ', x.shape)
    for out in outputs:
        print(' - the shape of output : ', out.shape)

    x = torch.randn(1, 3, 640, 640)
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('============== FLOPs & Params ================')
    print(' - FLOPs  : {:.2f} G'.format(flops / 1e9 * 2))
    print(' - Params : {:.2f} M'.format(params / 1e6))
