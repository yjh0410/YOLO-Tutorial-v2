import torch
import torch.nn as nn

try:
    from .yolov8_basic import BasicConv, ELANLayer
except:
    from  yolov8_basic import BasicConv, ELANLayer


# IN1K pretrained weight
pretrained_urls = {
    'n': "https://github.com/yjh0410/ICLab/releases/download/in1k_pretrained/rtcnet_n_in1k_62.1.pth",
    's': "https://github.com/yjh0410/ICLab/releases/download/in1k_pretrained/rtcnet_s_in1k_71.3.pth",
    'm': None,
    'l': None,
    'x': None,
}


# ---------------------------- Basic functions ----------------------------
class Yolov8Backbone(nn.Module):
    def __init__(self, cfg):
        super(Yolov8Backbone, self).__init__()
        # ------------------ Basic setting ------------------
        self.model_scale = cfg.scale
        self.feat_dims = [round(64  * cfg.width),
                          round(128 * cfg.width),
                          round(256 * cfg.width),
                          round(512 * cfg.width),
                          round(512 * cfg.width * cfg.ratio)]
        
        # ------------------ Network setting ------------------
        ## P1/2
        self.layer_1 = BasicConv(3, self.feat_dims[0],
                                 kernel_size=3, padding=1, stride=2,
                                 act_type=cfg.bk_act, norm_type=cfg.bk_norm, depthwise=cfg.bk_depthwise)
        # P2/4
        self.layer_2 = nn.Sequential(
            BasicConv(self.feat_dims[0], self.feat_dims[1],
                      kernel_size=3, padding=1, stride=2,
                      act_type=cfg.bk_act, norm_type=cfg.bk_norm, depthwise=cfg.bk_depthwise),
            ELANLayer(in_dim     = self.feat_dims[1],
                      out_dim    = self.feat_dims[1],
                      num_blocks = round(3*cfg.depth),
                      expansion  = 0.5,
                      shortcut   = True,
                      act_type   = cfg.bk_act,
                      norm_type  = cfg.bk_norm,
                      depthwise  = cfg.bk_depthwise)
        )
        # P3/8
        self.layer_3 = nn.Sequential(
            BasicConv(self.feat_dims[1], self.feat_dims[2],
                      kernel_size=3, padding=1, stride=2,
                      act_type=cfg.bk_act, norm_type=cfg.bk_norm, depthwise=cfg.bk_depthwise),
            ELANLayer(in_dim     = self.feat_dims[2],
                      out_dim    = self.feat_dims[2],
                      num_blocks = round(6*cfg.depth),
                      expansion  = 0.5,
                      shortcut   = True,
                      act_type   = cfg.bk_act,
                      norm_type  = cfg.bk_norm,
                      depthwise  = cfg.bk_depthwise)
        )
        # P4/16
        self.layer_4 = nn.Sequential(
            BasicConv(self.feat_dims[2], self.feat_dims[3],
                      kernel_size=3, padding=1, stride=2,
                      act_type=cfg.bk_act, norm_type=cfg.bk_norm, depthwise=cfg.bk_depthwise),
            ELANLayer(in_dim     = self.feat_dims[3],
                      out_dim    = self.feat_dims[3],
                      num_blocks = round(6*cfg.depth),
                      expansion  = 0.5,
                      shortcut   = True,
                      act_type   = cfg.bk_act,
                      norm_type  = cfg.bk_norm,
                      depthwise  = cfg.bk_depthwise)
        )
        # P5/32
        self.layer_5 = nn.Sequential(
            BasicConv(self.feat_dims[3], self.feat_dims[4],
                      kernel_size=3, padding=1, stride=2,
                      act_type=cfg.bk_act, norm_type=cfg.bk_norm, depthwise=cfg.bk_depthwise),
            ELANLayer(in_dim     = self.feat_dims[4],
                      out_dim    = self.feat_dims[4],
                      num_blocks = round(3*cfg.depth),
                      expansion  = 0.5,
                      shortcut   = True,
                      act_type   = cfg.bk_act,
                      norm_type  = cfg.bk_norm,
                      depthwise  = cfg.bk_depthwise)
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
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
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


# ---------------------------- Functions ----------------------------
## build Yolo's Backbone
def build_backbone(cfg): 
    # model
    backbone = Yolov8Backbone(cfg)
        
    return backbone


if __name__ == '__main__':
    import time
    from thop import profile
    class BaseConfig(object):
        def __init__(self) -> None:
            self.bk_act = 'silu'
            self.bk_norm = 'BN'
            self.bk_depthwise = False
            self.width = 1.0
            self.depth = 1.0
            self.ratio = 1.0
            self.scale = "n"
            self.use_pretrained = True

    cfg = BaseConfig()
    model = build_backbone(cfg)
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