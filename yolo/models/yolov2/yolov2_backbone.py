import torch
import torch.nn as nn

try:
    from .resnet import build_resnet
except:
    from  resnet import build_resnet


# --------------------- Yolov2's Backbone -----------------------
class Yolov2Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone, self.feat_dim = build_resnet(cfg.backbone, cfg.use_pretrained)

    def forward(self, x):
        c5 = self.backbone(x)

        return c5


if __name__=='__main__':
    from thop import profile

    # YOLOv1 configuration
    class Yolov2BaseConfig(object):
        def __init__(self) -> None:
            # ---------------- Model config ----------------
            self.out_stride = 32
            self.max_stride = 32
            ## Backbone
            self.backbone       = 'resnet18'
            self.use_pretrained = True
    cfg = Yolov2BaseConfig()

    # Build backbone
    model = Yolov2Backbone(cfg)

    # Randomly generate a input data
    x = torch.randn(2, 3, 640, 640)

    # Inference
    output = model(x)
    print(' - the shape of input :  ', x.shape)
    print(' - the shape of output : ', output.shape)

    x = torch.randn(1, 3, 640, 640)
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('============== FLOPs & Params ================')
    print(' - FLOPs  : {:.2f} G'.format(flops / 1e9 * 2))
    print(' - Params : {:.2f} M'.format(params / 1e6))