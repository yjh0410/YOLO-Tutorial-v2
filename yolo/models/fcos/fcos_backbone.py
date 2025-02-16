import torch
import torch.nn as nn

try:
    from .resnet import build_resnet
except:
    from  resnet import build_resnet


# --------------------- FCOS's Backbone -----------------------
class FcosBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone, self.feat_dims = build_resnet(cfg.backbone, cfg.use_pretrained)

    def forward(self, x):
        pyramid_feats = self.backbone(x)

        return pyramid_feats # [C3, C4, C5]


if __name__=='__main__':
    from thop import profile

    # YOLOv1 configuration
    class FcosBaseConfig(object):
        def __init__(self) -> None:
            # ---------------- Model config ----------------
            self.out_stride = [8, 16, 32]
            self.max_stride = 32
            ## Backbone
            self.backbone = 'resnet18'
            self.use_pretrained = True
    cfg = FcosBaseConfig()

    # Build backbone
    model = FcosBackbone(cfg)

    # Randomly generate a input data
    x = torch.randn(2, 3, 640, 640)

    # Inference
    outputs = model(x)
    print(' - the shape of input :  ', x.shape)
    for i, out in enumerate(outputs):
        print(f' - the shape of level-{i} output : ', out.shape)

    x = torch.randn(1, 3, 640, 640)
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('============== FLOPs & Params ================')
    print(' - FLOPs  : {:.2f} G'.format(flops / 1e9 * 2))
    print(' - Params : {:.2f} M'.format(params / 1e6))
