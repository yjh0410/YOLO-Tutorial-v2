import torch
import torch.nn as nn

try:
    from .resnet import build_resnet
except:
    from  resnet import build_resnet


# --------------------- Yolov1's Backbone -----------------------
class Yolov1Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone, self.feat_dim = build_resnet(cfg.backbone, cfg.use_pretrained)

    def forward(self, x):
        c5 = self.backbone(x)

        return c5


if __name__=='__main__':
    import time
    from thop import profile
    # YOLOv8-Base config
    class Yolov1BaseConfig(object):
        def __init__(self) -> None:
            # ---------------- Model config ----------------
            self.out_stride = 32
            self.max_stride = 32
            ## Backbone
            self.backbone       = 'resnet18'
            self.use_pretrained = True

    cfg = Yolov1BaseConfig()
    # Build backbone
    model = Yolov1Backbone(cfg)

    # Inference
    x = torch.randn(1, 3, 640, 640)
    t0 = time.time()
    output = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
    print(output.shape)

    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))    