import torch
import torch.nn as nn

try:
    from .yolov8_basic import BasicConv
except:
    from  yolov8_basic import BasicConv
    

# Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
class SPPF(nn.Module):
    """
        This code referenced to https://github.com/ultralytics/yolov5
    """
    def __init__(self, cfg, in_dim, out_dim):
        super().__init__()
        ## ----------- Basic Parameters -----------
        inter_dim = round(in_dim * cfg.neck_expand_ratio)
        self.out_dim = out_dim
        ## ----------- Network Parameters -----------
        self.cv1 = BasicConv(in_dim, inter_dim,
                             kernel_size=1, padding=0, stride=1,
                             act_type=cfg.neck_act, norm_type=cfg.neck_norm)
        self.cv2 = BasicConv(inter_dim * 4, out_dim,
                             kernel_size=1, padding=0, stride=1,
                             act_type=cfg.neck_act, norm_type=cfg.neck_norm)
        self.m = nn.MaxPool2d(kernel_size=cfg.spp_pooling_size,
                              stride=1,
                              padding=cfg.spp_pooling_size // 2)

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
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)

        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


if __name__=='__main__':
    import time
    from thop import profile
    # Model config
    
    # YOLOv8-Base config
    class Yolov8BaseConfig(object):
        def __init__(self) -> None:
            # ---------------- Model config ----------------
            self.out_stride = 32
            self.max_stride = 32
            ## Neck
            self.neck_act       = 'lrelu'
            self.neck_norm      = 'BN'
            self.neck_depthwise = False
            self.neck_expand_ratio = 0.5
            self.spp_pooling_size  = 5

    cfg = Yolov8BaseConfig()
    # Build a head
    in_dim  = 512
    out_dim = 512
    neck = SPPF(cfg, in_dim, out_dim)

    # Inference
    x = torch.randn(1, in_dim, 20, 20)
    t0 = time.time()
    output = neck(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
    print('Neck output: ', output.shape)

    flops, params = profile(neck, inputs=(x, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
