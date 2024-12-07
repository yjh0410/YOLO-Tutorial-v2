import torch
import torch.nn as nn

try:
    from .modules import ConvModule
except:
    from  modules import ConvModule


# Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv7-AF by Glenn Jocher
class SPPF(nn.Module):
    """
        This code referenced to https://github.com/ultralytics/yolov7-AF
    """
    def __init__(self, in_dim, out_dim, expansion=0.5):
        super().__init__()
        ## ----------- Basic Parameters -----------
        inter_dim = int(in_dim * expansion)
        self.out_dim = out_dim
        ## ----------- Network Parameters -----------
        self.cv1 = ConvModule(in_dim, inter_dim, kernel_size=1)
        self.cv2 = ConvModule(inter_dim * 4, out_dim, kernel_size=1,)
        self.m = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

        # Initialize all layers
        self.init_weights()

    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.reset_parameters()

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)

        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

# SPPF block with CSP module
class SPPFBlockCSP(nn.Module):
    """
        CSP Spatial Pyramid Pooling Block
    """
    def __init__(self, in_dim, out_dim):
        super(SPPFBlockCSP, self).__init__()
        inter_dim = in_dim // 2
        self.out_dim = out_dim
        self.cv1 = ConvModule(in_dim, inter_dim, kernel_size=1)
        self.cv2 = ConvModule(in_dim, inter_dim, kernel_size=1)
        self.module = nn.Sequential(
            ConvModule(inter_dim, inter_dim, kernel_size=3, padding=1),
            SPPF(inter_dim, inter_dim, expansion=1.0),
            ConvModule(inter_dim, inter_dim, kernel_size=3, padding=1),
            )
        self.cv3 = ConvModule(inter_dim * 2, self.out_dim, kernel_size=1)

        # Initialize all layers
        self.init_weights()

    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.reset_parameters()

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.module(self.cv2(x))
        y = self.cv3(torch.cat([x1, x2], dim=1))

        return y


if __name__=='__main__':
    import time
    from thop import profile
    # Model config
    
    # YOLOv7-AF-Base config
    class Yolov7AFBaseConfig(object):
        def __init__(self) -> None:
            # ---------------- Model config ----------------
            self.out_stride = 32
            self.max_stride = 32
            ## Neck
            self.neck_expand_ratio = 0.5
            self.spp_pooling_size  = 5

    cfg = Yolov7AFBaseConfig()
    # Build a head
    in_dim  = 512
    out_dim = 512
    neck = SPPF(in_dim, out_dim)

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
