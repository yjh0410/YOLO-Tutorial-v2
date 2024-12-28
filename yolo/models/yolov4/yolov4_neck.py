import torch
import torch.nn as nn

try:
    from .modules import ConvModule
except:
    from  modules import ConvModule


class SPPF(nn.Module):
    """
        This code referenced to https://github.com/ultralytics/yolov5
    """
    def __init__(self, in_dim, out_dim, expand_ratio=0.5):
        super().__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.out_dim = out_dim
        self.cv1 = ConvModule(in_dim, inter_dim, kernel_size=1)
        self.cv2 = ConvModule(inter_dim * 4, out_dim, kernel_size=1)
        self.m = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)

        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class SPPFBlockCSP(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 expand_ratio: float = 0.5,
                 ):
        super(SPPFBlockCSP, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.out_dim = out_dim
        self.cv1 = ConvModule(in_dim, inter_dim, kernel_size=1)
        self.cv2 = ConvModule(in_dim, inter_dim, kernel_size=1)
        self.m = nn.Sequential(
            ConvModule(inter_dim, inter_dim, kernel_size=3),
            SPPF(inter_dim, inter_dim, expand_ratio=1.0),
            ConvModule(inter_dim, inter_dim, kernel_size=3)
        )
        self.cv3 = ConvModule(inter_dim * 2, self.out_dim, kernel_size=1)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m(x2)
        y = self.cv3(torch.cat([x1, x3], dim=1))

        return y


if __name__=='__main__':
    from thop import profile
    
    # Build a neck
    in_dim  = 512
    out_dim = 512
    model = SPPFBlockCSP(512, 512, expand_ratio=0.5)

    # Randomly generate a input data
    x = torch.randn(2, in_dim, 20, 20)

    # Inference
    output = model(x)
    print(' - the shape of input :  ', x.shape)
    print(' - the shape of output : ', output.shape)

    x = torch.randn(1, in_dim, 20, 20)
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('============== FLOPs & Params ================')
    print(' - FLOPs  : {:.2f} G'.format(flops / 1e9 * 2))
    print(' - Params : {:.2f} M'.format(params / 1e6))
