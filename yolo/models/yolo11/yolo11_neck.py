import torch
import torch.nn as nn

try:
    from .modules import ConvModule, PSABlock
except:
    from  modules import ConvModule, PSABlock


class SPPF(nn.Module):
    def __init__(self, in_dim, out_dim, spp_pooling_size: int = 5, neck_expand_ratio:float = 0.5):
        super().__init__()
        ## ----------- Basic Parameters -----------
        inter_dim = round(in_dim * neck_expand_ratio)
        self.out_dim = out_dim
        ## ----------- Network Parameters -----------
        self.cv1 = ConvModule(in_dim, inter_dim, kernel_size=1, stride=1)
        self.cv2 = ConvModule(inter_dim * 4, out_dim, kernel_size=1, stride=1)
        self.m = nn.MaxPool2d(kernel_size=spp_pooling_size, stride=1, padding=spp_pooling_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)

        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
    
class C2PSA(nn.Module):
    def __init__(self, in_dim, out_dim, num_blocks=1, expansion=0.5):
        super().__init__()
        assert in_dim == out_dim
        inter_dim = int(in_dim * expansion)
        self.cv1 = ConvModule(in_dim, 2 * inter_dim, kernel_size=1)
        self.cv2 = ConvModule(2 * inter_dim, in_dim, kernel_size=1)
        self.m = nn.Sequential(*[
            PSABlock(in_dim     = inter_dim,
                     attn_ratio = 0.5,
                     num_heads  = inter_dim // 64
                     ) for _ in range(num_blocks)])

    def forward(self, x):
        x1, x2 = torch.chunk(self.cv1(x), chunks=2, dim=1)
        x2 = self.m(x2)

        return self.cv2(torch.cat([x1, x2], dim=1))
