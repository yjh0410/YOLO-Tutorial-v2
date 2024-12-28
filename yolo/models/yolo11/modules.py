import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ----------------- CNN modules -----------------
class ConvModule(nn.Module):
    def __init__(self, 
                 in_dim,        # in channels
                 out_dim,       # out channels 
                 kernel_size=1, # kernel size 
                 stride=1,      # padding
                 groups=1,      # groups
                 use_act: bool = True,
                ):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, padding=kernel_size//2, stride=stride, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(out_dim)
        self.act = nn.SiLU(inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self,
                 in_dim      :int,
                 out_dim     :int,
                 kernel_size :List = [3, 3],
                 shortcut    :bool = False,
                 expansion   :float = 0.5,
                 ) -> None:
        super(Bottleneck, self).__init__()
        # ----------------- Network setting -----------------
        inter_dim = int(out_dim * expansion)
        self.cv1 = ConvModule(in_dim,  inter_dim, kernel_size=kernel_size[0], stride=1)
        self.cv2 = ConvModule(inter_dim, out_dim, kernel_size=kernel_size[1], stride=1)
        self.shortcut = shortcut and in_dim == out_dim

    def forward(self, x):
        h = self.cv2(self.cv1(x))

        return x + h if self.shortcut else h

class C3kBlock(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 num_blocks: int = 1,
                 shortcut: bool = True,
                 expansion: float = 0.5,
                 ):
        super().__init__()
        inter_dim = int(out_dim * expansion)  # hidden channels
        self.cv1 = ConvModule(in_dim, inter_dim, kernel_size=1)
        self.cv2 = ConvModule(in_dim, inter_dim, kernel_size=1)
        self.cv3 = ConvModule(2 * inter_dim, out_dim, kernel_size=1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*[
            Bottleneck(in_dim      = inter_dim,
                       out_dim     = inter_dim,
                       kernel_size = [3, 3],
                       shortcut    = shortcut,
                       expansion   = 1.0,
                       ) for _ in range(num_blocks)])

    def forward(self, x):
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], dim=1))

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
    

# ----------------- Attention modules  -----------------
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv  = ConvModule(dim, h, kernel_size=1, use_act=False)
        self.proj = ConvModule(dim, dim, kernel_size=1, use_act=False)
        self.pe   = ConvModule(dim, dim, kernel_size=3, groups=dim, use_act=False)

    def forward(self, x):
        bs, c, h, w = x.shape
        seq_len = h * w

        qkv = self.qkv(x)
        q, k, v = qkv.view(bs, self.num_heads, self.key_dim * 2 + self.head_dim, seq_len).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(bs, c, h, w) + self.pe(v.reshape(bs, c, h, w))
        x = self.proj(x)

        return x

class PSABlock(nn.Module):
    def __init__(self, in_dim, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(in_dim, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(ConvModule(in_dim, in_dim * 2, kernel_size=1),
                                 ConvModule(in_dim * 2, in_dim, kernel_size=1, use_act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x)  if self.add else self.ffn(x)
        return x

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


# ----------------- YOLO11 components -----------------
class YoloStage(nn.Module):
    def __init__(self, in_dim, out_dim, num_blocks=1, use_c3k=True, expansion=0.5, shortcut=True):
        super().__init__()
        inter_dim = int(out_dim * expansion)  # hidden channels
        self.cv1 = ConvModule(in_dim, 2 * inter_dim, kernel_size=1)
        self.cv2 = ConvModule((2 + num_blocks) * inter_dim, out_dim, kernel_size=1)

        if use_c3k:
            self.m = nn.ModuleList(
                C3kBlock(inter_dim, inter_dim, 2, shortcut)
                for _ in range(num_blocks)
            )
        else:
            self.m = nn.ModuleList(
                Bottleneck(inter_dim, inter_dim, [3, 3], shortcut, expansion=0.5)
                for _ in range(num_blocks)
            )

    def _forward_impl(self, x):
        # Input proj
        x1, x2 = torch.chunk(self.cv1(x), 2, dim=1)
        out = list([x1, x2])

        # Bottlenecl
        out.extend(m(out[-1]) for m in self.m)

        # Output proj
        out = self.cv2(torch.cat(out, dim=1))

        return out

    def forward(self, x):
        return self._forward_impl(x)

class DflLayer(nn.Module):
    def __init__(self, reg_max=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.reg_max = reg_max
        proj_init = torch.arange(reg_max, dtype=torch.float)
        self.proj_weight = nn.Parameter(proj_init.view([1, reg_max, 1, 1]), requires_grad=False)

    def forward(self, pred_reg, anchor, stride):
        bs, hw = pred_reg.shape[:2]
        # [bs, hw, 4*rm] -> [bs, 4*rm, hw] -> [bs, 4, rm, hw]
        pred_reg = pred_reg.permute(0, 2, 1).reshape(bs, 4, -1, hw)

        # [bs, 4, rm, hw] -> [bs, rm, 4, hw]
        pred_reg = pred_reg.permute(0, 2, 1, 3).contiguous()

        # [bs, rm, 4, hw] -> [bs, 1, 4, hw]
        delta_pred = F.conv2d(F.softmax(pred_reg, dim=1), self.proj_weight)

        # [bs, 1, 4, hw] -> [bs, 4, hw] -> [bs, hw, 4]
        delta_pred = delta_pred.view(bs, 4, hw).permute(0, 2, 1).contiguous()
        delta_pred *= stride

        # Decode bbox: tlbr -> xyxy
        x1y1_pred = anchor - delta_pred[..., :2]
        x2y2_pred = anchor + delta_pred[..., 2:]
        box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

        return box_pred
