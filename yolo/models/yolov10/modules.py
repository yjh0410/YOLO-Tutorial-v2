import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# --------------------- Basic modules ---------------------
class ConvModule(nn.Module):
    def __init__(self, 
                 in_dim,
                 out_dim,
                 kernel_size=1,
                 stride=1,
                 groups=1,
                 use_act=True,
                ):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(out_dim)
        self.act  = nn.SiLU(inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class YoloBottleneck(nn.Module):
    def __init__(self,
                 in_dim      :int,
                 out_dim     :int,
                 kernel_size :List  = [1, 3],
                 expansion   :float = 0.5,
                 shortcut    :bool  = False,
                 ):
        super(YoloBottleneck, self).__init__()
        inter_dim = int(out_dim * expansion)
        # ----------------- Network setting -----------------
        self.conv_layer1 = ConvModule(in_dim, inter_dim, kernel_size=kernel_size[0], stride=1)
        self.conv_layer2 = ConvModule(inter_dim, out_dim, kernel_size=kernel_size[1], stride=1)
        self.shortcut = shortcut and in_dim == out_dim

    def forward(self, x):
        h = self.conv_layer2(self.conv_layer1(x))

        return x + h if self.shortcut else h

class CIBBlock(nn.Module):
    def __init__(self,
                 in_dim   :int,
                 out_dim  :int,
                 shortcut :bool  = False,
                 ) -> None:
        super(CIBBlock, self).__init__()
        # ----------------- Network setting -----------------
        self.cv1 = ConvModule(in_dim, in_dim, kernel_size=3, groups=in_dim)
        self.cv2 = ConvModule(in_dim, in_dim * 2, kernel_size=1)
        self.cv3 = ConvModule(in_dim * 2, in_dim * 2, kernel_size=3, groups=in_dim * 2)
        self.cv4 = ConvModule(in_dim * 2, out_dim, kernel_size=1)
        self.cv5 = ConvModule(out_dim, out_dim, kernel_size=3, groups=out_dim)
        self.shortcut = shortcut and in_dim == out_dim

    def forward(self, x):
        h = self.cv5(self.cv4(self.cv3(self.cv2(self.cv1(x)))))

        return x + h if self.shortcut else h


# --------------------- Yolov10 modules ---------------------
class C2fBlock(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 expansion : float = 0.5,
                 num_blocks : int = 1,
                 shortcut: bool = False,
                 use_cib: bool = False,
                 ):
        super(C2fBlock, self).__init__()
        inter_dim = round(out_dim * expansion)
        self.input_proj  = ConvModule(in_dim, inter_dim * 2, kernel_size=1)
        self.output_proj = ConvModule((2 + num_blocks) * inter_dim, out_dim, kernel_size=1)

        if use_cib:
            self.blocks = nn.ModuleList([
                CIBBlock(in_dim = inter_dim,
                         out_dim = inter_dim,
                         shortcut = shortcut,
                         ) for _ in range(num_blocks)])
        else:
            self.blocks = nn.ModuleList([
                YoloBottleneck(in_dim = inter_dim,
                               out_dim = inter_dim,
                               kernel_size = [3, 3],
                               expansion = 1.0,
                               shortcut = shortcut,
                               ) for _ in range(num_blocks)])

    def forward(self, x):
        # Input proj
        x1, x2 = torch.chunk(self.input_proj(x), 2, dim=1)
        out = list([x1, x2])

        # Bottlenecl
        out.extend(m(out[-1]) for m in self.blocks)

        # Output proj
        out = self.output_proj(torch.cat(out, dim=1))

        return out

class SCDown(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        self.cv1 = ConvModule(in_dim, out_dim, kernel_size=1)
        self.cv2 = ConvModule(out_dim, out_dim, kernel_size=kernel_size, stride=stride, groups=out_dim, use_act=False)

    def forward(self, x):
        return self.cv2(self.cv1(x))

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads                      # number of the attention heads
        self.head_dim = dim // num_heads                # per head dim of v
        self.key_dim = int(self.head_dim * attn_ratio)  # per head dim of qk
        self.scale = self.key_dim**-0.5
        
        qkv_dims = dim + self.key_dim * num_heads * 2   # total dims of qkv
        self.qkv  = ConvModule(dim, qkv_dims, kernel_size=1, use_act=False)  # qkv projection
        self.proj = ConvModule(dim, dim, kernel_size=1, use_act=False)       # output projection
        self.pe   = ConvModule(dim, dim, kernel_size=3, groups=dim, use_act=False)  # position embedding conv

    def forward(self, x):
        bs, c, h, w = x.shape
        seq_len = h * w

        qkv = self.qkv(x)

        # q, k -> [bs, nh, c_kdh, hw]; v -> [bs, nh, c_vh, hw]
        q, k, v = qkv.view(bs, self.num_heads, self.key_dim * 2 + self.head_dim, seq_len).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        # [bs, nh, hw(q), c_kdh] x [bs, nh, c_kdh, hw(k)] -> [bs, nh, hw(q), hw(k)]
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        # [bs, nh, c_vh, hw(v)] x [bs, nh, hw(k), hw(q)] -> [bs, nh, c_vh, hw]
        x = (v @ attn.transpose(-2, -1)).view(bs, c, h, w) + self.pe(v.reshape(bs, c, h, w))
        x = self.proj(x)

        return x

class PSABlock(nn.Module):
    def __init__(self, in_dim, out_dim, expansion=0.5):
        super().__init__()
        assert(in_dim == out_dim)
        self.inter_dim = int(in_dim * expansion)
        self.cv1 = ConvModule(in_dim, 2 * self.inter_dim, kernel_size=1)
        self.cv2 = ConvModule(2 * self.inter_dim, in_dim, kernel_size=1)
        
        self.attn = Attention(self.inter_dim, attn_ratio=0.5, num_heads=self.inter_dim // 64)
        self.ffn = nn.Sequential(
            ConvModule(self.inter_dim, self.inter_dim * 2, kernel_size=1),
            ConvModule(self.inter_dim * 2, self.inter_dim, kernel_size=1, use_act=False)
        )
        
    def forward(self, x):
        a, b = self.cv1(x).split((self.inter_dim, self.inter_dim), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))

class SPPF(nn.Module):
    """
        This code referenced to https://github.com/ultralytics/yolov5
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        ## ----------- Basic Parameters -----------
        inter_dim = in_dim // 2
        self.out_dim = out_dim
        ## ----------- Network Parameters -----------
        self.cv1 = ConvModule(in_dim, inter_dim, kernel_size=1, stride=1)
        self.cv2 = ConvModule(inter_dim * 4, out_dim, kernel_size=1, stride=1)
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
