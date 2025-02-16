import math
import torch
import torch.nn as nn
from typing import List

try:
    from .modules import ConvModule, DflLayer
except:
    from  modules import ConvModule, DflLayer


# YOLOv10 detection head
class Yolov10DetHead(nn.Module):
    def __init__(self, cfg, fpn_dims: List = [64, 128, 245]):
        super().__init__()
        self.out_stride = cfg.out_stride
        self.reg_max = cfg.reg_max
        self.num_classes = cfg.num_classes

        self.cls_dim = max(fpn_dims[0], min(cfg.num_classes, 128))
        self.reg_dim = max(fpn_dims[0]//4, 16, 4*cfg.reg_max)

        # classification head
        self.cls_heads = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(ConvModule(dim, dim, kernel_size=3, stride=1, groups=dim),
                              ConvModule(dim, self.cls_dim, kernel_size=1)),
                nn.Sequential(ConvModule(self.cls_dim, self.cls_dim, kernel_size=3, stride=1, groups=self.cls_dim),
                              ConvModule(self.cls_dim, self.cls_dim, kernel_size=1)),
                nn.Conv2d(self.cls_dim, cfg.num_classes, kernel_size=1),
            )
            for dim in fpn_dims
        )

        # bbox regression head
        self.reg_heads = nn.ModuleList(
            nn.Sequential(
                ConvModule(dim, self.reg_dim, kernel_size=3, stride=1),
                ConvModule(self.reg_dim, self.reg_dim, kernel_size=3, stride=1),
                nn.Conv2d(self.reg_dim, 4*cfg.reg_max, kernel_size=1),
            )
            for dim in fpn_dims
        )

        # DFL layer for decoding bbox
        self.dfl_layer = DflLayer(cfg.reg_max)
        for p in self.dfl_layer.parameters():
            p.requires_grad = False

        self.init_bias()
        
    def init_bias(self):
        # cls pred
        for i, m in enumerate(self.cls_heads):
            b = m[-1].bias.view(1, -1)
            b.data.fill_(math.log(5 / self.num_classes / (640. / self.out_stride[i]) ** 2))
            m[-1].bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        # reg pred
        for m in self.reg_heads:
            b = m[-1].bias.view(-1, )
            b.data.fill_(1.0)
            m[-1].bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            
            w = m[-1].weight
            w.data.fill_(0.)
            m[-1].weight = torch.nn.Parameter(w, requires_grad=True)

    def generate_anchors(self, fmp_size, level):
        """
            fmp_size: (List) [H, W]
        """
        # generate grid cells
        fmp_h, fmp_w = fmp_size
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        anchors = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2)
        anchors += 0.5  # add center offset
        anchors *= self.out_stride[level]

        return anchors

    def forward(self, fpn_feats):
        anchors = []
        strides = []
        cls_preds = []
        reg_preds = []
        box_preds = []

        for lvl, (feat, cls_head, reg_head) in enumerate(zip(fpn_feats, self.cls_heads, self.reg_heads)):
            bs, c, h, w = feat.size()
            device = feat.device
            
            # Prediction
            cls_pred = cls_head(feat)
            reg_pred = reg_head(feat)

            # [bs, c, h, w] -> [bs, c, hw] -> [bs, hw, c]
            cls_pred = cls_pred.flatten(2).permute(0, 2, 1).contiguous()
            reg_pred = reg_pred.flatten(2).permute(0, 2, 1).contiguous()

            # anchor points: [M, 2]
            anchor = self.generate_anchors(fmp_size=[h, w], level=lvl).to(device)
            stride = torch.ones_like(anchor[..., :1]) * self.out_stride[lvl]

            # Decode bbox coords
            box_pred = self.dfl_layer(reg_pred, anchor[None], self.out_stride[lvl])

            # collect results
            anchors.append(anchor)
            strides.append(stride)
            cls_preds.append(cls_pred)
            reg_preds.append(reg_pred)
            box_preds.append(box_pred)

        # output dict
        outputs = {"pred_cls":       cls_preds,        # List(Tensor) [B, M, C]
                   "pred_reg":       reg_preds,        # List(Tensor) [B, M, 4*(reg_max)]
                   "pred_box":       box_preds,        # List(Tensor) [B, M, 4]
                   "anchors":        anchors,          # List(Tensor) [M, 2]
                   "stride_tensors": strides,          # List(Tensor) [M, 1]
                   "strides":        self.out_stride,  # List(Int) = [8, 16, 32]
                   }

        return outputs


if __name__=='__main__':
    from thop import profile

    # YOLOv10-Base config
    class Yolov10BaseConfig(object):
        def __init__(self) -> None:
            # ---------------- Model config ----------------
            self.width    = 0.50
            self.depth    = 0.34
            self.ratio    = 2.0
            
            self.reg_max  = 16
            self.out_stride = [8, 16, 32]
            self.max_stride = 32
            self.num_levels = 3
            self.num_classes = 80

    cfg = Yolov10BaseConfig()

    # Random data
    fpn_dims = [256, 512, 512]
    x = [torch.randn(1, fpn_dims[0], 80, 80),
         torch.randn(1, fpn_dims[1], 40, 40),
         torch.randn(1, fpn_dims[2], 20, 20)]

    # Neck model
    model = Yolov10DetHead(cfg, fpn_dims)

    # Inference
    outputs = model(x)

    print('============ FLOPs & Params ===========')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
    