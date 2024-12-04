import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Single-level pred layer
class SingleLevelPredLayer(nn.Module):
    def __init__(self,
                 cls_dim     :int = 256,
                 reg_dim     :int = 256,
                 stride      :int = 32,
                 reg_max     :int = 16,
                 num_classes :int = 80,
                 num_coords  :int = 4):
        super().__init__()
        # --------- Basic Parameters ----------
        self.stride = stride
        self.cls_dim = cls_dim
        self.reg_dim = reg_dim
        self.reg_max = reg_max
        self.num_classes = num_classes
        self.num_coords = num_coords

        # --------- Network Parameters ----------
        self.cls_pred = nn.Conv2d(cls_dim, num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(reg_dim, num_coords, kernel_size=1, groups=4)                

        self.init_bias()
        
    def init_bias(self):
        # cls pred bias
        b = self.cls_pred.bias.view(1, -1)
        b.data.fill_(math.log(5 / self.num_classes / (640. / self.stride) ** 2))
        self.cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        # reg pred bias
        b = self.reg_pred.bias.view(-1, )
        b.data.fill_(1.0)
        self.reg_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        w = self.reg_pred.weight
        w.data.fill_(0.)
        self.reg_pred.weight = torch.nn.Parameter(w, requires_grad=True)

    def generate_anchors(self, fmp_size):
        """
            fmp_size: (List) [H, W]
        """
        # generate grid cells
        fmp_h, fmp_w = fmp_size
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        anchors = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2)
        anchors += 0.5  # add center offset
        anchors *= self.stride

        return anchors
        
    def forward(self, cls_feat, reg_feat):
        # pred
        cls_pred = self.cls_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)

        # generate anchor boxes: [M, 4]
        B, _, H, W = cls_pred.size()
        fmp_size = [H, W]
        anchors = self.generate_anchors(fmp_size)
        anchors = anchors.to(cls_pred.device)
        # stride tensor: [M, 1]
        stride_tensor = torch.ones_like(anchors[..., :1]) * self.stride
        
        # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4*self.reg_max)
        
        # output dict
        outputs = {"pred_cls": cls_pred,            # List(Tensor) [B, M, C]
                   "pred_reg": reg_pred,            # List(Tensor) [B, M, 4*(reg_max)]
                   "anchors": anchors,              # List(Tensor) [M, 2]
                   "strides": self.stride,          # List(Int) = [8, 16, 32]
                   "stride_tensor": stride_tensor   # List(Tensor) [M, 1]
                   }

        return outputs

# Multi-level pred layer
class GElanPredLayer(nn.Module):
    def __init__(self,
                 cfg,
                 cls_dim,
                 reg_dim,
                 ):
        super().__init__()
        # --------- Basic Parameters ----------
        self.cfg = cfg
        self.cls_dim = cls_dim
        self.reg_dim = reg_dim

        # ----------- Network Parameters -----------
        ## pred layers
        self.multi_level_preds = nn.ModuleList(
            [SingleLevelPredLayer(cls_dim     = cls_dim,
                                  reg_dim     = reg_dim,
                                  stride      = cfg.out_stride[level],
                                  reg_max     = cfg.reg_max,
                                  num_classes = cfg.num_classes,
                                  num_coords  = 4 * cfg.reg_max)
                                  for level in range(cfg.num_levels)
                                  ])
        ## proj conv
        proj_init = torch.arange(cfg.reg_max, dtype=torch.float)
        self.proj_conv = nn.Conv2d(cfg.reg_max, 1, kernel_size=1, bias=False).requires_grad_(False)
        self.proj_conv.weight.data[:] = nn.Parameter(proj_init.view([1, cfg.reg_max, 1, 1]), requires_grad=False)

    def forward(self, cls_feats, reg_feats):
        all_anchors = []
        all_strides = []
        all_cls_preds = []
        all_reg_preds = []
        all_box_preds = []
        for level in range(self.cfg.num_levels):
            # -------------- Single-level prediction --------------
            outputs = self.multi_level_preds[level](cls_feats[level], reg_feats[level])

            # -------------- Decode bbox --------------
            B, M = outputs["pred_reg"].shape[:2]
            # [B, M, 4*(reg_max)] -> [B, M, 4, reg_max]
            delta_pred = outputs["pred_reg"].reshape([B, M, 4, self.cfg.reg_max])
            # [B, M, 4, reg_max] -> [B, reg_max, 4, M]
            delta_pred = delta_pred.permute(0, 3, 2, 1).contiguous()
            # [B, reg_max, 4, M] -> [B, 1, 4, M]
            delta_pred = self.proj_conv(F.softmax(delta_pred, dim=1))
            # [B, 1, 4, M] -> [B, 4, M] -> [B, M, 4]
            delta_pred = delta_pred.view(B, 4, M).permute(0, 2, 1).contiguous()
            ## tlbr -> xyxy
            x1y1_pred = outputs["anchors"][None] - delta_pred[..., :2] * self.cfg.out_stride[level]
            x2y2_pred = outputs["anchors"][None] + delta_pred[..., 2:] * self.cfg.out_stride[level]
            box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

            # collect results
            all_cls_preds.append(outputs["pred_cls"])
            all_reg_preds.append(outputs["pred_reg"])
            all_box_preds.append(box_pred)
            all_anchors.append(outputs["anchors"])
            all_strides.append(outputs["stride_tensor"])
        
        # output dict
        outputs = {"pred_cls":      all_cls_preds,         # List(Tensor) [B, M, C]
                   "pred_reg":      all_reg_preds,         # List(Tensor) [B, M, 4*(reg_max)]
                   "pred_box":      all_box_preds,         # List(Tensor) [B, M, 4]
                   "anchors":       all_anchors,           # List(Tensor) [M, 2]
                   "stride_tensor": all_strides,           # List(Tensor) [M, 1]
                   "strides":       self.cfg.out_stride,   # List(Int) = [8, 16, 32]
                   }

        return outputs