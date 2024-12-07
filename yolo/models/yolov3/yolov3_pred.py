import torch
import torch.nn as nn
from typing import List

# -------------------- Detection Pred Layer --------------------
## Single-level pred layer
class DetPredLayer(nn.Module):
    def __init__(self,
                 cls_dim      :int,
                 reg_dim      :int,
                 stride       :int,
                 num_classes  :int,
                 anchor_sizes :List,
                 ):
        super().__init__()
        # --------- Basic Parameters ----------
        self.stride  = stride
        self.cls_dim = cls_dim
        self.reg_dim = reg_dim
        self.num_classes = num_classes
        # ------------------- Anchor box -------------------
        self.anchor_size = torch.as_tensor(anchor_sizes).float().view(-1, 2) # [A, 2]
        self.num_anchors = self.anchor_size.shape[0]

        # --------- Network Parameters ----------
        self.obj_pred = nn.Conv2d(self.cls_dim, 1 * self.num_anchors, kernel_size=1)
        self.cls_pred = nn.Conv2d(self.cls_dim, num_classes * self.num_anchors, kernel_size=1)
        self.reg_pred = nn.Conv2d(self.reg_dim, 4 * self.num_anchors, kernel_size=1)                

        self.init_bias()
        
    def init_bias(self):
        # Init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        # obj pred
        b = self.obj_pred.bias.view(1, -1)
        b.data.fill_(bias_value.item())
        self.obj_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        # cls pred
        b = self.cls_pred.bias.view(1, -1)
        b.data.fill_(bias_value.item())
        self.cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        # reg pred
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
        # 特征图的宽和高
        fmp_h, fmp_w = fmp_size

        # 生成网格的x坐标和y坐标
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])

        # 将xy两部分的坐标拼起来：[H, W, 2] -> [HW, 2]
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2)
        # [HW, 2] -> [HW, A, 2] -> [M, 2], M=HWA
        anchor_xy = anchor_xy.unsqueeze(1).repeat(1, self.num_anchors, 1)
        anchor_xy = anchor_xy.view(-1, 2)

        # [A, 2] -> [1, A, 2] -> [HW, A, 2] -> [M, 2], M=HWA
        anchor_wh = self.anchor_size.unsqueeze(0).repeat(fmp_h*fmp_w, 1, 1)
        anchor_wh = anchor_wh.view(-1, 2)

        anchors = torch.cat([anchor_xy, anchor_wh], dim=-1)

        return anchors
        
    def forward(self, cls_feat, reg_feat):
        # 预测层
        obj_pred = self.obj_pred(reg_feat)
        cls_pred = self.cls_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)

        # 生成网格坐标
        B, _, H, W = cls_pred.size()
        fmp_size = [H, W]
        anchors = self.generate_anchors(fmp_size)
        anchors = anchors.to(cls_pred.device)

        # 对 pred 的size做一些view调整，便于后续的处理
        # [B, C*A, H, W] -> [B, H, W, C*A] -> [B, H*W*A, C]
        obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
        
        # 解算边界框坐标
        cxcy_pred = (torch.sigmoid(reg_pred[..., :2]) + anchors[..., :2]) * self.stride
        bwbh_pred = torch.exp(reg_pred[..., 2:]) * anchors[..., 2:]
        pred_x1y1 = cxcy_pred - bwbh_pred * 0.5
        pred_x2y2 = cxcy_pred + bwbh_pred * 0.5
        box_pred = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        # output dict
        outputs = {"pred_obj": obj_pred,       # (torch.Tensor) [B, M, 1]
                   "pred_cls": cls_pred,       # (torch.Tensor) [B, M, C]
                   "pred_reg": reg_pred,       # (torch.Tensor) [B, M, 4]
                   "pred_box": box_pred,       # (torch.Tensor) [B, M, 4]
                   "anchors" : anchors,        # (torch.Tensor) [M, 2]
                   "fmp_size": fmp_size,
                   "stride"  : self.stride,    # (Int)
                   }

        return outputs

## Multi-level pred layer
class Yolov3DetPredLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # --------- Basic Parameters ----------
        self.cfg = cfg
        self.num_levels = len(cfg.out_stride)

        # ----------- Network Parameters -----------
        ## pred layers
        self.multi_level_preds = nn.ModuleList(
            [DetPredLayer(cls_dim      = round(cfg.head_dim * cfg.width),
                          reg_dim      = round(cfg.head_dim * cfg.width),
                          stride       = cfg.out_stride[level],
                          anchor_sizes = cfg.anchor_size[level],
                          num_classes  = cfg.num_classes,)
                          for level in range(self.num_levels)
                          ])

    def forward(self, cls_feats, reg_feats):
        all_anchors = []
        all_strides = []
        all_fmp_sizes = []
        all_obj_preds = []
        all_cls_preds = []
        all_reg_preds = []
        all_box_preds = []
        for level in range(self.num_levels):
            # -------------- Single-level prediction --------------
            outputs = self.multi_level_preds[level](cls_feats[level], reg_feats[level])

            # collect results
            all_obj_preds.append(outputs["pred_obj"])
            all_cls_preds.append(outputs["pred_cls"])
            all_reg_preds.append(outputs["pred_reg"])
            all_box_preds.append(outputs["pred_box"])
            all_fmp_sizes.append(outputs["fmp_size"])
            all_anchors.append(outputs["anchors"])
        
        # output dict
        outputs = {"pred_obj":  all_obj_preds,         # List(Tensor) [B, M, 1]
                   "pred_cls":  all_cls_preds,         # List(Tensor) [B, M, C]
                   "pred_reg":  all_reg_preds,         # List(Tensor) [B, M, 4*(reg_max)]
                   "pred_box":  all_box_preds,         # List(Tensor) [B, M, 4]
                   "fmp_sizes": all_fmp_sizes,         # List(Tensor) [M, 1]
                   "anchors":   all_anchors,           # List(Tensor) [M, 2]
                   "strides":   self.cfg.out_stride,   # List(Int) = [8, 16, 32]
                   }

        return outputs


if __name__=='__main__':
    import time
    from thop import profile
    # Model config
    
    # YOLOv8-Base config
    class Yolov3BaseConfig(object):
        def __init__(self) -> None:
            # ---------------- Model config ----------------
            self.width    = 1.0
            self.depth    = 1.0
            self.out_stride = [8, 16, 32]
            self.max_stride = 32
            self.num_levels = 3
            ## Head
            self.head_dim  = 256
            self.anchor_size = {0: [[10, 13],   [16, 30],   [33, 23]],
                                1: [[30, 61],   [62, 45],   [59, 119]],
                                2: [[116, 90],  [156, 198], [373, 326]]}

    cfg = Yolov3BaseConfig()
    cfg.num_classes = 20
    # Build a pred layer
    pred = Yolov3DetPredLayer(cfg)

    # Inference
    cls_feats = [torch.randn(1, cfg.head_dim, 80, 80),
                 torch.randn(1, cfg.head_dim, 40, 40),
                 torch.randn(1, cfg.head_dim, 20, 20),]
    reg_feats = [torch.randn(1, cfg.head_dim, 80, 80),
                 torch.randn(1, cfg.head_dim, 40, 40),
                 torch.randn(1, cfg.head_dim, 20, 20),]
    t0 = time.time()
    output = pred(cls_feats, reg_feats)
    t1 = time.time()
    print('Time: ', t1 - t0)
    print('====== Pred output ======= ')
    pred_obj = output["pred_obj"]
    pred_cls = output["pred_cls"]
    pred_reg = output["pred_reg"]
    pred_box = output["pred_box"]
    anchors  = output["anchors"]
    
    for level in range(cfg.num_levels):
        print("- Level-{} : objectness       -> {}".format(level, pred_obj[level].shape))
        print("- Level-{} : classification   -> {}".format(level, pred_cls[level].shape))
        print("- Level-{} : delta regression -> {}".format(level, pred_reg[level].shape))
        print("- Level-{} : bbox regression  -> {}".format(level, pred_box[level].shape))
        print("- Level-{} : anchor boxes     -> {}".format(level, anchors[level].shape))

    flops, params = profile(pred, inputs=(cls_feats, reg_feats, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
