import math
import torch
import torch.nn as nn

try:
    from .yolof_basic import BasicConv
except:
    from  yolof_basic import BasicConv
    

class YolofDecoder(nn.Module):
    def __init__(self, cfg, in_dim):
        super().__init__()
        # ------------------ Basic parameters -------------------
        self.cfg = cfg
        self.in_dim = in_dim
        self.stride       = cfg.out_stride
        self.num_classes  = cfg.num_classes
        self.num_cls_head = cfg.num_cls_head
        self.num_reg_head = cfg.num_reg_head
        # Anchor config
        self.anchor_size = torch.as_tensor(cfg.anchor_size)
        self.num_anchors = len(cfg.anchor_size)

        # ------------------ Network parameters -------------------
        ## cls head
        cls_heads = []
        self.cls_head_dim = cfg.head_dim
        for i in range(self.num_cls_head):
            if i == 0:
                cls_heads.append(
                    BasicConv(in_dim, self.cls_head_dim,
                              kernel_size=3, padding=1, stride=1, 
                              act_type=cfg.head_act, norm_type=cfg.head_norm, depthwise=cfg.head_depthwise)
                              )
            else:
                cls_heads.append(
                    BasicConv(self.cls_head_dim, self.cls_head_dim,
                              kernel_size=3, padding=1, stride=1, 
                              act_type=cfg.head_act, norm_type=cfg.head_norm, depthwise=cfg.head_depthwise)
                              )
        ## reg head
        reg_heads = []
        self.reg_head_dim = cfg.head_dim
        for i in range(self.num_reg_head):
            if i == 0:
                reg_heads.append(
                    BasicConv(in_dim, self.reg_head_dim,
                              kernel_size=3, padding=1, stride=1, 
                              act_type=cfg.head_act, norm_type=cfg.head_norm, depthwise=cfg.head_depthwise)
                              )
            else:
                reg_heads.append(
                    BasicConv(self.reg_head_dim, self.reg_head_dim,
                              kernel_size=3, padding=1, stride=1, 
                              act_type=cfg.head_act, norm_type=cfg.head_norm, depthwise=cfg.head_depthwise)
                              )
        self.cls_heads = nn.Sequential(*cls_heads)
        self.reg_heads = nn.Sequential(*reg_heads)

        # pred layer
        self.cls_pred = nn.Conv2d(self.cls_head_dim, self.num_classes * self.num_anchors, kernel_size=1)
        self.reg_pred = nn.Conv2d(self.reg_head_dim, 4 * self.num_anchors, kernel_size=1)

        self.init_weights()
        
    def init_weights(self):
        # Init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
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
        anchor_xy = anchor_xy.view(-1, 2) + 0.5
        anchor_xy *= self.stride

        # [A, 2] -> [1, A, 2] -> [HW, A, 2] -> [M, 2], M=HWA
        anchor_wh = self.anchor_size.unsqueeze(0).repeat(fmp_h*fmp_w, 1, 1)
        anchor_wh = anchor_wh.view(-1, 2)

        anchors = torch.cat([anchor_xy, anchor_wh], dim=-1)

        return anchors
        
    def decode_boxes(self, anchors, reg_pred):
        """
            anchors:  (List[tensor]) [1, M, 4]
            reg_pred: (List[tensor]) [B, M, 4]
        """
        cxcy_pred = anchors[..., :2] + reg_pred[..., :2] * self.stride
        bwbh_pred = anchors[..., 2:] * torch.exp(reg_pred[..., 2:])
        pred_x1y1 = cxcy_pred - bwbh_pred * 0.5
        pred_x2y2 = cxcy_pred + bwbh_pred * 0.5
        box_pred = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return box_pred

    def forward(self, x):
        # ------------------- Decoupled head -------------------
        cls_feats = self.cls_heads(x)
        reg_feats = self.reg_heads(x)

        # ------------------- Prediction -------------------
        cls_pred = self.cls_pred(cls_feats)
        reg_pred = self.reg_pred(reg_feats)

        # ------------------- Generate anchor box -------------------
        B, _, H, W = cls_pred.size()
        anchors = self.generate_anchors([H, W])   # [M, 4]
        anchors = anchors.to(cls_feats.device)

        # ------------------- Precoess preds -------------------
        # [B, C*A, H, W] -> [B, H, W, C*A] -> [B, H*W*A, C]
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)

        ## Decode bbox
        box_pred = self.decode_boxes(anchors[None], reg_pred)  # [B, M, 4]

        outputs = {"pred_cls": cls_pred,   # (torch.Tensor) [B, M, C]
                   "pred_reg": reg_pred,   # (torch.Tensor) [B, M, 4]
                   "pred_box": box_pred,   # (torch.Tensor) [B, M, 4]
                   "stride":   self.stride,
                   "anchors":  anchors,    # (torch.Tensor) [M, C]
                   }

        return outputs 
