import math
import torch
import torch.nn as nn

from ..basic.conv import BasicConv


class YolofHead(nn.Module):
    def __init__(self, cfg, in_dim, out_dim,):
        super().__init__()
        self.fmp_size = None
        self.ctr_clamp = cfg.center_clamp
        self.DEFAULT_EXP_CLAMP = math.log(1e8)
        self.DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)
        # ------------------ Basic parameters -------------------
        self.cfg = cfg
        self.in_dim = in_dim
        self.stride       = cfg.out_stride
        self.num_classes  = cfg.num_classes
        self.num_cls_head = cfg.num_cls_head
        self.num_reg_head = cfg.num_reg_head
        self.act_type     = cfg.head_act
        self.norm_type    = cfg.head_norm
        # Anchor config
        self.anchor_size = torch.as_tensor(cfg.anchor_size)
        self.num_anchors = len(cfg.anchor_size)

        # ------------------ Network parameters -------------------
        ## cls head
        cls_heads = []
        self.cls_head_dim = out_dim
        for i in range(self.num_cls_head):
            if i == 0:
                cls_heads.append(
                    BasicConv(in_dim, self.cls_head_dim,
                              kernel_size=3, padding=1, stride=1, 
                              act_type=self.act_type, norm_type=self.norm_type)
                              )
            else:
                cls_heads.append(
                    BasicConv(self.cls_head_dim, self.cls_head_dim,
                              kernel_size=3, padding=1, stride=1, 
                              act_type=self.act_type, norm_type=self.norm_type)
                              )
        ## reg head
        reg_heads = []
        self.reg_head_dim = out_dim
        for i in range(self.num_reg_head):
            if i == 0:
                reg_heads.append(
                    BasicConv(in_dim, self.reg_head_dim,
                              kernel_size=3, padding=1, stride=1, 
                              act_type=self.act_type, norm_type=self.norm_type)
                              )
            else:
                reg_heads.append(
                    BasicConv(self.reg_head_dim, self.reg_head_dim,
                              kernel_size=3, padding=1, stride=1, 
                              act_type=self.act_type, norm_type=self.norm_type)
                              )
        self.cls_heads = nn.Sequential(*cls_heads)
        self.reg_heads = nn.Sequential(*reg_heads)

        # pred layer
        self.obj_pred = nn.Conv2d(self.reg_head_dim, 1 * self.num_anchors, kernel_size=3, padding=1)
        self.cls_pred = nn.Conv2d(self.cls_head_dim, self.num_classes * self.num_anchors, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(self.reg_head_dim, 4 * self.num_anchors, kernel_size=3, padding=1)

        # init bias
        self._init_pred_layers()

    def _init_pred_layers(self):  
        # init cls pred
        nn.init.normal_(self.cls_pred.weight, mean=0, std=0.01)
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.cls_pred.bias, bias_value)
        # init reg pred
        nn.init.normal_(self.reg_pred.weight, mean=0, std=0.01)
        nn.init.constant_(self.reg_pred.bias, 0.0)
        # init obj pred
        nn.init.normal_(self.obj_pred.weight, mean=0, std=0.01)
        nn.init.constant_(self.obj_pred.bias, 0.0)

    def get_anchors(self, fmp_size):
        """fmp_size: list -> [H, W] \n
           stride: int -> output stride
        """
        # check anchor boxes
        if self.fmp_size is not None and self.fmp_size == fmp_size:
            return self.anchor_boxes
        else:
            # generate grid cells
            fmp_h, fmp_w = fmp_size
            anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
            # [H, W, 2] -> [HW, 2]
            anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2) + 0.5
            # [HW, 2] -> [HW, 1, 2] -> [HW, KA, 2] 
            anchor_xy = anchor_xy[:, None, :].repeat(1, self.num_anchors, 1)
            anchor_xy *= self.stride

            # [KA, 2] -> [1, KA, 2] -> [HW, KA, 2]
            anchor_wh = self.anchor_size[None, :, :].repeat(fmp_h*fmp_w, 1, 1)

            # [HW, KA, 4] -> [M, 4]
            anchor_boxes = torch.cat([anchor_xy, anchor_wh], dim=-1)
            anchor_boxes = anchor_boxes.view(-1, 4)

            self.anchor_boxes = anchor_boxes
            self.fmp_size = fmp_size

            return anchor_boxes
        
    def decode_boxes(self, anchor_boxes, pred_reg):
        """
            anchor_boxes: (List[tensor]) [1, M, 4]
            pred_reg: (List[tensor]) [B, M, 4]
        """
        # x = x_anchor + dx * w_anchor
        # y = y_anchor + dy * h_anchor
        pred_ctr_offset = pred_reg[..., :2] * anchor_boxes[..., 2:]
        pred_ctr_offset = torch.clamp(pred_ctr_offset, min=-self.ctr_clamp, max=self.ctr_clamp)
        pred_ctr_xy = anchor_boxes[..., :2] + pred_ctr_offset

        # w = w_anchor * exp(tw)
        # h = h_anchor * exp(th)
        pred_dwdh = pred_reg[..., 2:]
        pred_dwdh = torch.clamp(pred_dwdh, max=self.DEFAULT_SCALE_CLAMP)
        pred_wh = anchor_boxes[..., 2:] * pred_dwdh.exp()

        # convert [x, y, w, h] -> [x1, y1, x2, y2]
        pred_x1y1 = pred_ctr_xy - 0.5 * pred_wh
        pred_x2y2 = pred_ctr_xy + 0.5 * pred_wh
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box

    def forward(self, x, mask=None):
        # ------------------- Decoupled head -------------------
        cls_feats = self.cls_heads(x)
        reg_feats = self.reg_heads(x)

        # ------------------- Generate anchor box -------------------
        fmp_size = cls_feats.shape[2:]
        anchor_boxes = self.get_anchors(fmp_size)   # [M, 4]
        anchor_boxes = anchor_boxes.to(cls_feats.device)

        # ------------------- Predict -------------------
        obj_pred = self.obj_pred(reg_feats)
        cls_pred = self.cls_pred(cls_feats)
        reg_pred = self.reg_pred(reg_feats)

        # ------------------- Precoess preds -------------------
        ## implicit objectness
        B, _, H, W = obj_pred.size()
        obj_pred = obj_pred.view(B, -1, 1, H, W)
        cls_pred = cls_pred.view(B, -1, self.num_classes, H, W)
        normalized_cls_pred = cls_pred + obj_pred - torch.log(
                1. + 
                torch.clamp(cls_pred, max=self.DEFAULT_EXP_CLAMP).exp() + 
                torch.clamp(obj_pred, max=self.DEFAULT_EXP_CLAMP).exp())
        # [B, KA, C, H, W] -> [B, H, W, KA, C] -> [B, M, C], M = HxWxKA
        normalized_cls_pred = normalized_cls_pred.permute(0, 3, 4, 1, 2).contiguous()
        normalized_cls_pred = normalized_cls_pred.view(B, -1, self.num_classes)
        # [B, KA*4, H, W] -> [B, KA, 4, H, W] -> [B, H, W, KA, 4] -> [B, M, 4]
        reg_pred = reg_pred.view(B, -1, 4, H, W).permute(0, 3, 4, 1, 2).contiguous()
        reg_pred = reg_pred.view(B, -1, 4)
        ## Decode bbox
        box_pred = self.decode_boxes(anchor_boxes[None], reg_pred)  # [B, M, 4]
        ## adjust mask
        if mask is not None:
            # [B, H, W]
            mask = torch.nn.functional.interpolate(mask[None].float(), size=fmp_size).bool()[0]
            # [B, H, W] -> [B, HW]
            mask = mask.flatten(1)
            # [B, HW] -> [B, HW, KA] -> [BM,], M= HW x KA
            mask = mask[..., None].repeat(1, 1, self.num_anchors).flatten()

        outputs = {"pred_cls": normalized_cls_pred,
                   "pred_reg": reg_pred,
                   "pred_box": box_pred,
                   "anchors": anchor_boxes,
                   "mask": mask}

        return outputs 
