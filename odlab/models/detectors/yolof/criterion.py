# ---------------------------------------------------------------------
# Copyright (c) Megvii Inc. All rights reserved.
# ---------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.box_ops import *
from utils.misc import sigmoid_focal_loss
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized

from .matcher import UniformMatcher


class SetCriterion(nn.Module):
    """
        This code referenced to https://github.com/megvii-model/YOLOF/blob/main/playground/detection/coco/yolof/yolof_base/yolof.py
    """
    def __init__(self, cfg):
        super().__init__()
        # ------------- Basic parameters -------------
        self.cfg = cfg
        self.num_classes = cfg.num_classes
        # ------------- Focal loss -------------
        self.alpha = cfg.focal_loss_alpha
        self.gamma = cfg.focal_loss_gamma
        # ------------- Loss weight -------------
        self.weight_dict = {'loss_cls': cfg.loss_cls_weight,
                            'loss_reg': cfg.loss_reg_weight}
        # ------------- Matcher -------------
        self.matcher_cfg = cfg.matcher_hpy
        self.matcher = UniformMatcher(self.matcher_cfg['topk_candidates'])

    def loss_labels(self, pred_cls, tgt_cls, num_boxes):
        """
            pred_cls: (Tensor) [N, C]
            tgt_cls:  (Tensor) [N, C]
        """
        # cls loss: [V, C]
        loss_cls = sigmoid_focal_loss(pred_cls, tgt_cls, self.alpha, self.gamma)

        return loss_cls.sum() / num_boxes

    def loss_bboxes(self, pred_box, tgt_box, num_boxes):
        """
            pred_box: (Tensor) [N, 4]
            tgt_box:  (Tensor) [N, 4]
        """
        # giou
        pred_giou = generalized_box_iou(pred_box, tgt_box)  # [N, M]
        # giou loss
        loss_reg = 1. - torch.diag(pred_giou)

        return loss_reg.sum() / num_boxes

    def forward(self, outputs, targets):
        """
            outputs['pred_cls']: (Tensor) [B, M, C]
            outputs['pred_box']: (Tensor) [B, M, 4]
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]
        """
        # -------------------- Pre-process --------------------
        pred_box = outputs['pred_box']
        pred_cls = outputs['pred_cls'].reshape(-1, self.num_classes)
        anchor_boxes = outputs['anchors']
        masks = ~outputs['mask']
        device = pred_box.device
        B = len(targets)

        # -------------------- Label assignment --------------------
        indices = self.matcher(pred_box, anchor_boxes, targets)

        # [M, 4] -> [1, M, 4] -> [B, M, 4]
        anchor_boxes = box_cxcywh_to_xyxy(anchor_boxes)
        anchor_boxes = anchor_boxes[None].repeat(B, 1, 1)

        ious = []
        pos_ious = []
        for i in range(B):
            src_idx, tgt_idx = indices[i]
            # iou between predbox and tgt box
            iou, _ = box_iou(pred_box[i, ...], (targets[i]['boxes']).clone())
            if iou.numel() == 0:
                max_iou = iou.new_full((iou.size(0),), 0)
            else:
                max_iou = iou.max(dim=1)[0]
            # iou between anchorbox and tgt box
            a_iou, _ = box_iou(anchor_boxes[i], (targets[i]['boxes']).clone())
            if a_iou.numel() == 0:
                pos_iou = a_iou.new_full((0,), 0)
            else:
                pos_iou = a_iou[src_idx, tgt_idx]
            ious.append(max_iou)
            pos_ious.append(pos_iou)

        ious = torch.cat(ious)
        ignore_idx = ious > self.matcher_cfg['ignore_thresh']
        pos_ious = torch.cat(pos_ious)
        pos_ignore_idx = pos_ious < self.matcher_cfg['iou_thresh']

        src_idx = torch.cat(
            [src + idx * anchor_boxes[0].shape[0] for idx, (src, _) in
             enumerate(indices)])
        # [BM,]
        gt_cls = torch.full(pred_cls.shape[:1],
                                self.num_classes,
                                dtype=torch.int64,
                                device=device)
        gt_cls[ignore_idx] = -1
        tgt_cls_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        tgt_cls_o[pos_ignore_idx] = -1

        gt_cls[src_idx] = tgt_cls_o.to(device)

        foreground_idxs = (gt_cls >= 0) & (gt_cls != self.num_classes)
        num_foreground = foreground_idxs.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foreground)
        num_foreground = torch.clamp(num_foreground / get_world_size(), min=1).item()

        # -------------------- Classification loss --------------------
        gt_cls_target = torch.zeros_like(pred_cls)
        gt_cls_target[foreground_idxs, gt_cls[foreground_idxs]] = 1
        valid_idxs = (gt_cls >= 0) & masks
        loss_labels = self.loss_labels(pred_cls[valid_idxs], gt_cls_target[valid_idxs], num_foreground)

        # -------------------- Regression loss --------------------
        tgt_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(device)
        tgt_boxes = tgt_boxes[~pos_ignore_idx]
        matched_pred_box = pred_box.reshape(-1, 4)[src_idx[~pos_ignore_idx.cpu()]]
        loss_bboxes = self.loss_bboxes(matched_pred_box, tgt_boxes, num_foreground)

        total_loss = loss_labels * self.weight_dict["loss_cls"] + \
                     loss_bboxes * self.weight_dict["loss_reg"]
        loss_dict = dict(
                loss_cls = loss_labels,
                loss_reg = loss_bboxes,
                losses   = total_loss,
        )

        return loss_dict


if __name__ == "__main__":
    pass
