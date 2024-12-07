import torch
import torch.nn.functional as F
from utils.box_ops import get_ious
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized

from .matcher import YoloxMatcher


class SetCriterion(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_classes = cfg.num_classes
        self.loss_obj_weight = cfg.loss_obj
        self.loss_cls_weight = cfg.loss_cls
        self.loss_box_weight = cfg.loss_box
        # matcher
        self.matcher = YoloxMatcher(cfg.num_classes, cfg.ota_center_sampling_radius, cfg.ota_topk_candidate)

    def loss_objectness(self, pred_obj, gt_obj):
        loss_obj = F.binary_cross_entropy_with_logits(pred_obj, gt_obj, reduction='none')

        return loss_obj
    
    def loss_classes(self, pred_cls, gt_label):
        loss_cls = F.binary_cross_entropy_with_logits(pred_cls, gt_label, reduction='none')

        return loss_cls

    def loss_bboxes(self, pred_box, gt_box):
        # regression loss
        ious = get_ious(pred_box, gt_box, "xyxy", 'giou')
        loss_box = 1.0 - ious

        return loss_box

    def __call__(self, outputs, targets):        
        """
            outputs['pred_obj']: List(Tensor) [B, M, 1]
            outputs['pred_cls']: List(Tensor) [B, M, C]
            outputs['pred_reg']: List(Tensor) [B, M, 4]
            outputs['pred_box']: List(Tensor) [B, M, 4]
            outputs['strides']: List(Int) [8, 16, 32] output stride
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]
        """
        bs = outputs['pred_cls'][0].shape[0]
        device = outputs['pred_cls'][0].device
        fpn_strides = outputs['strides']
        anchors = outputs['anchors']
        # preds: [B, M, C]
        obj_preds = torch.cat(outputs['pred_obj'], dim=1)
        cls_preds = torch.cat(outputs['pred_cls'], dim=1)
        box_preds = torch.cat(outputs['pred_box'], dim=1)

        # label assignment
        cls_targets = []
        box_targets = []
        obj_targets = []
        fg_masks = []

        for batch_idx in range(bs):
            tgt_labels = targets[batch_idx]["labels"].to(device)
            tgt_bboxes = targets[batch_idx]["boxes"].to(device)

            # check target
            if len(tgt_labels) == 0 or tgt_bboxes.max().item() == 0.:
                num_anchors = sum([ab.shape[0] for ab in anchors])
                # There is no valid gt
                cls_target = obj_preds.new_zeros((0, self.num_classes))
                box_target = obj_preds.new_zeros((0, 4))
                obj_target = obj_preds.new_zeros((num_anchors, 1))
                fg_mask = obj_preds.new_zeros(num_anchors).bool()
            else:
                (
                    fg_mask,
                    assigned_labels,
                    assigned_ious,
                    assigned_indexs
                ) = self.matcher(
                    fpn_strides = fpn_strides,
                    anchors = anchors,
                    pred_obj = obj_preds[batch_idx],
                    pred_cls = cls_preds[batch_idx], 
                    pred_box = box_preds[batch_idx],
                    tgt_labels = tgt_labels,
                    tgt_bboxes = tgt_bboxes
                    )

                obj_target = fg_mask.unsqueeze(-1)
                cls_target = F.one_hot(assigned_labels.long(), self.num_classes)
                cls_target = cls_target * assigned_ious.unsqueeze(-1)
                box_target = tgt_bboxes[assigned_indexs]

            cls_targets.append(cls_target)
            box_targets.append(box_target)
            obj_targets.append(obj_target)
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        box_targets = torch.cat(box_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        num_fgs = fg_masks.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_fgs)
        num_fgs = (num_fgs / get_world_size()).clamp(1.0)

        # ------------------ Objecntness loss ------------------
        loss_obj = self.loss_objectness(obj_preds.view(-1, 1), obj_targets.float())
        loss_obj = loss_obj.sum() / num_fgs
        
        # ------------------ Classification loss ------------------
        cls_preds_pos = cls_preds.view(-1, self.num_classes)[fg_masks]
        loss_cls = self.loss_classes(cls_preds_pos, cls_targets)
        loss_cls = loss_cls.sum() / num_fgs

        # ------------------ Regression loss ------------------
        box_preds_pos = box_preds.view(-1, 4)[fg_masks]
        loss_box = self.loss_bboxes(box_preds_pos, box_targets)
        loss_box = loss_box.sum() / num_fgs

        # total loss
        losses = self.loss_obj_weight * loss_obj + \
                 self.loss_cls_weight * loss_cls + \
                 self.loss_box_weight * loss_box

        # Loss dict
        loss_dict = dict(
                loss_obj = loss_obj,
                loss_cls = loss_cls,
                loss_box = loss_box,
                losses = losses
        )

        return loss_dict


if __name__ == "__main__":
    pass