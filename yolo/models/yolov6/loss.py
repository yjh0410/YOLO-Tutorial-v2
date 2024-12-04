import torch
import torch.nn.functional as F

from utils.box_ops import bbox_iou
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized

from .matcher import TaskAlignedAssigner


class SetCriterion(object):
    def __init__(self, cfg):
        # --------------- Basic parameters ---------------
        self.cfg = cfg
        self.reg_max = cfg.reg_max
        self.num_classes = cfg.num_classes
        # --------------- Loss config ---------------
        self.loss_cls_weight = cfg.loss_cls
        self.loss_box_weight = cfg.loss_box
        # --------------- Matcher config ---------------
        self.matcher = TaskAlignedAssigner(num_classes     = cfg.num_classes,
                                           topk_candidates = cfg.tal_topk_candidates,
                                           alpha           = cfg.tal_alpha,
                                           beta            = cfg.tal_beta
                                           )

    def loss_classes(self, pred_cls, labels, scores):
        # compute bce loss
        alpha = 0.75
        gamma = 2.0
        # pred and target should be of the same size
        bg_class_ind = pred_cls.shape[-1]
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)

        new_scores = pred_cls.new_zeros(pred_cls.shape)
        pos_labels = labels[pos_inds]
        new_scores[pos_inds, pos_labels] = scores[pos_inds].clone().detach()

        pred_sigmoid = pred_cls.sigmoid()
        focal_weight = new_scores * (new_scores > 0.0).float() + \
            alpha * (pred_sigmoid - new_scores).abs().pow(gamma) * \
            (new_scores <= 0.0).float()
        
        loss_cls = F.binary_cross_entropy_with_logits(
            pred_cls, new_scores, reduction='none') * focal_weight
    
        return loss_cls
    
    def loss_bboxes(self, pred_box, gt_box, bbox_weight):
        # regression loss
        ious = bbox_iou(pred_box, gt_box, xywh=False, GIoU=True)
        loss_box = (1.0 - ious.squeeze(-1)) * bbox_weight

        return loss_box
    
    def __call__(self, outputs, targets):        
        """
            outputs['pred_cls']: List(Tensor) [B, M, C]
            outputs['pred_reg']: List(Tensor) [B, M, 4*(reg_max+1)]
            outputs['pred_box']: List(Tensor) [B, M, 4]
            outputs['anchors']: List(Tensor) [M, 2]
            outputs['strides']: List(Int) [8, 16, 32] output stride
            outputs['stride_tensor']: List(Tensor) [M, 1]
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]
        """
        # preds: [B, M, C]
        cls_preds = torch.cat(outputs['pred_cls'], dim=1)
        box_preds = torch.cat(outputs['pred_box'], dim=1)
        bs, num_anchors = cls_preds.shape[:2]
        device = cls_preds.device
        anchors = torch.cat(outputs['anchors'], dim=0)
        
        # --------------- label assignment ---------------
        gt_label_targets = []
        gt_score_targets = []
        gt_bbox_targets = []
        fg_masks = []
        for batch_idx in range(bs):
            tgt_labels = targets[batch_idx]["labels"].to(device)     # [Mp,]
            tgt_boxs = targets[batch_idx]["boxes"].to(device)        # [Mp, 4]

            # check target
            if len(tgt_labels) == 0 or tgt_boxs.max().item() == 0.:
                # There is no valid gt
                fg_mask  = cls_preds.new_zeros(1, num_anchors).bool()                       # [1, M,]
                gt_label = cls_preds.new_zeros((1, num_anchors)).long()                     # [1, M,]
                gt_score = cls_preds.new_zeros((1, num_anchors, self.num_classes)).float()  # [1, M, C]
                gt_box   = cls_preds.new_zeros((1, num_anchors, 4)).float()                 # [1, M, 4]
            else:
                tgt_labels = tgt_labels[None, :, None]      # [1, Mp, 1]
                tgt_boxs = tgt_boxs[None]                   # [1, Mp, 4]
                (
                    gt_label,   # 
                    gt_box,     # [1, M, 4]
                    gt_score,   # [1, M, C]
                    fg_mask,    # [1, M,]
                    _
                ) = self.matcher(
                    pd_scores = cls_preds[batch_idx:batch_idx+1].detach().sigmoid(), 
                    pd_bboxes = box_preds[batch_idx:batch_idx+1].detach(),
                    anc_points = anchors,
                    gt_labels = tgt_labels,
                    gt_bboxes = tgt_boxs
                    )
            gt_label_targets.append(gt_label)
            gt_score_targets.append(gt_score)
            gt_bbox_targets.append(gt_box)
            fg_masks.append(fg_mask)

        # List[B, 1, M, C] -> Tensor[B, M, C] -> Tensor[BM, C]
        fg_masks = torch.cat(fg_masks, 0).view(-1)                                    # [BM,]
        gt_label_targets = torch.cat(gt_label_targets, 0).view(-1)                    # [BM,]
        gt_score_targets = torch.cat(gt_score_targets, 0).view(-1, self.num_classes)  # [BM, C]
        gt_bbox_targets = torch.cat(gt_bbox_targets, 0).view(-1, 4)                   # [BM, 4]
        num_fgs = gt_score_targets.sum()

        # Average loss normalizer across all the GPUs
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_fgs)
        num_fgs = (num_fgs / get_world_size()).clamp(1.0)

        # ------------------ Classification loss ------------------
        target_labels = torch.where(fg_masks > 0, gt_label_targets,
                                    torch.full_like(gt_label_targets, self.num_classes))
        target_scores = gt_score_targets.new_zeros(gt_score_targets.shape[0])
        target_scores[fg_masks] = gt_score_targets[fg_masks, target_labels[fg_masks]]
        cls_preds = cls_preds.view(-1, self.num_classes)
        loss_cls = self.loss_classes(cls_preds, target_labels, target_scores)
        loss_cls = loss_cls.sum() / num_fgs

        # ------------------ Regression loss ------------------
        box_preds_pos = box_preds.view(-1, 4)[fg_masks]
        box_targets_pos = gt_bbox_targets.view(-1, 4)[fg_masks]
        bbox_weight = gt_score_targets[fg_masks].sum(-1)
        loss_box = self.loss_bboxes(box_preds_pos, box_targets_pos, bbox_weight)
        loss_box = loss_box.sum() / num_fgs

        # total loss
        losses = loss_cls * self.loss_cls_weight + loss_box * self.loss_box_weight
        loss_dict = dict(
                loss_cls = loss_cls,
                loss_box = loss_box,
                losses = losses
        )

        return loss_dict
    

if __name__ == "__main__":
    pass
