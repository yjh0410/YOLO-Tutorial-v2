import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.box_ops import get_ious
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized

from .matcher import AlignedOTAMatcher


class SetCriterion(nn.Module):
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
        # ------------- Matcher & Loss weight -------------
        self.matcher_cfg = cfg.matcher_hpy
        self.matcher = AlignedOTAMatcher(cfg.num_classes,
                                         cfg.matcher_hpy['soft_center_radius'],
                                         cfg.matcher_hpy['topk_candidates'],
                                         )

    def loss_labels(self, pred_cls, target, beta=2.0, num_boxes=1.0):
        # Quality FocalLoss
        """
            pred_cls: (torch.Tensor): [N, C]。
            target:   (tuple([torch.Tensor], [torch.Tensor])): label -> (N,), score -> (N)
        """
        label, score = target
        pred_sigmoid = pred_cls.sigmoid()
        scale_factor = pred_sigmoid
        zerolabel = scale_factor.new_zeros(pred_cls.shape)

        ce_loss = F.binary_cross_entropy_with_logits(
            pred_cls, zerolabel, reduction='none') * scale_factor.pow(beta)
        
        bg_class_ind = pred_cls.shape[-1]
        pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
        if pos.shape[0] > 0:
            pos_label = label[pos].long()

            scale_factor = score[pos] - pred_sigmoid[pos, pos_label]

            ce_loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
                pred_cls[pos, pos_label], score[pos],
                reduction='none') * scale_factor.abs().pow(beta)

        return ce_loss.sum() / num_boxes
    
    def loss_bboxes(self, pred_box, gt_box, num_boxes=1.0, box_weight=None):
        ious = get_ious(pred_box, gt_box, box_mode="xyxy", iou_type='giou')
        loss_box = 1.0 - ious

        if box_weight is not None:
            loss_box = loss_box.squeeze(-1) * box_weight

        return loss_box.sum() / num_boxes

    def compute_loss(self, outputs, targets):
        """
            outputs['pred_cls']: (Tensor) [B, M, C]
            outputs['pred_reg']: (Tensor) [B, M, 4]
            outputs['pred_box']: (Tensor) [B, M, 4]
            outputs['strides']: (List) [8, 16, 32, ...] stride of the model output
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]
        """
        # -------------------- Pre-process --------------------
        bs          = outputs['pred_cls'][0].shape[0]
        device      = outputs['pred_cls'][0].device
        fpn_strides = outputs['strides']
        anchors     = outputs['anchors']
        # preds: [B, M, C]
        cls_preds = torch.cat(outputs['pred_cls'], dim=1)
        box_preds = torch.cat(outputs['pred_box'], dim=1)
        masks = ~torch.cat(outputs['mask'], dim=1).view(-1)

        # -------------------- Label Assignment --------------------
        cls_targets = []
        box_targets = []
        assign_metrics = []
        for batch_idx in range(bs):
            tgt_labels = targets[batch_idx]["labels"].to(device)  # [N,]
            tgt_bboxes = targets[batch_idx]["boxes"].to(device)   # [N, 4]
            # refine target
            tgt_boxes_wh = tgt_bboxes[..., 2:] - tgt_bboxes[..., :2]
            min_tgt_size = torch.min(tgt_boxes_wh, dim=-1)[0]
            keep = (min_tgt_size >= 8)
            tgt_bboxes = tgt_bboxes[keep]
            tgt_labels = tgt_labels[keep]
            # label assignment
            assigned_result = self.matcher(fpn_strides=fpn_strides,
                                           anchors=anchors,
                                           pred_cls=cls_preds[batch_idx].detach(),
                                           pred_box=box_preds[batch_idx].detach(),
                                           gt_labels=tgt_labels,
                                           gt_bboxes=tgt_bboxes
                                           )
            cls_targets.append(assigned_result['assigned_labels'])
            box_targets.append(assigned_result['assigned_bboxes'])
            assign_metrics.append(assigned_result['assign_metrics'])

        # List[B, M, C] -> Tensor[BM, C]
        cls_targets = torch.cat(cls_targets, dim=0)
        box_targets = torch.cat(box_targets, dim=0)
        assign_metrics = torch.cat(assign_metrics, dim=0)

        valid_idxs = (cls_targets >= 0) & masks
        foreground_idxs = (cls_targets >= 0) & (cls_targets != self.num_classes)
        num_fgs = assign_metrics.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_fgs)
        num_fgs = torch.clamp(num_fgs / get_world_size(), min=1).item()

        # -------------------- Classification loss --------------------
        cls_preds = cls_preds.view(-1, self.num_classes)[valid_idxs]
        qfl_targets = (cls_targets[valid_idxs], assign_metrics[valid_idxs])
        loss_labels = self.loss_labels(cls_preds, qfl_targets, 2.0, num_fgs)

        # -------------------- Regression loss --------------------
        box_preds_pos = box_preds.view(-1, 4)[foreground_idxs]
        box_targets_pos = box_targets[foreground_idxs]
        box_weight = assign_metrics[foreground_idxs]
        loss_bboxes = self.loss_bboxes(box_preds_pos, box_targets_pos, num_fgs, box_weight)

        total_loss = loss_labels * self.weight_dict["loss_cls"] + \
                         loss_bboxes * self.weight_dict["loss_reg"]
        loss_dict = dict(
                loss_cls = loss_labels,
                loss_reg = loss_bboxes,
                loss     = total_loss,
        )

        return loss_dict
    
    def forward(self, outputs, targets):
        """
            outputs['pred_cls']: (Tensor) [B, M, C]
            outputs['pred_reg']: (Tensor) [B, M, 4]
            outputs['pred_box']: (Tensor) [B, M, 4]
            outputs['strides']: (List) [8, 16, 32, ...] stride of the model output
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]
        """
        self.matcher.topk_candidates = self.cfg.matcher_hpy['topk_candidates']
        o2m_loss_dict = self.compute_loss(outputs["outputs_o2m"], targets)
        self.matcher.topk_candidates = 1
        o2o_loss_dict = self.compute_loss(outputs["outputs_o2o"], targets)

        loss_dict = {}
        loss_dict["loss"] = o2o_loss_dict["loss"] + o2m_loss_dict["loss"]
        for k in o2m_loss_dict:
            loss_dict['o2m_' + k] = o2m_loss_dict[k]
        for k in o2o_loss_dict:
            loss_dict['o2o_' + k] = o2o_loss_dict[k]

        return loss_dict
    

if __name__ == "__main__":
    pass
