import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import sigmoid_focal_loss
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized

from .matcher import FcosMatcher


class SetCriterion(object):
    def __init__(self, cfg):
        # ------------- Basic parameters -------------
        self.cfg = cfg
        self.num_classes = cfg.num_classes

        # ------------- Focal loss -------------
        self.alpha = cfg.focal_loss_alpha
        self.gamma = cfg.focal_loss_gamma

        # ------------- Loss weight -------------
        self.weight_dict = {'loss_cls': cfg.loss_cls,
                            'loss_reg': cfg.loss_reg,
                            'loss_ctn': cfg.loss_ctn,}
        
        # ------------- Matcher -------------
        self.matcher = FcosMatcher(cfg.num_classes,
                                   center_sampling_radius=cfg.center_sampling_radius,
                                   object_sizes_of_interest=cfg.object_sizes_of_interest,
                                   box_weights=[1., 1., 1., 1.],
                                   )

    def loss_labels(self, pred_cls, tgt_cls, num_boxes=1.0):
        """
            pred_cls: (Tensor) [N, C]
            tgt_cls:  (Tensor) [N, C]
        """
        # cls loss: [V, C]
        loss_cls = sigmoid_focal_loss(pred_cls, tgt_cls, self.alpha, self.gamma)

        return loss_cls.sum() / num_boxes

    def loss_bboxes(self, pred_delta, tgt_delta, bbox_quality=None, num_boxes=1.0):
        """
            pred_box: (Tensor) [N, 4]
            tgt_box:  (Tensor) [N, 4]
        """
        pred_delta = torch.cat((-pred_delta[..., :2], pred_delta[..., 2:]), dim=-1)
        tgt_delta = torch.cat((-tgt_delta[..., :2], tgt_delta[..., 2:]), dim=-1)

        eps = torch.finfo(torch.float32).eps

        pred_area = (pred_delta[..., 2] - pred_delta[..., 0]).clamp_(min=0) \
            * (pred_delta[..., 3] - pred_delta[..., 1]).clamp_(min=0)
        tgt_area = (tgt_delta[..., 2] - tgt_delta[..., 0]).clamp_(min=0) \
            * (tgt_delta[..., 3] - tgt_delta[..., 1]).clamp_(min=0)

        w_intersect = (torch.min(pred_delta[..., 2], tgt_delta[..., 2])
                    - torch.max(pred_delta[..., 0], tgt_delta[..., 0])).clamp_(min=0)
        h_intersect = (torch.min(pred_delta[..., 3], tgt_delta[..., 3])
                    - torch.max(pred_delta[..., 1], tgt_delta[..., 1])).clamp_(min=0)

        area_intersect = w_intersect * h_intersect
        area_union = tgt_area + pred_area - area_intersect
        ious = area_intersect / area_union.clamp(min=eps)

        # giou
        g_w_intersect = torch.max(pred_delta[..., 2], tgt_delta[..., 2]) \
            - torch.min(pred_delta[..., 0], tgt_delta[..., 0])
        g_h_intersect = torch.max(pred_delta[..., 3], tgt_delta[..., 3]) \
            - torch.min(pred_delta[..., 1], tgt_delta[..., 1])
        ac_uion = g_w_intersect * g_h_intersect
        gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)
        loss_box = 1 - gious

        if bbox_quality is not None:
            loss_box = loss_box * bbox_quality.view(loss_box.size())

        return loss_box.sum() / num_boxes

    def __call__(self, outputs, targets):
        """
            outputs['pred_cls']: (Tensor) [B, M, C]
            outputs['pred_reg']: (Tensor) [B, M, 4]
            outputs['pred_ctn']: (Tensor) [B, M, 1]
            outputs['strides']: (List) [8, 16, 32, ...] stride of the model output
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]
        """
        # -------------------- Pre-process --------------------
        device = outputs['pred_cls'][0].device
        fpn_strides = outputs['strides']
        anchors = outputs['anchors']

        pred_cls   = torch.cat(outputs['pred_cls'], dim=1).view(-1, self.num_classes)
        pred_delta = torch.cat(outputs['pred_reg'], dim=1).view(-1, 4)
        pred_ctn   = torch.cat(outputs['pred_ctn'], dim=1).view(-1, 1)

        # -------------------- Label Assignment --------------------
        gt_classes, gt_deltas, gt_centerness = self.matcher(fpn_strides, anchors, targets)
        gt_classes = gt_classes.flatten().to(device)
        gt_deltas = gt_deltas.view(-1, 4).to(device)
        gt_centerness = gt_centerness.view(-1, 1).to(device)

        fg_masks = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_fgs = fg_masks.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_fgs)
        num_fgs = torch.clamp(num_fgs / get_world_size(), min=1).item()

        num_fgs_ctn = gt_centerness[fg_masks].sum()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_fgs_ctn)
        num_targets = torch.clamp(num_fgs_ctn / get_world_size(), min=1).item()

        # -------------------- classification loss --------------------
        gt_classes_target = torch.zeros_like(pred_cls)
        gt_classes_target[fg_masks, gt_classes[fg_masks]] = 1
        loss_labels = self.loss_labels(pred_cls, gt_classes_target, num_fgs)

        # -------------------- regression loss --------------------
        loss_bboxes = self.loss_bboxes(
            pred_delta[fg_masks], gt_deltas[fg_masks], gt_centerness[fg_masks], num_targets)

        # -------------------- centerness loss --------------------
        loss_centerness = F.binary_cross_entropy_with_logits(
            pred_ctn[fg_masks],  gt_centerness[fg_masks], reduction='none')
        loss_centerness = loss_centerness.sum() / num_fgs

        total_loss = loss_labels * self.weight_dict["loss_cls"] + \
                     loss_bboxes * self.weight_dict["loss_reg"] + \
                     loss_centerness * self.weight_dict["loss_ctn"]
        loss_dict = dict(
                loss_cls = loss_labels,
                loss_reg = loss_bboxes,
                loss_ctn = loss_centerness,
                losses   = total_loss,
        )

        return loss_dict
