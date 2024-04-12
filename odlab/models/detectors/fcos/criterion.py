import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.box_ops import get_ious
from utils.misc import sigmoid_focal_loss
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized

from .matcher import FcosMatcher, SimOtaMatcher


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
        # ------------- Matcher & Loss weight -------------
        self.matcher_cfg = cfg.matcher_hpy
        if cfg.matcher == 'fcos_matcher':
            self.weight_dict = {'loss_cls': cfg.loss_cls_weight,
                                'loss_reg': cfg.loss_reg_weight,
                                'loss_ctn': cfg.loss_ctn_weight}
            self.matcher = FcosMatcher(cfg.num_classes,
                                       self.matcher_cfg['center_sampling_radius'],
                                       self.matcher_cfg['object_sizes_of_interest'],
                                       [1., 1., 1., 1.]
                                       )
        elif cfg.matcher == 'simota':
            self.weight_dict = {'loss_cls': cfg.loss_cls_weight,
                                'loss_reg': cfg.loss_reg_weight}
            self.matcher = SimOtaMatcher(cfg.num_classes,
                                         self.matcher_cfg['soft_center_radius'],
                                         self.matcher_cfg['topk_candidates'])
        else:
            raise NotImplementedError("Unknown matcher: {}.".format(cfg.matcher))

    def loss_labels(self, pred_cls, tgt_cls, num_boxes=1.0):
        """
            pred_cls: (Tensor) [N, C]
            tgt_cls:  (Tensor) [N, C]
        """
        # cls loss: [V, C]
        loss_cls = sigmoid_focal_loss(pred_cls, tgt_cls, self.alpha, self.gamma)

        return loss_cls.sum() / num_boxes

    def loss_labels_qfl(self, pred_cls, target, beta=2.0, num_boxes=1.0):
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
    
    def loss_bboxes_ltrb(self, pred_delta, tgt_delta, bbox_quality=None, num_boxes=1.0):
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

    def loss_bboxes_xyxy(self, pred_box, gt_box, num_boxes=1.0, box_weight=None):
        ious = get_ious(pred_box, gt_box, box_mode="xyxy", iou_type='giou')
        loss_box = 1.0 - ious

        if box_weight is not None:
            loss_box = loss_box.squeeze(-1) * box_weight

        return loss_box.sum() / num_boxes
    
    def fcos_loss(self, outputs, targets):
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
        pred_cls = torch.cat(outputs['pred_cls'], dim=1).view(-1, self.num_classes)
        pred_delta = torch.cat(outputs['pred_reg'], dim=1).view(-1, 4)
        pred_ctn = torch.cat(outputs['pred_ctn'], dim=1).view(-1, 1)
        masks = ~torch.cat(outputs['mask'], dim=1).view(-1)

        # -------------------- Label Assignment --------------------
        gt_classes, gt_deltas, gt_centerness = self.matcher(fpn_strides, anchors, targets)
        gt_classes = gt_classes.flatten().to(device)
        gt_deltas = gt_deltas.view(-1, 4).to(device)
        gt_centerness = gt_centerness.view(-1, 1).to(device)

        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foreground)
        num_foreground = torch.clamp(num_foreground / get_world_size(), min=1).item()

        num_foreground_centerness = gt_centerness[foreground_idxs].sum()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foreground_centerness)
        num_targets = torch.clamp(num_foreground_centerness / get_world_size(), min=1).item()

        # -------------------- classification loss --------------------
        gt_classes_target = torch.zeros_like(pred_cls)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1
        valid_idxs = (gt_classes >= 0) & masks
        loss_labels = self.loss_labels(
            pred_cls[valid_idxs], gt_classes_target[valid_idxs], num_foreground)

        # -------------------- regression loss --------------------
        loss_bboxes = self.loss_bboxes_ltrb(
            pred_delta[foreground_idxs], gt_deltas[foreground_idxs], gt_centerness[foreground_idxs], num_targets)

        # -------------------- centerness loss --------------------
        loss_centerness = F.binary_cross_entropy_with_logits(
            pred_ctn[foreground_idxs],  gt_centerness[foreground_idxs], reduction='none')
        loss_centerness = loss_centerness.sum() / num_foreground

        loss_dict = dict(
                loss_cls = loss_labels,
                loss_reg = loss_bboxes,
                loss_ctn = loss_centerness,
        )

        return loss_dict
    
    def ota_loss(self, outputs, targets):
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

        # -------------------- classification loss --------------------
        cls_preds = cls_preds.view(-1, self.num_classes)[valid_idxs]
        qfl_targets = (cls_targets[valid_idxs], assign_metrics[valid_idxs])
        loss_labels = self.loss_labels_qfl(cls_preds, qfl_targets, 2.0, num_fgs)

        # -------------------- regression loss --------------------
        box_preds_pos = box_preds.view(-1, 4)[foreground_idxs]
        box_targets_pos = box_targets[foreground_idxs]
        box_weight = assign_metrics[foreground_idxs]
        loss_bboxes = self.loss_bboxes_xyxy(box_preds_pos, box_targets_pos, num_fgs, box_weight)

        loss_dict = dict(
                loss_cls = loss_labels,
                loss_reg = loss_bboxes,
        )

        return loss_dict
    
    def forward(self, outputs, targets):
        """
            outputs['pred_cls']: (Tensor) [B, M, C]
            outputs['pred_reg']: (Tensor) [B, M, 4]
            outputs['pred_ctn']: (Tensor) [B, M, 1]
            outputs['strides']: (List) [8, 16, 32, ...] stride of the model output
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]
        """
        if self.cfg.matcher == "fcos_matcher":
            return self.fcos_loss(outputs, targets)
        elif self.cfg.matcher == "simota":
            return self.ota_loss(outputs, targets)
        else:
            raise NotImplementedError


if __name__ == "__main__":
    pass
