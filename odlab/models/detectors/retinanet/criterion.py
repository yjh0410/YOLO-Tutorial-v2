import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from utils.misc import sigmoid_focal_loss
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized

from .matcher import RetinaNetMatcher


class Criterion(nn.Module):
    def __init__(self, cfg, num_classes=80):
        super().__init__()
        # ------------- Basic parameters -------------
        self.cfg = cfg
        self.num_classes = num_classes
        # ------------- Focal loss -------------
        self.alpha = cfg['focal_loss_alpha']
        self.gamma = cfg['focal_loss_gamma']
        # ------------- Loss weight -------------
        self.weight_dict = {'loss_cls': cfg['loss_cls_weight'],
                            'loss_reg': cfg['loss_reg_weight']}
        # ------------- Matcher -------------
        self.matcher_cfg = cfg['matcher_hpy']
        self.matcher = RetinaNetMatcher(num_classes,
                                        iou_threshold=self.matcher_cfg['iou_thresh'],
                                        iou_labels=self.matcher_cfg['iou_labels'],
                                        allow_low_quality_matches=self.matcher_cfg['allow_low_quality_matches']
                                        )

    def loss_labels(self, pred_cls, tgt_cls, num_boxes):
        """
            pred_cls: (Tensor) [N, C]
            tgt_cls:  (Tensor) [N, C]
        """
        # cls loss: [V, C]
        loss_cls = sigmoid_focal_loss(pred_cls, tgt_cls, self.alpha, self.gamma)

        return loss_cls.sum() / num_boxes

    def loss_bboxes(self, pred_reg=None, pred_box=None, tgt_box=None, anchors=None, num_boxes=1, use_giou=False):
        """
            pred_reg: (Tensor) [Nq, 4]
            tgt_box:  (Tensor) [Nq, 4]
            anchors:  (Tensor) [Nq, 4]
        """
        # GIoU loss
        if use_giou:
            pred_giou = generalized_box_iou(pred_box, tgt_box)  # [N, M]
            loss_reg = 1. - torch.diag(pred_giou)
        
        # L1 loss
        else:
            # xyxy -> cxcy&bwbh
            tgt_cxcy = (tgt_box[..., :2] + tgt_box[..., 2:]) * 0.5
            tgt_bwbh = tgt_box[..., 2:] - tgt_box[..., :2]
            # encode gt box
            tgt_offsets = (tgt_cxcy - anchors[..., :2]) / anchors[..., 2:]
            tgt_sizes = torch.log(tgt_bwbh / anchors[..., 2:])
            tgt_box_encode = torch.cat([tgt_offsets, tgt_sizes], dim=-1)
            # compute l1 loss
            loss_reg = F.l1_loss(pred_reg, tgt_box_encode, reduction='none')

        return loss_reg.sum() / num_boxes

    def forward(self, outputs, targets):
        """
            outputs['pred_cls']: (Tensor) [B, M, C]
            outputs['pred_reg']: (Tensor) [B, M, 4]
            outputs['strides']: (List) [8, 16, 32, ...] stride of the model output
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]
            anchors: (Tensor) [M, 4]
        """
        # -------------------- Pre-process --------------------
        cls_preds = torch.cat(outputs['pred_cls'], dim=1).view(-1, self.num_classes)
        reg_preds = torch.cat(outputs['pred_reg'], dim=1).view(-1, 4)
        box_preds = torch.cat(outputs['pred_box'], dim=1).view(-1, 4)
        masks = ~torch.cat(outputs['mask'], dim=1).view(-1)
        B = len(targets)
       
        # process anchor boxes
        anchor_boxes = torch.cat(outputs['anchors'])
        anchor_boxes = anchor_boxes[None].repeat(B, 1, 1)
        anchor_boxes_xyxy = box_cxcywh_to_xyxy(anchor_boxes)

        # -------------------- Label Assignment --------------------
        tgt_classes, tgt_boxes = self.matcher(anchor_boxes_xyxy, targets)
        tgt_classes = tgt_classes.flatten()
        tgt_boxes = tgt_boxes.view(-1, 4)
        del anchor_boxes_xyxy

        foreground_idxs = (tgt_classes >= 0) & (tgt_classes != self.num_classes)
        valid_idxs = (tgt_classes >= 0) & masks
        num_foreground = foreground_idxs.sum()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foreground)
        num_foreground = torch.clamp(num_foreground / get_world_size(), min=1).item()

        # -------------------- Classification loss --------------------
        gt_cls_target = torch.zeros_like(cls_preds)
        gt_cls_target[foreground_idxs, tgt_classes[foreground_idxs]] = 1
        loss_labels = self.loss_labels(
            cls_preds[valid_idxs], gt_cls_target[valid_idxs], num_foreground)

        # -------------------- Regression loss --------------------
        if self.cfg['use_giou_loss']:
            box_preds_pos = box_preds[foreground_idxs]
            tgt_boxes_pos = tgt_boxes[foreground_idxs].to(reg_preds.device)
            loss_bboxes = self.loss_bboxes(
                pred_box=box_preds_pos, tgt_box=tgt_boxes_pos, num_boxes=num_foreground, use_giou=self.cfg['use_giou_loss'])
        else:
            reg_preds_pos = reg_preds[foreground_idxs]
            tgt_boxes_pos = tgt_boxes[foreground_idxs].to(reg_preds.device)
            anchors_pos = anchor_boxes.view(-1, 4)[foreground_idxs]
            loss_bboxes = self.loss_bboxes(
                pred_reg=reg_preds_pos, tgt_box=tgt_boxes_pos, anchors=anchors_pos, num_boxes=num_foreground, use_giou=self.cfg['use_giou_loss'])

        loss_dict = dict(
                loss_cls = loss_labels,
                loss_reg = loss_bboxes,
        )

        return loss_dict

    
# build criterion
def build_criterion(cfg, num_classes=80):
    criterion = Criterion(cfg=cfg, num_classes=num_classes)
    return criterion


if __name__ == "__main__":
    pass
