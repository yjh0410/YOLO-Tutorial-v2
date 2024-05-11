"""
reference: 
https://github.com/facebookresearch/detr/blob/main/models/detr.py

by lyuwenyu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized
from .matcher import HungarianMatcher


# --------------- Criterion for DETR ---------------
class SetCriterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg.num_classes
        self.losses = ['labels', 'boxes']
        # -------- Loss weights --------
        self.weight_dict = {'loss_cls':  cfg.loss_cls,
                            'loss_box':  cfg.loss_box,
                            'loss_giou': cfg.loss_giou}
        for i in range(cfg.num_dec_layers - 1):
            self.weight_dict.update({k + f'_aux_{i}': v for k, v in self.weight_dict.items()})
        # -------- Matcher --------
        self.matcher = HungarianMatcher(cfg.cost_class, cfg.cost_bbox, cfg.cost_giou)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_cls = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)

        return {'loss_cls': loss_cls.sum() / num_boxes}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes),
                                                       box_cxcywh_to_xyxy(target_boxes)))

        return {'loss_box': loss_bbox.sum() / num_boxes,
                'loss_giou': loss_giou.sum() / num_boxes}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])

        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])

        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'labels': self.loss_labels,
        }
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        '''get_cdn_matched_indices
        '''
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))
        
        return dn_match_indices
