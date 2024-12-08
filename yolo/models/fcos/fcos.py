import torch
import torch.nn as nn

# --------------- Model components ---------------
from .fcos_backbone import FcosBackbone
from .fcos_fpn import FcosFPN
from .fcos_head import FcosHead

# --------------- External components ---------------
from utils.misc import multiclass_nms


# ------------------------ Fully Convolutional One-Stage Detector ------------------------
class Fcos(nn.Module):
    def __init__(self, 
                 cfg,
                 is_val = False,
                 ) -> None:
        super(Fcos, self).__init__()
        # ---------------------- Basic setting ----------------------
        self.cfg = cfg
        self.num_classes = cfg.num_classes
        ## Post-process parameters
        self.topk_candidates  = cfg.val_topk        if is_val else cfg.test_topk
        self.conf_thresh      = cfg.val_conf_thresh if is_val else cfg.test_conf_thresh
        self.nms_thresh       = cfg.val_nms_thresh  if is_val else cfg.test_nms_thresh
        self.no_multi_labels  = False if is_val else True

        # ---------------------- Network Parameters ----------------------
        self.backbone = FcosBackbone(cfg)
        self.fpn      = FcosFPN(cfg, self.backbone.feat_dims[-3:])
        self.head     = FcosHead(cfg, self.fpn.out_dim)

    def post_process(self, cls_preds, ctn_preds, box_preds):
        """
        Input:
            cls_preds: List(Tensor) [[B, H x W, C], ...]
            ctn_preds: List(Tensor) [[B, H x W, 1], ...]
            box_preds: List(Tensor) [[B, H x W, 4], ...]
        """
        all_scores = []
        all_labels = []
        all_bboxes = []
        
        for cls_pred_i, ctn_pred_i, box_pred_i in zip(cls_preds, ctn_preds, box_preds):
            cls_pred_i = cls_pred_i[0]
            ctn_pred_i = ctn_pred_i[0]
            box_pred_i = box_pred_i[0]
            
            # (H x W x C,)
            scores_i = torch.sqrt(cls_pred_i.sigmoid() * ctn_pred_i.sigmoid()).flatten()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_pred_i.size(0))

            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = scores_i.sort(descending=True)
            topk_scores = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = topk_scores > self.conf_thresh
            topk_idxs = topk_idxs[keep_idxs]

            # final scores
            scores = topk_scores[keep_idxs]
            # final labels
            labels = topk_idxs % self.num_classes
            # final bboxes
            anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
            bboxes = box_pred_i[anchor_idxs]

            all_scores.append(scores)
            all_labels.append(labels)
            all_bboxes.append(bboxes)

        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        bboxes = torch.cat(all_bboxes)

        # to cpu & numpy
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes)

        return bboxes, scores, labels

    def forward(self, x):
        # ---------------- Backbone ----------------
        pyramid_feats = self.backbone(x)

        # ---------------- Neck ----------------
        pyramid_feats = self.fpn(pyramid_feats)

        # ---------------- Heads ----------------
        outputs = self.head(pyramid_feats)

        if not self.training:
            # ---------------- PostProcess ----------------
            cls_pred = outputs["pred_cls"]
            ctn_pred = outputs["pred_ctn"]
            box_pred = outputs["pred_box"]
            bboxes, scores, labels = self.post_process(cls_pred, ctn_pred, box_pred)

            outputs = {
                'scores': scores,
                'labels': labels,
                'bboxes': bboxes
            }

        return outputs 
