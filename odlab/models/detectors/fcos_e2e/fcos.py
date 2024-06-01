import copy
import torch
import torch.nn as nn

# --------------- Model components ---------------
from ...backbone import build_backbone
from ...neck import build_neck
from ...head import build_head


# --------------------- End-to-End RT-FCOS ---------------------
class FcosE2E(nn.Module):
    def __init__(self, 
                 cfg,
                 conf_thresh  :float = 0.05,
                 topk_results :int   = 1000,
                 ):
        super(FcosE2E, self).__init__()
        # ---------------------- Basic Parameters ----------------------
        self.conf_thresh  = conf_thresh
        self.num_classes  = cfg.num_classes
        self.topk_results = topk_results

        # ---------------------- Network Parameters ----------------------
        ## Backbone
        self.backbone, pyramid_feats = build_backbone(cfg)

        ## Neck
        self.backbone_fpn = build_neck(cfg, pyramid_feats, cfg.head_dim)

        ## Heads (one-to-many)
        self.detection_head_o2m = build_head(cfg, cfg.head_dim, cfg.head_dim)

        ## Heads (one-to-one)
        self.detection_head_o2o = copy.deepcopy(self.detection_head_o2m)

    def post_process(self, cls_preds, box_preds):
        """
        Input:
            cls_preds: List(Tensor) [[B, H x W, C], ...]
            box_preds: List(Tensor) [[B, H x W, 4], ...]
        """
        all_scores = []
        all_labels = []
        all_bboxes = []
        
        for cls_pred_i, box_pred_i in zip(cls_preds, box_preds):
            cls_pred_i = cls_pred_i[0]
            box_pred_i = box_pred_i[0]
            
            # (H x W x C,)
            scores_i = cls_pred_i.sigmoid().flatten()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_results, box_pred_i.size(0))

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

        return bboxes, scores, labels

    def inference_o2o(self, src):
        # ---------------- Backbone ----------------
        pyramid_feats = self.backbone(src)

        # ---------------- Neck ----------------
        pyramid_feats = self.backbone_fpn(pyramid_feats)

        # ---------------- Heads ----------------
        outputs = self.detection_head_o2o(pyramid_feats)
        cls_pred = outputs["pred_cls"]
        box_pred = outputs["pred_box"]

        # PostProcess (no NMS)
        bboxes, scores, labels = self.post_process(cls_pred, box_pred)

        # Normalize bbox
        bboxes[..., 0::2] /= src.shape[-1]
        bboxes[..., 1::2] /= src.shape[-2]
        bboxes = bboxes.clip(0., 1.)

        outputs = {
            'scores': scores,
            'labels': labels,
            'bboxes': bboxes
        }

        return outputs

    def forward(self, src, src_mask=None):
        if not self.training:
            return self.inference_o2o(src)
        else:
            # ---------------- Backbone ----------------
            pyramid_feats = self.backbone(src)

            # ---------------- Neck ----------------
            pyramid_feats = self.backbone_fpn(pyramid_feats)

            # ---------------- Heads ----------------
            outputs = {}
            ## One-to-many detection
            outputs_o2m = self.detection_head_o2m(pyramid_feats, src_mask)
            outputs["outputs_o2m"] = outputs_o2m
            ## One-to-one  detection
            pyramid_feats_detach = [feat.detach() for feat in pyramid_feats]
            outputs_o2o = self.detection_head_o2o(pyramid_feats_detach, src_mask)
            outputs["outputs_o2o"] = outputs_o2o

            return outputs 
