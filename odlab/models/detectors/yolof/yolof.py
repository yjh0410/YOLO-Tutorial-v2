import torch
import torch.nn as nn

# --------------- Model components ---------------
from ...backbone import build_backbone
from ...neck import build_neck
from ...head import build_head

# --------------- External components ---------------
from utils.misc import multiclass_nms


# ------------------------ You Only Look One-level Feature ------------------------
class YOLOF(nn.Module):
    def __init__(self, 
                 cfg,
                 num_classes :int   = 80, 
                 conf_thresh :float = 0.05,
                 nms_thresh  :float = 0.6,
                 topk        :int   = 1000,
                 ca_nms      :bool  = False):
        super(YOLOF, self).__init__()
        # ---------------------- Basic Parameters ----------------------
        self.cfg = cfg
        self.topk = topk
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.ca_nms = ca_nms

        # ---------------------- Network Parameters ----------------------
        ## Backbone
        self.backbone, feat_dims = build_backbone(cfg)

        ## Neck
        self.neck = build_neck(cfg, feat_dims[-1], cfg['head_dim'])
        
        ## Heads
        self.head = build_head(cfg, cfg['head_dim'], cfg['head_dim'], num_classes)

    def post_process(self, cls_pred, box_pred):
        """
        Input:
            cls_pred: (Tensor) [[H x W x KA, C]
            box_pred: (Tensor)  [H x W x KA, 4]
        """
        cls_pred = cls_pred[0]
        box_pred = box_pred[0]
        
        # (H x W x KA x C,)
        scores_i = cls_pred.sigmoid().flatten()

        # Keep top k top scoring indices only.
        num_topk = min(self.topk, box_pred.size(0))

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
        bboxes = box_pred[anchor_idxs]

        # to cpu & numpy
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, self.ca_nms)

        return bboxes, scores, labels

    def forward(self, src, src_mask=None, targets=None):
        # ---------------- Backbone ----------------
        pyramid_feats = self.backbone(src)

        # ---------------- Neck ----------------
        feat = self.neck(pyramid_feats[-1])

        # ---------------- Heads ----------------
        outputs = self.head(feat, src_mask)

        if not self.training:
            # ---------------- PostProcess ----------------
            cls_pred = outputs["pred_cls"]
            box_pred = outputs["pred_box"]
            bboxes, scores, labels = self.post_process(cls_pred, box_pred)
            # normalize bbox
            bboxes[..., 0::2] /= src.shape[-1]
            bboxes[..., 1::2] /= src.shape[-2]
            bboxes = bboxes.clip(0., 1.)

            return bboxes, scores, labels

        return outputs 
