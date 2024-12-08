import torch
import torch.nn as nn

# --------------- Model components ---------------
from .yolof_backbone import YolofBackbone
from .yolof_encoder  import DilatedEncoder
from .yolof_decoder  import YolofHead

# --------------- External components ---------------
from utils.misc import multiclass_nms


# ------------------------ You Only Look One-level Feature ------------------------
class Yolof(nn.Module):
    def __init__(self, cfg, is_val: bool = False):
        super(Yolof, self).__init__()
        # ---------------------- Basic setting ----------------------
        self.cfg = cfg
        self.num_classes = cfg.num_classes
        ## Post-process parameters
        self.topk_candidates  = cfg.val_topk        if is_val else cfg.test_topk
        self.conf_thresh      = cfg.val_conf_thresh if is_val else cfg.test_conf_thresh
        self.nms_thresh       = cfg.val_nms_thresh  if is_val else cfg.test_nms_thresh
        self.no_multi_labels  = False if is_val else True

        # ---------------------- Network Parameters ----------------------
        self.backbone = YolofBackbone(cfg)
        self.encoder  = DilatedEncoder(cfg, self.backbone.feat_dim, cfg.head_dim)
        self.decoder  = YolofHead(cfg, self.encoder.out_dim, cfg.head_dim)

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
        num_topk = min(self.topk_candidates, box_pred.size(0))

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
            scores, labels, bboxes, self.nms_thresh, self.num_classes)

        return bboxes, scores, labels

    def forward(self, x):
        x = self.backbone(x)
        x = self.encoder(x)
        outputs = self.decoder(x)

        if not self.training:
            # ---------------- PostProcess ----------------
            cls_pred = outputs["pred_cls"]
            box_pred = outputs["pred_box"]
            bboxes, scores, labels = self.post_process(cls_pred, box_pred)

            outputs = {
                'scores': scores,
                'labels': labels,
                'bboxes': bboxes
            }

        return outputs 
