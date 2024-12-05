# --------------- Torch components ---------------
import torch
import torch.nn as nn

# --------------- Model components ---------------
from .yolov1_backbone import Yolov1Backbone
from .yolov1_neck     import SPPF
from .yolov1_head     import Yolov1DetHead
from .yolov1_pred     import Yolov1DetPredLayer

# --------------- External components ---------------
from utils.misc import multiclass_nms


# YOLOv1
class Yolov1(nn.Module):
    def __init__(self,
                 cfg,
                 is_val = False,
                 ) -> None:
        super(Yolov1, self).__init__()
        # ---------------------- Basic setting ----------------------
        self.cfg = cfg
        self.num_classes = cfg.num_classes
        ## Post-process parameters
        self.topk_candidates  = cfg.val_topk        if is_val else cfg.test_topk
        self.conf_thresh      = cfg.val_conf_thresh if is_val else cfg.test_conf_thresh
        self.nms_thresh       = cfg.val_nms_thresh  if is_val else cfg.test_nms_thresh
        self.no_multi_labels  = False if is_val else True
        
        # ---------------------- Network Parameters ----------------------
        self.backbone = Yolov1Backbone(cfg)
        self.neck     = SPPF(self.backbone.feat_dim, cfg.head_dim)
        self.head     = Yolov1DetHead(cfg, self.neck.out_dim)
        self.pred     = Yolov1DetPredLayer(cfg)

    def post_process(self, obj_preds, cls_preds, box_preds):
        """
        We process predictions at each scale hierarchically
        Input:
            obj_preds: torch.Tensor -> [B, M, 1], B=1
            cls_preds: torch.Tensor -> [B, M, C], B=1
            box_preds: torch.Tensor -> [B, M, 4], B=1
        Output:
            bboxes: np.array -> [N, 4]
            scores: np.array -> [N,]
            labels: np.array -> [N,]
        """        
        obj_preds = obj_preds[0]
        cls_preds = cls_preds[0]
        box_preds = box_preds[0]
        if self.no_multi_labels:
            # [M,]
            scores, labels = torch.max(
                torch.sqrt(obj_preds.sigmoid() * cls_preds.sigmoid()), dim=1)

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_preds.size(0))

            # topk candidates
            predicted_prob, topk_idxs = scores.sort(descending=True)
            topk_scores = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = topk_scores > self.conf_thresh
            scores = topk_scores[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            labels = labels[topk_idxs]
            bboxes = box_preds[topk_idxs]
        else:
            # [M, C] -> [MC,]
            scores = torch.sqrt(obj_preds.sigmoid() * cls_preds.sigmoid()).flatten()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_preds.size(0))

            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = scores.sort(descending=True)
            topk_scores = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = topk_scores > self.conf_thresh
            scores = topk_scores[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
            labels = topk_idxs % self.num_classes

            bboxes = box_preds[anchor_idxs]

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
        x = self.backbone(x)

        # ---------------- Neck ----------------
        x = self.neck(x)

        # ---------------- Heads ----------------
        cls_feats, reg_feats = self.head(x)

        # ---------------- Preds ----------------
        outputs = self.pred(cls_feats, reg_feats)
        outputs['image_size'] = [x.shape[2], x.shape[3]]

        if not self.training:
            obj_preds = outputs['pred_obj']
            cls_preds = outputs['pred_cls']
            box_preds = outputs['pred_box']

            # post process
            bboxes, scores, labels = self.post_process(
                obj_preds, cls_preds, box_preds)
            outputs = {
                "scores": scores,
                "labels": labels,
                "bboxes": bboxes
            }
        
        return outputs 
