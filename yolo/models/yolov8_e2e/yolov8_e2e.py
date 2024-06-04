# --------------- Torch components ---------------
import copy
import torch
import torch.nn as nn

# --------------- Model components ---------------
from .yolov8_backbone import Yolov8Backbone
from .yolov8_neck     import SPPF
from .yolov8_pafpn    import Yolov8PaFPN
from .yolov8_head     import Yolov8DetHead
from .yolov8_pred     import Yolov8DetPredLayer


# End-to-End YOLOv8
class Yolov8E2E(nn.Module):
    def __init__(self, cfg, is_val = False):
        super(Yolov8E2E, self).__init__()
        # ---------------------- Basic setting ----------------------
        self.cfg = cfg
        self.num_classes = cfg.num_classes
        ## Post-process parameters
        self.topk_candidates  = cfg.val_topk        if is_val else cfg.test_topk
        self.conf_thresh      = cfg.val_conf_thresh if is_val else cfg.test_conf_thresh
        self.no_multi_labels  = False if is_val else True
        
        # ---------------------- Model Parameters ----------------------
        ## Backbone
        self.backbone = Yolov8Backbone(cfg)
        self.pyramid_feat_dims = self.backbone.feat_dims[-3:]
        ## Neck
        self.neck     = SPPF(cfg, self.pyramid_feat_dims[-1], self.pyramid_feat_dims[-1])
        self.pyramid_feat_dims[-1] = self.neck.out_dim
        ## Neck: PaFPN
        self.fpn      = Yolov8PaFPN(cfg, self.backbone.feat_dims)
        ## Head (one-to-one)
        self.head_o2o = Yolov8DetHead(cfg, self.fpn.out_dims)
        ## Pred (one-to-one)
        self.pred_o2o = Yolov8DetPredLayer(cfg, self.head_o2o.cls_head_dim, self.head_o2o.reg_head_dim)

        ## Aux head (one-to-many)
        self.head_o2m = copy.deepcopy(self.head_o2o)
        ## Aux Pred (one-to-many)
        self.pred_o2m = copy.deepcopy(self.pred_o2o)

    def post_process(self, cls_preds, box_preds):
        """
        We process predictions at each scale hierarchically
        Input:
            cls_preds: List[torch.Tensor] -> [[B, M, C], ...], B=1
            box_preds: List[torch.Tensor] -> [[B, M, 4], ...], B=1
        Output:
            bboxes: np.array -> [N, 4]
            scores: np.array -> [N,]
            labels: np.array -> [N,]
        """
        all_scores = []
        all_labels = []
        all_bboxes = []
        
        for cls_pred_i, box_pred_i in zip(cls_preds, box_preds):
            cls_pred_i = cls_pred_i[0]
            box_pred_i = box_pred_i[0]
            if self.no_multi_labels:
                # [M,]
                scores, labels = torch.max(cls_pred_i.sigmoid(), dim=1)

                # Keep top k top scoring indices only.
                num_topk = min(self.topk_candidates, box_pred_i.size(0))

                # topk candidates
                predicted_prob, topk_idxs = scores.sort(descending=True)
                topk_scores = predicted_prob[:num_topk]
                topk_idxs = topk_idxs[:num_topk]

                # filter out the proposals with low confidence score
                keep_idxs = topk_scores > self.conf_thresh
                scores = topk_scores[keep_idxs]
                topk_idxs = topk_idxs[keep_idxs]

                labels = labels[topk_idxs]
                bboxes = box_pred_i[topk_idxs]
            else:
                # [M, C] -> [MC,]
                scores_i = cls_pred_i.sigmoid().flatten()

                # Keep top k top scoring indices only.
                num_topk = min(self.topk_candidates, box_pred_i.size(0))

                # torch.sort is actually faster than .topk (at least on GPUs)
                predicted_prob, topk_idxs = scores_i.sort(descending=True)
                topk_scores = predicted_prob[:num_topk]
                topk_idxs = topk_idxs[:num_topk]

                # filter out the proposals with low confidence score
                keep_idxs = topk_scores > self.conf_thresh
                scores = topk_scores[keep_idxs]
                topk_idxs = topk_idxs[keep_idxs]

                anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
                labels = topk_idxs % self.num_classes

                bboxes = box_pred_i[anchor_idxs]

            all_scores.append(scores)
            all_labels.append(labels)
            all_bboxes.append(bboxes)

        scores = torch.cat(all_scores, dim=0)
        labels = torch.cat(all_labels, dim=0)
        bboxes = torch.cat(all_bboxes, dim=0)

        # to cpu & numpy
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        return bboxes, scores, labels
    
    def inference_o2o(self, x):
        # ---------------- Backbone ----------------
        pyramid_feats = self.backbone(x)
        # ---------------- Neck: SPP ----------------
        pyramid_feats[-1] = self.neck(pyramid_feats[-1])

        # ---------------- Neck: PaFPN ----------------
        pyramid_feats = self.fpn(pyramid_feats)

        # ---------------- Heads ----------------
        cls_feats, reg_feats = self.head_o2o(pyramid_feats)

        # ---------------- Preds ----------------
        outputs = self.pred_o2o(cls_feats, reg_feats)
        outputs['image_size'] = [x.shape[2], x.shape[3]]

        all_cls_preds = outputs['pred_cls']
        all_box_preds = outputs['pred_box']

        # post process (no NMS)
        bboxes, scores, labels = self.post_process(all_cls_preds, all_box_preds)
        outputs = {
            "scores": scores,
            "labels": labels,
            "bboxes": bboxes
        }
        
        return outputs 

    def forward(self, x):
        if not self.training:
            return self.inference_o2o(x)
        else:
            # ---------------- Backbone ----------------
            pyramid_feats = self.backbone(x)
            # ---------------- Neck: SPP ----------------
            pyramid_feats[-1] = self.neck(pyramid_feats[-1])

            # ---------------- Neck: PaFPN ----------------
            pyramid_feats = self.fpn(pyramid_feats)

            # ---------------- Heads ----------------
            o2m_cls_feats, o2m_reg_feats = self.head_o2m(pyramid_feats)

            # ---------------- Preds ----------------
            outputs_o2m = self.pred_o2m(o2m_cls_feats, o2m_reg_feats)
            outputs_o2m['image_size'] = [x.shape[2], x.shape[3]]
            
            # ---------------- Heads (one-to-one) ----------------
            o2o_cls_feats, o2o_reg_feats = self.head_o2o([feat.detach() for feat in pyramid_feats])

            # ---------------- Preds (one-to-one) ----------------
            outputs_o2o = self.pred_o2o(o2o_cls_feats, o2o_reg_feats)
            outputs_o2o['image_size'] = [x.shape[2], x.shape[3]]

            outputs = {
                "outputs_o2m": outputs_o2m,
                "outputs_o2o": outputs_o2o,
            }
            
            return outputs 
