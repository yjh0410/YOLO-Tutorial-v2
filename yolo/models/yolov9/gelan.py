# --------------- Torch components ---------------
import torch
import torch.nn as nn

# --------------- Model components ---------------
from .gelan_backbone import GElanBackbone
from .gelan_neck     import SPPElan
from .gelan_pafpn    import GElanPaFPN
from .gelan_head     import GElanDetHead
from .gelan_pred     import GElanPredLayer

# --------------- External components ---------------
from utils.misc import multiclass_nms


# G-ELAN proposed by YOLOv9
class GElan(nn.Module):
    def __init__(self,
                 cfg,
                 is_val = False,
                 deploy = False,
                 ) -> None:
        super(GElan, self).__init__()
        # ---------------------- Basic setting ----------------------
        self.cfg = cfg
        self.deploy = deploy
        self.num_classes = cfg.num_classes
        ## Post-process parameters
        self.topk_candidates = cfg.val_topk        if is_val else cfg.test_topk
        self.conf_thresh     = cfg.val_conf_thresh if is_val else cfg.test_conf_thresh
        self.nms_thresh      = cfg.val_nms_thresh  if is_val else cfg.test_nms_thresh
        self.no_multi_labels = False if is_val else True
        
        # ---------------------- Network Parameters ----------------------
        ## Backbone
        self.backbone = GElanBackbone(cfg)
        self.neck = SPPElan(cfg, self.backbone.feat_dims[-1])
        self.backbone.feat_dims[-1] = self.neck.out_dim
        ## PaFPN
        self.fpn = GElanPaFPN(cfg, self.backbone.feat_dims)
        ## Detection head
        self.head = GElanDetHead(cfg, self.fpn.out_dims)
        self.pred = GElanPredLayer(cfg, self.head.cls_head_dim, self.head.reg_head_dim)

    def switch_to_deploy(self,):
        for m in self.modules():
            if hasattr(m, "fuse_convs"):
                m.fuse_convs()

    def post_process(self, cls_preds, box_preds):
        """
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

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes)

        return bboxes, scores, labels
    
    def forward(self, x):
        # ---------------- Backbone ----------------
        pyramid_feats = self.backbone(x)
        # ---------------- Neck: SPP ----------------
        pyramid_feats[-1] = self.neck(pyramid_feats[-1])

        # ---------------- Neck: PaFPN ----------------
        pyramid_feats = self.fpn(pyramid_feats)

        # ---------------- Heads ----------------
        cls_feats, reg_feats = self.head(pyramid_feats)

        # ---------------- Preds ----------------
        outputs = self.pred(cls_feats, reg_feats)
        outputs['image_size'] = [x.shape[2], x.shape[3]]

        if not self.training:
            all_cls_preds = outputs['pred_cls']
            all_box_preds = outputs['pred_box']

            # post process
            bboxes, scores, labels = self.post_process(all_cls_preds, all_box_preds)
            outputs = {
                "scores": scores,
                "labels": labels,
                "bboxes": bboxes
            }
        
        return outputs
    