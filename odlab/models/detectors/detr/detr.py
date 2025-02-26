import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------- Model components ---------------
from ...backbone    import build_backbone
from ...transformer import build_transformer
from ...basic.mlp   import MLP


# Detection with Transformer
class DETR(nn.Module):
    def __init__(self, 
                 cfg,
                 num_classes :int   = 90, 
                 conf_thresh :float = 0.05,
                 topk        :int   = 1000,
                 ):
        super().__init__()
        # ---------------------- Basic Parameters ----------------------
        self.cfg = cfg
        self.topk = topk
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh

        # ---------------------- Network Parameters ----------------------
        ## Backbone
        backbone, feat_dims = build_backbone(cfg)
        self.backbone = nn.Sequential(backbone)
        ## Input proj
        self.input_proj = nn.Conv2d(feat_dims[-1], cfg.hidden_dim, kernel_size=1)

        ## Transformer
        self.transformer = build_transformer(cfg, return_intermediate_dec=True)

        ## Object queries
        self.query_embed = nn.Embedding(cfg.num_queries, cfg.hidden_dim)
        
        ## Output
        self.class_embed = nn.Linear(cfg.hidden_dim, num_classes + 1)
        self.bbox_embed  = MLP(cfg.hidden_dim, cfg.hidden_dim, 4, 3)

    @torch.jit.unused
    def set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def post_process(self, cls_pred, box_pred):
        """
        Input:
            cls_pred: (Tensor) [Nq, C]
            box_pred: (Tensor) [Nq, 4]
        """        
        # [Nq x C,]
        scores_i = cls_pred.flatten()

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

        return bboxes, scores, labels

    def forward(self, src, src_mask=None):
        # ---------------- Backbone ----------------
        pyramid_feats = self.backbone(src)
        feat = self.input_proj(pyramid_feats[-1])

        if src_mask is not None:
            src_mask = F.interpolate(src_mask[None].float(), size=feat.shape[-2:]).bool()[0]
        else:
            src_mask = torch.zeros([feat.shape[0], *feat.shape[-2:]], device=feat.device, dtype=torch.bool)

        # ---------------- Transformer ----------------
        hs = self.transformer(feat, src_mask, self.query_embed.weight)[0]

        # ---------------- Head ----------------
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        if self.training:
            outputs = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            outputs['aux_outputs'] = self.set_aux_loss(outputs_class, outputs_coord)
        else:
            cls_pred = outputs_class[-1].softmax(-1)[..., :-1]
            box_pred = outputs_coord[-1]

            # [B, N, C] -> [N, C]
            cls_pred = cls_pred[0]
            box_pred = box_pred[0]

            # xywh -> xyxy
            cxcy_pred = box_pred[..., :2]
            bwbh_pred = box_pred[..., 2:]
            x1y1_pred = cxcy_pred - 0.5 * bwbh_pred
            x2y2_pred = cxcy_pred + 0.5 * bwbh_pred
            box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

            # Post-process (no NMS)
            bboxes, scores, labels = self.post_process(cls_pred, box_pred)

            outputs = {
                'scores': scores,
                'labels': labels,
                'bboxes': bboxes
            }

        return outputs 
