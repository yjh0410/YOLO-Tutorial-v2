import torch
import torch.nn as nn

from .rtdetr_encoder import ImageEncoder
from .rtdetr_decoder import RTDetrTransformer

from .basic_modules.nms_ops import multiclass_nms


# Real-time DETR
class RTDETR(nn.Module):
    def __init__(self,
                 cfg,
                 is_val = False,
                 use_nms = False,
                 onnx_deploy = False,
                 ) -> None:
        super(RTDETR, self).__init__()
        # ---------------------- Basic setting ----------------------
        self.cfg = cfg
        self.use_nms = use_nms
        self.onnx_deploy = onnx_deploy
        self.num_classes = cfg.num_classes
        ## Post-process parameters
        self.topk_candidates = cfg.val_topk        if is_val else cfg.test_topk
        self.conf_thresh     = cfg.val_conf_thresh if is_val else cfg.test_conf_thresh
        self.nms_thresh      = cfg.val_nms_thresh  if is_val else cfg.test_nms_thresh
        self.no_multi_labels = False if is_val else True


        # ----------- Network setting -----------
        ## Image encoder
        self.image_encoder = ImageEncoder(cfg)
        ## Detect decoder
        self.detect_decoder = RTDetrTransformer(in_dims             = self.image_encoder.fpn_dims,
                                                hidden_dim          = cfg.hidden_dim,
                                                strides             = cfg.out_stride,
                                                num_classes         = cfg.num_classes,
                                                num_queries         = cfg.num_queries,
                                                num_heads           = cfg.de_num_heads,
                                                num_layers          = cfg.de_num_layers,
                                                num_levels          = len(cfg.out_stride),
                                                num_points          = cfg.de_num_points,
                                                ffn_dim             = cfg.de_ffn_dim,
                                                dropout             = cfg.de_dropout,
                                                act_type            = cfg.de_act,
                                                return_intermediate = True,
                                                num_denoising       = cfg.dn_num_denoising,
                                                label_noise_ratio   = cfg.dn_label_noise_ratio,
                                                box_noise_scale     = cfg.dn_box_noise_scale,
                                                learnt_init_query   = cfg.learnt_init_query,
                                                )

    def post_process(self, box_pred, cls_pred):
        # xywh -> xyxy
        box_preds_x1y1 = box_pred[..., :2] - 0.5 * box_pred[..., 2:]
        box_preds_x2y2 = box_pred[..., :2] + 0.5 * box_pred[..., 2:]
        box_pred = torch.cat([box_preds_x1y1, box_preds_x2y2], dim=-1)
        
        cls_pred = cls_pred[0]
        box_pred = box_pred[0]
        if self.no_multi_labels:
            # [M,]
            scores, labels = torch.max(cls_pred.sigmoid(), dim=1)

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_pred.size(0))

            # Topk candidates
            predicted_prob, topk_idxs = scores.sort(descending=True)
            topk_scores = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # Filter out the proposals with low confidence score
            keep_idxs = topk_scores > self.conf_thresh
            topk_idxs = topk_idxs[keep_idxs]

            # Top-k results
            topk_scores = topk_scores[keep_idxs]
            topk_labels = labels[topk_idxs]
            topk_bboxes = box_pred[topk_idxs]

        else:
            # Top-k select
            cls_pred = cls_pred.flatten().sigmoid_()
            box_pred = box_pred

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_pred.size(0))

            # Topk candidates
            predicted_prob, topk_idxs = cls_pred.sort(descending=True)
            topk_scores = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # Filter out the proposals with low confidence score
            keep_idxs = topk_scores > self.conf_thresh
            topk_scores = topk_scores[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]
            topk_box_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')

            ## Top-k results
            topk_labels = topk_idxs % self.num_classes
            topk_bboxes = box_pred[topk_box_idxs]

        if not self.onnx_deploy:
            topk_scores = topk_scores.cpu().numpy()
            topk_labels = topk_labels.cpu().numpy()
            topk_bboxes = topk_bboxes.cpu().numpy()

            # nms
            if self.use_nms:
                topk_scores, topk_labels, topk_bboxes = multiclass_nms(
                    topk_scores, topk_labels, topk_bboxes, self.nms_thresh, self.num_classes)

        return topk_bboxes, topk_scores, topk_labels
    
    def forward(self, x, targets=None):
        # ----------- Image Encoder -----------
        pyramid_feats = self.image_encoder(x)

        # ----------- Transformer -----------
        outputs = self.detect_decoder(pyramid_feats, targets)

        if not self.training:
            img_h, img_w = x.shape[2:]
            box_pred = outputs["pred_boxes"]
            cls_pred = outputs["pred_logits"]

            # rescale bbox
            box_pred[..., [0, 2]] *= img_h
            box_pred[..., [1, 3]] *= img_w
            
            # post-process
            bboxes, scores, labels = self.post_process(box_pred, cls_pred)

            outputs = {
                "scores": scores,
                "labels": labels,
                "bboxes": bboxes,
            }

        return outputs
