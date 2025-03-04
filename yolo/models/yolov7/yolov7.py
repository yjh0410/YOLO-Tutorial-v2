import torch
import torch.nn as nn

from utils.misc import multiclass_nms

# --------------- Model components ---------------
from .yolov7_backbone import Yolov7Backbone
from .yolov7_neck     import SPPFBlockCSP
from .yolov7_pafpn    import Yolov7PaFPN
from .yolov7_head     import DecoupledHead

# --------------- External components ---------------
from utils.misc import multiclass_nms


class Yolov7(nn.Module):
    def __init__(self, cfg, is_val: bool = False) -> None:
        super(Yolov7, self).__init__()
        # ---------------------- Basic setting ----------------------
        self.cfg = cfg
        self.num_classes = cfg.num_classes
        self.out_stride = cfg.out_stride
        self.num_levels = len(cfg.out_stride)

        ## Post-process parameters
        self.topk_candidates  = cfg.val_topk        if is_val else cfg.test_topk
        self.conf_thresh      = cfg.val_conf_thresh if is_val else cfg.test_conf_thresh
        self.nms_thresh       = cfg.val_nms_thresh  if is_val else cfg.test_nms_thresh
        self.no_multi_labels  = False if is_val else True
        
        # ------------------- Network Structure -------------------
        self.backbone = Yolov7Backbone(use_pretrained=cfg.use_pretrained)
        self.neck     = SPPFBlockCSP(self.backbone.feat_dims[-1], self.backbone.feat_dims[-1] // 2, expansion=0.5)
        self.backbone.feat_dims[-1] = self.backbone.feat_dims[-1] // 2
        self.fpn      = Yolov7PaFPN(self.backbone.feat_dims[-3:], head_dim=cfg.head_dim)
        self.non_shared_heads = nn.ModuleList([DecoupledHead(cfg, in_dim)
                                               for in_dim in self.fpn.fpn_out_dims
                                               ])

        ## 预测层
        self.obj_preds = nn.ModuleList(
                            [nn.Conv2d(head.reg_head_dim, 1, kernel_size=1)
                             for head in self.non_shared_heads
                             ]) 
        self.cls_preds = nn.ModuleList(
                            [nn.Conv2d(head.cls_head_dim, self.num_classes, kernel_size=1) 
                             for head in self.non_shared_heads
                             ]) 
        self.reg_preds = nn.ModuleList(
                            [nn.Conv2d(head.reg_head_dim, 4, kernel_size=1) 
                             for head in self.non_shared_heads
                             ])
        
        # init pred layers
        self.init_weight()
    
    def init_weight(self):
        # Init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        # obj pred
        for obj_pred in self.obj_preds:
            b = obj_pred.bias.view(1, -1)
            b.data.fill_(bias_value.item())
            obj_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        # cls pred
        for cls_pred in self.cls_preds:
            b = cls_pred.bias.view(1, -1)
            b.data.fill_(bias_value.item())
            cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        # reg pred
        for reg_pred in self.reg_preds:
            b = reg_pred.bias.view(-1, )
            b.data.fill_(1.0)
            reg_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = reg_pred.weight
            w.data.fill_(0.)
            reg_pred.weight = torch.nn.Parameter(w, requires_grad=True)

    def generate_anchors(self, level, fmp_size):
        """
            fmp_size: (List) [H, W]
        """
        # generate grid cells
        fmp_h, fmp_w = fmp_size
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        anchors = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2)
        anchors += 0.5  # add center offset
        anchors *= self.out_stride[level]

        return anchors
        
    def post_process(self, obj_preds, cls_preds, box_preds):
        """
        We process predictions at each scale hierarchically
        Input:
            obj_preds: List[torch.Tensor] -> [[B, M, 1], ...], B=1
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
        
        for obj_pred_i, cls_pred_i, box_pred_i in zip(obj_preds, cls_preds, box_preds):
            obj_pred_i = obj_pred_i[0]
            cls_pred_i = cls_pred_i[0]
            box_pred_i = box_pred_i[0]
            if self.no_multi_labels:
                # [M,]
                scores, labels = torch.max(torch.sqrt(obj_pred_i.sigmoid() * cls_pred_i.sigmoid()), dim=1)

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
                scores_i = torch.sqrt(obj_pred_i.sigmoid() * cls_pred_i.sigmoid()).flatten()

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
        bs = x.shape[0]
        pyramid_feats = self.backbone(x)
        pyramid_feats[-1] = self.neck(pyramid_feats[-1])
        pyramid_feats = self.fpn(pyramid_feats)

        all_anchors = []
        all_obj_preds = []
        all_cls_preds = []
        all_box_preds = []
        all_reg_preds = []
        for level, (feat, head) in enumerate(zip(pyramid_feats, self.non_shared_heads)):
            cls_feat, reg_feat = head(feat)

            # [B, C, H, W]
            obj_pred = self.obj_preds[level](reg_feat)
            cls_pred = self.cls_preds[level](cls_feat)
            reg_pred = self.reg_preds[level](reg_feat)

            B, _, H, W = cls_pred.size()
            fmp_size = [H, W]
            # generate anchor boxes: [M, 4]
            anchors = self.generate_anchors(level, fmp_size)
            anchors = anchors.to(x.device)
            
            # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
            obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)

            # decode bbox
            ctr_pred = reg_pred[..., :2] * self.out_stride[level] + anchors[..., :2]
            wh_pred = torch.exp(reg_pred[..., 2:]) * self.out_stride[level]
            pred_x1y1 = ctr_pred - wh_pred * 0.5
            pred_x2y2 = ctr_pred + wh_pred * 0.5
            box_pred = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

            all_obj_preds.append(obj_pred)
            all_cls_preds.append(cls_pred)
            all_box_preds.append(box_pred)
            all_reg_preds.append(reg_pred)
            all_anchors.append(anchors)

        if not self.training:
            bboxes, scores, labels = self.post_process(all_obj_preds, all_cls_preds, all_box_preds)
            outputs = {
                "scores": scores,
                "labels": labels,
                "bboxes": bboxes
            }
        else:
            outputs = {"pred_obj": all_obj_preds,        # List(Tensor) [B, M, 1]
                       "pred_cls": all_cls_preds,        # List(Tensor) [B, M, C]
                       "pred_box": all_box_preds,        # List(Tensor) [B, M, 4]
                       "pred_reg": all_reg_preds,        # List(Tensor) [B, M, 4]
                       "anchors": all_anchors,           # List(Tensor) [M, 2]
                       "strides": self.out_stride,       # List(Int) [8, 16, 32]
                       }

        return outputs 
