import torch
import torch.nn as nn

from utils.misc import multiclass_nms

from .yolov4_backbone import build_backbone
from .yolov4_neck import build_neck
from .yolov4_pafpn import build_fpn
from .yolov4_head import build_head


# YOLOv4
class YOLOv4(nn.Module):
    def __init__(self,
                 cfg,
                 device,
                 num_classes=20,
                 conf_thresh=0.01,
                 nms_thresh=0.5,
                 topk=100,
                 trainable=False,
                 deploy=False,
                 no_multi_labels=False,
                 nms_class_agnostic=False):
        super(YOLOv4, self).__init__()
        # ------------------- Basic parameters -------------------
        self.cfg = cfg                                 # 模型配置文件
        self.device = device                           # cuda或者是cpu
        self.num_classes = num_classes                 # 类别的数量
        self.trainable = trainable                     # 训练的标记
        self.conf_thresh = conf_thresh                 # 得分阈值
        self.nms_thresh = nms_thresh                   # NMS阈值
        self.topk_candidates = topk                    # topk
        self.stride = [8, 16, 32]                      # 网络的输出步长
        self.deploy = deploy
        self.no_multi_labels = no_multi_labels
        self.nms_class_agnostic = nms_class_agnostic
        # ------------------- Anchor box -------------------
        self.num_levels = 3
        self.num_anchors = len(cfg['anchor_size']) // self.num_levels
        self.anchor_size = torch.as_tensor(
            cfg['anchor_size']
            ).float().view(self.num_levels, self.num_anchors, 2) # [S, A, 2]
        
        # ------------------- Network Structure -------------------
        ## 主干网络
        self.backbone, feats_dim = build_backbone(
            cfg['backbone'], trainable&cfg['pretrained'])

        ## 颈部网络: SPP模块
        self.neck = build_neck(cfg, in_dim=feats_dim[-1], out_dim=feats_dim[-1])
        feats_dim[-1] = self.neck.out_dim

        ## 颈部网络: 特征金字塔
        self.fpn = build_fpn(cfg=cfg, in_dims=feats_dim, out_dim=int(256*cfg['width']))
        self.head_dim = self.fpn.out_dim

        ## 检测头
        self.non_shared_heads = nn.ModuleList(
            [build_head(cfg, head_dim, head_dim, num_classes) 
            for head_dim in self.head_dim
            ])

        ## 预测层
        self.obj_preds = nn.ModuleList(
                            [nn.Conv2d(head.reg_out_dim, 1 * self.num_anchors, kernel_size=1) 
                                for head in self.non_shared_heads
                              ]) 
        self.cls_preds = nn.ModuleList(
                            [nn.Conv2d(head.cls_out_dim, self.num_classes * self.num_anchors, kernel_size=1) 
                                for head in self.non_shared_heads
                              ]) 
        self.reg_preds = nn.ModuleList(
                            [nn.Conv2d(head.reg_out_dim, 4 * self.num_anchors, kernel_size=1) 
                                for head in self.non_shared_heads
                              ])                 
    

    # ---------------------- Basic Functions ----------------------
    ## generate anchor points
    def generate_anchors(self, level, fmp_size):
        """
            fmp_size: (List) [H, W]
        """
        fmp_h, fmp_w = fmp_size
        # [KA, 2]
        anchor_size = self.anchor_size[level]

        # generate grid cells
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2)
        # [HW, 2] -> [HW, KA, 2] -> [M, 2]
        anchor_xy = anchor_xy.unsqueeze(1).repeat(1, self.num_anchors, 1) + 0.5
        anchor_xy = anchor_xy.view(-1, 2).to(self.device)

        # [KA, 2] -> [1, KA, 2] -> [HW, KA, 2] -> [M, 2]
        anchor_wh = anchor_size.unsqueeze(0).repeat(fmp_h*fmp_w, 1, 1)
        anchor_wh = anchor_wh.view(-1, 2).to(self.device)

        anchors = torch.cat([anchor_xy, anchor_wh], dim=-1)

        return anchors
        
    ## post-process
    def post_process(self, obj_preds, cls_preds, box_preds):
        """
        Input:
            cls_preds: List[np.array] -> [[M, C], ...]
            box_preds: List[np.array] -> [[M, 4], ...]
            obj_preds: List[np.array] -> [[M, 1], ...] or None
        Output:
            bboxes: np.array -> [N, 4]
            scores: np.array -> [N,]
            labels: np.array -> [N,]
        """
        assert len(cls_preds) == self.num_levels
        all_scores = []
        all_labels = []
        all_bboxes = []
        
        for obj_pred_i, cls_pred_i, box_pred_i in zip(obj_preds, cls_preds, box_preds):
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
                scores_i = (torch.sqrt(obj_pred_i.sigmoid() * cls_pred_i.sigmoid())).flatten()

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

        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        bboxes = torch.cat(all_bboxes)

        # to cpu & numpy
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, self.nms_class_agnostic)

        return bboxes, scores, labels
    

    # ---------------------- Main Process for Inference ----------------------
    @torch.no_grad()
    def inference(self, x):
        # 主干网络
        pyramid_feats = self.backbone(x)

        # 颈部网络
        pyramid_feats[-1] = self.neck(pyramid_feats[-1])

        # 特征金字塔
        pyramid_feats = self.fpn(pyramid_feats)

        # 检测头
        all_anchors = []
        all_obj_preds = []
        all_cls_preds = []
        all_box_preds = []
        for level, (feat, head) in enumerate(zip(pyramid_feats, self.non_shared_heads)):
            cls_feat, reg_feat = head(feat)

            # [1, C, H, W]
            obj_pred = self.obj_preds[level](reg_feat)
            cls_pred = self.cls_preds[level](cls_feat)
            reg_pred = self.reg_preds[level](reg_feat)

            # anchors: [M, 2]
            fmp_size = cls_pred.shape[-2:]
            anchors = self.generate_anchors(level, fmp_size)

            # [1, AC, H, W] -> [H, W, AC] -> [M, C]
            obj_pred = obj_pred[0].permute(1, 2, 0).contiguous().view(-1, 1)
            cls_pred = cls_pred[0].permute(1, 2, 0).contiguous().view(-1, self.num_classes)
            reg_pred = reg_pred[0].permute(1, 2, 0).contiguous().view(-1, 4)

            # decode bbox
            ctr_pred = (torch.sigmoid(reg_pred[..., :2]) * 3.0 - 1.5 + anchors[..., :2]) * self.stride[level]
            wh_pred = torch.exp(reg_pred[..., 2:]) * anchors[..., 2:]
            pred_x1y1 = ctr_pred - wh_pred * 0.5
            pred_x2y2 = ctr_pred + wh_pred * 0.5
            box_pred = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

            all_obj_preds.append(obj_pred)
            all_cls_preds.append(cls_pred)
            all_box_preds.append(box_pred)
            all_anchors.append(anchors)

        if self.deploy:
            obj_preds = torch.cat(all_obj_preds, dim=0)
            cls_preds = torch.cat(all_cls_preds, dim=0)
            box_preds = torch.cat(all_box_preds, dim=0)
            scores = torch.sqrt(obj_preds.sigmoid() * cls_preds.sigmoid())
            bboxes = box_preds
            # [n_anchors_all, 4 + C]
            outputs = torch.cat([bboxes, scores], dim=-1)

        else:
            # post process
            bboxes, scores, labels = self.post_process(
                all_obj_preds, all_cls_preds, all_box_preds)
            outputs = {
                "scores": scores,
                "labels": labels,
                "bboxes": bboxes
            }

        return outputs


    # ---------------------- Main Process for Training ----------------------
    def forward(self, x):
        if not self.trainable:
            return self.inference(x)
        else:
            bs = x.shape[0]
            # 主干网络
            pyramid_feats = self.backbone(x)

            # 颈部网络
            pyramid_feats[-1] = self.neck(pyramid_feats[-1])

            # 特征金字塔
            pyramid_feats = self.fpn(pyramid_feats)

            # 检测头
            all_fmp_sizes = []
            all_obj_preds = []
            all_cls_preds = []
            all_box_preds = []
            for level, (feat, head) in enumerate(zip(pyramid_feats, self.non_shared_heads)):
                cls_feat, reg_feat = head(feat)

                # [B, C, H, W]
                obj_pred = self.obj_preds[level](reg_feat)
                cls_pred = self.cls_preds[level](cls_feat)
                reg_pred = self.reg_preds[level](reg_feat)

                fmp_size = cls_pred.shape[-2:]

                # generate anchor boxes: [M, 4]
                anchors = self.generate_anchors(level, fmp_size)
                
                # [B, AC, H, W] -> [B, H, W, AC] -> [B, M, C]
                obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, 1)
                cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, self.num_classes)
                reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, 4)

                # decode bbox
                ctr_pred = (torch.sigmoid(reg_pred[..., :2]) * 3.0 - 1.5 + anchors[..., :2]) * self.stride[level]
                wh_pred = torch.exp(reg_pred[..., 2:]) * anchors[..., 2:]
                pred_x1y1 = ctr_pred - wh_pred * 0.5
                pred_x2y2 = ctr_pred + wh_pred * 0.5
                box_pred = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

                all_obj_preds.append(obj_pred)
                all_cls_preds.append(cls_pred)
                all_box_preds.append(box_pred)
                all_fmp_sizes.append(fmp_size)

            # output dict
            outputs = {"pred_obj": all_obj_preds,        # List [B, M, 1]
                       "pred_cls": all_cls_preds,        # List [B, M, C]
                       "pred_box": all_box_preds,        # List [B, M, 4]
                       'fmp_sizes': all_fmp_sizes,       # List
                       'strides': self.stride,           # List
                       }

            return outputs 
