# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# https://github.com/facebookresearch/detr

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # [B * num_queries, C] = [N, C]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        # [M,] where M is number of all targets in this batch
        tgt_ids = torch.cat([v["labels"] for v in targets])
        # [M, 4]
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # [N, M]
        cost_class = -out_prob[:, tgt_ids] 

        # [N, M]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # [N, M]
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix: [N, M]
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        # [N, M] -> [B, num_queries, M]
        C = C.view(bs, num_queries, -1).cpu()

        # Optimziee cost
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64),   # tgt indexes
                 torch.as_tensor(j, dtype=torch.int64))   # pred indexes
                 for i, j in indices]

