# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def get_ious(bboxes1,
             bboxes2,
             box_mode="xyxy",
             iou_type="iou"):
    """
    Compute iou loss of type ['iou', 'giou', 'linear_iou']

    Args:
        inputs (tensor): pred values
        targets (tensor): target values
        weight (tensor): loss weight
        box_mode (str): 'xyxy' or 'ltrb', 'ltrb' is currently supported.
        loss_type (str): 'giou' or 'iou' or 'linear_iou'
        reduction (str): reduction manner

    Returns:
        loss (tensor): computed iou loss.
    """
    if box_mode == "ltrb":
        bboxes1 = torch.cat((-bboxes1[..., :2], bboxes1[..., 2:]), dim=-1)
        bboxes2 = torch.cat((-bboxes2[..., :2], bboxes2[..., 2:]), dim=-1)
    elif box_mode != "xyxy":
        raise NotImplementedError

    eps = torch.finfo(torch.float32).eps

    bboxes1_area = (bboxes1[..., 2] - bboxes1[..., 0]).clamp_(min=0) \
        * (bboxes1[..., 3] - bboxes1[..., 1]).clamp_(min=0)
    bboxes2_area = (bboxes2[..., 2] - bboxes2[..., 0]).clamp_(min=0) \
        * (bboxes2[..., 3] - bboxes2[..., 1]).clamp_(min=0)

    w_intersect = (torch.min(bboxes1[..., 2], bboxes2[..., 2])
                   - torch.max(bboxes1[..., 0], bboxes2[..., 0])).clamp_(min=0)
    h_intersect = (torch.min(bboxes1[..., 3], bboxes2[..., 3])
                   - torch.max(bboxes1[..., 1], bboxes2[..., 1])).clamp_(min=0)

    area_intersect = w_intersect * h_intersect
    area_union = bboxes2_area + bboxes1_area - area_intersect
    ious = area_intersect / area_union.clamp(min=eps)

    if iou_type == "iou":
        return ious
    elif iou_type == "giou":
        g_w_intersect = torch.max(bboxes1[..., 2], bboxes2[..., 2]) \
            - torch.min(bboxes1[..., 0], bboxes2[..., 0])
        g_h_intersect = torch.max(bboxes1[..., 3], bboxes2[..., 3]) \
            - torch.min(bboxes1[..., 1], bboxes2[..., 1])
        ac_uion = g_w_intersect * g_h_intersect
        gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)
        return gious
    else:
        raise NotImplementedError


def delta2bbox(proposals,
               deltas,
               max_shape=None,
               wh_ratio_clip=16 / 1000,
               clip_border=True,
               add_ctr_clamp=False,
               ctr_clamp=32):

    dxy = deltas[..., :2]
    dwh = deltas[..., 2:]

    # Compute width/height of each roi
    pxy = proposals[..., :2]
    pwh = proposals[..., 2:]

    dxy_wh = pwh * dxy
    wh_ratio_clip = torch.tensor(wh_ratio_clip).to(deltas.device)
    max_ratio = torch.abs(torch.log(wh_ratio_clip))
    if add_ctr_clamp:
        dxy_wh = torch.clamp(dxy_wh, max=ctr_clamp, min=-ctr_clamp)
        dwh = torch.clamp(dwh, max=max_ratio)
    else:
        dwh = dwh.clamp(min=-max_ratio, max=max_ratio)

    gxy = pxy + dxy_wh
    gwh = pwh * dwh.exp()
    x1y1 = gxy - (gwh * 0.5)
    x2y2 = gxy + (gwh * 0.5)
    bboxes = torch.cat([x1y1, x2y2], dim=-1)
    if clip_border and max_shape is not None:
        bboxes[..., 0::2].clamp_(min=0).clamp_(max=max_shape[1])
        bboxes[..., 1::2].clamp_(min=0).clamp_(max=max_shape[0])
        
    return bboxes


def bbox2delta(proposals, gt, means=(0., 0., 0., 0.), stds=(1., 1., 1., 1.)):
    # hack for matcher
    if proposals.size() != gt.size():
        proposals = proposals[:, None]
        gt = gt[None]

    proposals = proposals.float()
    gt = gt.float()
    px, py, pw, ph = proposals.unbind(-1)
    gx, gy, gw, gh = gt.unbind(-1)

    dx = (gx - px) / (pw + 0.1)
    dy = (gy - py) / (ph + 0.1)
    dw = torch.log(gw / (pw + 0.1))
    dh = torch.log(gh / (ph + 0.1))
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    union[union == 0.0] = 1.0

    iou = inter / union
    
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
