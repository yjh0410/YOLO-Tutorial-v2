#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from .criterion import build_criterion
from .detr import DETR


# build object detector
def build_detr(cfg, num_classes=80, is_val=False):
    # -------------- Build RT-DETR --------------
    model = DETR(cfg         = cfg,
                 num_classes = num_classes,
                 conf_thresh = cfg['train_conf_thresh'] if is_val else cfg['test_conf_thresh'],
                 nms_thresh  = cfg['train_nms_thresh']  if is_val else cfg['test_nms_thresh'],
                 topk        = cfg['train_topk']        if is_val else cfg['test_topk'],
                 use_nms     = False,
                 )
            
    # -------------- Build criterion --------------
    criterion = None
    if is_val:
        # build criterion for training
        criterion = build_criterion(cfg, num_classes, aux_loss=True)
        
    return model, criterion