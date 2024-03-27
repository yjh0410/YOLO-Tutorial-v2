#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from .criterion import build_criterion
from .yolof import YOLOF


# build YOLOF
def build_yolof(cfg, num_classes=80, is_val=False):
    # -------------- Build YOLOF --------------
    model = YOLOF(cfg         = cfg,
                  num_classes = num_classes,
                  conf_thresh = cfg['train_conf_thresh'] if is_val else cfg['test_conf_thresh'],
                  nms_thresh  = cfg['train_nms_thresh']  if is_val else cfg['test_nms_thresh'],
                  topk        = cfg['train_topk']        if is_val else cfg['test_topk'],
                  )
            
    # -------------- Build Criterion --------------
    criterion = None
    if is_val:
        # build criterion for training
        criterion = build_criterion(cfg, num_classes)

    return model, criterion