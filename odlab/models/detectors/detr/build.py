#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from .criterion import SetCriterion
from .detr import DETR


# build DETR
def build_detr(cfg, is_val=False):
    # -------------- Build DETR --------------
    model = DETR(cfg         = cfg,
                 num_classes = cfg.num_classes,
                 conf_thresh = cfg.train_conf_thresh if is_val else cfg.test_conf_thresh,
                 topk        = cfg.train_topk        if is_val else cfg.test_topk,
                 )
            
    # -------------- Build Criterion --------------
    criterion = None
    if is_val:
        # build criterion for training
        criterion = SetCriterion(cfg)

    return model, criterion