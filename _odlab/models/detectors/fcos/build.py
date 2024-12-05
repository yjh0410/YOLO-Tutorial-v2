#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from .criterion import SetCriterion
from .fcos import FCOS, FcosRT


# build FCOS
def build_fcos(cfg, is_val=False):
    # -------------- Build FCOS --------------
    model = FCOS(cfg         = cfg,
                 num_classes = cfg.num_classes,
                 conf_thresh = cfg.train_conf_thresh if is_val else cfg.test_conf_thresh,
                 nms_thresh  = cfg.train_nms_thresh  if is_val else cfg.test_nms_thresh,
                 topk        = cfg.train_topk        if is_val else cfg.test_topk,
                 )
            
    # -------------- Build Criterion --------------
    criterion = None
    if is_val:
        # build criterion for training
        criterion = SetCriterion(cfg)

    return model, criterion

# build FCOS
def build_fcos_rt(cfg, is_val=False):
    # -------------- Build FCOS --------------
    model = FcosRT(cfg         = cfg,
                   num_classes = cfg.num_classes,
                   conf_thresh = cfg.train_conf_thresh if is_val else cfg.test_conf_thresh,
                   nms_thresh  = cfg.train_nms_thresh  if is_val else cfg.test_nms_thresh,
                   topk        = cfg.train_topk        if is_val else cfg.test_topk,
                   )
            
    # -------------- Build Criterion --------------
    criterion = None
    if is_val:
        # build criterion for training
        criterion = SetCriterion(cfg)

    return model, criterion