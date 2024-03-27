#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from .criterion import build_criterion
from .retinanet import RetinaNet


# build RetinaNet
def build_retinanet(cfg, num_classes=80, is_val=False):
    # -------------- Build RetinaNet --------------
    model = RetinaNet(cfg         = cfg,
                      num_classes = num_classes,
                      conf_thresh = cfg['train_conf_thresh'] if is_val else cfg['test_conf_thresh'],
                      nms_thresh  = cfg['train_nms_thresh']  if is_val else cfg['test_nms_thresh'],
                      topk        = cfg['train_topk']        if is_val else cfg['test_topk'],
                      ca_nms      = False if is_val else cfg['nms_class_agnostic'])
            
    # -------------- Build Criterion --------------
    criterion = None
    if is_val:
        # build criterion for training
        criterion = build_criterion(cfg, num_classes)

    return model, criterion