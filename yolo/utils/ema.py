# =====================================================================
# Copyright 2021 RangiLyu. All rights reserved.
# =====================================================================
# Modified from: https://github.com/facebookresearch/d2go
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Licensed under the Apache License, Version 2.0 (the "License")
import math
from copy import deepcopy

import torch
import torch.nn as nn


# Modified from the YOLOv5 project
class ModelEMA(object):
    def __init__(self, model, ema_decay=0.9999, ema_tau=2000, resume=None):
        # Create EMA
        self.ema = deepcopy(self.de_parallel(model)).eval()  # FP32 EMA
        self.updates = 0  # number of EMA updates
        self.decay = lambda x: ema_decay * (1 - math.exp(-x / ema_tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

        if resume is not None and resume.lower() != "none":
            self.load_resume(resume)

        print("Initialize ModelEMA's updates: {}".format(self.updates))

    def load_resume(self, resume):
        checkpoint = torch.load(resume)
        if 'model_ema' in checkpoint.keys():
            print('--Load ModelEMA state dict from the checkpoint: ', resume)
            model_ema_state_dict = checkpoint["model_ema"]
            self.ema.load_state_dict(model_ema_state_dict)
        if 'ema_updates' in checkpoint.keys():
            print('--Load ModelEMA updates from the checkpoint: ', resume)
            # checkpoint state dict
            self.updates = checkpoint.pop("ema_updates")

    def is_parallel(self, model):
        # Returns True if model is of type DP or DDP
        return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

    def de_parallel(self, model):
        # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
        return model.module if self.is_parallel(model) else model

    def copy_attr(self, a, b, include=(), exclude=()):
        # Copy attributes from b to a, options to only include [...] and to exclude [...]
        for k, v in b.__dict__.items():
            if (len(include) and k not in include) or k.startswith('_') or k in exclude:
                continue
            else:
                setattr(a, k, v)

    def update(self, model):
        # Update EMA parameters
        self.updates += 1
        d = self.decay(self.updates)

        msd = self.de_parallel(model).state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        self.copy_attr(self.ema, model, include, exclude)
