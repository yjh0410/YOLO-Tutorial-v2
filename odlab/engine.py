# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable

import torch
from utils import distributed_utils
from utils.misc import MetricLogger, SmoothedValue
from utils.vis_tools import vis_data


def train_one_epoch(cfg,
                    model       : torch.nn.Module,
                    criterion   : torch.nn.Module,
                    data_loader : Iterable,
                    optimizer   : torch.optim.Optimizer,
                    device      : torch.device,
                    epoch       : int,
                    vis_target  : bool,
                    warmup_lr_scheduler,
                    debug       :bool = False
                    ):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{} / {}]'.format(epoch, cfg.max_epoch)
    epoch_size = len(data_loader)
    print_freq = 10

    for iter_i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        ni = iter_i + epoch * epoch_size
        # WarmUp
        if ni < cfg.warmup_iters:
            warmup_lr_scheduler(ni, optimizer)
        elif ni == cfg.warmup_iters:
            print('Warmup stage is over.')
            warmup_lr_scheduler.set_lr(optimizer, cfg.base_lr)

        # To device
        images, masks = samples
        images = images.to(device)
        masks  = masks.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Visualize train targets
        if vis_target:
            vis_data(images, targets, masks, cfg.class_labels, cfg.normalize_coords, cfg.box_format)

        # Inference
        outputs = model(images, masks)

        # Compute loss
        loss_dict = criterion(outputs, targets)
        loss_weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * loss_weight_dict[k] for k in loss_dict.keys() if k in loss_weight_dict)
        loss_value = losses.item()
        losses /= cfg.grad_accumulate

        # Reduce losses over all GPUs for logging purposes
        loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)

        # Check loss
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # Backward
        losses.backward()

        # Optimize
        if (iter_i + 1) % cfg.grad_accumulate == 0:
            if cfg.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_max_norm)
            optimizer.step()
            optimizer.zero_grad()

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if debug:
            print("For debug mode, we only train the model with 1 iteration.")
            break
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
