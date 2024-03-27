# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import random
import argparse
import numpy as np
from copy import deepcopy

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import distributed_utils
from utils.misc import compute_flops, collate_fn
from utils.misc import get_param_dict, ModelEMA
from utils.optimizer import build_optimizer
from utils.lr_scheduler import build_wp_lr_scheduler, build_lr_scheduler

from config import build_config
from evaluator import build_evluator
from datasets import build_dataset, build_dataloader, build_transform

from models.detectors import build_model
from engine import train_one_epoch


def parse_args():
    parser = argparse.ArgumentParser('General 2D Object Detection', add_help=False)
    # Random seed
    parser.add_argument('--seed', default=42, type=int)
    # GPU
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    # Batch size
    parser.add_argument('-bs', '--batch_size', default=16, type=int, 
                        help='total batch size on all GPUs.')
    # Model
    parser.add_argument('-m', '--model', default='yolof_r18_c5_1x',
                        help='build object detector')
    parser.add_argument('-p', '--pretrained', default=None, type=str,
                        help='load pretrained weight')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='keep training')
    parser.add_argument('--ema', default=None, type=str,
                        help='use Model EMA trick.')
    # Dataset
    parser.add_argument('--root', default='/Users/liuhaoran/Desktop/python_work/object-detection/dataset/COCO/',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc, widerface, crowdhuman')
    parser.add_argument('--vis_tgt', action="store_true", default=False,
                        help="visualize input data.")
    # Dataloader
    parser.add_argument('--num_workers', default=2, type=int, 
                        help='Number of workers used in dataloading')
    # Epoch
    parser.add_argument('--eval_epoch', default=2, type=int,
                        help='interval between evaluations')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='path to save weight')
    parser.add_argument('--eval_first', action="store_true", default=False,
                        help="visualize input data.")
    # DDP train
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--sybn', action='store_true', default=False, 
                        help='use sybn.')
    parser.add_argument('--find_unused_parameters', action='store_true', default=False, 
                        help='set find_unused_parameters as True.')
    # Debug setting
    parser.add_argument('--debug', action='store_true', default=False, 
                        help='debug codes.')

    return parser.parse_args()


def fix_random_seed(args):
    seed = args.seed + distributed_utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    # path to save model
    path_to_save = os.path.join(args.save_folder, args.dataset, args.model)
    os.makedirs(path_to_save, exist_ok=True)


    # ---------------------------- Build DDP ----------------------------
    distributed_utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(distributed_utils.get_sha()))
    world_size = distributed_utils.get_world_size()
    print('World size: {}'.format(world_size))
    per_gpu_batch = args.batch_size // world_size


    # ---------------------------- Build CUDA ----------------------------
    if args.cuda and torch.cuda.is_available():
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    # ---------------------------- Fix random seed ----------------------------
    fix_random_seed(args)


    # ---------------------------- Build config ----------------------------
    cfg = build_config(args)
    print('Model config: ', cfg)


    # ---------------------------- Build Dataset ----------------------------
    transforms = build_transform(cfg, is_train=True)
    dataset, dataset_info = build_dataset(args, transforms, is_train=True)


    # ---------------------------- Build Dataloader ----------------------------
    train_loader = build_dataloader(args, dataset, per_gpu_batch, collate_fn, is_train=True)


    # ---------------------------- Build model ----------------------------
    ## Build model
    model, criterion = build_model(args, cfg, dataset_info['num_classes'], is_val=True)
    model.to(device)
    model_without_ddp = model
    ## Calcute Params & GFLOPs
    if distributed_utils.is_main_process():
        model_copy = deepcopy(model_without_ddp)
        model_copy.trainable = False
        model_copy.eval()
        compute_flops(model=model_copy,
                      min_size=cfg['test_min_size'],
                      max_size=cfg['test_max_size'],
                      device=device)
        del model_copy
    if args.distributed:
        dist.barrier()


    # ---------------------------- Build Optimizer ----------------------------
    cfg['base_lr'] = cfg['base_lr'] * args.batch_size
    param_dicts = None
    if 'param_dict_type' in cfg.keys() and cfg['param_dict_type'] != 'default':
        print("- Param dict type: {}".format(cfg['param_dict_type']))
        param_dicts = get_param_dict(model_without_ddp, cfg)
    optimizer, start_epoch = build_optimizer(cfg, model_without_ddp, param_dicts, args.resume)


    # ---------------------------- Build LR Scheduler ----------------------------
    wp_lr_scheduler = build_wp_lr_scheduler(cfg, cfg['base_lr'])
    lr_scheduler    = build_lr_scheduler(cfg, optimizer, args.resume)


    # ---------------------------- Build Model EMA ----------------------------
    model_ema = None
    if 'use_ema' in cfg.keys() and cfg['use_ema']:
        print("Build Model EMA for {}".format(args.model))
        model_ema = ModelEMA(cfg, model, start_epoch * len(train_loader))


    # ---------------------------- Build DDP model ----------------------------
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_parameters)
        model_without_ddp = model.module


    # ---------------------------- Build Evaluator ----------------------------
    evaluator = build_evluator(args, cfg, device)


    # ----------------------- Eval before training -----------------------
    if args.eval_first and distributed_utils.is_main_process():
        evaluator.evaluate(model_without_ddp)
        return


    # ----------------------- Training -----------------------
    print("Start training")
    best_map = -1.
    for epoch in range(start_epoch, cfg['max_epoch']):
        if args.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)

        # Train one epoch
        train_one_epoch(cfg,
                        model,
                        criterion,
                        train_loader,
                        optimizer,
                        device,
                        epoch,
                        args.vis_tgt,
                        wp_lr_scheduler,
                        dataset_info['class_labels'],
                        model_ema=model_ema,
                        debug=args.debug)
        
        # LR Scheduler
        lr_scheduler.step()

        # Evaluate
        if distributed_utils.is_main_process():
            model_eval = model_ema.ema if model_ema is not None else model_without_ddp
            if (epoch % args.eval_epoch) == 0 or (epoch == cfg['max_epoch'] - 1):
                if evaluator is None:
                    cur_map = 0.
                else:
                    evaluator.evaluate(model_eval)
                    cur_map = evaluator.map
                # Save model
                if cur_map > best_map:
                    # update best-map
                    best_map = cur_map
                    # save model
                    print('Saving state, epoch:', epoch + 1)
                    torch.save({'model':        model_eval.state_dict(),
                                'optimizer':    optimizer.state_dict(),
                                'lr_scheduler': lr_scheduler.state_dict(),
                                'mAP':          round(cur_map*100, 1),
                                'epoch':        epoch,
                                'args':         args}, 
                                os.path.join(path_to_save, '{}_best.pth'.format(args.model)))
        if args.distributed:
            dist.barrier()

        if args.debug:
            print("For debug mode, we only train the model with 1 epoch.")
            exit(0)


if __name__ == '__main__':
    main()
