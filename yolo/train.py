from __future__ import division

import os
import random
import numpy as np
import argparse
from copy import deepcopy

# ----------------- Torch Components -----------------
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ----------------- Extra Components -----------------
from utils import distributed_utils
from utils.misc import compute_flops, build_dataloader, CollateFunc
from utils.ema  import ModelEMA

# ----------------- Config Components -----------------
from config import build_config

# ----------------- Data Components -----------------
from dataset.build import build_dataset, build_transform

# ----------------- Evaluator Components -----------------
from evaluator.build import build_evluator

# ----------------- Model Components -----------------
from models import build_model

# ----------------- Train Components -----------------
from engine import build_trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Real-time Object Detection LAB')
    # Random seed
    parser.add_argument('--seed', default=42, type=int)

    # GPU
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    
    # Image size
    parser.add_argument('--eval_first', action='store_true', default=False,
                        help='evaluate model before training.')
    
    # Outputs
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='path to save weight')
    parser.add_argument('--vis_tgt', action="store_true", default=False,
                        help="visualize training data.")
    parser.add_argument('--vis_aux_loss', action="store_true", default=False,
                        help="visualize aux loss.")
    
    # Mixing precision
    parser.add_argument('--fp16', dest="fp16", action="store_true", default=False,
                        help="Adopting mix precision training.")
    
    # Batchsize
    parser.add_argument('-bs', '--batch_size', default=16, type=int, 
                        help='batch size on all the GPUs.')

    # Model
    parser.add_argument('-m', '--model', default='yolo_n', type=str,
                        help='build yolo')
    parser.add_argument('-p', '--pretrained', default=None, type=str,
                        help='load pretrained weight')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='keep training')

    # Dataset
    parser.add_argument('--root', default='D:/python_work/dataset/VOCdevkit/',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc')
    parser.add_argument('--num_workers', default=4, type=int, 
                        help='Number of workers used in dataloading')
    
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
    
    # Debug mode
    parser.add_argument('--debug', action='store_true', default=False, 
                        help='debug mode.')

    return parser.parse_args()


def fix_random_seed(args):
    seed = args.seed + distributed_utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    # ---------------------------- Build DDP ----------------------------
    local_rank = local_process_rank = -1
    if args.distributed:
        distributed_utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(distributed_utils.get_sha()))
        try:
            # Multiple Mechine & Multiple GPUs (world size > 8)
            local_rank = torch.distributed.get_rank()
            local_process_rank = int(os.getenv('LOCAL_PROCESS_RANK', '0'))
        except:
            # Single Mechine & Multiple GPUs (world size <= 8)
            local_rank = local_process_rank = torch.distributed.get_rank()
    world_size = distributed_utils.get_world_size()
    print("LOCAL RANK: ", local_rank)
    print("LOCAL_PROCESS_RANL: ", local_process_rank)
    print('WORLD SIZE: {}'.format(world_size))

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

    # ---------------------------- Build Transform ----------------------------
    train_transform = build_transform(cfg, is_train=True)
    val_transform   = build_transform(cfg, is_train=False)

    # ---------------------------- Build Dataset & Dataloader ----------------------------
    dataset      = build_dataset(args, cfg, train_transform, is_train=True)
    train_loader = build_dataloader(args, dataset, args.batch_size // world_size, CollateFunc())

    # ---------------------------- Build Evaluator ----------------------------
    evaluator = build_evluator(args, cfg, val_transform, device)

    # ---------------------------- Build model ----------------------------
    ## Build model
    model, criterion = build_model(args, cfg, is_val=True)
    model = model.to(device).train()
    model_without_ddp = model

    # ---------------------------- Build Model-EMA ----------------------------
    if cfg.use_ema and distributed_utils.get_rank() in [-1, 0]:
        print('Build ModelEMA for {} ...'.format(args.model))
        model_ema = ModelEMA(model, cfg.ema_decay, cfg.ema_tau, args.resume)
    else:
        model_ema = None

    ## Calcute Params & GFLOPs
    if distributed_utils.is_main_process:
        model_copy = deepcopy(model_without_ddp)
        model_copy.trainable = False
        model_copy.eval()
        compute_flops(model=model_copy,
                      img_size=cfg.test_img_size,
                      device=device)
        del model_copy
    if args.distributed:
        dist.barrier()

    ## Build DDP model
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_parameters)
        if args.sybn:
            print('use SyncBatchNorm ...')
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model_without_ddp = model.module

    if args.distributed:
        dist.barrier()

    # ---------------------------- Build Trainer ----------------------------
    trainer = build_trainer(args, cfg, device, model, model_ema, criterion, train_transform, val_transform, dataset, train_loader, evaluator)

    ## Eval before training
    if args.eval_first and distributed_utils.is_main_process():
        # to check whether the evaluator can work
        model_eval = model_without_ddp
        trainer.eval(model_eval)
        return

    # garbage = torch.randn(640, 1024, 73, 73).to(device) # 15 G

    # ---------------------------- Train pipeline ----------------------------
    trainer.train(model)

    # Empty cache after train loop
    del trainer
    del garbage
    if args.cuda:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    train()
