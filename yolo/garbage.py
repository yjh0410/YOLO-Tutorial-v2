from __future__ import division

import os
import random
import numpy as np
import argparse
import time

# ----------------- Torch Components -----------------
import torch

# ----------------- Extra Components -----------------
from utils import distributed_utils



def parse_args():
    parser = argparse.ArgumentParser(description='Real-time Object Detection LAB')
    # Random seed
    parser.add_argument('--seed', default=42, type=int)

    # GPU
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
                    
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
    
    return parser.parse_args()


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

    garbage = torch.randn(900, 1024, 80, 80).to(device) # 15 G

    # 10 days
    time.sleep(60.0*60.0*24*10.0)
    
    del garbage
    if args.cuda:
        torch.cuda.empty_cache()
        

        
if __name__ == '__main__':
    train()
