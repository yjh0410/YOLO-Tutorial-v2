import argparse
import cv2
import os
import time
import numpy as np
from copy import deepcopy
import torch

# load transform
from dataset.build import build_dataset, build_transform

# load some utils
from utils.misc import load_weight, compute_flops
from utils.box_ops import rescale_bboxes
from utils.vis_tools import visualize

from config import build_config
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Real-time Object Detection LAB')
    # Basic setting
    parser.add_argument('-size', '--img_size', default=640, type=int,
                        help='the max size of input image')
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='use cuda.')

    # Model setting
    parser.add_argument('-m', '--model', default='yolov1_r18', type=str,
                        help='build yolo')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False,
                        help='fuse Conv & BN')

    # Data setting
    parser.add_argument('--root', default='D:/python_work/dataset/COCO/',
                        help='data root')

    return parser.parse_args()


@torch.no_grad()
def test_det(model, 
             device, 
             dataset,
             transform=None
             ):
    # Step-1: Compute FLOPs and Params
    compute_flops(model, cfg.test_img_size, device)

    # Step-2: Compute FPS
    num_images = 2002
    total_time = 0
    count = 0
    with torch.no_grad():
        for index in range(num_images):
            if index % 500 == 0:
                print('Testing image {:d}/{:d}....'.format(index+1, num_images))

            # Load an image
            image, _ = dataset.pull_image(index)

            # Preprocess
            x, _, ratio = transform(image)
            x = x.unsqueeze(0).to(device)

            # Start
            torch.cuda.synchronize()
            start_time = time.perf_counter()   

            # Inference
            outputs = model(x)

            # End
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
        
            if index > 1:
                total_time += elapsed
                count += 1

        print('- FPS :', 1.0 / (total_time / count))

if __name__ == '__main__':
    args = parse_args()
    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Model Config
    cfg = build_config(args)

    # Transform
    transform = build_transform(cfg, is_train=False)

    # Dataset
    args.dataset = 'coco'
    dataset = build_dataset(args, cfg, transform, is_train=False)

    # Build model
    model = build_model(args, cfg, is_val=False)

    # Load trained weight
    model = load_weight(model, args.weight, args.fuse_conv_bn)
    model.to(device).eval()
        
    # Run
    test_det(model     = model, 
             device    = device, 
             dataset   = dataset,
             transform = transform,
             )
