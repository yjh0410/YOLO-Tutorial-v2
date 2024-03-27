import argparse
import numpy as np
import time
import os
import torch

from datasets import build_dataset, build_transform
from utils.misc import compute_flops, fuse_conv_bn
from utils.misc import load_weight

from config import build_config
from models.detectors import build_model


parser = argparse.ArgumentParser(description='Benchmark')
# Model
parser.add_argument('-m', '--model', default='fcos_r18_1x',
                    help='build detector')
parser.add_argument('--fuse_conv_bn', action='store_true', default=False,
                    help='fuse conv and bn')
parser.add_argument('--topk', default=100, type=int,
                    help='NMS threshold')
parser.add_argument('--weight', default=None, type=str,
                    help='Trained state_dict file path to open')
# Data root
parser.add_argument('--root', default='/data/datasets/COCO',
                    help='data root')
# cuda
parser.add_argument('--cuda', action='store_true', default=False, 
                    help='use cuda.')

args = parser.parse_args()


def test(cfg, model, device, dataset, transform):
    # Step-1: Compute FLOPs and Params
    compute_flops(
        model=model,
        min_size=cfg['test_min_size'],
        max_size=cfg['test_max_size'],
        device=device)

    # Step-2: Compute FPS
    num_images = 2002
    total_time = 0
    count = 0
    with torch.no_grad():
        for index in range(num_images):
            if index % 500 == 0:
                print('Testing image {:d}/{:d}....'.format(index+1, num_images))
            image, _ = dataset[index]
            orig_h, orig_w = image.height, image.width

            # PreProcess
            x, _ = transform(image)
            x = x.unsqueeze(0).to(device)

            # star time
            torch.cuda.synchronize()
            start_time = time.perf_counter()    

            # inference
            bboxes, scores, labels = model(x)
            
            # Rescale bboxes
            bboxes[..., 0::2] *= orig_w
            bboxes[..., 1::2] *= orig_h

            # end time
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            # print("detection time used ", elapsed, "s")
            if index > 1:
                total_time += elapsed
                count += 1
            
        print('- FPS :', 1.0 / (total_time / count))



if __name__ == '__main__':
    # get device
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Dataset & Model Config
    cfg = build_config(args)

    # Transform
    transform = build_transform(cfg, is_train=False)

    # Dataset
    args.dataset = 'coco'
    dataset, dataset_info = build_dataset(args, is_train=False)

    # Model
    model = build_model(args, cfg, device, dataset_info['num_classes'], False)
    model = load_weight(model, args.weight, args.fuse_conv_bn)
    model.to(device).eval()

    # fuse conv bn
    if args.fuse_conv_bn:
        print('fuse conv and bn ...')
        model = fuse_conv_bn(model)

    # run
    test(cfg, model, device, dataset, transform)
