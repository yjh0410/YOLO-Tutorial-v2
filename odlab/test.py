import argparse
import cv2
import os
import time
import numpy as np
from copy import deepcopy
import torch

# load transform
from datasets import build_dataset, build_transform
from datasets.coco import coco_labels_91

# load some utils
from utils.misc import load_weight, compute_flops
from utils.vis_tools import visualize

from config import build_config
from models.detectors import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Object Detection Lab')
    # Basic
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='use cuda.')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show the visulization results.')
    parser.add_argument('--save', action='store_true', default=False,
                        help='save the visulization results.')
    parser.add_argument('--save_folder', default='det_results/', type=str,
                        help='Dir to save results')
    parser.add_argument('-vt', '--visual_threshold', default=0.3, type=float,
                        help='Final confidence threshold')
    parser.add_argument('-ws', '--window_scale', default=1.0, type=float,
                        help='resize window of cv2 for visualization.')
    # Model
    parser.add_argument('-m', '--model', default='yolof_r18_c5_1x', type=str,
                        help='build detector')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False,
                        help='fuse Conv & BN')
    # Dataset
    parser.add_argument('--root', default='/Users/liuhaoran/Desktop/python_work/object-detection/dataset/COCO/',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc.')

    return parser.parse_args()

@torch.no_grad()
def test_det(args, model, device, dataset, transform, class_colors, class_names):
    num_images = len(dataset)
    save_path = os.path.join('det_results/', args.dataset, args.model)
    os.makedirs(save_path, exist_ok=True)

    for index, (image, _) in enumerate(dataset):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        orig_h, orig_w = image.height, image.width

        # PreProcess
        x, _ = transform(image)
        x = x.unsqueeze(0).to(device)

        # Inference
        t0 = time.time()
        outputs = model(x)
        scores = outputs['scores']
        labels = outputs['labels']
        bboxes = outputs['bboxes']
        print("Infer. time: {}".format(time.time() - t0, "s"))
        
        # Rescale bboxes
        bboxes[..., 0::2] *= orig_w
        bboxes[..., 1::2] *= orig_h

        # Convert PIL.Image to numpy
        image = np.array(image).astype(np.uint8)
        image = image[..., (2, 1, 0)].copy()

        # Visualize results
        img_processed = visualize(image=image,
                                  bboxes=bboxes,
                                  scores=scores,
                                  labels=labels,
                                  class_colors=class_colors,
                                  class_names=class_names)
        if args.show:
            h, w = img_processed.shape[:2]
            sw, sh = int(w*args.window_scale), int(h*args.window_scale)
            cv2.namedWindow('detection', 0)
            cv2.resizeWindow('detection', sw, sh)
            cv2.imshow('detection', img_processed)
            cv2.waitKey(0)

        if args.save:
            # save result
            cv2.imwrite(os.path.join(save_path, str(index).zfill(6) +'.jpg'), img_processed)


if __name__ == '__main__':
    args = parse_args()
    # cuda
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
    dataset = build_dataset(args, cfg, is_train=False)
    if args.model == "detr_r50":
        # Test official DETR model
        cfg.class_labels = coco_labels_91
        cfg.num_classes = 91

    # Model
    model = build_model(args, cfg, is_val=False)
    model = load_weight(model, args.weight, args.fuse_conv_bn)
    model.to(device).eval()

    # Compute FLOPs and Params
    model_copy = deepcopy(model)
    model_copy.trainable = False
    model_copy.eval()
    compute_flops(
        model=model_copy,
        min_size=cfg.test_min_size,
        max_size=cfg.test_max_size,
        device=device)
    del model_copy
        
    print("================= DETECT =================")
    # Color for beautiful visualization
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255))
                     for _ in range(cfg.num_classes)]
    # Run
    test_det(args         = args,
             model        = model, 
             device       = device, 
             dataset      = dataset,
             transform    = transform,
             class_colors = class_colors,
             class_names  = cfg.class_labels,
             )
