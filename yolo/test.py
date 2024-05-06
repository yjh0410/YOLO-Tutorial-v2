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
    parser.add_argument('--show', action='store_true', default=False,
                        help='show the visulization results.')
    parser.add_argument('--save', action='store_true', default=False,
                        help='save the visulization results.')
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='use cuda.')
    parser.add_argument('--save_folder', default='det_results/', type=str,
                        help='Dir to save results')
    parser.add_argument('-ws', '--window_scale', default=1.0, type=float,
                        help='resize window of cv2 for visualization.')

    # Model setting
    parser.add_argument('-m', '--model', default='yolo_n', type=str,
                        help='build yolo')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False,
                        help='fuse Conv & BN')

    # Data setting
    parser.add_argument('--root', default='D:/python_work/dataset/COCO/',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc.')

    return parser.parse_args()


@torch.no_grad()
def test_det(args,
             model, 
             device, 
             dataset,
             transform=None,
             class_colors=None, 
             class_names=None):
    num_images = len(dataset)
    save_path = os.path.join('det_results/', args.dataset, args.model)
    os.makedirs(save_path, exist_ok=True)

    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        image, _ = dataset.pull_image(index)

        orig_h, orig_w, _ = image.shape
        orig_size = [orig_w, orig_h]

        # prepare
        x, _, ratio = transform(image)
        x = x.unsqueeze(0).to(device)

        t0 = time.time()
        # inference
        outputs = model(x)
        scores = outputs['scores']
        labels = outputs['labels']
        bboxes = outputs['bboxes']
        print("detection time used ", time.time() - t0, "s")
        
        # rescale bboxes
        bboxes = rescale_bboxes(bboxes, orig_size, ratio)

        # vis detection
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
    dataset = build_dataset(args, cfg, transform, is_train=False)

    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(cfg.num_classes)]

    # build model
    model = build_model(args, cfg, is_val=False)

    # load trained weight
    model = load_weight(model, args.weight, args.fuse_conv_bn)
    model.to(device).eval()

    # compute FLOPs and Params
    model_copy = deepcopy(model)
    model_copy.trainable = False
    model_copy.eval()
    compute_flops(model_copy, cfg.test_img_size, device)
    del model_copy
        
    print("================= DETECT =================")
    # run
    test_det(args         = args,
             model        = model, 
             device       = device, 
             dataset      = dataset,
             transform    = transform,
             class_colors = class_colors,
             class_names  = cfg.class_labels,
             )
