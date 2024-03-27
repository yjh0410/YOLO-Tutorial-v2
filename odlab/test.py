import argparse
import cv2
import os
import time
import numpy as np
from copy import deepcopy
import torch

# load transform
from datasets import build_dataset, build_transform

# load some utils
from utils.misc import load_weight, compute_flops

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
    parser.add_argument('--resave', action='store_true', default=False, 
                        help='resave checkpoints without optimizer state dict.')
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

def plot_bbox_labels(img, bbox, label=None, cls_color=None, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    
    if label is not None:
        # plot title bbox
        cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
        # put the test on the title bbox
        cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img

def visualize(img, 
              bboxes, 
              scores, 
              labels, 
              vis_thresh, 
              class_colors, 
              class_names):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            cls_id = int(labels[i])
            cls_color = class_colors[cls_id]
                
            mess = '%s: %.2f' % (class_names[cls_id], scores[i])
            img = plot_bbox_labels(img, bbox, mess, cls_color, text_scale=ts)

    return img
        
@torch.no_grad()
def run(args, model, device, dataset, transform, class_colors, class_names):
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
        bboxes, scores, labels = model(x)
        print("Infer. time: {}".format(time.time() - t0, "s"))
        
        # Rescale bboxes
        bboxes[..., 0::2] *= orig_w
        bboxes[..., 1::2] *= orig_h

        # vis detection
        image = np.array(image).astype(np.uint8)
        image = image[..., (2, 1, 0)].copy()
        img_processed = visualize(
            image, bboxes, scores, labels, args.visual_threshold, class_colors, class_names)
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
    dataset, dataset_info = build_dataset(args, is_train=False)

    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255))
                     for _ in range(dataset_info['num_classes'])]

    # Model
    model = build_model(args, cfg, dataset_info['num_classes'], is_val=False)
    model = load_weight(model, args.weight, args.fuse_conv_bn)
    model.to(device).eval()

    # Compute FLOPs and Params
    model_copy = deepcopy(model)
    model_copy.trainable = False
    model_copy.eval()
    compute_flops(
        model=model_copy,
        min_size=cfg['test_min_size'],
        max_size=cfg['test_max_size'],
        device=device)
    del model_copy

    # Resave model weight
    if args.resave:
        print('Resave: {}'.format(args.model.upper()))
        checkpoint = torch.load(args.weight, map_location='cpu')
        output_dir = 'weights/{}/{}/'.format(args.dataset, args.model)
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, "{}_pure.pth".format(args.model))
        torch.save({'model': model.state_dict(),
                    'mAP': checkpoint.pop("mAP"),
                    'epoch': checkpoint.pop("epoch")}, 
                    checkpoint_path)
        
    print("================= DETECT =================")
    # run
    run(args, model, device, dataset, transform, class_colors, dataset_info['class_labels'])
