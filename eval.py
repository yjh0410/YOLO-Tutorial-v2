import argparse
import torch

from evaluator.voc_evaluator import VOCAPIEvaluator
from evaluator.coco_evaluator import COCOAPIEvaluator
from evaluator.customed_evaluator import CustomedEvaluator

# load transform
from dataset.build import build_dataset, build_transform

# load some utils
from utils.misc import load_weight

from config import build_config
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Real-time Object Detection LAB')
    # Basic setting
    parser.add_argument('-size', '--img_size', default=640, type=int,
                        help='the max size of input image')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use cuda')

    # Model setting
    parser.add_argument('-m', '--model', default='yolov1', type=str,
                        help='build yolo')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('-p', '--pretrained', default=None, type=str,
                        help='load pretrained weight')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='keep training')
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False,
                        help='fuse Conv & BN')
    parser.add_argument('--fuse_rep_conv', action='store_true', default=False,
                        help='fuse Conv & BN')

    # Data setting
    parser.add_argument('--root', default='/Users/liuhaoran/Desktop/python_work/object-detection/dataset/',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc.')

    # TTA
    parser.add_argument('-tta', '--test_aug', action='store_true', default=False,
                        help='use test augmentation.')

    return parser.parse_args()



def voc_test(cfg, model, data_dir, device, transform):
    evaluator = VOCAPIEvaluator(cfg=cfg,
                                data_dir=data_dir,
                                device=device,
                                transform=transform,
                                display=True)

    # VOC evaluation
    evaluator.evaluate(model)

def coco_test(cfg, model, data_dir, device, transform):
    # eval
    evaluator = COCOAPIEvaluator(
                    cfg=cfg,
                    data_dir=data_dir,
                    device=device,
                    transform=transform)

    # COCO evaluation
    evaluator.evaluate(model)

def customed_test(cfg, model, data_dir, device, transform):
    evaluator = CustomedEvaluator(
        cfg=cfg,
        data_dir=data_dir,
        device=device,
        image_set='val',
        transform=transform)

    # WiderFace evaluation
    evaluator.evaluate(model)


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

    # build model
    model, _ = build_model(args, cfg, is_val=True)

    # load trained weight
    model = load_weight(model, args.weight, args.fuse_conv_bn)
    model.to(device).eval()

    # evaluation
    with torch.no_grad():
        if args.dataset == 'voc':
            voc_test(cfg, model, args.root, device, transform)
        elif args.dataset == 'coco':
            coco_test(cfg, model, args.root, device, transform)
        elif args.dataset == 'customed':
            customed_test(cfg, model, args.root, device, transform)
