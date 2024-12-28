import argparse
import torch

from map_evaluator import MapEvaluator
from dataset.build import build_dataset, build_transform
from utils.misc import load_weight

from config import build_config
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Real-time Object Detection LAB')
    # Basic setting
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use cuda')

    # Model setting
    parser.add_argument('--model', default='yolov1', type=str,
                        help='build yolo')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--resume', default=None, type=str,
                        help='keep training')
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False,
                        help='fuse Conv & BN')

    # Data setting
    parser.add_argument('--root', default='/Users/liuhaoran/Desktop/python_work/object-detection/dataset/',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc.')

    return parser.parse_args()


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
    evaluator = MapEvaluator(cfg = cfg,
                             dataset_name = args.dataset,
                             data_dir  = args.root,
                             device    = device,
                             transform = transform
                             )
    evaluator.evaluate(model)
