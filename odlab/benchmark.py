import argparse
import time
import torch

# load transform
from datasets import build_dataset, build_transform

# load some utils
from utils.misc import compute_flops, load_weight

from config import build_config
from models.detectors import build_model


parser = argparse.ArgumentParser(description='Benchmark')
# Model
parser.add_argument('-m', '--model', default='fcos_r18_1x',
                    help='build detector')
parser.add_argument('--fuse_conv_bn', action='store_true', default=False,
                    help='fuse conv and bn')
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
        min_size=cfg.test_min_size,
        max_size=cfg.test_max_size,
        device=device)

    # Step-2: Compute FPS
    num_images = 2002
    total_time = 0
    count = 0
    with torch.no_grad():
        for index in range(num_images):
            if index % 500 == 0:
                print('Testing image {:d}/{:d}....'.format(index+1, num_images))

            # Load an image
            image, _ = dataset[index]

            # Preprocess
            x, _ = transform(image)
            x = x.unsqueeze(0).to(device)

            # Star
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
    dataset = build_dataset(args, cfg, is_train=False)

    # Model
    model = build_model(args, cfg, is_val=False)
    model = load_weight(model, args.weight, args.fuse_conv_bn)
    model.to(device).eval()

    print("================= DETECT =================")
    # Run
    test(cfg, model, device, dataset, transform)
