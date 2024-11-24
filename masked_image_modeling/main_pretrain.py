import os
import cv2
import time
import datetime
import argparse
import numpy as np

# ---------------- Torch compoments ----------------
import torch
import torch.backends.cudnn as cudnn

# ---------------- Dataset compoments ----------------
from data import build_dataset, build_dataloader
from models import build_model

# ---------------- Utils compoments ----------------
from utils.misc import setup_seed
from utils.misc import load_model, save_model, unpatchify
from utils.optimizer import build_optimizer
from utils.lr_scheduler import build_lr_scheduler, LinearWarmUpLrScheduler

# ---------------- Training engine ----------------
from engine_pretrain import train_one_epoch


def parse_args():
    parser = argparse.ArgumentParser()
    # Basic
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed.')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size on all GPUs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--path_to_save', type=str, default='weights/',
                        help='path to save trained model.')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate model.')
    # Epoch
    parser.add_argument('--wp_epoch', type=int, default=20, 
                        help='warmup epoch for finetune with MAE pretrained')
    parser.add_argument('--start_epoch', type=int, default=0, 
                        help='start epoch for finetune with MAE pretrained')
    parser.add_argument('--eval_epoch', type=int, default=10, 
                        help='warmup epoch for finetune with MAE pretrained')
    parser.add_argument('--max_epoch', type=int, default=200, 
                        help='max epoch')
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset name')
    parser.add_argument('--root', type=str, default='/mnt/share/ssd2/dataset',
                        help='path to dataset folder')
    parser.add_argument('--num_classes', type=int, default=None, 
                        help='number of classes.')
    # Model
    parser.add_argument('-m', '--model', type=str, default='vit_t',
                        help='model name')
    parser.add_argument('--resume', default=None, type=str,
                        help='keep training')
    parser.add_argument('--drop_path', type=float, default=0.,
                        help='drop_path')
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                        help='mask ratio.')    
    # Optimizer
    parser.add_argument('-opt', '--optimizer', type=str, default='adamw',
                        help='sgd, adam')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.05,
                        help='weight decay')
    parser.add_argument('--base_lr', type=float, default=0.00015,
                        help='learning rate for training model')
    parser.add_argument('--min_lr', type=float, default=0,
                        help='the final lr')
    # Optimizer
    parser.add_argument('-lrs', '--lr_scheduler', type=str, default='cosine',
                        help='step, cosine')

    return parser.parse_args()

    
def main():
    args = parse_args()
    # set random seed
    setup_seed(args.seed)

    # Path to save model
    path_to_save = os.path.join(args.path_to_save, args.dataset, "pretrained", args.model)
    os.makedirs(path_to_save, exist_ok=True)
    args.output_dir = path_to_save
    
    # ------------------------- Build CUDA -------------------------
    if args.cuda:
        if torch.cuda.is_available():
            cudnn.benchmark = True
            device = torch.device("cuda")
        else:
            print('There is no available GPU.')
            args.cuda = False
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    # ------------------------- Build Dataset -------------------------
    train_dataset = build_dataset(args, is_train=True)

    # ------------------------- Build Dataloader -------------------------
    train_dataloader = build_dataloader(args, train_dataset, is_train=True)
    print('=================== Dataset Information ===================')
    print('Train dataset size : {}'.format(len(train_dataset)))

   # ------------------------- Build Model -------------------------
    model = build_model(args, model_type='mae')
    model.train().to(device)
    print(model)

    # ------------------------- Build Optimzier -------------------------
    optimizer = build_optimizer(args, model)

    # ------------------------- Build Lr Scheduler -------------------------
    lr_scheduler_warmup = LinearWarmUpLrScheduler(args.base_lr, wp_iter=args.wp_epoch * len(train_dataloader))
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # ------------------------- Build checkpoint -------------------------
    load_model(args, model, optimizer, lr_scheduler)

    # ------------------------- Eval before Train Pipeline -------------------------
    if args.eval:
        print('visualizing ...')
        visualize(args, device, model)
        return

    # ------------------------- Training Pipeline -------------------------
    start_time = time.time()
    print("=================== Start training for {} epochs ===================".format(args.max_epoch))
    for epoch in range(args.start_epoch, args.max_epoch):
        # Train one epoch
        train_one_epoch(args, device, model, train_dataloader,
                        optimizer, epoch, lr_scheduler_warmup)

        # LR scheduler
        if (epoch + 1) > args.wp_epoch:
            lr_scheduler.step()

        # Evaluate
        if epoch % args.eval_epoch == 0 or epoch + 1 == args.max_epoch:
            print('- saving the model after {} epochs ...'.format(epoch))
            save_model(args, epoch, model, optimizer, lr_scheduler, mae_task=True)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def visualize(args, device, model):
    # test dataset
    val_dataset = build_dataset(args, is_train=False)
    val_dataloader = build_dataloader(args, val_dataset, is_train=False)

    # save path
    save_path = "vis_results/{}/{}".format(args.dataset, args.model)
    os.makedirs(save_path, exist_ok=True)

    # switch to evaluate mode
    model.eval()
    patch_size = args.patch_size
    pixel_mean = val_dataloader.dataset.pixel_mean
    pixel_std  = val_dataloader.dataset.pixel_std

    with torch.no_grad():
        for i, (images, target) in enumerate(val_dataloader):
            # To device
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # Inference
            output = model(images)

            # Denormalize input image
            org_img = images[0].permute(1, 2, 0).cpu().numpy()
            org_img = (org_img * pixel_std + pixel_mean) * 255.
            org_img = org_img.astype(np.uint8)

            # 调整mask的格式：[B, H*W] -> [B, H*W, p*p*3]
            mask = output['mask'].unsqueeze(-1).repeat(1, 1, patch_size**2 *3)  # [B, H*W] -> [B, H*W, p*p*3]
            # 将序列格式的mask逆转回二维图像格式
            mask = unpatchify(mask, patch_size)
            mask = mask[0].permute(1, 2, 0).cpu().numpy()
            # 掩盖图像中被遮掩的图像patch区域
            masked_img = org_img * (1 - mask)  # 1 is removing, 0 is keeping
            masked_img = masked_img.astype(np.uint8)

            # 将序列格式的重构图像逆转回二维图像格式
            pred_img = unpatchify(output['x_pred'], patch_size)
            pred_img = pred_img[0].permute(1, 2, 0).cpu().numpy()
            pred_img = (pred_img * pixel_std + pixel_mean) * 255.
            # 将原图中被保留的图像patch和网络预测的重构的图像patch拼在一起
            pred_img = org_img * (1 - mask) + pred_img * mask
            pred_img = pred_img.astype(np.uint8)

            # visualize
            vis_image = np.concatenate([masked_img, org_img, pred_img], axis=1)
            vis_image = vis_image[..., (2, 1, 0)]
            cv2.imshow('masked | origin | reconstruct ', vis_image)
            cv2.waitKey(0)

            # save
            cv2.imwrite('{}/{:06}.png'.format(save_path, i), vis_image)


if __name__ == "__main__":
    main()