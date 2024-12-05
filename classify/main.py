import os
import time
import matplotlib.pyplot as plt
import argparse
import datetime

# ---------------- Torch compoments ----------------
import torch
import torch.backends.cudnn as cudnn

# ---------------- Dataset compoments ----------------
from data import build_dataset, build_dataloader

# ---------------- Model compoments ----------------
from models import build_model

# ---------------- Utils compoments ----------------
from utils.misc import setup_seed, load_model, save_model
from utils.optimzer import build_optimizer
from utils.lr_scheduler import build_lr_scheduler, LinearWarmUpLrScheduler

# ---------------- Training engine ----------------
from engine import train_one_epoch, evaluate


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
    parser.add_argument('--wp_epoch', type=int, default=1, 
                        help='warmup epoch for finetune with MAE pretrained')
    parser.add_argument('--start_epoch', type=int, default=0, 
                        help='start epoch for finetune with MAE pretrained')
    parser.add_argument('--max_epoch', type=int, default=50, 
                        help='max epoch')
    parser.add_argument('--eval_epoch', type=int, default=5, 
                        help='max epoch')
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset name')
    parser.add_argument('--root', type=str, default='/mnt/share/ssd2/dataset',
                        help='path to dataset folder')
    parser.add_argument('--img_dim', type=int, default=3, 
                        help='input image dimension')
    parser.add_argument('--num_classes', type=int, default=1000, 
                        help='number of the classes')
    # Model
    parser.add_argument('-m', '--model', type=str, default='mlp4',
                        help='model name')
    parser.add_argument('--resume', default=None, type=str,
                        help='keep training')
    # Optimizer
    parser.add_argument('-opt', '--optimizer', type=str, default='adamw',
                        help='sgd, adam')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.05,
                        help='weight decay')
    parser.add_argument('--base_lr', type=float, default=1e-3,
                        help='learning rate for training model')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='the final lr')
    # Lr scheduler
    parser.add_argument('-lrs', '--lr_scheduler', type=str, default='step',
                        help='lr scheduler: cosine, step')

    return parser.parse_args()

    
def main():
    args = parse_args()
    print(args)
    # set random seed
    setup_seed(args.seed)

    # Path to save model
    path_to_save = os.path.join(args.path_to_save, args.dataset, args.model)
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
    val_dataset   = build_dataset(args, is_train=False)

    # ------------------------- Build Dataloader -------------------------
    train_dataloader = build_dataloader(args, train_dataset, is_train=True)
    val_dataloader   = build_dataloader(args, val_dataset,   is_train=False)

    print('=================== Dataset Information ===================')
    print("Dataset: ", args.dataset)
    print('- train dataset size : ', len(train_dataset))
    print('- val dataset size   : ', len(val_dataset))

    # ------------------------- Build Model -------------------------
    model = build_model(args)
    model.train().to(device)
    print(model)

    # ------------------------- Build Criterion -------------------------
    criterion = torch.nn.CrossEntropyLoss()

    # ------------------------- Build Optimzier -------------------------
    optimizer = build_optimizer(args, model)

    # ------------------------- Build Lr Scheduler -------------------------
    lr_scheduler_warmup = LinearWarmUpLrScheduler(args.base_lr, wp_iter=args.wp_epoch * len(train_dataloader))
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # ------------------------- Build Criterion -------------------------
    load_model(args, model, optimizer, lr_scheduler)

    # ------------------------- Eval before Train Pipeline -------------------------
    if args.eval:
        print('evaluating ...')
        test_stats = evaluate(val_dataloader, model, device)
        print('Eval Results: [loss: %.2f][acc1: %.2f][acc5 : %.2f]' %
                (test_stats['loss'], test_stats['acc1'], test_stats['acc5']), flush=True)
        return

    # ------------------------- Training Pipeline -------------------------
    start_time = time.time()
    max_accuracy = -1.0
    print("=============== Start training for {} epochs ===============".format(args.max_epoch))
    train_loss_logs = []
    valid_loss_logs = []
    valid_acc1_logs = []
    for epoch in range(args.start_epoch, args.max_epoch):
        # train one epoch
        train_stats = train_one_epoch(args, device, model, train_dataloader, optimizer,
                                      epoch, lr_scheduler_warmup, criterion)

        # LR scheduler
        if (epoch + 1) > args.wp_epoch:
            lr_scheduler.step()

        train_loss_logs.append((epoch, train_stats["loss"]))

        # Evaluate
        if (epoch % args.eval_epoch) == 0 or (epoch + 1 == args.max_epoch):
            print("Evaluating ...")
            test_stats = evaluate(val_dataloader, model, device)
            print(f"Accuracy of the network on the {len(val_dataset)} test images: {test_stats['acc1']:.1f}%")
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f'Max accuracy: {max_accuracy:.2f}%')

            # Save model
            print('- saving the model after {} epochs ...'.format(epoch))
            save_model(args, epoch, model, optimizer, lr_scheduler, test_stats["acc1"])

            valid_acc1_logs.append((epoch, test_stats["acc1"]))
            valid_loss_logs.append((epoch, test_stats["loss"]))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # --------------- Plot log curve ---------------
    ## Training loss
    epochs = [sample[0] for sample in train_loss_logs]
    tloss  = [sample[1] for sample in train_loss_logs]
    plt.plot(epochs, tloss, c='r', label='training loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training & Validation loss curve')
    ## Valid loss
    epochs = [sample[0] for sample in valid_loss_logs]
    vloss  = [sample[1] for sample in valid_loss_logs]
    plt.plot(epochs, vloss, c='b', label='validation loss')
    plt.show()
    ## Valid acc1
    epochs = [sample[0] for sample in valid_acc1_logs]
    acc1   = [sample[1] for sample in valid_acc1_logs]
    plt.plot(epochs, acc1, label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('top1 accuracy')
    plt.title('Validation top-1 accuracy curve')
    plt.show()



if __name__ == "__main__":
    main()