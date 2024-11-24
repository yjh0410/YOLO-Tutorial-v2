import sys
import math
import torch

from utils.misc import MetricLogger, SmoothedValue, accuracy


def train_one_epoch(args,
                    device,
                    model,
                    data_loader,
                    optimizer,
                    epoch,
                    lr_scheduler_warmup,
                    criterion,
                    ):
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    epoch_size = len(data_loader)

    optimizer.zero_grad()

    # train one epoch
    for iter_i, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        ni = iter_i + epoch * epoch_size
        nw = args.wp_epoch * epoch_size

        # Warmup
        if nw > 0 and ni < nw:
            lr_scheduler_warmup(ni, optimizer)
        elif ni == nw:
            print("Warmup stage is over.")
            lr_scheduler_warmup.set_lr(optimizer, args.base_lr)

        # To device
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Inference
        output = model(images)

        # Compute loss
        loss = criterion(output, targets)

        # Check loss
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Backward
        loss.backward()

        # Optimize
        optimizer.step()
        optimizer.zero_grad()

        # Logs
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=lr)

    # gather the stats from all processes
    print("Averaged stats: {}".format(metric_logger))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Inference
        output = model(images)

        # Compute loss
        loss = criterion(output, target)

        # Compute accuracy
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss),
          )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
