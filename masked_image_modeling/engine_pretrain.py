import sys
import math

from utils.misc import MetricLogger, SmoothedValue


def train_one_epoch(args,
                    device,
                    model,
                    data_loader,
                    optimizer,
                    epoch,
                    lr_scheduler_warmup,
                    ):
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    epoch_size = len(data_loader)

    # Train one epoch
    for iter_i, (images, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
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

        # Inference
        output = model(images)

        # Compute loss
        loss = output["loss"]

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
