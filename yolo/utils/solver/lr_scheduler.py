import numpy as np
import math
import torch


# ------------------------- WarmUp LR Scheduler -------------------------
## Warmup LR Scheduler
class LinearWarmUpLrScheduler(object):
    def __init__(self, wp_iter=500, base_lr=0.01, warmup_bias_lr=0.1, warmup_momentum=0.8):
        self.wp_iter = wp_iter
        self.warmup_momentum = warmup_momentum
        self.base_lr = base_lr
        self.warmup_bias_lr = warmup_bias_lr

    def set_lr(self, optimizer, cur_lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = cur_lr

    def __call__(self, iter, optimizer):
        # warmup
        xi = [0, self.wp_iter]
        for j, x in enumerate(optimizer.param_groups):
            # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            x['lr'] = np.interp(
                iter, xi, [self.warmup_bias_lr if j == 0 else 0.0, x['initial_lr']])
     
                           
# ------------------------- LR Scheduler -------------------------
def build_lr_scheduler(cfg, optimizer, resume=None):
    print('==============================')
    print('LR Scheduler: {}'.format(cfg.lr_scheduler))

    if cfg.lr_scheduler == "step":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.lr_step, gamma=0.1)
    elif cfg.lr_scheduler == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epoch - cfg.warmup_epoch - 1, eta_min=cfg.min_lr)
    else:
        raise NotImplementedError("Unknown lr scheduler: {}".format(cfg.lr_scheduler))
        
    if resume is not None and resume.lower() != "none":
        checkpoint = torch.load(resume)
        if 'lr_scheduler' in checkpoint.keys():
            print('--Load lr scheduler from the checkpoint: ', resume)
            # checkpoint state dict
            checkpoint_state_dict = checkpoint.pop("lr_scheduler")
            lr_scheduler.load_state_dict(checkpoint_state_dict)

    return lr_scheduler

def build_lambda_lr_scheduler(cfg, optimizer, epochs):
    """Build learning rate scheduler from cfg file."""
    print('==============================')
    print('Lr Scheduler: {}'.format(cfg.lr_scheduler))
    # Cosine LR scheduler
    if cfg.lr_scheduler == 'cosine':
        lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (cfg.min_lr_ratio - 1) + 1
    # Linear LR scheduler
    elif cfg.lr_scheduler == 'linear':
        lf = lambda x: (1 - x / epochs) * (1.0 - cfg.min_lr_ratio) + cfg.min_lr_ratio

    else:
        print('unknown lr scheduler.')
        exit(0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    return scheduler, lf
