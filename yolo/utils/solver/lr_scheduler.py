import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

# ------------------------- WarmUp LR Scheduler -------------------------
## Warmup LR Scheduler
class LinearWarmUpLrScheduler(object):
    def __init__(self, wp_iter=500, base_lr=0.01, warmup_bias_lr=0.0):
        self.wp_iter = wp_iter
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
        lr_step = [cfg.max_epoch // 2, cfg.max_epoch // 3 * 4]
        lr_scheduler = MultiStepLR(optimizer, milestones=lr_step, gamma=0.1)

    elif cfg.lr_scheduler == "cosine":
        if hasattr(cfg, "warmup_epoch"):
            total_epochs = cfg.max_epoch - cfg.warmup_epoch - 1
        else:
            total_epochs = cfg.max_epoch - 1
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=cfg.min_lr)
    
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