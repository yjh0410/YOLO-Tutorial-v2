import math
import torch


# ------------------------- WarmUp LR Scheduler -------------------------
## Warmup LR Scheduler
class LinearWarmUpLrScheduler(object):
    def __init__(self, base_lr=0.01, wp_iter=500):
        self.base_lr = base_lr
        self.wp_iter = wp_iter
        self.warmup_factor = 0.00066667

    def set_lr(self, optimizer, cur_lr):
        for param_group in optimizer.param_groups:
            init_lr = param_group['initial_lr']
            ratio = init_lr / self.base_lr
            param_group['lr'] = cur_lr * ratio

    def __call__(self, iter, optimizer):
        # warmup
        assert iter < self.wp_iter
        alpha = iter / self.wp_iter
        warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        tmp_lr = self.base_lr * warmup_factor
        self.set_lr(optimizer, tmp_lr)
        
                           
# ------------------------- LR Scheduler -------------------------
def build_lr_scheduler(cfg, optimizer, resume=None):
    print('==============================')
    print('LR Scheduler: {}'.format(cfg.lr_scheduler))

    if cfg.lr_scheduler == "step":
        lr_step = [cfg.max_epoch // 3, cfg.max_epoch // 3 * 2]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_step, gamma=0.1)
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
