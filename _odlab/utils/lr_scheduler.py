import torch


# ------------------------- WarmUp LR Scheduler -------------------------
## Warmup LR Scheduler
class LinearWarmUpScheduler(object):
    def __init__(self, base_lr=0.01, wp_iter=500, warmup_factor=0.00066667):
        self.base_lr = base_lr
        self.wp_iter = wp_iter
        self.warmup_factor = warmup_factor

    def set_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            init_lr = param_group['initial_lr']
            ratio = init_lr / self.base_lr
            param_group['lr'] = lr * ratio

    def __call__(self, iter, optimizer):
        # warmup
        alpha = iter / self.wp_iter
        warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        tmp_lr = self.base_lr * warmup_factor
        self.set_lr(optimizer, tmp_lr)
        
## Build WP LR Scheduler
def build_wp_lr_scheduler(cfg):
    print('==============================')
    print('WarmUpScheduler: {}'.format(cfg.warmup))
    print('--base_lr: {}'.format(cfg.base_lr))
    print('--warmup_iters: {} ({})'.format(cfg.warmup_iters, cfg.warmup_iters * cfg.grad_accumulate))
    print('--warmup_factor: {}'.format(cfg.warmup_factor))

    if cfg.warmup == 'linear':
        wp_lr_scheduler = LinearWarmUpScheduler(cfg.base_lr, cfg.warmup_iters, cfg.warmup_factor)
    
    return wp_lr_scheduler

                           
# ------------------------- LR Scheduler -------------------------
def build_lr_scheduler(cfg, optimizer, resume=None):
    print('==============================')
    print('LR Scheduler: {}'.format(cfg.lr_scheduler))

    if cfg.lr_scheduler == 'step':
        assert hasattr(cfg, 'lr_epoch')
        print('--lr_epoch: {}'.format(cfg.lr_epoch))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=cfg.lr_epoch)
    elif cfg.lr_scheduler == 'cosine':
        pass
        
    if resume is not None and resume.lower() != "none":
        print('Load lr scheduler from the checkpoint: ', resume)
        checkpoint = torch.load(resume)
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("lr_scheduler")
        lr_scheduler.load_state_dict(checkpoint_state_dict)

    return lr_scheduler
