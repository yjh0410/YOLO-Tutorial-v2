import torch
from torch import optim


def build_optimizer(cfg, model, resume=None):
    print('==============================')
    print('Optimizer: {}'.format(cfg.optimizer))
    print('--base_lr: {}'.format(cfg.base_lr))
    print('--backbone_lr_ratio: {}'.format(cfg.bk_lr_ratio))
    print('--momentum: {}'.format(cfg.momentum))
    print('--weight_decay: {}'.format(cfg.weight_decay))

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": cfg.base_lr * cfg.bk_lr_ratio,
        },
    ]

    if cfg.optimizer == 'sgd':
        optimizer = optim.SGD(
            params=param_dicts, 
            lr=cfg.base_lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay
            )
                                
    elif cfg.optimizer == 'adamw':
        optimizer = optim.AdamW(
            params=param_dicts, 
            lr=cfg.base_lr,
            weight_decay=cfg.weight_decay
            )
                                
    start_epoch = 0
    cfg.best_map = -1.
    if resume is not None and resume.lower() != "none":
        print('Load optimzier from the checkpoint: ', resume)
        checkpoint = torch.load(resume)
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("optimizer")
        optimizer.load_state_dict(checkpoint_state_dict)
        start_epoch = checkpoint.pop("epoch") + 1
        if "mAP" in checkpoint:
            print('--Load best metric from the checkpoint: ', resume)
            best_map = checkpoint["mAP"]
            cfg.best_map = best_map / 100.0
        del checkpoint, checkpoint_state_dict
                                                        
    return optimizer, start_epoch
