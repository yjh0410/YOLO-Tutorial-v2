import torch
from torch import optim


def build_optimizer(optimizer_cfg, model, param_dicts=None, resume=None):
    print('==============================')
    print('Optimizer: {}'.format(optimizer_cfg['optimizer']))
    print('--base_lr: {}'.format(optimizer_cfg['base_lr']))
    print('--backbone_lr_ratio: {}'.format(optimizer_cfg['backbone_lr_ratio']))
    print('--momentum: {}'.format(optimizer_cfg['momentum']))
    print('--weight_decay: {}'.format(optimizer_cfg['weight_decay']))

    if param_dicts is None:
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": optimizer_cfg['base_lr'] * optimizer_cfg['backbone_lr_ratio'],
            },
        ]

    if optimizer_cfg['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            params=param_dicts, 
            lr=optimizer_cfg['base_lr'],
            momentum=optimizer_cfg['momentum'],
            weight_decay=optimizer_cfg['weight_decay']
            )
                                
    elif optimizer_cfg['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            params=param_dicts, 
            lr=optimizer_cfg['base_lr'],
            weight_decay=optimizer_cfg['weight_decay']
            )
                                
    start_epoch = 0
    if resume is not None:
        print('keep training: ', resume)
        checkpoint = torch.load(resume)
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("optimizer")
        optimizer.load_state_dict(checkpoint_state_dict)
        start_epoch = checkpoint.pop("epoch") + 1
                                                        
    return optimizer, start_epoch


def build_detr_optimizer(optimizer_cfg, model, resume=None):
    print('==============================')
    print('Optimizer: {}'.format(optimizer_cfg['optimizer']))
    print('--base_lr: {}'.format(optimizer_cfg['base_lr']))
    print('--backbone_lr_ratio: {}'.format(optimizer_cfg['backbone_lr_ratio']))
    print('--weight_decay: {}'.format(optimizer_cfg['weight_decay']))

    # ------------- Divide model's parameters -------------
    param_dicts = [], [], [], [], [], [], []
    norm_names = ["norm"] + ["norm{}".format(i) for i in range(10000)]
    for n, p in model.named_parameters():
        # Non-Backbone's learnable parameters
        if "backbone" not in n and p.requires_grad:
            if "bias" == n.split(".")[-1]:
                param_dicts[0].append(p)      # no weight decay for all layers' bias
            else:
                if n.split(".")[-2] in norm_names:
                    param_dicts[1].append(p)  # no weight decay for all NormLayers' weight
                elif "cpb_mlp1" in n.split(".") or "cpb_mlp2" in n.split("."):
                    param_dicts[2].append(p)  # no weight decay for plain-detr cpb_mlp weight
                else:
                    param_dicts[3].append(p)  # weight decay for all Non-NormLayers' weight
        # Backbone's learnable parameters
        elif "backbone" in n and p.requires_grad:
            if "bias" == n.split(".")[-1]:
                param_dicts[4].append(p)      # no weight decay for all layers' bias
            else:
                if n.split(".")[-2] in norm_names:
                    param_dicts[5].append(p)  # no weight decay for all NormLayers' weight
                else:
                    param_dicts[6].append(p)  # weight decay for all Non-NormLayers' weight

    # Non-Backbone's learnable parameters
    optimizer = torch.optim.AdamW(param_dicts[0], lr=optimizer_cfg['base_lr'], weight_decay=0.0)
    optimizer.add_param_group({"params": param_dicts[1], "weight_decay": 0.0})
    optimizer.add_param_group({"params": param_dicts[2], "weight_decay": 0.0})
    optimizer.add_param_group({"params": param_dicts[3], "weight_decay": optimizer_cfg['weight_decay']})

    # Backbone's learnable parameters
    backbone_lr = optimizer_cfg['base_lr'] * optimizer_cfg['backbone_lr_ratio']
    optimizer.add_param_group({"params": param_dicts[4], "lr": backbone_lr, "weight_decay": 0.0})
    optimizer.add_param_group({"params": param_dicts[5], "lr": backbone_lr, "weight_decay": 0.0})
    optimizer.add_param_group({"params": param_dicts[6], "lr": backbone_lr, "weight_decay": optimizer_cfg['weight_decay']})

    start_epoch = 0
    if resume is not None:
        print('keep training: ', resume)
        checkpoint = torch.load(resume)
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("optimizer")
        optimizer.load_state_dict(checkpoint_state_dict)
        start_epoch = checkpoint.pop("epoch") + 1
                                                        
    return optimizer, start_epoch
