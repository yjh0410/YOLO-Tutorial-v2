import torch


def build_optimizer(args, model):
    print("=================== Optimizer information ===================")
    print("Optimizer: ", args.optimizer)
    
    ## learning rate
    if args.optimizer == "adamw":
        batch_base = 256 if "vit" in args.model else 1024
        args.base_lr = args.base_lr / batch_base * args.batch_size
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.base_lr,
                                      weight_decay=args.weight_decay)
        print('- base lr: ', args.base_lr)
        print('- min  lr: ', args.min_lr)
        print('- weight_decay: ', args.weight_decay)
    elif args.optimizer == "sgd":
        batch_base = 128
        args.base_lr = args.base_lr / batch_base * args.batch_size
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.base_lr,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)
        print('- base lr: ', args.base_lr)
        print('- min  lr: ', args.min_lr)
        print('- momentum: ', 0.9)
        print('- weight decay: ', args.weight_decay)
    else:
        raise NotImplementedError("Unknown optimizer: {}".format(args.optimizer))

    print('- min  lr: ', args.min_lr)

    return optimizer
