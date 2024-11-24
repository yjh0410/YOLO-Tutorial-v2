import torch


def build_optimizer(args, model):
    ## learning rate
    if args.optimizer == "adamw":
        args.base_lr = args.base_lr / 256 * args.batch_size
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.base_lr,
                                      weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        args.base_lr = args.base_lr / 256 * args.batch_size
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.base_lr,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        raise NotImplementedError("Unknown optimizer: {}".format(args.optimizer))

    print("=================== Optimizer information ===================")
    print("Optimizer: ", args.optimizer)
    print('- base lr: ', args.base_lr)
    print('- min  lr: ', args.min_lr)

    return optimizer
