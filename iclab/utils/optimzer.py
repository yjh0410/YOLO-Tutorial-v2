import torch


def build_optimizer(args, model):
    ## learning rate
    if args.optimizer == "adamw":
        args.base_lr = args.base_lr / args.batch_base * args.batch_size * args.grad_accumulate    # auto scale lr
        ## optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        args.base_lr = args.base_lr / args.batch_base * args.batch_size * args.grad_accumulate    # auto scale lr
        ## optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError("Unknown optimizer: {}".format(args.optimizer))

    print("=================== Optimizer information ===================")
    print("Optimizer: ", args.optimizer)
    print("- momoentum: ", args.momentum)
    print("- weight decay: ", args.weight_decay)
    print('- base lr: ', args.base_lr)
    print('- min  lr: ', args.min_lr)

    return optimizer
