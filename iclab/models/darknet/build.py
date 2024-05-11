from .darknet import darknet_n, darknet_s, darknet_m, darknet_l, darknet_x

def build_darknet(args):
    # build vit model
    if   args.model == 'darknet_n':
        model = darknet_n(args.img_dim, args.num_classes)
    elif args.model == 'darknet_s':
        model = darknet_s(args.img_dim, args.num_classes)
    elif args.model == 'darknet_m':
        model = darknet_m(args.img_dim, args.num_classes)
    elif args.model == 'darknet_l':
        model = darknet_l(args.img_dim, args.num_classes)
    elif args.model == 'darknet_x':
        model = darknet_x(args.img_dim, args.num_classes)
    else:
        raise NotImplementedError("Unknown darknet: {}".format(args.model))
    
    return model
