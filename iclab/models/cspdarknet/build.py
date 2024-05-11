from .cspdarknet import cspdarknet_n, cspdarknet_s, cspdarknet_m, cspdarknet_l, cspdarknet_x

def build_cspdarknet(args):
    # build vit model
    if   args.model == 'cspdarknet_n':
        model = cspdarknet_n(args.img_dim, args.num_classes)
    elif args.model == 'cspdarknet_s':
        model = cspdarknet_s(args.img_dim, args.num_classes)
    elif args.model == 'cspdarknet_m':
        model = cspdarknet_m(args.img_dim, args.num_classes)
    elif args.model == 'cspdarknet_l':
        model = cspdarknet_l(args.img_dim, args.num_classes)
    elif args.model == 'cspdarknet_x':
        model = cspdarknet_x(args.img_dim, args.num_classes)
    else:
        raise NotImplementedError("Unknown cspdarknet: {}".format(args.model))
    
    return model
