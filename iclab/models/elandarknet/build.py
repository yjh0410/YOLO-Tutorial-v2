from .elandarknet import elandarknet_n, elandarknet_s, elandarknet_m, elandarknet_l, elandarknet_x

def build_elandarknet(args):
    # build vit model
    if   args.model == 'elandarknet_n':
        model = elandarknet_n(args.img_dim, args.num_classes)
    elif args.model == 'elandarknet_s':
        model = elandarknet_s(args.img_dim, args.num_classes)
    elif args.model == 'elandarknet_m':
        model = elandarknet_m(args.img_dim, args.num_classes)
    elif args.model == 'elandarknet_l':
        model = elandarknet_l(args.img_dim, args.num_classes)
    elif args.model == 'elandarknet_x':
        model = elandarknet_x(args.img_dim, args.num_classes)
    
    return model
