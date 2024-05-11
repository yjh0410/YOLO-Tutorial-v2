from .gelan import gelan_s, gelan_c

def build_gelan(args):
    # build vit model
    if   args.model == 'gelan_s':
        model = gelan_s(args.img_dim, args.num_classes)
    elif args.model == 'gelan_c':
        model = gelan_c(args.img_dim, args.num_classes)
    else:
        raise NotImplementedError("Unknown elannet: {}".format(args.model))
    
    return model
