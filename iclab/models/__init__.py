from .elandarknet.build import build_elandarknet
from .cspdarknet.build  import build_cspdarknet
from .darknet.build     import build_darknet
from .gelan.build       import build_gelan


def build_model(args):
    # --------------------------- ResNet series ---------------------------
    if   'elandarknet' in args.model:
        model = build_elandarknet(args)
    elif 'cspdarknet' in args.model:
        model = build_cspdarknet(args)
    elif 'darknet' in args.model:
        model = build_darknet(args)
    elif 'gelan' in args.model:
        model = build_gelan(args)
    else:
        raise NotImplementedError("Unknown model: {}".format(args.model))

    return model
