from .mlp.build     import build_mlp
from .convnet.build import build_convnet
from .vit.build     import build_vit


def build_model(args):
    # --------------------------- ResNet series ---------------------------
    if   'mlp' in args.model:
        model = build_mlp(args)
    elif 'convnet' in args.model:
        model = build_convnet(args)
    elif 'vit' in args.model:
        model = build_vit(args)
    else:
        raise NotImplementedError("Unknown model: {}".format(args.model))

    return model
