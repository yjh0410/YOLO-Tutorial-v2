from .vit.build import build_vision_transformer


def build_model(args, model_type='default'):
    # ----------- Vision Transformer -----------
    if "vit" in args.model:
        return build_vision_transformer(args, model_type)
    else:
        raise NotImplementedError("Unknown model: {}".format(args.model))
