# ----------------------- Model Config -----------------------
from .fcos_config      import build_fcos_config
from .yolof_config     import build_yolof_config
from .detr_config      import build_detr_config

def build_config(args):
    # FCOS
    if "fcos" in args.model:
        return build_fcos_config(args)
    # YOLOF
    elif "yolof" in args.model:
        return build_yolof_config
    # DETR
    elif "detr" in args.model:
        return build_detr_config
    
    else:
        raise NotImplementedError('Unknown Model: {}'.format(args.model))
