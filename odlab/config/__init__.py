# ----------------------- Model Config -----------------------
from .fcos_config      import fcos_cfg
from .yolof_config     import yolof_cfg
from .detr_config      import detr_cfg

def build_config(args):
    # FCOS
    if   args.model in fcos_cfg.keys():
        return fcos_cfg[args.model]
    # YOLOF
    elif args.model in yolof_cfg.keys():
        return yolof_cfg[args.model]
    # DETR
    elif args.model in detr_cfg.keys():
        return detr_cfg[args.model]
    else:
        raise NotImplementedError('Unknown Model: {}'.format(args.model))
