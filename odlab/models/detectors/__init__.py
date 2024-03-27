# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

from .retinanet.build import build_retinanet
from .fcos.build      import build_fcos
from .yolof.build     import build_yolof
from .detr.build      import build_detr


def build_model(args, cfg, num_classes=80, is_val=False):
    # ------------ build object detector ------------
    ## RetinaNet    
    if 'retinanet' in args.model:
        model, criterion = build_retinanet(cfg, num_classes, is_val)
    ## FCOS    
    elif 'fcos' in args.model:
        model, criterion = build_fcos(cfg, num_classes, is_val)
    ## YOLOF    
    elif 'yolof' in args.model:
        model, criterion = build_yolof(cfg, num_classes, is_val)
    ## DETR    
    elif 'detr' in args.model:
        model, criterion = build_detr(cfg, num_classes, is_val)
    else:
        raise NotImplementedError("Unknown detector: {}".args.model)
    
    if is_val:
        # ------------ Keep training from the given weight ------------
        if args.resume is not None:
            print('keep training: ', args.resume)
            checkpoint = torch.load(args.resume, map_location='cpu')
            # checkpoint state dict
            checkpoint_state_dict = checkpoint.pop("model")
            model.load_state_dict(checkpoint_state_dict)

        return model, criterion

    else:      
        return model
    