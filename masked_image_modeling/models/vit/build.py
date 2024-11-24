import os
import torch

from .vit     import build_vit
from .vit_mae import build_vit_mae
from .vit_cls import ViTForImageClassification


def build_vision_transformer(args, model_type='default'):
    assert args.model in ['vit_t', 'vit_s', 'vit_b', 'vit_l', 'vit_h'], "Unknown vit model: {}".format(args.model)

    # ----------- Masked Image Modeling task -----------
    if model_type == 'mae':
        model = build_vit_mae(args.model, args.img_size, args.patch_size, args.img_dim, args.mask_ratio)
    
    # ----------- Image Classification task -----------
    elif model_type == 'cls':
        image_encoder = build_vit(args.model, args.img_size, args.patch_size, args.img_dim)
        model = ViTForImageClassification(image_encoder, num_classes=args.num_classes, qkv_bias=True)
        load_mae_pretrained(model.encoder, args.pretrained)

    # ----------- Vison Backbone -----------
    elif model_type == 'default':
        model = build_vit(args.model, args.img_size, args.patch_size, args.img_dim)
        load_mae_pretrained(model, args.pretrained)
        
    else:
        raise NotImplementedError("Unknown model type: {}".format(model_type))
    
    return model


def load_mae_pretrained(model, ckpt=None):
    if ckpt is not None:
        # check path
        if not os.path.exists(ckpt):
            print("No pretrained model.")
            return model
        print('- Loading pretrained from: {}'.format(ckpt))
        checkpoint = torch.load(ckpt, map_location='cpu')
        # checkpoint state dict
        encoder_state_dict = checkpoint.pop("encoder")

        # load encoder weight into ViT's encoder
        model.load_state_dict(encoder_state_dict)
