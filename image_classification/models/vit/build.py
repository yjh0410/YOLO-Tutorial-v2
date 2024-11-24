import torch.nn as nn
from .vit import ImageEncoderViT, ViTForImageClassification


def build_vit(args):
    if args.model == "vit_t":
        img_encoder = ImageEncoderViT(img_size=args.img_size,
                                      patch_size=args.patch_size,
                                      in_chans=args.img_dim,
                                      patch_embed_dim=192,
                                      depth=12,
                                      num_heads=3,
                                      mlp_ratio=4.0,
                                      act_layer=nn.GELU,
                                      dropout = 0.1)
    elif args.model == "vit_s":
        img_encoder = ImageEncoderViT(img_size=args.img_size,
                                      patch_size=args.patch_size,
                                      in_chans=args.img_dim,
                                      patch_embed_dim=384,
                                      depth=12,
                                      num_heads=6,
                                      mlp_ratio=4.0,
                                      act_layer=nn.GELU,
                                      dropout = 0.1)
    else:
        raise NotImplementedError("Unknown vit: {}".format(args.model))
    
    # Build ViT for classification
    return ViTForImageClassification(img_encoder, args.num_classes, qkv_bias=True)
