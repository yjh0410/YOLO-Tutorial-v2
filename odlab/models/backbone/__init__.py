from .resnet           import build_resnet


def build_backbone(cfg):
    print('==============================')
    print('Backbone: {}'.format(cfg.backbone))
    # ResNet
    if "resnet" in cfg.backbone:
        return build_resnet(cfg)
    else:
        raise NotImplementedError("unknown backbone: {}".format(cfg.backbone))
    
                           