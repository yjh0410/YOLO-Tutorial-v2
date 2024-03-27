from .resnet           import build_resnet
from .swin_transformer import build_swin_transformer


def build_backbone(cfg):
    print('==============================')
    print('Backbone: {}'.format(cfg['backbone']))
    # ResNet
    if cfg['backbone'] in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        return build_resnet(cfg)
    # SwinTransformer
    elif cfg['backbone'] in ['swin_T_224_1k', 'swin_S_224_22k', 'swin_B_224_22k', 'swin_B_384_22k', 'swin_L_224_22k', 'swin_L_384_22k']:
        return build_swin_transformer(cfg)
    
                           