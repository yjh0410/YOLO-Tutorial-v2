from .resnet import ResNet
from .modules import PlainResBlock, BottleneckResBlock


def build_resnet(args):
    if args.model == 'resnet18':
        model = ResNet(in_dim=args.img_dim,
                       block=PlainResBlock,
                       expansion=1.0,
                       num_blocks=[2, 2, 2, 2],
                       )
    elif args.model == 'resnet50':
        model = ResNet(in_dim=args.img_dim,
                       block=BottleneckResBlock,
                       expansion=4.0,
                       num_blocks=[3, 4, 6, 3],
                       )
    else:
        raise NotImplementedError("Unknown resnet: {}".format(args.model))
    
    return model
