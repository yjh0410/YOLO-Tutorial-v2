import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

try:
    from .yolov1_basic import conv1x1, BasicBlock, Bottleneck
except:
    from  yolov1_basic import conv1x1, BasicBlock, Bottleneck

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# --------------------- Yolov1's Backbone -----------------------
class Yolov1Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone, self.feat_dim = build_resnet(cfg.backbone, cfg.use_pretrained)

    def forward(self, x):
        c5 = self.backbone(x)

        return c5


# --------------------- ResNet -----------------------
class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Input:
            x: (Tensor) -> [B, C, H, W]
        Output:
            c5: (Tensor) -> [B, C, H/32, W/32]
        """
        c1 = self.conv1(x)     # [B, C, H/2, W/2]
        c1 = self.bn1(c1)      # [B, C, H/2, W/2]
        c1 = self.relu(c1)     # [B, C, H/2, W/2]
        c2 = self.maxpool(c1)  # [B, C, H/4, W/4]

        c2 = self.layer1(c2)   # [B, C, H/4, W/4]
        c3 = self.layer2(c2)   # [B, C, H/8, W/8]
        c4 = self.layer3(c3)   # [B, C, H/16, W/16]
        c5 = self.layer4(c4)   # [B, C, H/32, W/32]

        return c5


# --------------------- Functions -----------------------
def build_resnet(model_name="resnet18", pretrained=False):
    if model_name == 'resnet18':
        model = resnet18(pretrained)
        feat_dim = 512
    elif model_name == 'resnet34':
        model = resnet34(pretrained)
        feat_dim = 512
    elif model_name == 'resnet50':
        model = resnet50(pretrained)
        feat_dim = 2048
    elif model_name == 'resnet101':
        model = resnet34(pretrained)
        feat_dim = 2048
    else:
        raise NotImplementedError("Unknown resnet: {}".format(model_name))
    
    return model, feat_dim

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        # strict = False as we don't need fc layer params.
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model

def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model

def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=False)
    return model


if __name__=='__main__':
    import time
    from thop import profile
    # YOLOv8-Base config
    class Yolov1BaseConfig(object):
        def __init__(self) -> None:
            # ---------------- Model config ----------------
            self.out_stride = 32
            self.max_stride = 32
            ## Backbone
            self.backbone       = 'resnet18'
            self.use_pretrained = True

    cfg = Yolov1BaseConfig()
    # Build backbone
    model = Yolov1Backbone(cfg)

    # Inference
    x = torch.randn(1, 3, 640, 640)
    t0 = time.time()
    output = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
    print(output.shape)

    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))    