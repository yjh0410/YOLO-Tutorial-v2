# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""

import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import (ResNet18_Weights,
                                       ResNet34_Weights,
                                       ResNet50_Weights,
                                       ResNet101_Weights)

model_urls = {
    # IN1K-Cls pretrained weights
    'resnet18':  ResNet18_Weights,
    'resnet34':  ResNet34_Weights,
    'resnet50':  ResNet50_Weights,
    'resnet101': ResNet101_Weights,
}


# Frozen BatchNormazlizarion
class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

# -------------------- ResNet series --------------------
class ResNet(nn.Module):
    """Standard ResNet backbone."""
    def __init__(self,
                 name           :str  = "resnet50",
                 res5_dilation  :bool = False,
                 norm_type      :str  = "BN",
                 freeze_at      :int  = 0,
                 use_pretrained :bool = False):
        super().__init__()
        # Pretrained
        if use_pretrained:
            pretrained_weights = model_urls[name].IMAGENET1K_V1
        else:
            pretrained_weights = None

        # Norm layer
        print("- Norm layer of backbone: {}".format(norm_type))
        if norm_type == 'BN':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'FrozeBN':
            norm_layer = FrozenBatchNorm2d
        else:
            raise NotImplementedError("Unknown norm type: {}".format(norm_type))

        # Backbone
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, res5_dilation],
            norm_layer=norm_layer, weights=pretrained_weights)
        return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.feat_dims = [128, 256, 512] if name in ('resnet18', 'resnet34') else [512, 1024, 2048]
 
        # Freeze
        print("- Freeze at {}".format(freeze_at))
        if freeze_at >= 0:
            for name, parameter in backbone.named_parameters():
                if freeze_at == 0: # Only freeze stem layer
                    if 'layer1' not in name and 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                        parameter.requires_grad_(False)
                elif freeze_at == 1: # Freeze stem layer + layer1
                    if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                        parameter.requires_grad_(False)
                elif freeze_at == 2: # Freeze stem layer + layer1 + layer2
                    if 'layer3' not in name and 'layer4' not in name:
                        parameter.requires_grad_(False)
                elif freeze_at == 3: # Freeze stem layer + layer1 + layer2 + layer3
                    if 'layer4' not in name:
                        parameter.requires_grad_(False)
                else: # Freeze all resnet's layers
                    parameter.requires_grad_(False)

    def forward(self, x):
        xs = self.body(x)
        fmp_list = []
        for name, fmp in xs.items():
            fmp_list.append(fmp)

        return fmp_list


# build backbone
def build_resnet(cfg):
    # ResNet series
    backbone = ResNet(
        name           = cfg.backbone,
        res5_dilation  = cfg.res5_dilation,
        norm_type      = cfg.bk_norm,
        use_pretrained = cfg.use_pretrained,
        freeze_at      = cfg.freeze_at)

    return backbone, backbone.feat_dims


if __name__ == '__main__':

    class FcosBaseConfig(object):
        def __init__(self):
            self.backbone = "resnet18"
            self.bk_norm = "FrozeBN"
            self.res5_dilation = False
            self.use_pretrained = True
            self.freeze_at = 0

    cfg = FcosBaseConfig()
    model, feat_dim = build_resnet(cfg)
    print(feat_dim)

    x = torch.randn(2, 3, 320, 320)
    output = model(x)
    for k in model.state_dict():
        print(k)
    for y in output:
        print(y.size())
