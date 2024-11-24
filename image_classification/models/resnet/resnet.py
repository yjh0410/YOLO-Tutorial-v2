import torch
import torch.nn as nn

try:
    from .modules import ConvModule, PlainResBlock, BottleneckResBlock
except:
    from  modules import ConvModule, PlainResBlock, BottleneckResBlock


class ResNet(nn.Module):
    def __init__(self,
                 in_dim,
                 block,
                 expansion = 1.0,
                 num_blocks = [2, 2, 2, 2],
                 num_classes = 1000,
                 ) -> None:
        super().__init__()
        # ----------- Basic parameters -----------
        self.expansion = expansion
        self.num_blocks = num_blocks
        self.feat_dims  = [64,                      # C2 level
                           round(64 * expansion),   # C2 level
                           round(128 * expansion),  # C3 level
                           round(256 * expansion),  # C4 level
                           round(512 * expansion),  # C5 level
                           ]
        # ----------- Model parameters -----------
        ## Backbone
        self.layer_1 = nn.Sequential(
            ConvModule(in_dim, self.feat_dims[0],
                       kernel_size=7, padding=3, stride=2,
                       act_type='relu', norm_type='bn', depthwise=False),
            nn.MaxPool2d(kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))
        )
        self.layer_2 = self.make_layer(block, self.feat_dims[0], self.feat_dims[1], depth=num_blocks[0], downsample=False)
        self.layer_3 = self.make_layer(block, self.feat_dims[1], self.feat_dims[2], depth=num_blocks[1], downsample=True)
        self.layer_4 = self.make_layer(block, self.feat_dims[2], self.feat_dims[3], depth=num_blocks[2], downsample=True)
        self.layer_5 = self.make_layer(block, self.feat_dims[3], self.feat_dims[4], depth=num_blocks[3], downsample=True)

        ## Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(self.feat_dims[4] , num_classes)
        
    def make_layer(self, block, in_dim, out_dim, depth=1, downsample=False):
        stage_blocks = []
        for i in range(depth):
            if i == 0:
                stride = 2 if downsample else 1
                inter_dim = round(out_dim / self.expansion)
                stage_blocks.append(block(in_dim, inter_dim, out_dim, stride))
            else:
                stride = 1
                inter_dim = round(out_dim / self.expansion)
                stage_blocks.append(block(out_dim, inter_dim, out_dim, stride))
        
        layers = nn.Sequential(*stage_blocks)

        return layers
    
    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x


def build_resnet(model_name='resnet18', img_dim=3):
    if model_name == 'resnet18':
        model = ResNet(in_dim=img_dim,
                       block=PlainResBlock,
                       expansion=1.0,
                       num_blocks=[2, 2, 2, 2],
                       )
    elif model_name == 'resnet50':
        model = ResNet(in_dim=img_dim,
                       block=BottleneckResBlock,
                       expansion=4.0,
                       num_blocks=[3, 4, 6, 3],
                       )
    else:
        raise NotImplementedError("Unknown resnet: {}".format(model_name))
    
    return model


if __name__=='__main__':
    import time

    # 构建ResNet模型
    model = build_resnet(model_name='resnet18')

    # 打印模型结构
    print(model)

    # 随即成生数据
    x = torch.randn(1, 3, 224, 224)

    # 模型前向推理
    t0 = time.time()
    output = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
