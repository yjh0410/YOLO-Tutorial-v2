import torch
import torch.nn as nn

# ImageNet pretrained weight
pretrained_urls = {
    "darknet19": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/darknet19.pth",
}


# --------------------- Basic Module -----------------------
class ConvModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: int = 0,
                 stride: int = 1,
                 dilation: int = 1,
                 ):
        super(ConvModule, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class DarkNet19(nn.Module):
    def __init__(self, use_pretrained=False):
        super(DarkNet19, self).__init__()

        # output : stride = 2, c = 32
        self.conv_1 = nn.Sequential(
            ConvModule(3, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )

        # output : stride = 4, c = 64
        self.conv_2 = nn.Sequential(
            ConvModule(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        # output : stride = 8, c = 128
        self.conv_3 = nn.Sequential(
            ConvModule(64, 128, kernel_size=3, padding=1),
            ConvModule(128, 64, 1),
            ConvModule(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        # output : stride = 8, c = 256
        self.conv_4 = nn.Sequential(
            ConvModule(128, 256, kernel_size=3, padding=1),
            ConvModule(256, 128, 1),
            ConvModule(128, 256, kernel_size=3, padding=1),
        )

        # output : stride = 16, c = 512
        self.maxpool_4 = nn.MaxPool2d((2, 2), 2)
        self.conv_5 = nn.Sequential(
            ConvModule(256, 512, kernel_size=3, padding=1),
            ConvModule(512, 256, 1),
            ConvModule(256, 512, kernel_size=3, padding=1),
            ConvModule(512, 256, 1),
            ConvModule(256, 512, kernel_size=3, padding=1),
        )
        
        # output : stride = 32, c = 1024
        self.maxpool_5 = nn.MaxPool2d((2, 2), 2)
        self.conv_6 = nn.Sequential(
            ConvModule(512, 1024, kernel_size=3, padding=1),
            ConvModule(1024, 512, 1),
            ConvModule(512, 1024, kernel_size=3, padding=1),
            ConvModule(1024, 512, 1),
            ConvModule(512, 1024, kernel_size=3, padding=1)
        )

        if use_pretrained:
            self.load_pretrained()

    def load_pretrained(self):
        url = pretrained_urls["darknet19"]
        if url is not None:
            print(' - Loading backbone pretrained weight from : {}'.format(url))
            # checkpoint state dict
            checkpoint_state_dict = torch.hub.load_state_dict_from_url(
                url=url, map_location="cpu", check_hash=True)

            # model state dict
            model_state_dict = self.state_dict()

            # check
            for k in list(checkpoint_state_dict.keys()):
                if k in model_state_dict:
                    shape_model = tuple(model_state_dict[k].shape)
                    shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                    if shape_model != shape_checkpoint:
                        checkpoint_state_dict.pop(k)
                else:
                    checkpoint_state_dict.pop(k)
                    print('Unused key: ', k)
            # load the weight
            self.load_state_dict(checkpoint_state_dict)
        else:
            print(' - No pretrained weight for darknet-19.')

    def forward(self, x):
        c1 = self.conv_1(x)                    # c1
        c2 = self.conv_2(c1)                   # c2
        c3 = self.conv_3(c2)                   # c3
        c3 = self.conv_4(c3)                   # c3
        c4 = self.conv_5(self.maxpool_4(c3))   # c4
        c5 = self.conv_6(self.maxpool_5(c4))   # c5

        return c5


if __name__ == '__main__':
    from thop import profile

    # Build model
    model = DarkNet19(use_pretrained=True)

    # Randomly generate a input data
    x = torch.randn(2, 3, 640, 640)

    # Inference
    output = model(x)
    print(' - the shape of input :  ', x.shape)
    print(' - the shape of output : ', output.shape)

    x = torch.randn(1, 3, 640, 640)
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('============== FLOPs & Params ================')
    print(' - FLOPs  : {:.2f} G'.format(flops / 1e9 * 2))
    print(' - Params : {:.2f} M'.format(params / 1e6))
