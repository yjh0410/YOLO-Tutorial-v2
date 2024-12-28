import torch
import torch.nn as nn

try:
    from .modules import ConvModule, ResBlock
except:
    from  modules import ConvModule, ResBlock
    

in1k_pretrained_urls = {
    "darknet53": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/darknet53_silu.pth",
}


# --------------------- Yolov3 backbone (DarkNet-53 with SiLU) -----------------------
class Yolov3Backbone(nn.Module):
    def __init__(self, use_pretrained: bool = False):
        super(Yolov3Backbone, self).__init__()
        self.feat_dims = [256, 512, 1024]
        self.use_pretrained = use_pretrained

        # P1
        self.layer_1 = nn.Sequential(
            ConvModule(3, 32, kernel_size=3),
            ConvModule(32, 64, kernel_size=3, stride=2),
            ResBlock(64, 64, num_blocks=1)
        )
        # P2
        self.layer_2 = nn.Sequential(
            ConvModule(64, 128, kernel_size=3, stride=2),
            ResBlock(128, 128, num_blocks=2)
        )
        # P3
        self.layer_3 = nn.Sequential(
            ConvModule(128, 256, kernel_size=3, stride=2),
            ResBlock(256, 256, num_blocks=8)
        )
        # P4
        self.layer_4 = nn.Sequential(
            ConvModule(256, 512, kernel_size=3, stride=2),
            ResBlock(512, 512, num_blocks=8)
        )
        # P5
        self.layer_5 = nn.Sequential(
            ConvModule(512, 1024, kernel_size=3, stride=2),
            ResBlock(1024, 1024, num_blocks=4)
        )

        # Initialize all layers
        self.init_weights()
        
    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.reset_parameters()

        # Load imagenet pretrained weight
        if self.use_pretrained:
            self.load_pretrained()

    def load_pretrained(self):
        url = in1k_pretrained_urls["darknet53"]
        if url is not None:
            print('Loading backbone pretrained weight from : {}'.format(url))
            # checkpoint state dict
            checkpoint = torch.hub.load_state_dict_from_url(
                url=url, map_location="cpu", check_hash=True)
            checkpoint_state_dict = checkpoint.pop("model")
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
            print('No pretrained weight for model scale: {}.'.format(self.model_scale))

    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        outputs = [c3, c4, c5]

        return outputs


if __name__=='__main__':
    from thop import profile

    # Build backbone
    model = Yolov3Backbone(use_pretrained=True)

    # Randomly generate a input data
    x = torch.randn(2, 3, 640, 640)

    # Inference
    outputs = model(x)
    print(' - the shape of input :  ', x.shape)
    for out in outputs:
        print(' - the shape of output : ', out.shape)

    x = torch.randn(1, 3, 640, 640)
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('============== FLOPs & Params ================')
    print(' - FLOPs  : {:.2f} G'.format(flops / 1e9 * 2))
    print(' - Params : {:.2f} M'.format(params / 1e6))
