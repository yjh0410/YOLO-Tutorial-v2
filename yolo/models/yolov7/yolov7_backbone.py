import torch
import torch.nn as nn

try:
    from .modules import ConvModule, ELANBlock, DownSample
except:
    from  modules import ConvModule, ELANBlock, DownSample
    

in1k_pretrained_urls = {
    "elannet_large": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/yolov7_elannet_large.pth",
}

# --------------------- Yolov7 backbone -----------------------
class Yolov7Backbone(nn.Module):
    def __init__(self, use_pretrained: bool = False):
        super(Yolov7Backbone, self).__init__()
        self.feat_dims = [32, 64, 128, 256, 512, 1024, 1024]
        self.squeeze_ratios = [0.5, 0.5, 0.5, 0.25]  # Stage-1 -> Stage-4
        self.branch_depths = [2, 2, 2, 2]            # Stage-1 -> Stage-4
        self.use_pretrained = use_pretrained

        # -------------------- Network parameters --------------------
        ## P1/2
        self.layer_1 = nn.Sequential(
            ConvModule(3, self.feat_dims[0], kernel_size=3),      
            ConvModule(self.feat_dims[0], self.feat_dims[1], kernel_size=3, stride=2),
            ConvModule(self.feat_dims[1], self.feat_dims[1], kernel_size=3)
        )
        ## P2/4: Stage-1
        self.layer_2 = nn.Sequential(   
            ConvModule(self.feat_dims[1], self.feat_dims[2], kernel_size=3, stride=2),             
            ELANBlock(in_dim = self.feat_dims[2],
                      out_dim = self.feat_dims[3],
                      expansion = self.squeeze_ratios[0],
                      branch_depth = self.branch_depths[0],
                      )
        )
        ## P3/8: Stage-2
        self.layer_3 = nn.Sequential(
            DownSample(self.feat_dims[3], self.feat_dims[3]),
            ELANBlock(in_dim = self.feat_dims[3],
                      out_dim = self.feat_dims[4],
                      expansion = self.squeeze_ratios[1],
                      branch_depth = self.branch_depths[1],
                      )
        )
        ## P4/16: Stage-3
        self.layer_4 = nn.Sequential(
            DownSample(self.feat_dims[4], self.feat_dims[4]),
            ELANBlock(in_dim = self.feat_dims[4],
                      out_dim = self.feat_dims[5],
                      expansion = self.squeeze_ratios[2],
                      branch_depth = self.branch_depths[2],
                      )
        )
        ## P5/32: Stage-4
        self.layer_5 = nn.Sequential(
            DownSample(self.feat_dims[5], self.feat_dims[5]),
            ELANBlock(in_dim = self.feat_dims[5],
                      out_dim = self.feat_dims[6],
                      expansion = self.squeeze_ratios[3],
                      branch_depth = self.branch_depths[3],
                      )
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
        url = in1k_pretrained_urls["elannet_large"]
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
    model = Yolov7Backbone(use_pretrained=True)

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
