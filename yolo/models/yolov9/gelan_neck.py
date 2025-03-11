import torch
import torch.nn as nn

try:
    from .modules import ConvModule
except:
    from  modules import ConvModule

# SPP-ELAN (from yolov9)
class SPPElan(nn.Module):
    def __init__(self, cfg, in_dim):
        """SPPElan looks like the SPPF."""
        super().__init__()
        ## ----------- Basic Parameters -----------
        self.in_dim = in_dim
        self.inter_dim = cfg.spp_inter_dim
        self.out_dim   = cfg.spp_out_dim
        ## ----------- Network Parameters -----------
        self.conv_layer_1 = ConvModule(in_dim, self.inter_dim, kernel_size=1)
        self.conv_layer_2 = ConvModule(self.inter_dim * 4, self.out_dim, kernel_size=1)
        self.pool_layer   = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

        # Initialize all layers
        self.init_weights()

    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.reset_parameters()

    def forward(self, x):
        y = [self.conv_layer_1(x)]
        y.extend(self.pool_layer(y[-1]) for _ in range(3))
        
        return self.conv_layer_2(torch.cat(y, 1))


if __name__=='__main__':
    from thop import profile

    class BaseConfig(object):
        def __init__(self) -> None:
            self.spp_inter_dim = 512
            self.spp_out_dim = 512

    # 定义模型配置文件
    cfg = BaseConfig()

    # Build a neck
    in_dim  = 512
    model = SPPElan(cfg, in_dim)

    # Randomly generate a input data
    x = torch.randn(2, in_dim, 20, 20)

    # Inference
    output = model(x)
    print(' - the shape of input :  ', x.shape)
    print(' - the shape of output : ', output.shape)

    x = torch.randn(1, in_dim, 20, 20)
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('============== FLOPs & Params ================')
    print(' - FLOPs  : {:.2f} G'.format(flops / 1e9 * 2))
    print(' - Params : {:.2f} M'.format(params / 1e6))
