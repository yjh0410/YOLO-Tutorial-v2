import torch
import torch.nn as nn

try:
    from .modules import ConvModule
except:
    from  modules import ConvModule


class Yolov2DetHead(nn.Module):
    def __init__(self, cfg, in_dim: int = 256):
        super().__init__()
        # --------- Basic Parameters ----------
        self.in_dim = in_dim
        self.cls_head_dim = cfg.head_dim
        self.reg_head_dim = cfg.head_dim
        self.num_cls_head = cfg.num_cls_head
        self.num_reg_head = cfg.num_reg_head
        
        # --------- Network Parameters ----------
        ## cls head
        cls_feats = []
        for i in range(self.num_cls_head):
            if i == 0:
                cls_feats.append(ConvModule(in_dim, self.cls_head_dim, kernel_size=3, padding=1, stride=1))
            else:
                cls_feats.append(ConvModule(self.cls_head_dim, self.cls_head_dim, kernel_size=3, padding=1, stride=1))
        ## reg head
        reg_feats = []
        for i in range(self.num_reg_head):
            if i == 0:
                reg_feats.append(ConvModule(in_dim, self.reg_head_dim, kernel_size=3, padding=1, stride=1))
            else:
                reg_feats.append(ConvModule(self.reg_head_dim, self.reg_head_dim, kernel_size=3, padding=1, stride=1))
        self.cls_feats = nn.Sequential(*cls_feats)
        self.reg_feats = nn.Sequential(*reg_feats)

        self.init_weights()
        
    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.reset_parameters()

    def forward(self, x):
        """
            in_feats: (Tensor) [B, C, H, W]
        """
        cls_feats = self.cls_feats(x)
        reg_feats = self.reg_feats(x)

        return cls_feats, reg_feats


if __name__=='__main__':
    import time
    from thop import profile
    # Model config
    
    # YOLOv8-Base config
    class Yolov2BaseConfig(object):
        def __init__(self) -> None:
            # ---------------- Model config ----------------
            self.out_stride = 32
            self.max_stride = 32
            ## Head
            self.head_act  = 'lrelu'
            self.head_norm = 'BN'
            self.head_depthwise = False
            self.head_dim  = 256
            self.num_cls_head   = 2
            self.num_reg_head   = 2

    cfg = Yolov2BaseConfig()
    # Build a head
    head = Yolov2DetHead(cfg, 512)


    # Inference
    x = torch.randn(1, 512, 20, 20)
    t0 = time.time()
    cls_feat, reg_feat = head(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
    print(cls_feat.shape, reg_feat.shape)

    print('==============================')
    flops, params = profile(head, inputs=(x, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))    
