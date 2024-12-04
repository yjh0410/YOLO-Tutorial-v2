import torch
import torch.nn as nn

try:
    from .gelan_basic import BasicConv
except:
    from  gelan_basic import BasicConv
    

# Single-level Head
class SingleLevelHead(nn.Module):
    def __init__(self,
                 in_dim       :int  = 256,
                 cls_head_dim :int  = 256,
                 reg_head_dim :int  = 256,
                 num_cls_head :int  = 2,
                 num_reg_head :int  = 2,
                 act_type     :str  = "silu",
                 norm_type    :str  = "BN",
                 depthwise    :bool = False):
        super().__init__()
        # --------- Basic Parameters ----------
        self.in_dim = in_dim
        self.num_cls_head = num_cls_head
        self.num_reg_head = num_reg_head
        self.act_type = act_type
        self.norm_type = norm_type
        self.depthwise = depthwise
        
        # --------- Network Parameters ----------
        ## cls head
        cls_feats = []
        self.cls_head_dim = cls_head_dim
        for i in range(num_cls_head):
            if i == 0:
                cls_feats.append(
                    BasicConv(in_dim, self.cls_head_dim,
                              kernel_size=3, padding=1, stride=1, 
                              act_type=act_type,
                              norm_type=norm_type,
                              depthwise=depthwise)
                              )
            else:
                cls_feats.append(
                    BasicConv(self.cls_head_dim, self.cls_head_dim,
                              kernel_size=3, padding=1, stride=1, 
                              act_type=act_type,
                              norm_type=norm_type,
                              depthwise=depthwise)
                              )
        ## reg head
        reg_feats = []
        self.reg_head_dim = reg_head_dim
        for i in range(num_reg_head):
            if i == 0:
                reg_feats.append(
                    BasicConv(in_dim, self.reg_head_dim,
                              kernel_size=3, padding=1, stride=1, 
                              act_type=act_type,
                              norm_type=norm_type,
                              depthwise=depthwise)
                              )
            else:
                reg_feats.append(
                    BasicConv(self.reg_head_dim, self.reg_head_dim,
                              kernel_size=3, padding=1, stride=1, group=4,
                              act_type=act_type,
                              norm_type=norm_type,
                              depthwise=depthwise)
                              )
        self.cls_feats = nn.Sequential(*cls_feats)
        self.reg_feats = nn.Sequential(*reg_feats)

        self.init_weights()
        
    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()

    def forward(self, x):
        """
            in_feats: (Tensor) [B, C, H, W]
        """
        cls_feats = self.cls_feats(x)
        reg_feats = self.reg_feats(x)

        return cls_feats, reg_feats
    
# Multi-level Head
class GElanDetHead(nn.Module):
    def __init__(self, cfg, in_dims):
        super().__init__()
        ## ----------- Network Parameters -----------
        self.multi_level_heads = nn.ModuleList(
            [SingleLevelHead(in_dim       = in_dims[level],
                             cls_head_dim = max(in_dims[0], min(cfg.num_classes * 2, 128)),
                             reg_head_dim = max(in_dims[0]//4, 16, 4*cfg.reg_max),
                             num_cls_head = cfg.num_cls_head,
                             num_reg_head = cfg.num_reg_head,
                             act_type     = cfg.head_act,
                             norm_type    = cfg.head_norm,
                             depthwise    = cfg.head_depthwise)
                             for level in range(cfg.num_levels)
                             ])
        # --------- Basic Parameters ----------
        self.in_dims = in_dims
        self.cls_head_dim = self.multi_level_heads[0].cls_head_dim
        self.reg_head_dim = self.multi_level_heads[0].reg_head_dim


    def forward(self, feats):
        """
            feats: List[(Tensor)] [[B, C, H, W], ...]
        """
        cls_feats = []
        reg_feats = []
        for feat, head in zip(feats, self.multi_level_heads):
            # ---------------- Pred ----------------
            cls_feat, reg_feat = head(feat)

            cls_feats.append(cls_feat)
            reg_feats.append(reg_feat)

        return cls_feats, reg_feats
    


if __name__=='__main__':
    import time
    from thop import profile
    # Model config
    
    # GElan-Base config
    class GElanBaseConfig(object):
        def __init__(self) -> None:
            # ---------------- Model config ----------------
            self.reg_max  = 16
            self.out_stride = [8, 16, 32]
            self.max_stride = 32
            self.num_levels = 3
            ## Head
            self.head_act  = 'lrelu'
            self.head_norm = 'BN'
            self.head_depthwise = False
            self.num_cls_head   = 2
            self.num_reg_head   = 2

    cfg = GElanBaseConfig()
    cfg.num_classes = 20

    # Build a head
    fpn_dims = [128, 256, 256]
    pyramid_feats = [torch.randn(1, fpn_dims[0], 80, 80),
                     torch.randn(1, fpn_dims[1], 40, 40),
                     torch.randn(1, fpn_dims[2], 20, 20)]
    head = GElanDetHead(cfg, fpn_dims)


    # Inference
    t0 = time.time()
    cls_feats, reg_feats = head(pyramid_feats)
    t1 = time.time()
    print('Time: ', t1 - t0)
    print("====== GElan Head output ======")
    for level, (cls_f, reg_f) in enumerate(zip(cls_feats, reg_feats)):
        print("- Level-{} : ".format(level), cls_f.shape, reg_f.shape)

    flops, params = profile(head, inputs=(pyramid_feats, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
    