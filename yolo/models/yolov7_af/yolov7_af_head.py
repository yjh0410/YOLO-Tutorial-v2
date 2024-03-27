import torch
import torch.nn as nn

from .yolov7_af_basic import BasicConv


# -------------------- Detection Head --------------------
## Single-level Detection Head
class DetHead(nn.Module):
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
                              kernel_size=3, padding=1, stride=1, 
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
    
## Multi-level Detection Head
class Yolov7DetHead(nn.Module):
    def __init__(self, cfg, in_dims):
        super().__init__()
        ## ----------- Network Parameters -----------
        self.multi_level_heads = nn.ModuleList(
            [DetHead(in_dim       = in_dims[level],
                     cls_head_dim = max(in_dims[0], min(cfg.num_classes, 128)),
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
