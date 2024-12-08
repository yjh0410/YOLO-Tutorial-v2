import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .modules import ConvModule
except:
    from  modules import ConvModule


class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """
    def __init__(self, init_value=1.0):
        """
        init_value : initial value for the scalar
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        """
        input -> scale * input
        """
        return x * self.scale

class FcosHead(nn.Module):
    def __init__(self, cfg, in_dim: int = 256,):
        super().__init__()
        # ------------------ Basic parameters -------------------
        self.cfg = cfg
        self.in_dim = in_dim
        self.out_dim = cfg.head_dim
        self.out_stride  = cfg.out_stride
        self.num_classes = cfg.num_classes

        # ------------------ Network parameters -------------------
        ## classification head
        cls_heads = []
        self.cls_head_dim = cfg.head_dim
        for i in range(cfg.num_cls_head):
            if i == 0:
                cls_heads.append(ConvModule(in_dim, self.cls_head_dim, kernel_size=3, padding=1, stride=1))
            else:
                cls_heads.append(ConvModule(self.cls_head_dim, self.cls_head_dim, kernel_size=3, padding=1, stride=1))
        
        ## bbox regression head
        reg_heads = []
        self.reg_head_dim = cfg.head_dim
        for i in range(cfg.num_reg_head):
            if i == 0:
                reg_heads.append(ConvModule(in_dim, self.reg_head_dim, kernel_size=3, padding=1, stride=1))
            else:
                reg_heads.append(ConvModule(self.reg_head_dim, self.reg_head_dim, kernel_size=3, padding=1, stride=1))
        
        self.cls_heads = nn.Sequential(*cls_heads)
        self.reg_heads = nn.Sequential(*reg_heads)

        ## pred layers
        self.cls_pred = nn.Conv2d(self.cls_head_dim, cfg.num_classes, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(self.reg_head_dim, 4, kernel_size=3, padding=1)
        self.ctn_pred = nn.Conv2d(self.reg_head_dim, 1, kernel_size=3, padding=1)
        
        ## scale layers
        self.scales = nn.ModuleList(
            Scale() for _ in range(len(self.out_stride))
        )
        
        # init bias
        self._init_layers()

    def _init_layers(self):
        for module in [self.cls_heads, self.reg_heads, self.cls_pred, self.reg_pred, self.ctn_pred]:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0)
                if isinstance(layer, nn.GroupNorm):
                    torch.nn.init.constant_(layer.weight, 1)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0)
        # init the bias of cls pred
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        torch.nn.init.constant_(self.cls_pred.bias, bias_value)
        
    def get_anchors(self, level, fmp_size):
        """
            fmp_size: (List) [H, W]
        """
        # generate grid cells
        fmp_h, fmp_w = fmp_size
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        anchors = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2)
        anchors += 0.5
        anchors *= self.out_stride[level]

        return anchors
        
    def decode_boxes(self, pred_deltas, anchors):
        """
            pred_deltas: (List[Tensor]) [B, M, 4] or [M, 4] (l, t, r, b)
            anchors:     (List[Tensor]) [1, M, 2] or [M, 2]
        """
        # x1 = x_anchor - l, x2 = x_anchor + r
        # y1 = y_anchor - t, y2 = y_anchor + b
        pred_x1y1 = anchors - pred_deltas[..., :2]
        pred_x2y2 = anchors + pred_deltas[..., 2:]
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box
    
    def forward(self, pyramid_feats):
        all_anchors = []
        all_cls_preds = []
        all_reg_preds = []
        all_box_preds = []
        all_ctn_preds = []
        for level, feat in enumerate(pyramid_feats):
            # ------------------- Decoupled head -------------------
            cls_feat = self.cls_heads(feat)
            reg_feat = self.reg_heads(feat)

            # ------------------- Generate anchor box -------------------
            B, _, H, W = cls_feat.size()
            fmp_size = [H, W]
            anchors = self.get_anchors(level, fmp_size)   # [M, 4]
            anchors = anchors.to(cls_feat.device)

            # ------------------- Predict -------------------
            cls_pred = self.cls_pred(cls_feat)
            reg_pred = self.reg_pred(reg_feat)
            ctn_pred = self.ctn_pred(reg_feat)

            # ------------------- Process preds -------------------
            ## [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
            ctn_pred = ctn_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
            reg_pred = F.relu(self.scales[level](reg_pred)) * self.out_stride[level]

            ## Decode bbox
            box_pred = self.decode_boxes(reg_pred, anchors)
                
            all_anchors.append(anchors)
            all_cls_preds.append(cls_pred)
            all_reg_preds.append(reg_pred)
            all_box_preds.append(box_pred)
            all_ctn_preds.append(ctn_pred)

        outputs = {"pred_cls": all_cls_preds,  # List [B, M, C]
                   "pred_reg": all_reg_preds,  # List [B, M, 4]
                   "pred_box": all_box_preds,  # List [B, M, 4]
                   "pred_ctn": all_ctn_preds,  # List [B, M, 1]
                   "anchors": all_anchors,     # List [B, M, 2]
                   "strides": self.out_stride,
                   }

        return outputs 


if __name__=='__main__':
    import time
    from thop import profile
    # Model config
    
    # YOLOv3-Base config
    class FcosBaseConfig(object):
        def __init__(self) -> None:
            # ---------------- Model config ----------------
            self.width = 0.50
            self.depth = 0.34

            self.out_stride = [8, 16, 32, 64]
            self.num_classes = 20

            ## Head
            self.head_dim  = 256
            self.num_cls_head = 4
            self.num_reg_head = 4

    cfg = FcosBaseConfig()
    feat_dim = 256
    pyramid_feats = [torch.randn(1, feat_dim, 80, 80),
                     torch.randn(1, feat_dim, 40, 40),
                     torch.randn(1, feat_dim, 20, 20),
                     torch.randn(1, feat_dim, 10, 10)]

    # Build a head
    head = FcosHead(cfg, feat_dim)

    # Inference
    t0 = time.time()
    outputs = head(pyramid_feats)
    t1 = time.time()
    print('Time: ', t1 - t0)
    print("====== FCOS Head output ======")
    for k in outputs:
        print(f" - shape of {k}: ", outputs[k].shape )

    flops, params = profile(head, inputs=(pyramid_feats, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
