import torch
import torch.nn as nn


# -------------------- Detection Pred Layer --------------------
## Single-level pred layer
class Yolov1DetPredLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # --------- Basic Parameters ----------
        self.stride  = cfg.out_stride
        self.cls_dim = cfg.head_dim
        self.reg_dim = cfg.head_dim
        self.num_classes = cfg.num_classes

        # --------- Network Parameters ----------
        self.obj_pred = nn.Conv2d(self.cls_dim, 1, kernel_size=1)
        self.cls_pred = nn.Conv2d(self.cls_dim, self.num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(self.reg_dim, 4, kernel_size=1)                

        self.init_bias()
        
    def init_bias(self):
        # Init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        # obj pred
        b = self.obj_pred.bias.view(1, -1)
        b.data.fill_(bias_value.item())
        self.obj_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        # cls pred
        b = self.cls_pred.bias.view(1, -1)
        b.data.fill_(bias_value.item())
        self.cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        # reg pred
        b = self.reg_pred.bias.view(-1, )
        b.data.fill_(1.0)
        self.reg_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        w = self.reg_pred.weight
        w.data.fill_(0.)
        self.reg_pred.weight = torch.nn.Parameter(w, requires_grad=True)

    def generate_anchors(self, fmp_size):
        """
            fmp_size: (List) [H, W]
        """
        # 特征图的宽和高
        fmp_h, fmp_w = fmp_size

        # 生成网格的x坐标和y坐标
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # 将xy两部分的坐标拼起来：[H, W, 2]
        anchors = torch.stack([anchor_x, anchor_y], dim=-1).float()
        
        # [H, W, 2] -> [HW, 2]
        anchors = anchors.view(-1, 2)

        return anchors
        
    def forward(self, cls_feat, reg_feat):
        # 预测层
        obj_pred = self.obj_pred(cls_feat)
        cls_pred = self.cls_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)

        # 生成网格坐标
        B, _, H, W = cls_pred.size()
        fmp_size = [H, W]
        anchors = self.generate_anchors(fmp_size)
        anchors = anchors.to(cls_pred.device)

        # 对 pred 的size做一些view调整，便于后续的处理
        # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
        obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        
        # 解算边界框坐标
        cxcy_pred = (torch.sigmoid(reg_pred[..., :2]) + anchors[..., :2]) * self.stride
        bwbh_pred = torch.exp(reg_pred[..., 2:]) * self.stride
        pred_x1y1 = cxcy_pred - bwbh_pred * 0.5
        pred_x2y2 = cxcy_pred + bwbh_pred * 0.5
        box_pred = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        # output dict
        outputs = {"pred_obj": obj_pred,       # (torch.Tensor) [B, M, 1]
                   "pred_cls": cls_pred,       # (torch.Tensor) [B, M, C]
                   "pred_reg": reg_pred,       # (torch.Tensor) [B, M, 4]
                   "pred_box": box_pred,       # (torch.Tensor) [B, M, 4]
                   "anchors" : anchors,        # (torch.Tensor) [M, 2]
                   "fmp_size": fmp_size,
                   "stride"  : self.stride,    # (Int)
                   }

        return outputs


if __name__=='__main__':
    from thop import profile
    # Model config
    
    # YOLOv1 configuration
    class Yolov1BaseConfig(object):
        def __init__(self) -> None:
            # ---------------- Model config ----------------
            self.out_stride = 32
            self.max_stride = 32
            ## Head
            self.head_dim  = 512
    cfg = Yolov1BaseConfig()
    cfg.num_classes = 20

    # Build a pred layer
    model = Yolov1DetPredLayer(cfg)

    # Randomly generate a input data
    cls_feat = torch.randn(2, cfg.head_dim, 20, 20)
    reg_feat = torch.randn(2, cfg.head_dim, 20, 20)

    # Inference
    output = model(cls_feat, reg_feat)

    print('====== Pred output ======= ')
    for k in output:
        if isinstance(output[k], torch.Tensor):
            print("-{}: ".format(k), output[k].shape)
        else:
            print("-{}: ".format(k), output[k])

    cls_feat = torch.randn(1, cfg.head_dim, 20, 20)
    reg_feat = torch.randn(1, cfg.head_dim, 20, 20)
    flops, params = profile(model, inputs=(cls_feat, reg_feat, ), verbose=False)
    print('============== FLOPs & Params ================')
    print(' - FLOPs  : {:.2f} G'.format(flops / 1e9 * 2))
    print(' - Params : {:.2f} M'.format(params / 1e6))