# You Only Look One-level Feature

def build_yolof_config(args):
    if   args.model == 'yolof_r18_c5_1x':
        return Yolof_R18_C5_1x_Config()
    elif args.model == 'yolof_r50_c5_1x':
        return Yolof_R50_C5_1x_Config()
    elif args.model == 'yolof_r50_dc5_1x':
        return Yolof_R50_DC5_1x_Config()
    else:
        raise NotImplementedError("No config for model: {}".format(args.model))

class YolofBaseConfig(object):
    def __init__(self):
        # --------- Backbone ---------
        self.backbone = "resnet50"
        self.bk_norm  = "FrozeBN"
        self.res5_dilation = False
        self.use_pretrained = True
        self.freeze_at = 1
        self.max_stride = 32
        self.out_stride = 32

        # --------- Neck ---------
        self.neck = 'dilated_encoder'
        self.neck_dilations = [2, 4, 6, 8]
        self.neck_expand_ratio = 0.25
        self.neck_act = 'relu'
        self.neck_norm = 'GN'

        # --------- Head ---------
        self.head         = 'yolof_head'
        self.head_dim     = 512
        self.num_cls_head = 2
        self.num_reg_head = 4
        self.head_act     = 'relu'
        self.head_norm    = 'GN'
        self.center_clamp = 32
        self.anchor_size  = [[32, 32],
                             [64, 64],
                             [128, 128],
                             [256, 256],
                             [512, 512]]

        # --------- Post-process ---------
        self.train_topk = 1000
        self.train_conf_thresh = 0.05
        self.train_nms_thresh  = 0.6
        self.test_topk = 300
        self.test_conf_thresh = 0.3
        self.test_nms_thresh  = 0.45
        self.nms_class_agnostic = True

        # --------- Label Assignment ---------
        self.matcher = 'yolof_matcher'
        self.matcher_hpy = {'topk_candidates': 4,
                            'iou_thresh': 0.15,
                            'ignore_thresh': 0.7,
                              }

        # --------- Loss weight ---------
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2.0
        self.loss_cls_weight  = 1.0
        self.loss_reg_weight  = 1.0

        # --------- Optimizer ---------
        self.optimizer = 'sgd'
        self.batch_size_base = 64
        self.per_image_lr  = 0.12 / 64
        self.bk_lr_ratio   = 1.0 / 3.0
        self.momentum      = 0.9
        self.weight_decay  = 1e-4
        self.clip_max_norm = 10.0


        # --------- LR Scheduler ---------
        self.lr_scheduler = 'step'
        self.warmup = 'linear'
        self.warmup_iters = 1500
        self.warmup_factor = 0.00066667

        # --------- Train epoch ---------
        self.max_epoch = 12        # 1x
        self.lr_epoch  = [8, 11]   # 1x
        self.eval_epoch = 2

        # --------- Data process ---------
        ## input size
        self.train_min_size = [800]   # short edge of image
        self.train_max_size = 1333
        self.test_min_size  = [800]
        self.test_max_size  = 1333
        ## Pixel mean & std
        self.pixel_mean = [0.485, 0.456, 0.406]
        self.pixel_std  = [0.229, 0.224, 0.225]
        ## Transforms
        self.box_format = 'xyxy'
        self.normalize_coords = False
        self.detr_style = False
        self.trans_config = [
            {'name': 'RandomHFlip'},
            {'name': 'RandomResize'},
            {'name': 'RandomShift', 'max_shift': 32},
        ]

    def print_config(self):
        config_dict = {key: value for key, value in self.__dict__.items() if not key.startswith('__')}
        for k, v in config_dict.items():
            print("{} : {}".format(k, v))

class Yolof_R18_C5_1x_Config(YolofBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        ## Backbone
        # --------- Backbone ---------
        self.backbone = "resnet18"

class Yolof_R50_C5_1x_Config(YolofBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        ## Backbone
        # --------- Backbone ---------
        self.backbone = "resnet50"

class Yolof_R50_DC5_1x_Config(YolofBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        ## Backbone
        # --------- Backbone ---------
        self.backbone = "resnet50"
        self.res5_dilation = True
        self.use_pretrained = True
        self.max_stride = 16
        self.out_stride = 16

        # --------- Neck ---------
        self.neck = 'dilated_encoder'
        self.neck_dilations = [4, 8, 12, 16]
        self.neck_expand_ratio = 0.25
        self.neck_act = 'relu'
        self.neck_norm = 'GN'

        # --------- Head ---------
        self.anchor_size  = [[16, 16],
                             [32, 32],
                             [64, 64],
                             [128, 128],
                             [256, 256],
                             [512, 512]],

        # --------- Label Assignment ---------
        self.matcher = 'yolof_matcher'
        self.matcher_hpy = {'topk_candidates': 8,
                            'iou_thresh': 0.1,
                            'ignore_thresh': 0.7,
                              }
