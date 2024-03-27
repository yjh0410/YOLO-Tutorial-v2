# Fully Convolutional One-Stage object detector

def build_fcos_config(args):
    if   args.model == 'fcos_r18_1x':
        return Fcos_R18_1x_Config()
    elif args.model == 'fcos_r50_1x':
        return Fcos_R50_1x_Config()
    elif args.model == 'fcos_rt_r18_1x':
        return FcosRT_R18_1x_Config()
    elif args.model == 'fcos_rt_r50_1x':
        return FcosRT_R50_1x_Config()
    else:
        raise NotImplementedError("No config for model: {}".format(args.model))

class FcosBaseConfig(object):
    def __init__(self):
        # --------- Backbone ---------
        self.backbone = "resnet50"
        self.bk_norm  = "FrozeBN"
        self.res5_dilation = False
        self.use_pretrained = True
        self.freeze_at = 1
        self.max_stride = 128
        self.out_stride = [8, 16, 32, 64, 128]

        # --------- Neck ---------
        self.neck = 'basic_fpn'
        self.fpn_p6_feat = True
        self.fpn_p7_feat = True
        self.fpn_p6_from_c5  = False

        # --------- Head ---------
        self.head = 'fcos_head'
        self.head_dim = 256
        self.num_cls_head = 4
        self.num_reg_head = 4
        self.head_act     = 'relu'
        self.head_norm    = 'GN'

        # --------- Post-process ---------
        self.train_topk = 1000
        self.train_conf_thresh = 0.05
        self.train_nms_thresh  = 0.6
        self.test_topk = 100
        self.test_conf_thresh = 0.5
        self.test_nms_thresh  = 0.45
        self.nms_class_agnostic = True

        # --------- Label Assignment ---------
        self.matcher = 'fcos_matcher'
        self.matcher_hpy = {'center_sampling_radius': 1.5,
                            'object_sizes_of_interest': [[-1, 64],
                                                         [64, 128],
                                                         [128, 256],
                                                         [256, 512],
                                                         [512, float('inf')]]
                                                         },

        # --------- Loss weight ---------
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2.0
        self.loss_cls_weight  = 1.0
        self.loss_reg_weight  = 1.0
        self.loss_ctn_weight  = 1.0

        # --------- Optimizer ---------
        self.optimizer = 'sgd'
        self.per_image_lr  = 0.01 / 16
        self.bk_lr_ratio   = 1.0 / 1.0
        self.momentum      = 0.9
        self.weight_decay  = 1e-4
        self.clip_max_norm = -1.0

        # --------- LR Scheduler ---------
        self.lr_scheduler = 'step'
        self.warmup = 'linear'
        self.warmup_iters = 500
        self.warmup_factor = 0.00066667

        # --------- Train epoch ---------
        self.max_epoch = 12,       # 1x
        self.lr_epoch  = [8, 11]   # 1x

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
        ]

    def print_config(self):
        config_dict = {key: value for key, value in self.__dict__.items() if not key.startswith('__')}
        for k, v in config_dict.items():
            print("{} : {}".format(k, v))

class Fcos_R18_1x_Config(FcosBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        ## Backbone
        self.backbone = "resnet18"

class Fcos_R50_1x_Config(FcosBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        ## Backbone
        self.backbone = "resnet50"

class FcosRT_R18_1x_Config(FcosBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        ## Backbone
        self.backbone = "resnet18"
        self.max_stride = 32
        self.out_stride = [8, 16, 32]

        # --------- Neck ---------
        self.neck = 'basic_fpn'
        self.fpn_p6_feat = False
        self.fpn_p7_feat = False
        self.fpn_p6_from_c5  = False

        # --------- Label Assignment ---------
        self.matcher = 'fcos_matcher'
        self.matcher_hpy = {'center_sampling_radius': 1.5,
                            'object_sizes_of_interest': [[-1, 64],
                                                         [64, 128],
                                                         [128, float('inf')]]
                                                         },

        # --------- Train epoch ---------
        self.max_epoch = 36,        # 3x
        self.lr_epoch  = [24, 33]   # 3x

        # --------- Data process ---------
        ## input size
        self.train_min_size = [256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608]   # short edge of image
        self.train_max_size = 900
        self.test_min_size  = [512]
        self.test_max_size  = 736
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
        ]

class FcosRT_R50_1x_Config(FcosBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        ## Backbone
        self.backbone = "resnet50"
        self.max_stride = 32
        self.out_stride = [8, 16, 32]

        # --------- Neck ---------
        self.neck = 'basic_fpn'
        self.fpn_p6_feat = False
        self.fpn_p7_feat = False
        self.fpn_p6_from_c5  = False

        # --------- Label Assignment ---------
        self.matcher = 'fcos_matcher'
        self.matcher_hpy = {'center_sampling_radius': 1.5,
                            'object_sizes_of_interest': [[-1, 64],
                                                         [64, 128],
                                                         [128, float('inf')]]
                                                         },

        # --------- Train epoch ---------
        self.max_epoch = 36,        # 3x
        self.lr_epoch  = [24, 33]   # 3x

        # --------- Data process ---------
        ## input size
        self.train_min_size = [256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608]   # short edge of image
        self.train_max_size = 900
        self.test_min_size  = [512]
        self.test_max_size  = 736
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
        ]
