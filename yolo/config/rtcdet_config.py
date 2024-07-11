# RTCDet config


def build_rtcdet_config(args):
    if   args.model == 'rtcdet_n':
        return RTCDet_Nano_Config()
    elif args.model == 'rtcdet_t':
        return RTCDet_Tiny_Config()
    elif args.model == 'rtcdet_s':
        return RTCDet_Small_Config()
    elif args.model == 'rtcdet_m':
        return RTCDet_Medium_Config()
    elif args.model == 'rtcdet_l':
        return RTCDet_Large_Config()
    elif args.model == 'rtcdet_x':
        return RTCDet_xLarge_Config()
    else:
        raise NotImplementedError("No config for model: {}".format(args.model))
    
# RTCDet-Base config
class RTCDetBaseConfig(object):
    def __init__(self) -> None:
        # ---------------- Model config ----------------
        self.stage_dims  = [64, 128, 256, 512, 512]
        self.stage_depth = [3, 6, 6, 3]
        self.width    = 1.0
        self.depth    = 1.0
        self.ratio    = 1.0
        self.reg_max  = 16
        self.out_stride = [8, 16, 32]
        self.max_stride = 32
        self.num_levels = 3
        ## Backbone
        self.bk_block    = 'elan_layer'
        self.bk_ds_block = 'conv'
        self.bk_act      = 'silu'
        self.bk_norm     = 'bn'
        self.bk_depthwise   = False
        ## Neck
        self.neck_act       = 'silu'
        self.neck_norm      = 'bn'
        self.neck_depthwise = False
        self.neck_expand_ratio = 0.5
        self.spp_pooling_size  = 5
        ## FPN
        self.fpn_block     = 'elan_layer'
        self.fpn_ds_block  = 'conv'
        self.fpn_act       = 'silu'
        self.fpn_norm      = 'bn'
        self.fpn_depthwise = False
        ## Head
        self.head_act  = 'silu'
        self.head_norm = 'bn'
        self.head_depthwise = False
        self.num_cls_head   = 2
        self.num_reg_head   = 2

        # ---------------- Post-process config ----------------
        ## Post process
        self.val_topk = 1000
        self.val_conf_thresh = 0.001
        self.val_nms_thresh  = 0.7
        self.test_topk = 100
        self.test_conf_thresh = 0.2
        self.test_nms_thresh  = 0.5

        # ---------------- Assignment & Loss config ----------------
        self.loss_cls_type = "bce"
        self.matcher_dict = {"tal_alpha": 0.5, "tal_beta": 6.0, "topk_candidates": 10}
        self.weight_dict  = {"loss_cls": 0.5, "loss_box": 7.5, "loss_dfl": 1.5}

        # ---------------- Assignment & Loss config ----------------
        # self.loss_cls_type = "vfl"
        # self.matcher_dict = {"tal_alpha": 1.0, "tal_beta": 6.0, "topk_candidates": 13}   # For VFL
        # self.weight_dict  = {"loss_cls": 1.0, "loss_box": 2.5, "loss_dfl": 0.5}   # For VFL

        # ---------------- ModelEMA config ----------------
        self.use_ema = True
        self.ema_decay = 0.9998
        self.ema_tau   = 2000

        # ---------------- Optimizer config ----------------
        self.trainer      = 'yolo'
        self.no_norm_decay = True
        self.no_bias_decay = True
        self.batch_size_base = 64
        self.optimizer    = 'adamw'
        self.base_lr      = 0.001
        self.min_lr_ratio = 0.05      # min_lr  = base_lr * min_lr_ratio
        self.momentum     = 0.9
        self.weight_decay = 0.05
        self.clip_max_norm   = 35.0
        self.warmup_bias_lr  = 0.1
        self.warmup_momentum = 0.8
        self.use_fp16        = True  # use mixing precision

        # ---------------- Lr Scheduler config ----------------
        self.warmup_epoch = 3
        self.lr_scheduler = "cosine"
        self.max_epoch    = 500
        self.eval_epoch   = 10
        self.no_aug_epoch = 15

        # ---------------- Data process config ----------------
        self.aug_type = 'yolo'
        self.box_format = 'xyxy'
        self.normalize_coords = False
        self.mosaic_prob = 0.0
        self.mixup_prob  = 0.0
        self.copy_paste  = 0.0           # approximated by the YOLOX's mixup
        self.multi_scale = [0.5, 1.5]   # multi scale: [img_size * 0.5, img_size * 1.5]
        ## Pixel mean & std
        self.pixel_mean = [0., 0., 0.]
        self.pixel_std  = [255., 255., 255.]
        ## Transforms
        self.train_img_size = 640
        self.test_img_size  = 640
        self.affine_params = {
            'degrees': 0.0,
            'translate': 0.2,
            'scale': [0.1, 2.0],
            'shear': 0.0,
            'perspective': 0.0,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
        }

    def print_config(self):
        config_dict = {key: value for key, value in self.__dict__.items() if not key.startswith('__')}
        for k, v in config_dict.items():
            print("{} : {}".format(k, v))

# RTCDet-N
class RTCDet_Nano_Config(RTCDetBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width = 0.25
        self.depth = 0.34
        self.ratio = 2.0

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.0
        self.copy_paste  = 1.0

# RTCDet-T
class RTCDet_Tiny_Config(RTCDetBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width = 0.375
        self.depth = 0.34
        self.ratio = 2.0

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.0
        self.copy_paste  = 1.0

# RTCDet-S
class RTCDet_Small_Config(RTCDetBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width = 0.50
        self.depth = 0.34
        self.ratio = 2.0

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.05
        self.copy_paste  = 1.0

# RTCDet-M
class RTCDet_Medium_Config(RTCDetBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width = 0.75
        self.depth = 0.67
        self.ratio = 1.5

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.1
        self.copy_paste  = 1.0

# RTCDet-L
class RTCDet_Large_Config(RTCDetBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width = 1.0
        self.depth = 1.0
        self.ratio = 1.0

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.15
        self.copy_paste  = 1.0

# RTCDet-X
class RTCDet_xLarge_Config(RTCDetBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width = 1.25
        self.depth = 1.0
        self.ratio = 1.0

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.2
        self.copy_paste  = 1.0
        