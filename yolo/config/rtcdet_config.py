# yolo Config


def build_rtcdet_config(args):
    if   args.model == 'rtcdet_n':
        return RTCDetNConfig()
    elif args.model == 'rtcdet_t':
        return RTCDetTConfig()
    elif args.model == 'rtcdet_s':
        return RTCDetSConfig()
    elif args.model == 'rtcdet_m':
        return RTCDetMConfig()
    elif args.model == 'rtcdet_l':
        return RTCDetLConfig()
    elif args.model == 'rtcdet_x':
        return RTCDetXConfig()
    else:
        raise NotImplementedError("No config for model: {}".format(args.model))
    
# RTCDet-Base config
class RTCDetBaseConfig(object):
    def __init__(self) -> None:
        # ---------------- Model config ----------------
        self.channel_width = 1.0
        self.last_stage_ratio = 1.0
        self.num_blocks = [3, 6, 6, 3]
        self.num_levels = 3
        self.out_stride = [8, 16, 32]
        self.max_stride = 32
        self.reg_max    = 16
        self.scale      = "b"
        ## Backbone
        self.bk_act   = 'silu'
        self.bk_norm  = 'BN'
        self.bk_depthwise = False
        self.use_pretrained = False
        ## Neck
        self.neck_act       = 'silu'
        self.neck_norm      = 'BN'
        self.neck_depthwise = False
        self.neck_expand_ratio = 0.5
        self.spp_pooling_size  = 5
        ## FPN
        self.fpn_num_blocks = 3
        self.fpn_act  = 'silu'
        self.fpn_norm = 'BN'
        self.fpn_depthwise = False
        ## Head
        self.head_act  = 'silu'
        self.head_norm = 'BN'
        self.head_depthwise = False
        self.num_cls_head   = 2
        self.num_reg_head   = 2

        # ---------------- Post-process config ----------------
        ## Post process
        self.val_topk = 1000
        self.val_conf_thresh = 0.001
        self.val_nms_thresh  = 0.7
        self.test_topk = 100
        self.test_conf_thresh = 0.3
        self.test_nms_thresh  = 0.5

        # ---------------- Assignment config ----------------
        ## Matcher
        self.ota_soft_center_radius = 3.0
        self.ota_topk_candidates = 13
        ## Loss weight
        self.loss_cls = 1.0
        self.loss_box = 2.0
        self.loss_dfl = 0.5

        # ---------------- ModelEMA config ----------------
        self.use_ema = True
        self.ema_decay = 0.9998
        self.ema_tau   = 2000

        # ---------------- Optimizer config ----------------
        self.trainer      = 'yolo'
        self.optimizer    = 'adamw'
        self.per_image_lr = 0.001 / 64
        self.base_lr      = None      # base_lr = per_image_lr * batch_size
        self.min_lr_ratio = 0.01      # min_lr  = base_lr * min_lr_ratio
        self.momentum     = 0.9
        self.weight_decay = 0.05
        self.clip_max_norm   = 35.0
        self.warmup_bias_lr  = 0.1
        self.warmup_momentum = 0.8

        # ---------------- Lr Scheduler config ----------------
        self.warmup_epoch = 3
        self.lr_scheduler = "cosine"
        self.max_epoch    = 300
        self.eval_epoch   = 10
        self.no_aug_epoch = 20

        # ---------------- Data process config ----------------
        self.aug_type = 'yolo'
        self.box_format = 'xyxy'
        self.normalize_coords = False
        self.mosaic_prob = 0.0
        self.mixup_prob  = 0.0
        self.copy_paste  = 0.0           # approximated by the YOLOX's mixup
        self.multi_scale = [0.5, 1.25]   # multi scale: [img_size * 0.5, img_size * 1.25]
        ## Pixel mean & std
        self.pixel_mean = [0., 0., 0.]
        self.pixel_std  = [255., 255., 255.]
        ## Transforms
        self.train_img_size = 640
        self.test_img_size  = 640
        self.use_ablu = True
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
class RTCDetNConfig(RTCDetBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.channel_width = 0.25
        self.last_stage_ratio = 2.0
        self.num_blocks = [1, 2, 2, 1]
        self.scale = "n"
        self.fpn_num_blocks = 1

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.0
        self.copy_paste  = 0.5

# RTCDet-N
class RTCDetTConfig(RTCDetBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.channel_width = 0.375
        self.last_stage_ratio = 2.0
        self.num_blocks = [1, 2, 2, 1]
        self.scale = "t"
        self.fpn_num_blocks = 1

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.0
        self.copy_paste  = 0.5

# RTCDet-S
class RTCDetSConfig(RTCDetBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.channel_width = 0.50
        self.num_blocks = [1, 2, 2, 1]
        self.last_stage_ratio = 2.0
        self.scale = "s"
        self.fpn_num_blocks = 1

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.0
        self.copy_paste  = 0.5

# RTCDet-M
class RTCDetMConfig(RTCDetBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.channel_width = 0.75
        self.last_stage_ratio = 1.5
        self.num_blocks = [2, 4, 4, 2]
        self.scale = "m"
        self.fpn_num_blocks = 2

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.1
        self.copy_paste  = 0.5

# RTCDet-L
class RTCDetLConfig(RTCDetBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.channel_width = 1.0
        self.last_stage_ratio = 1.0
        self.num_blocks = [3, 6, 6, 3]
        self.scale = "l"
        self.fpn_num_blocks = 3

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.1
        self.copy_paste  = 0.5

# RTCDet-X
class RTCDetXConfig(RTCDetBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.channel_width = 1.25
        self.last_stage_ratio = 1.0
        self.num_blocks = [3, 6, 6, 3]
        self.scale = "x"
        self.fpn_num_blocks = 4

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.1
        self.copy_paste  = 0.5
