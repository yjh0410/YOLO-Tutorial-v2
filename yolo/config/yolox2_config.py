# yolo Config


def build_yolox2_config(args):
    if   args.model == 'yolox2_n':
        return Yolox2NConfig()
    elif args.model == 'yolox2_s':
        return Yolox2SConfig()
    elif args.model == 'yolox2_m':
        return Yolox2MConfig()
    elif args.model == 'yolox2_l':
        return Yolox2LConfig()
    elif args.model == 'yolox2_x':
        return Yolox2XConfig()
    else:
        raise NotImplementedError("No config for model: {}".format(args.model))
    
# YOLOx2-Base config
class Yolox2BaseConfig(object):
    def __init__(self) -> None:
        # ---------------- Model config ----------------
        self.width    = 1.0
        self.depth    = 1.0
        self.out_stride = [8, 16, 32]
        self.max_stride = 32
        self.num_levels = 3
        self.scale      = "b"
        ## Backbone
        self.bk_act   = 'silu'
        self.bk_norm  = 'BN'
        self.bk_depthwise = False
        self.use_pretrained = True
        ## Neck
        self.neck_act       = 'silu'
        self.neck_norm      = 'BN'
        self.neck_depthwise = False
        self.neck_expand_ratio = 0.5
        self.spp_pooling_size  = 5
        ## FPN
        self.fpn_act  = 'silu'
        self.fpn_norm = 'BN'
        self.fpn_depthwise = False
        ## Head
        self.head_act  = 'silu'
        self.head_norm = 'BN'
        self.head_depthwise = False
        self.head_dim       = 256
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
        self.mosaic_prob = 1.0
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

# YOLOx2-N
class Yolox2NConfig(Yolox2BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width = 0.25
        self.depth = 0.34
        self.scale = "n"

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.0
        self.copy_paste  = 0.5

# YOLOx2-S
class Yolox2SConfig(Yolox2BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width = 0.50
        self.depth = 0.34
        self.scale = "s"

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.0
        self.copy_paste  = 0.5

# YOLOx2-M
class Yolox2MConfig(Yolox2BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width = 0.75
        self.depth = 0.67
        self.scale = "m"

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.1
        self.copy_paste  = 0.5

# YOLOx2-L
class Yolox2LConfig(Yolox2BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width = 1.0
        self.depth = 1.0
        self.scale = "l"

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.1
        self.copy_paste  = 0.5

# YOLOx2-X
class Yolox2XConfig(Yolox2BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width = 1.25
        self.depth = 1.34
        self.scale = "x"

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.1
        self.copy_paste  = 0.5
