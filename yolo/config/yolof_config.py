# Modified You Only Look One-level Feature

def build_yolof_config(args):
    if   args.model == 'yolof_n':
        return YolofNConfig()
    elif args.model == 'yolof_s':
        return YolofSConfig()
    elif args.model == 'yolof_m':
        return YolofMConfig()
    elif args.model == 'yolof_l':
        return YolofLConfig()
        
    else:
        raise NotImplementedError("No config for model: {}".format(args.model))


# --------------- Base configuration ---------------
class YolofBaseConfig(object):
    def __init__(self):
        # --------- Backbone ---------
        self.width = 1.0
        self.depth = 1.0
        self.ratio = 1.0
        self.scale = "b"
        self.max_stride = 32
        self.out_stride = 16
        ## Backbone
        self.bk_act   = 'silu'
        self.bk_norm  = 'BN'
        self.bk_depthwise = False
        self.use_pretrained = True

        # --------- Neck ---------
        self.upscale_factor = 2
        self.neck_dilations = [2, 4, 6, 8]
        self.neck_expand_ratio = 0.5
        self.neck_act = 'silu'
        self.neck_norm = 'BN'
        self.neck_depthwise = False

        # --------- Head ---------
        self.head_dim     = 512
        self.num_cls_head = 4
        self.num_reg_head = 4
        self.head_act     = 'silu'
        self.head_norm    = 'BN'
        self.head_depthwise = False
        self.anchor_size  = [[16, 16],
                             [32, 32],
                             [64, 64],
                             [128, 128],
                             [256, 256],
                             [512, 512]]

        # --------- Post-process ---------
        ## Post process
        self.val_topk = 1000
        self.val_conf_thresh = 0.001
        self.val_nms_thresh  = 0.7
        self.test_topk = 300
        self.test_conf_thresh = 0.4
        self.test_nms_thresh  = 0.5

        # --------- Label Assignment ---------
        ## Matcher
        self.ota_soft_center_radius = 3.0
        self.ota_topk_candidates = 8
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
        self.mosaic_prob = 0.0
        self.mixup_prob  = 0.0
        self.copy_paste  = 0.0           # approximated by the YOLOX's mixup
        self.multi_scale = [0.5, 1.25]   # multi scale: [img_size * 0.5, img_size * 1.25]
        ## Pixel mean & std
        self.pixel_mean = [123.675, 116.28, 103.53]   # RGB format
        self.pixel_std  = [58.395, 57.12, 57.375]     # RGB format
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

# --------------- Modified YOLOF ---------------
class YolofNConfig(YolofBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width = 0.25
        self.depth = 0.34
        self.ratio = 2.0
        self.scale = "n"
        self.head_dim = 128

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.0
        self.copy_paste  = 0.5

class YolofSConfig(YolofBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width = 0.50
        self.depth = 0.34
        self.ratio = 2.0
        self.scale = "s"
        self.head_dim = 256

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.0
        self.copy_paste  = 0.5

class YolofMConfig(YolofSConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width = 0.75
        self.depth = 0.67
        self.ratio = 1.5
        self.scale = "m"
        self.head_dim = 384

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.1
        self.copy_paste  = 0.5

class YolofLConfig(YolofSConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        self.width = 1.0
        self.depth = 1.0
        self.ratio = 1.0
        self.scale = "l"
        self.head_dim = 512

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.1
        self.copy_paste  = 0.5
