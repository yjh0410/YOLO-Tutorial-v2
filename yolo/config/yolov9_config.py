# Gelan (proposed by yolov9) config


def build_yolov9_config(args):
    if   args.model == 'yolov9_s':
        return GElanSConfig()
    elif args.model == 'yolov9_c':
        return GElanCConfig()
    else:
        raise NotImplementedError("No config for model: {}".format(args.model))
    
# GELAN-Base config
class GElanBaseConfig(object):
    def __init__(self) -> None:
        # ---------------- Model config ----------------
        self.reg_max  = 16
        self.out_stride = [8, 16, 32]
        self.max_stride = 32

        ## Backbone
        self.use_pretrained = True
        self.backbone_feats = {
            "c1": [64],
            "c2": [128, [128, 64],  256],
            "c3": [256, [256, 128], 512],
            "c4": [512, [512, 256], 512],
            "c5": [512, [512, 256], 512],
        }
        self.model_scale = "l"
        self.backbone_depth = 1

        ## Neck
        self.spp_inter_dim     = 256
        self.spp_out_dim       = 512

        ## FPN
        self.fpn_depth    = 1
        self.fpn_feats_td = {
            "p4": [[512, 256], 512],
            "p3": [[256, 128], 256],
        }
        self.fpn_feats_bu = {
            "p4": [[512, 256], 512],
            "p5": [[512, 256], 512],
        }

        ## Head
        self.num_cls_head = 2
        self.num_reg_head = 2

        # ---------------- Post-process config ----------------
        ## Post process
        self.val_topk = 1000
        self.val_conf_thresh = 0.001
        self.val_nms_thresh  = 0.7
        self.test_topk = 100
        self.test_conf_thresh = 0.2
        self.test_nms_thresh  = 0.5

        # ---------------- Assignment config ----------------
        ## Matcher
        self.tal_topk_candidates = 10
        self.tal_alpha = 0.5
        self.tal_beta  = 6.0
        ## Loss weight
        self.loss_cls = 0.5
        self.loss_box = 7.5
        self.loss_dfl = 1.5

        # ---------------- ModelEMA config ----------------
        self.use_ema = True
        self.ema_decay = 0.9998
        self.ema_tau   = 2000

        # ---------------- Optimizer config ----------------
        self.trainer      = 'yolo'
        self.optimizer    = 'adamw'
        self.base_lr      = 0.001     # base_lr = per_image_lr * batch_size
        self.min_lr_ratio = 0.01      # min_lr  = base_lr * min_lr_ratio
        self.batch_size_base = 64
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

# GELAN-C
class GElanCConfig(GElanBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self.use_pretrained = True
        self.model_scale = "c"
     
        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.1
        self.copy_paste  = 0.0

# GELAN-S
class GElanSConfig(GElanBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # ---------------- Model config ----------------
        ## Backbone
        self.use_pretrained = True
        self.backbone_feats = {
            "c1": [32],
            "c2": [64,  [64, 32],   64],
            "c3": [64,  [64, 32],   128],
            "c4": [128, [128, 64],  256],
            "c5": [256, [256, 128], 256],
        }
        self.model_scale = "s"
        self.backbone_depth = 3
        ## Neck
        self.spp_inter_dim = 128
        self.spp_out_dim   = 256
        ## FPN
        self.fpn_depth    = 3
        self.fpn_feats_td = {
            "p4": [[256, 128], 256],
            "p3": [[128, 64],  128],
        }
        self.fpn_feats_bu = {
            "p4": [[256, 128], 256],
            "p5": [[256, 128], 256],
        }

        # ---------------- Data process config ----------------
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.0
        self.copy_paste  = 0.0           # approximated by the YOLOX's mixup

    def print_config(self):
        config_dict = {key: value for key, value in self.__dict__.items() if not key.startswith('__')}
        for k, v in config_dict.items():
            print("{} : {}".format(k, v))
