# Fcos Config


def build_fcos_config(args):
    if args.model == 'fcos_r18':
        return FcosR18Config()
    else:
        raise NotImplementedError("No config for model: {}".format(args.model))
    
# Fcos-Base config
class FcosBaseConfig(object):
    def __init__(self) -> None:
        # ---------------- Model config ----------------
        self.out_stride = [8, 16, 32, 64]
        self.max_stride = 64
        ## Backbone
        self.backbone = 'resnet50'
        self.use_pretrained = True
        ## Head
        self.head_dim  = 256
        self.num_cls_head = 4
        self.num_reg_head = 4

        # ---------------- Post-process config ----------------
        ## Post process
        self.val_topk = 1000
        self.val_conf_thresh = 0.05
        self.val_nms_thresh  = 0.6
        self.test_topk = 100
        self.test_conf_thresh = 0.3
        self.test_nms_thresh  = 0.45

        # ---------------- Assignment config ----------------
        ## Matcher
        self.center_sampling_radius = 1.5
        self.object_sizes_of_interest = [[-1, 64], [64, 128], [128, 256], [256, float('inf')]]

        ## Loss weight
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2.0
        self.loss_cls = 1.0
        self.loss_reg = 1.0
        self.loss_ctn = 1.0

        # ---------------- ModelEMA config ----------------
        self.use_ema   = False
        self.ema_decay = 0.9998
        self.ema_tau   = 2000

        # ---------------- Optimizer config ----------------
        self.trainer      = 'simple'
        self.optimizer    = 'sgd'
        self.base_lr      = 0.01     # base_lr = per_image_lr * batch_size
        self.min_lr_ratio = 0.01     # min_lr  = base_lr * min_lr_ratio
        self.bk_lr_ratio  = 1.0
        self.batch_size_base = 16
        self.momentum     = 0.9
        self.weight_decay = 0.0001
        self.clip_max_norm   = 10.0
        self.warmup_bias_lr  = 0.0
        self.warmup_momentum = 0.9

        # ---------------- Lr Scheduler config ----------------
        self.warmup_iters = 500
        self.lr_scheduler = "cosine"
        self.max_epoch    = 150
        self.eval_epoch   = 10
        self.no_aug_epoch = -1

        # ---------------- Data process config ----------------
        self.aug_type = 'yolo'
        self.mosaic_prob = 0.0
        self.mixup_prob  = 0.0
        self.copy_paste  = 0.0          # approximated by the YOLOX's mixup
        self.multi_scale = [0.5, 1.5]   # multi scale: [img_size * 0.5, img_size * 1.5]
        ## Pixel mean & std
        self.pixel_mean = [0., 0., 0.]
        self.pixel_std  = [255., 255., 255.]
        ## Transforms
        self.train_img_size = 640
        self.test_img_size  = 640
        self.affine_params = {
            'degrees': 0.0,
            'translate': 0.1,
            'scale': [0.5, 1.5],
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

# YOLOv2-R18
class FcosR18Config(FcosBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = 'resnet18'

# YOLOv2-R50
class FcosR50Config(FcosBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # TODO: Try your best.
