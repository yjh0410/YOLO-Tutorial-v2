# yolo Config


def build_yolov2_config(args):
    if args.model == 'yolov2_r18':
        return Yolov2R18Config()
    else:
        raise NotImplementedError("No config for model: {}".format(args.model))
    
# YOLOv2-Base config
class Yolov2BaseConfig(object):
    def __init__(self) -> None:
        # ---------------- Model config ----------------
        self.out_stride = 32
        self.max_stride = 32
        ## Backbone
        self.backbone       = 'resnet50'
        self.use_pretrained = True
        ## Neck
        self.neck_act       = 'lrelu'
        self.neck_norm      = 'BN'
        self.neck_depthwise = False
        self.neck_expand_ratio = 0.5
        self.spp_pooling_size  = 5
        ## Head
        self.head_act  = 'lrelu'
        self.head_norm = 'BN'
        self.head_depthwise = False
        self.head_dim  = 512
        self.num_cls_head = 2
        self.num_reg_head = 2
        self.anchor_sizes = [[17, 25], [55, 75], [92, 206], [202, 21], [289, 311]]

        # ---------------- Post-process config ----------------
        ## Post process
        self.val_topk = 1000
        self.val_conf_thresh = 0.001
        self.val_nms_thresh  = 0.7
        self.test_topk = 300
        self.test_conf_thresh = 0.4
        self.test_nms_thresh  = 0.5

        # ---------------- Assignment config ----------------
        ## Matcher
        self.iou_thresh = 0.5
        ## Loss weight
        self.loss_obj = 1.0
        self.loss_cls = 1.0
        self.loss_box = 5.0

        # ---------------- ModelEMA config ----------------
        self.use_ema   = True
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
        self.lr_scheduler = "linear"
        self.max_epoch    = 150
        self.eval_epoch   = 10
        self.no_aug_epoch = 20

        # ---------------- Data process config ----------------
        self.aug_type = 'ssd'
        self.box_format = 'xyxy'
        self.normalize_coords = False
        self.mosaic_prob = 0.0
        self.mixup_prob  = 0.0
        self.copy_paste  = 0.0          # approximated by the YOLOX's mixup
        self.multi_scale = [0.5, 1.25]   # multi scale: [img_size * 0.5, img_size * 1.5]
        ## Pixel mean & std
        self.pixel_mean = [123.675, 116.28, 103.53]   # RGB format
        self.pixel_std  = [58.395, 57.12, 57.375]     # RGB format
        ## Transforms
        self.train_img_size = 640
        self.test_img_size  = 640

    def print_config(self):
        config_dict = {key: value for key, value in self.__dict__.items() if not key.startswith('__')}
        for k, v in config_dict.items():
            print("{} : {}".format(k, v))

# YOLOv2-R18
class Yolov2R18Config(Yolov2BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = 'resnet18'

# YOLOv2-R50
class Yolov2R50Config(Yolov2BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # TODO: Try your best.
