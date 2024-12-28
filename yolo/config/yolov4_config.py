# yolo Config


def build_yolov4_config(args):
    return Yolov4Config()
    
# YOLOv4 config
class Yolov4Config(object):
    def __init__(self) -> None:
        # ---------------- Model config ----------------
        self.out_stride = [8, 16, 32]
        self.max_stride = 32
        ## Backbone
        self.use_pretrained = True
        ## Head
        self.head_dim       = 256
        self.num_cls_head   = 2
        self.num_reg_head   = 2
        self.anchor_size  = [[10, 13],   [16, 30],   [33, 23],
                             [30, 61],   [62, 45],   [59, 119],
                             [116, 90],  [156, 198], [373, 326]]

        # ---------------- Post-process config ----------------
        ## Post process
        self.val_topk = 1000
        self.val_conf_thresh = 0.001
        self.val_nms_thresh  = 0.7
        self.test_topk = 100
        self.test_conf_thresh = 0.45
        self.test_nms_thresh  = 0.5

        # ---------------- Assignment config ----------------
        ## Matcher
        self.iou_thresh = 0.5
        ## Loss weight
        self.loss_obj = 1.0
        self.loss_cls = 1.0
        self.loss_box = 5.0

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
        self.mosaic_prob = 1.0
        self.mixup_prob  = 0.1
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
