# End-to-end Detection with Transformer

def build_detr_config(args):
    if   args.model == 'detr_r50':
        return Detr_R50_Config()
    else:
        raise NotImplementedError("No config for model: {}".format(args.model))


class DetrBaseConfig(object):
    def __init__(self):
        # --------- Backbone ---------
        self.backbone = "resnet50"
        self.bk_norm  = "FrozeBN"
        self.res5_dilation = False
        self.use_pretrained = True
        self.freeze_at = 1
        self.max_stride = 32
        self.out_stride = 32

        # --------- Transformer ---------
        self.transformer = "detr_transformer"
        self.hidden_dim = 256
        self.num_heads = 8
        self.feedforward_dim = 2048
        self.num_queries = 100
        self.num_enc_layers = 6
        self.num_dec_layers = 6
        self.dropout = 0.1
        self.tr_act = 'relu'
        self.pre_norm = False

        # --------- Post-process ---------
        self.train_topk = 300
        self.train_conf_thresh = 0.05
        self.test_topk = 300
        self.test_conf_thresh = 0.5

        # --------- Label Assignment ---------
        self.matcher_hpy = {'cost_class': 1.0,
                            'cost_bbox':  5.0,
                            'cost_giou':  2.0,
                              }

        # --------- Loss weight ---------
        self.loss_cls  = 1.0
        self.loss_box  = 5.0
        self.loss_giou = 2.0

        # --------- Optimizer ---------
        self.optimizer = 'adamw'
        self.batch_size_base = 16
        self.per_image_lr  = 0.0001 / 16
        self.bk_lr_ratio   = 0.1
        self.momentum      = None
        self.weight_decay  = 1e-4
        self.clip_max_norm = 0.1

        # --------- LR Scheduler ---------
        self.lr_scheduler = 'step'
        self.warmup = 'linear'
        self.warmup_iters = 100
        self.warmup_factor = 0.00066667

        # --------- Train epoch ---------
        self.max_epoch = 500
        self.lr_epoch  = [400]
        self.eval_epoch = 2

        # --------- Data process ---------
        ## input size
        self.train_min_size = [800]   # short edge of image
        self.train_min_size2 = [400, 500, 600]
        self.train_max_size = 1333
        self.test_min_size  = [800]
        self.test_max_size  = 1333
        self.random_crop_size = [320, 600]
        ## Pixel mean & std
        self.pixel_mean = [0.485, 0.456, 0.406]
        self.pixel_std  = [0.229, 0.224, 0.225]
        ## Transforms
        self.box_format = 'xywh'
        self.normalize_coords = True
        self.detr_style = True
        self.trans_config = None

    def print_config(self):
        config_dict = {key: value for key, value in self.__dict__.items() if not key.startswith('__')}
        for k, v in config_dict.items():
            print("{} : {}".format(k, v))

class Detr_R50_Config(DetrBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        # --------- Backbone ---------
        self.backbone = "resnet50"
        self.bk_norm  = "FrozeBN"
        self.res5_dilation = False
        self.use_pretrained = True
