# Fully Convolutional One-Stage object detector

def build_fcos_config(args):
    if   args.model == 'fcos_r18_1x':
        return Fcos_R18_1x_Config()
    else:
        raise NotImplementedError("No config for model: {}".format(args.model))

class FcosBaseConfig(object):
    def __init__(self):
        pass

    def print_config(self):
        config_dict = {key: value for key, value in self.__dict__.items() if not key.startswith('__')}
        for k, v in config_dict.items():
            print("{} : {}".format(k, v))

class Fcos_R18_1x_Config(FcosBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        ## Backbone
        pass


fcos_cfg = {
    'fcos_r18_1x':{
        # ----------------- Model-----------------
        ## Backbone
        'backbone': 'resnet18',
        'backbone_norm': 'FrozeBN',
        'res5_dilation': False,
        'pretrained': True,
        'freeze_at': 1,  # freeze stem layer + layer1 of the backbone
        'pretrained_weight': 'imagenet1k_v1',
        'max_stride': 128,
        'out_stride': [8, 16, 32, 64, 128],
        ## Neck
        'neck': 'basic_fpn',
        'fpn_p6_feat': True,
        'fpn_p7_feat': True,
        'fpn_p6_from_c5': False,
        ## Head
        'head': 'fcos_head',
        'head_dim': 256,
        'num_cls_head': 4,
        'num_reg_head': 4,
        'head_act': 'relu',
        'head_norm': 'GN',
        ## Post-process
        'train_topk': 1000,
        'train_conf_thresh': 0.05,
        'train_nms_thresh': 0.6,
        'test_topk': 100,
        'test_conf_thresh': 0.5,
        'test_nms_thresh': 0.45,
        'nms_class_agnostic': True,  # We prefer to use class-agnostic NMS in the demo.
        # ----------------- Label Assignment -----------------
        'matcher': 'fcos_matcher',
        'matcher_hpy':{'center_sampling_radius': 1.5,
                       'object_sizes_of_interest': [[-1, 64], [64, 128], [128, 256], [256, 512], [512, float('inf')]]
                       },
        # ----------------- Loss weight -----------------
        ## Loss hyper-parameters
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 2.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        'loss_ctn_weight': 1.0,
        # ----------------- Training -----------------
        ## Training scheduler
        'scheduler': '1x',
        ## Optimizer
        'optimizer': 'sgd',
        'base_lr': 0.01 / 16,
        'backbone_lr_ratio': 1.0 / 1.0,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'clip_max_norm': -1.0,
        'param_dict_type': 'default',
        ## LR Scheduler
        'lr_scheduler': 'step',
        'warmup': 'linear',
        'warmup_iters': 500,
        'warmup_factor': 0.00066667,
        ## Epoch
        'max_epoch': 12,      # 1x
        'lr_epoch': [8, 11],  # 1x
        # ----------------- Input -----------------
        ## Transforms
        'train_min_size': [800],   # short edge of image
        'train_max_size': 1333,
        'test_min_size': [800],
        'test_max_size': 1333,
        ## Pixel mean & std
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std':  [0.229, 0.224, 0.225],
        ## Transforms
        'detr_style': False,
        'trans_config': [
            {'name': 'RandomHFlip'},
            {'name': 'RandomResize'},
        ],
        'box_format': 'xyxy',
        'normalize_coords': False,
    },

    'fcos_r50_1x':{
        # ----------------- Model-----------------
        ## Backbone
        'backbone': 'resnet50',
        'backbone_norm': 'FrozeBN',
        'res5_dilation': False,
        'pretrained': True,
        'freeze_at': 1,  # freeze stem layer + layer1 of the backbone
        'pretrained_weight': 'imagenet1k_v1',
        'max_stride': 128,
        'out_stride': [8, 16, 32, 64, 128],
        ## Neck
        'neck': 'basic_fpn',
        'fpn_p6_feat': True,
        'fpn_p7_feat': True,
        'fpn_p6_from_c5': False,
        ## Head
        'head': 'fcos_head',
        'head_dim': 256,
        'num_cls_head': 4,
        'num_reg_head': 4,
        'head_act': 'relu',
        'head_norm': 'GN',
        ## Post-process
        'train_topk': 1000,
        'train_conf_thresh': 0.05,
        'train_nms_thresh': 0.65,
        'test_topk': 100,
        'test_conf_thresh': 0.5,
        'test_nms_thresh': 0.45,
        'nms_class_agnostic': True,  # We prefer to use class-agnostic NMS in the demo.
        # ----------------- Label Assignment -----------------
        'matcher': 'fcos_matcher',
        'matcher_hpy':{'center_sampling_radius': 1.5,
                       'object_sizes_of_interest': [[-1, 64], [64, 128], [128, 256], [256, 512], [512, float('inf')]]
                       },
        # ----------------- Loss weight -----------------
        ## Loss hyper-parameters
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 2.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        'loss_ctn_weight': 1.0,
        # ----------------- Training -----------------
        ## Training scheduler
        'scheduler': '1x',
        ## Optimizer
        'optimizer': 'sgd',
        'base_lr': 0.01 / 16,
        'backbone_lr_ratio': 1.0 / 1.0,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'clip_max_norm': -1.0,
        'param_dict_type': 'default',
        ## LR Scheduler
        'lr_scheduler': 'step',
        'warmup': 'linear',
        'warmup_iters': 500,
        'warmup_factor': 0.00066667,
        ## Epoch
        'max_epoch': 12,      # 1x
        'lr_epoch': [8, 11],  # 1x
        # ----------------- Input -----------------
        ## Transforms
        'train_min_size': [800],   # short edge of image
        'train_max_size': 1333,
        'test_min_size': [800],
        'test_max_size': 1333,
        ## Pixel mean & std
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std':  [0.229, 0.224, 0.225],
        ## Transforms
        'detr_style': False,
        'trans_config': [
            {'name': 'RandomHFlip'},
            {'name': 'RandomResize'},
        ],
        'box_format': 'xyxy',
        'normalize_coords': False,
    },

    'fcos_rt_r18_1x':{
        # ----------------- Model-----------------
        ## Backbone
        'backbone': 'resnet18',
        'backbone_norm': 'FrozeBN',
        'res5_dilation': False,
        'pretrained': True,
        'freeze_at': 1,  # freeze stem layer + layer1 of the backbone
        'pretrained_weight': 'imagenet1k_v1',
        'max_stride': 32,
        'out_stride': [8, 16, 32],
        ## Neck
        'neck': 'basic_fpn',
        'fpn_p6_feat': False,
        'fpn_p7_feat': False,
        'fpn_p6_from_c5': False,
        ## Head
        'head': 'fcos_head',
        'head_dim': 256,
        'num_cls_head': 4,
        'num_reg_head': 4,
        'head_act': 'relu',
        'head_norm': 'GN',
        ## Post-process
        'train_topk': 1000,
        'train_conf_thresh': 0.05,
        'train_nms_thresh': 0.6,
        'test_topk': 100,
        'test_conf_thresh': 0.5,
        'test_nms_thresh': 0.45,
        'nms_class_agnostic': True,  # We prefer to use class-agnostic NMS in the demo.
        # ----------------- Label Assignment -----------------
        'matcher': 'fcos_matcher',
        'matcher_hpy':{'center_sampling_radius': 1.5,
                       'object_sizes_of_interest': [[-1, 64], [64, 128], [128, float('inf')]]
                       },
        # ----------------- Loss weight -----------------
        ## Loss hyper-parameters
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 2.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        'loss_ctn_weight': 1.0,
        # ----------------- Training -----------------
        ## Training scheduler
        'scheduler': '1x',
        ## Optimizer
        'optimizer': 'sgd',
        'base_lr': 0.01 / 16,
        'backbone_lr_ratio': 1.0 / 1.0,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'clip_max_norm': -1.0,
        ## LR Scheduler
        'lr_scheduler': 'step',
        'warmup': 'linear',
        'warmup_iters': 500,
        'warmup_factor': 0.00066667,
        ## Epoch
        'max_epoch': 36,       # 1x
        'lr_epoch': [24, 33],  # 1x
        # ----------------- Input -----------------
        ## Transforms
        'train_min_size': [256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608],   # short edge of image
        'train_max_size': 900,
        'test_min_size': [512],
        'test_max_size': 736,
        ## Pixel mean & std
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std':  [0.229, 0.224, 0.225],
        ## Transforms
        'detr_style': False,
        'trans_config': [
            {'name': 'RandomHFlip'},
            {'name': 'RandomResize'},
        ],
        'box_format': 'xyxy',
        'normalize_coords': False,
    },

    'fcos_rt_r50_1x':{
        # ----------------- Model-----------------
        ## Backbone
        'backbone': 'resnet50',
        'backbone_norm': 'FrozeBN',
        'res5_dilation': False,
        'pretrained': True,
        'freeze_at': 1,  # freeze stem layer + layer1 of the backbone
        'pretrained_weight': 'imagenet1k_v1',
        'max_stride': 32,
        'out_stride': [8, 16, 32],
        ## Neck
        'neck': 'basic_fpn',
        'fpn_p6_feat': False,
        'fpn_p7_feat': False,
        'fpn_p6_from_c5': False,
        ## Head
        'head': 'fcos_head',
        'head_dim': 256,
        'num_cls_head': 4,
        'num_reg_head': 4,
        'head_act': 'relu',
        'head_norm': 'GN',
        ## Post-process
        'train_topk': 1000,
        'train_conf_thresh': 0.05,
        'train_nms_thresh': 0.6,
        'test_topk': 100,
        'test_conf_thresh': 0.5,
        'test_nms_thresh': 0.45,
        'nms_class_agnostic': True,  # We prefer to use class-agnostic NMS in the demo.
        # ----------------- Label Assignment -----------------
        'matcher': 'fcos_matcher',
        'matcher_hpy':{'center_sampling_radius': 1.5,
                       'object_sizes_of_interest': [[-1, 64], [64, 128], [128, float('inf')]]
                       },
        # ----------------- Loss weight -----------------
        ## Loss hyper-parameters
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 2.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        'loss_ctn_weight': 1.0,
        # ----------------- Training -----------------
        ## Training scheduler
        'scheduler': '1x',
        ## Optimizer
        'optimizer': 'sgd',
        'base_lr': 0.01 / 16,
        'backbone_lr_ratio': 1.0 / 1.0,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'clip_max_norm': -1.0,
        ## LR Scheduler
        'lr_scheduler': 'step',
        'warmup': 'linear',
        'warmup_iters': 500,
        'warmup_factor': 0.00066667,
        ## Epoch
        'max_epoch': 36,       # 1x
        'lr_epoch': [24, 33],  # 1x
        # ----------------- Input -----------------
        ## Transforms
        'train_min_size': [256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608],   # short edge of image
        'train_max_size': 900,
        'test_min_size': [512],
        'test_max_size': 736,
        ## Pixel mean & std
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std':  [0.229, 0.224, 0.225],
        ## Transforms
        'detr_style': False,
        'trans_config': [
            {'name': 'RandomHFlip'},
            {'name': 'RandomResize'},
        ],
        'box_format': 'xyxy',
        'normalize_coords': False,
    },

}