# Fully Convolutional One-Stage object detector


yolof_cfg = {
    # --------------- C5 level ---------------
    'yolof_r18_c5_1x':{
        # ----------------- Model-----------------
        ## Backbone
        'backbone': 'resnet18',
        'backbone_norm': 'FrozeBN',
        'res5_dilation': False,
        'pretrained': True,
        'pretrained_weight': 'imagenet1k_v1',
        'freeze_at': 1,  # freeze stem layer + layer1 of the backbone
        'max_stride': 32,
        'out_stride': 32,
        ## Neck
        'neck': 'dilated_encoder',
        'neck_dilations': [2, 4, 6, 8],
        'neck_expand_ratio': 0.25,
        'neck_act': 'relu',
        'neck_norm': 'BN',
        ## Head
        'head': 'yolof_head',
        'head_dim': 512,
        'num_cls_head': 2,
        'num_reg_head': 4,
        'head_act': 'relu',
        'head_norm': 'BN',
        'center_clamp': 32,         
        'anchor_size': [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        ## Post-process
        'train_topk': 1000,
        'train_conf_thresh': 0.05,
        'train_nms_thresh': 0.6,
        'test_topk': 300,
        'test_conf_thresh': 0.3,
        'test_nms_thresh': 0.45,
        'nms_class_agnostic': True,  # We prefer to use class-agnostic NMS in the demo.
        # ----------------- Label Assignment -----------------
        'matcher': 'yolof_matcher',
        'matcher_hpy': {'topk_candidates': 4,
                        'iou_thresh': 0.15,
                        'ignore_thresh': 0.7,
                        },
        # ----------------- Loss weight -----------------
        ## Loss hyper-parameters
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 2.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        # ----------------- Training -----------------
        ## Training scheduler
        'scheduler': '1x',
        ## Optimizer
        'optimizer': 'sgd',
        'base_lr': 0.12 / 64,
        'backbone_lr_ratio': 1.0 / 3.0,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'clip_max_norm': 10.0,
        'param_dict_type': 'default',
        ## LR Scheduler
        'lr_scheduler': 'step',
        'warmup': 'linear',
        'warmup_iters': 1500,
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
            {'name': 'RandomShift', 'max_shift': 32},
        ],
        'box_format': 'xyxy',
        'normalize_coords': False,
    },

    'yolof_r50_c5_1x':{
        # ----------------- Model-----------------
        ## Backbone
        'backbone': 'resnet50',
        'backbone_norm': 'FrozeBN',
        'res5_dilation': False,
        'pretrained': True,
        'pretrained_weight': 'imagenet1k_v1',
        'freeze_at': 1,  # freeze stem layer + layer1 of the backbone
        'max_stride': 32,
        'out_stride': 32,
        ## Neck
        'neck': 'dilated_encoder',
        'neck_dilations': [2, 4, 6, 8],
        'neck_expand_ratio': 0.25,
        'neck_act': 'relu',
        'neck_norm': 'BN',
        ## Head
        'head': 'yolof_head',
        'head_dim': 512,
        'num_cls_head': 2,
        'num_reg_head': 4,
        'head_act': 'relu',
        'head_norm': 'BN',
        'center_clamp': 32,         
        'anchor_size': [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        ## Post-process
        'train_topk': 1000,
        'train_conf_thresh': 0.05,
        'train_nms_thresh': 0.6,
        'test_topk': 300,
        'test_conf_thresh': 0.3,
        'test_nms_thresh': 0.45,
        'nms_class_agnostic': True,  # We prefer to use class-agnostic NMS in the demo.
        # ----------------- Label Assignment -----------------
        'matcher': 'yolof_matcher',
        'matcher_hpy': {'topk_candidates': 4,
                        'iou_thresh': 0.15,
                        'ignore_thresh': 0.7,
                        },
        # ----------------- Loss weight -----------------
        ## Loss hyper-parameters
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 2.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        # ----------------- Training -----------------
        ## Training scheduler
        'scheduler': '1x',
        ## Optimizer
        'optimizer': 'sgd',
        'base_lr': 0.12 / 64,
        'backbone_lr_ratio': 1.0 / 3.0,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'clip_max_norm': 10.0,
        'param_dict_type': 'default',
        ## LR Scheduler
        'lr_scheduler': 'step',
        'warmup': 'linear',
        'warmup_iters': 1500,
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
            {'name': 'RandomShift', 'max_shift': 32},
        ],
        'box_format': 'xyxy',
        'normalize_coords': False,
    },

    # --------------- Dilated C5 level ---------------
    'yolof_r50_dc5_1x':{
        # ----------------- Model-----------------
        ## Backbone
        'backbone': 'resnet50',
        'backbone_norm': 'FrozeBN',
        'res5_dilation': True,
        'pretrained': True,
        'pretrained_weight': 'imagenet1k_v1',
        'freeze_at': 1,  # freeze stem layer + layer1 of the backbone
        'max_stride': 16,
        'out_stride': 16,
        ## Neck
        'neck': 'dilated_encoder',
        'neck_dilations': [4, 8, 12, 16],
        'neck_expand_ratio': 0.25,
        'neck_act': 'relu',
        'neck_norm': 'BN',
        ## Head
        'head': 'yolof_head',
        'head_dim': 512,
        'num_cls_head': 2,
        'num_reg_head': 4,
        'head_act': 'relu',
        'head_norm': 'BN',
        'center_clamp': 32,         
        'anchor_size': [[16, 16], [32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        ## Post-process
        'train_topk': 1000,
        'train_conf_thresh': 0.05,
        'train_nms_thresh': 0.6,
        'test_topk': 300,
        'test_conf_thresh': 0.3,
        'test_nms_thresh': 0.45,
        'nms_class_agnostic': True,  # We prefer to use class-agnostic NMS in the demo.
        # ----------------- Label Assignment -----------------
        'matcher': 'yolof_matcher',
        'matcher_hpy': {'topk_candidates': 8,
                        'iou_thresh': 0.1,
                        'ignore_thresh': 0.7,
                        },
        # ----------------- Loss weight -----------------
        ## Loss hyper-parameters
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 2.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        # ----------------- Training -----------------
        ## Training scheduler
        'scheduler': '1x',
        ## Optimizer
        'optimizer': 'sgd',
        'base_lr': 0.12 / 64,
        'backbone_lr_ratio': 1.0 / 3.0,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'clip_max_norm': 10.0,
        'param_dict_type': 'default',
        ## LR Scheduler
        'lr_scheduler': 'step',
        'warmup': 'linear',
        'warmup_iters': 1500,
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
            {'name': 'RandomShift', 'max_shift': 32},
        ],
        'box_format': 'xyxy',
        'normalize_coords': False,
    },

}