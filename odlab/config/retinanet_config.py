# RetinaNet


retinanet_cfg = {
    'retinanet_r18_1x':{
        # ----------------- Model-----------------
        ## Backbone
        'backbone': 'resnet18',
        'backbone_norm': 'FrozeBN',
        'res5_dilation': False,
        'pretrained': True,
        'pretrained_weight': 'imagenet1k_v1',
        'freeze_at': 1,  # freeze stem layer + layer1 of the backbone        
        'max_stride': 128,
        'out_stride': [8, 16, 32, 64, 128],
        ## Neck
        'neck': 'basic_fpn',
        'fpn_p6_feat': True,
        'fpn_p7_feat': True,
        'fpn_p6_from_c5': True,
        ## Head
        'head': 'retinanet_head',
        'head_dim': 256,
        'num_cls_head': 4,
        'num_reg_head': 4,
        'head_act': 'relu',
        'head_norm': None,
        'anchor_config': {'basic_size': [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
                          'aspect_ratio': [0.5, 1.0, 2.0],
                          'area_scale': [2 ** 0, 2 ** (1. / 3.), 2 ** (2. / 3.)]},
        ## Post-process
        'train_topk': 1000,
        'train_conf_thresh': 0.05,
        'train_nms_thresh': 0.6,
        'test_topk': 100,
        'test_conf_thresh': 0.3,
        'test_nms_thresh': 0.45,
        'nms_class_agnostic': True,  # We prefer to use class-agnostic NMS in the demo.
        # ----------------- Label Assignment -----------------
        'matcher': 'retinanet_matcher',
        'matcher_hpy': {'iou_thresh': [0.4, 0.5],
                        'iou_labels': [0, -1, 1], # [negative sample, ignored sample, positive sample]
                        'allow_low_quality_matches': True,
                        },
        # ----------------- Loss weight -----------------
        ## Loss hyper-parameters
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 2.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        'use_giou_loss': False,
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

    'retinanet_r50_1x':{
        # ----------------- Model-----------------
        ## Backbone
        'backbone': 'resnet50',
        'backbone_norm': 'FrozeBN',
        'res5_dilation': False,
        'pretrained': True,
        'pretrained_weight': 'imagenet1k_v1',
        'freeze_at': 1,  # freeze stem layer + layer1 of the backbone        
        'max_stride': 128,
        'out_stride': [8, 16, 32, 64, 128],
        ## Neck
        'neck': 'basic_fpn',
        'fpn_p6_feat': True,
        'fpn_p7_feat': True,
        'fpn_p6_from_c5': True,
        ## Head
        'head': 'retinanet_head',
        'head_dim': 256,
        'num_cls_head': 4,
        'num_reg_head': 4,
        'head_act': 'relu',
        'head_norm': None,
        'anchor_config': {'basic_size': [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
                          'aspect_ratio': [0.5, 1.0, 2.0],
                          'area_scale': [2 ** 0, 2 ** (1. / 3.), 2 ** (2. / 3.)]},
        ## Post-process
        'train_topk': 1000,
        'train_conf_thresh': 0.05,
        'train_nms_thresh': 0.6,
        'test_topk': 100,
        'test_conf_thresh': 0.3,
        'test_nms_thresh': 0.45,
        'nms_class_agnostic': True,  # We prefer to use class-agnostic NMS in the demo.
        # ----------------- Label Assignment -----------------
        'matcher': 'retinanet_matcher',
        'matcher_hpy': {'iou_thresh': [0.4, 0.5],
                        'iou_labels': [0, -1, 1], # [negative sample, ignored sample, positive sample]
                        'allow_low_quality_matches': True,
                        },
        # ----------------- Loss weight -----------------
        ## Loss hyper-parameters
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 2.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        'use_giou_loss': False,
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

}