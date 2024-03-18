import os

try:
    # dataset class
    from .voc        import VOCDataset
    from .coco       import COCODataset
    from .customed   import CustomedDataset
    # transform class
    from .data_augment.yolo_augment import YOLOAugmentation, YOLOBaseTransform
    from .data_augment.ssd_augment  import SSDAugmentation, SSDBaseTransform

except:
    # dataset class
    from voc        import VOCDataset
    from coco       import COCODataset
    from customed   import CustomedDataset
    # transform class
    from data_augment.yolo_augment import YOLOAugmentation, YOLOBaseTransform
    from data_augment.ssd_augment  import SSDAugmentation, SSDBaseTransform


# ------------------------------ Dataset ------------------------------
def build_dataset(args, cfg, transform=None, is_train=False):
    # ------------------------- Build dataset -------------------------
    ## VOC dataset
    if args.dataset == 'voc':
        image_set = [('2007', 'trainval'), ('2012', 'trainval')] if is_train else [('2007', 'test')]
        cfg.num_classes  = 20
        dataset = VOCDataset(cfg       = cfg,
                             data_dir  = args.root,
                             image_set = image_set,
                             transform = transform,
                             is_train  = is_train,
                             )
    ## COCO dataset
    elif args.dataset == 'coco':
        image_set = 'train2017' if is_train else 'val2017'
        cfg.num_classes  = 80
        dataset = COCODataset(cfg       = cfg,
                              data_dir  = args.root,
                              image_set = image_set,
                              transform = transform,
                              is_train  = is_train,
                              )
    ## Custom dataset
    elif args.dataset == 'customed':
        image_set = 'train' if is_train else 'val'
        cfg.num_classes  = 20
        dataset = CustomedDataset(cfg       = cfg,
                                  data_dir  = args.root,
                                  image_set = image_set,
                                  transform = transform,
                                  is_train  = is_train,
                                  )

    cfg.class_labels = dataset.class_labels
    cfg.class_indexs = dataset.class_indexs
    cfg.num_classes  = dataset.num_classes

    return dataset


# ------------------------------ Transform ------------------------------
def build_transform(cfg, is_train=False):
    # ---------------- Build transform ----------------
    ## YOLO style transform
    if cfg.aug_type == 'yolo':
        if is_train:
            transform = YOLOAugmentation(cfg.train_img_size,
                                         cfg.affine_params,
                                         cfg.use_ablu,
                                         cfg.pixel_mean,
                                         cfg.pixel_std,
                                         cfg.box_format,
                                         cfg.normalize_coords)
        else:
            transform = YOLOBaseTransform(cfg.test_img_size,
                                          cfg.max_stride,
                                          cfg.pixel_mean,
                                          cfg.pixel_std,
                                          cfg.box_format,
                                          cfg.normalize_coords)

    ## RT-DETR style transform
    elif cfg.aug_type == 'ssd':
        if is_train:
            transform = SSDAugmentation(cfg.train_img_size,
                                           cfg.pixel_mean,
                                           cfg.pixel_std,
                                           cfg.box_format,
                                           cfg.normalize_coords)
        else:
            transform = SSDBaseTransform(cfg.test_img_size,
                                            cfg.pixel_mean,
                                            cfg.pixel_std,
                                            cfg.box_format,
                                            cfg.normalize_coords)

    return transform
