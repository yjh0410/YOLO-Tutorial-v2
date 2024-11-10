import os
import cv2
import time
import numpy as np
from pycocotools.coco import COCO

try:
    from .data_augment.strong_augment import MosaicAugment, MixupAugment
    from .coco import COCODataset
except:
    from  data_augment.strong_augment import MosaicAugment, MixupAugment
    from  coco import COCODataset


custom_class_indexs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
custom_class_labels = ('bird', 'butterfly', 'cat', 'cow', 'dog', 'lion', 'person', 'pig', 'tiger', )


class CustomDataset(COCODataset):
    def __init__(self, 
                 cfg,
                 data_dir     :str = None, 
                 transform    = None,
                 is_train     :bool =False,
                 ):
        # ----------- Basic parameters -----------
        self.image_set = "train" if is_train else "val"
        self.is_train  = is_train
        self.num_classes = len(custom_class_labels)
        # ----------- Path parameters -----------
        self.data_dir = data_dir
        self.json_file = '{}.json'.format(self.image_set)
        # ----------- Data parameters -----------
        self.coco = COCO(os.path.join(self.data_dir, self.image_set, 'annotations', self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        self.dataset_size = len(self.ids)
        self.class_labels = custom_class_labels
        self.class_indexs = custom_class_indexs
        # ----------- Transform parameters -----------
        self.transform = transform
        if is_train:
            if cfg.mosaic_prob == 0.:
                self.mosaic_augment = None
            else:
                self.mosaic_augment = MosaicAugment(cfg.train_img_size, cfg.affine_params, is_train)
            self.mosaic_prob = cfg.mosaic_prob
            if cfg.mixup_prob == 0.:
                self.mixup_augment = None
            else:
                self.mixup_augment = MixupAugment(cfg.train_img_size)
            self.mixup_prob  = cfg.mixup_prob
            self.copy_paste  = cfg.copy_paste
        else:
            self.mosaic_prob = 0.0
            self.mixup_prob  = 0.0
            self.copy_paste  = 0.0
            self.mosaic_augment = None
            self.mixup_augment  = None

        print(' ============ Strong augmentation info. ============ ')
        print('use Mosaic Augmentation: {}'.format(self.mosaic_prob))
        print('use Mixup Augmentation: {}'.format(self.mixup_prob))
        print('use Copy-paste Augmentation: {}'.format(self.copy_paste))

    def pull_image(self, index):
        id_ = self.ids[index]
        im_ann = self.coco.loadImgs(id_)[0] 
        img_file = os.path.join(
                self.data_dir, self.image_set, 'images', im_ann["file_name"])
        image = cv2.imread(img_file)

        return image, id_

    def pull_anno(self, index):
        img_id = self.ids[index]
        im_ann = self.coco.loadImgs(img_id)[0]
        anno_ids = self.coco.getAnnIds(imgIds=[int(img_id)], iscrowd=0)
        annotations = self.coco.loadAnns(anno_ids)
        
        # image infor
        width = im_ann['width']
        height = im_ann['height']
        
        #load a target
        bboxes = []
        labels = []
        for anno in annotations:
            if 'bbox' in anno and anno['area'] > 0:
                # bbox
                x1 = np.max((0, anno['bbox'][0]))
                y1 = np.max((0, anno['bbox'][1]))
                x2 = np.min((width - 1, x1 + np.max((0, anno['bbox'][2] - 1))))
                y2 = np.min((height - 1, y1 + np.max((0, anno['bbox'][3] - 1))))
                if x2 <= x1 or y2 <= y1:
                    continue
                # class label
                cls_id = self.class_ids.index(anno['category_id'])
                
                bboxes.append([x1, y1, x2, y2])
                labels.append(cls_id)

        # guard against no boxes via resizing
        bboxes = np.array(bboxes).reshape(-1, 4)
        labels = np.array(labels).reshape(-1)
        
        return bboxes, labels


if __name__ == "__main__":
    import time
    import argparse
    from build import build_transform

    parser = argparse.ArgumentParser(description='RT-ODLab')

    # opt
    parser.add_argument('--root', default='D:/python_work/dataset/AnimalDataset/',
                        help='data root')
    parser.add_argument('--is_train', action="store_true", default=False,
                        help='mixup augmentation.')
    parser.add_argument('--aug_type', default="yolo", type=str, choices=["yolo", "ssd"],
                        help='yolo, ssd.')
    
    args = parser.parse_args()

    class YoloBaseConfig(object):
        def __init__(self) -> None:
            self.max_stride = 32
            # ---------------- Data process config ----------------
            self.box_format = 'xywh'
            self.normalize_coords = False
            self.mosaic_prob = 1.0
            self.mixup_prob  = 0.15
            self.copy_paste  = 0.3
            ## Pixel mean & std
            self.pixel_mean = [0., 0., 0.]
            self.pixel_std  = [255., 255., 255.]
            ## Transforms
            self.train_img_size = 640
            self.test_img_size  = 640
            self.use_ablu = True
            self.aug_type = 'yolo'
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

    class SSDBaseConfig(object):
        def __init__(self) -> None:
            self.max_stride = 32
            # ---------------- Data process config ----------------
            self.box_format = 'xywh'
            self.normalize_coords = False
            self.mosaic_prob = 0.0
            self.mixup_prob  = 0.0
            self.copy_paste  = 0.0
            ## Pixel mean & std
            self.pixel_mean = [0., 0., 0.]
            self.pixel_std  = [255., 255., 255.]
            ## Transforms
            self.train_img_size = 640
            self.test_img_size  = 640
            self.aug_type = 'ssd'

    if args.aug_type == "yolo":
        cfg = YoloBaseConfig()
    elif args.aug_type == "ssd":
        cfg = SSDBaseConfig()

    transform = build_transform(cfg, args.is_train)
    dataset = CustomDataset(cfg, args.root, transform, args.is_train)
    
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(80)]
    print('Data length: ', len(dataset))

    for i in range(1000):
        t0 = time.time()
        image, target, deltas = dataset.pull_item(i)
        print("Load data: {} s".format(time.time() - t0))

        # to numpy
        image = image.permute(1, 2, 0).numpy()
        
        # denormalize
        image = image * cfg.pixel_std + cfg.pixel_mean

        # rgb -> bgr
        if transform.color_format == 'rgb':
            image = image[..., (2, 1, 0)]

        # to uint8
        image = image.astype(np.uint8)
        image = image.copy()
        img_h, img_w = image.shape[:2]

        boxes = target["boxes"]
        labels = target["labels"]

        for box, label in zip(boxes, labels):
            if cfg.box_format == 'xyxy':
                x1, y1, x2, y2 = box
            elif cfg.box_format == 'xywh':
                cx, cy, bw, bh = box
                x1 = cx - 0.5 * bw
                y1 = cy - 0.5 * bh
                x2 = cx + 0.5 * bw
                y2 = cy + 0.5 * bh
            
            if cfg.normalize_coords:
                x1 *= img_w
                y1 *= img_h
                x2 *= img_w
                y2 *= img_h

            cls_id = int(label)
            color = class_colors[cls_id]
            # class name
            label = custom_class_labels[cls_id]
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            # put the test on the bbox
            cv2.putText(image, label, (int(x1), int(y1 - 5)), 0, 0.5, color, 1, lineType=cv2.LINE_AA)
        cv2.imshow('gt', image)
        # cv2.imwrite(str(i)+'.jpg', img)
        cv2.waitKey(0)
