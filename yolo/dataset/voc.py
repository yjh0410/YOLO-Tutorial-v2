import os
import cv2
import time
import random
import numpy as np
import torch
from pycocotools.coco import COCO

try:
    from .data_augment.strong_augment import MosaicAugment, MixupAugment
except:
    from  data_augment.strong_augment import MosaicAugment, MixupAugment


voc_class_indexs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
voc_class_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 cfg,
                 data_dir  :str = None, 
                 transform = None,
                 is_train  :bool = False,
                 ):
        # ----------- Basic parameters -----------
        self.data_dir  = data_dir
        self.image_set = "train" if is_train else "val"
        self.is_train  = is_train
        self.num_classes = 20
        # ----------- Data parameters -----------
        self.json_file = "instances_{}.json".format(self.image_set)
        self.coco = COCO(os.path.join(self.data_dir, 'annotations', self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        self.dataset_size = len(self.ids)
        self.class_labels = voc_class_labels
        self.class_indexs = voc_class_indexs
        # ----------- Transform parameters -----------
        self.transform = transform
        if is_train:
            self.mosaic_augment = MosaicAugment(cfg.train_img_size, cfg.affine_params, is_train)
            self.mixup_augment = MixupAugment(cfg.train_img_size)
            self.mosaic_prob = cfg.mosaic_prob
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

    # ------------ Basic dataset function ------------
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.pull_item(index)

    # ------------ Mosaic & Mixup ------------
    def load_mosaic(self, index):
        # ------------ Prepare 4 indexes of images ------------
        ## Load 4x mosaic image
        index_list = np.arange(index).tolist() + np.arange(index+1, len(self.ids)).tolist()
        id1 = index
        id2, id3, id4 = random.sample(index_list, 3)
        indexs = [id1, id2, id3, id4]

        ## Load images and targets
        image_list = []
        target_list = []
        for index in indexs:
            img_i, target_i = self.load_image_target(index)
            image_list.append(img_i)
            target_list.append(target_i)

        # ------------ Mosaic augmentation ------------
        image, target = self.mosaic_augment(image_list, target_list)

        return image, target

    def load_mixup(self, origin_image, origin_target, yolox_style=False):
        # ------------ Load a new image & target ------------
        if yolox_style:
            new_index = np.random.randint(0, len(self.ids))
            new_image, new_target = self.load_image_target(new_index)
        else:
            new_index = np.random.randint(0, len(self.ids))
            new_image, new_target = self.load_mosaic(new_index)
            
        # ------------ Mixup augmentation ------------
        image, target = self.mixup_augment(origin_image, origin_target, new_image, new_target, yolox_style)

        return image, target
    
    # ------------ Load data function ------------
    def load_image_target(self, index):
        # load an image
        image, _ = self.pull_image(index)
        height, width, channels = image.shape

        # load a target
        bboxes, labels = self.pull_anno(index)
        target = {
            "boxes": bboxes,
            "labels": labels,
            "orig_size": [height, width]
        }

        return image, target

    def pull_item(self, index):
        if random.random() < self.mosaic_prob:
            # load a mosaic image
            mosaic = True
            image, target = self.load_mosaic(index)
        else:
            mosaic = False
            # load an image and target
            image, target = self.load_image_target(index)

        # Yolov5-MixUp
        mixup = False
        if random.random() < self.mixup_prob:
            mixup = True
            image, target = self.load_mixup(image, target)

        # Copy-paste (use Yolox-Mixup to approximate copy-paste)
        if not mixup and random.random() < self.copy_paste:
            image, target = self.load_mixup(image, target, yolox_style=True)

        # augment
        image, target, deltas = self.transform(image, target, mosaic)

        return image, target, deltas

    def pull_image(self, index):
        # get the image file name
        image_dict = self.coco.dataset['images'][index]
        image_id = image_dict["id"]
        filename = image_dict["file_name"]

        # load the image
        image_path = os.path.join(self.data_dir, "images", filename)
        image = cv2.imread(image_path)

        assert image is not None

        return image, image_id

    def pull_anno(self, index):
        img_id = self.ids[index]        
        # image infor
        im_ann = self.coco.loadImgs(img_id)[0]
        width = im_ann['width']
        height = im_ann['height']

        # annotation infor
        anno_ids = self.coco.getAnnIds(imgIds=[int(img_id)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)

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
                if x2 < x1 or y2 < y1:
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
    
    parser = argparse.ArgumentParser(description='COCO-Dataset')

    # opt
    parser.add_argument('--root', default="D:/python_work/dataset/VOC0712/",
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
            self.mixup_prob  = 0.0
            self.copy_paste  = 1.0
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
            self.affine_params = None

    if args.aug_type == "yolo":
        cfg = YoloBaseConfig()
    elif args.aug_type == "ssd":
        cfg = SSDBaseConfig()

    transform = build_transform(cfg, args.is_train)
    dataset = VOCDataset(cfg, args.root, transform, args.is_train)
    
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
            x1, y1, x2, y2 = box
            
            cls_id = int(label)
            color = class_colors[cls_id]
            # class name
            label = voc_class_labels[cls_id]
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            # put the test on the bbox
            cv2.putText(image, label, (int(x1), int(y1 - 5)), 0, 0.5, color, 1, lineType=cv2.LINE_AA)
        cv2.imshow('gt', image)
        # cv2.imwrite(str(i)+'.jpg', img)
        cv2.waitKey(0)