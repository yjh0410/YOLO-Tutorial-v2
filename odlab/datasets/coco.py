# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision

try:
    from .transforms import build_transform
except:
    from transforms import build_transform


coco_labels_91 = ('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
coco_labels_80 = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',  'traffic light',  'fire hydrant',  'stop sign',  'parking meter',  'bench',  'bird',  'cat',  'dog',  'horse',  'sheep',  'cow',  'elephant',  'bear',  'zebra',  'giraffe',  'backpack',  'umbrella',  'handbag',  'tie',  'suitcase',  'frisbee',  'skis',  'snowboard',  'sports ball',  'kite',  'baseball bat',  'baseball glove',  'skateboard',  'surfboard',  'tennis racket',  'bottle',  'wine glass',  'cup',  'fork',  'knife',  'spoon',  'bowl',  'banana',  'apple',  'sandwich',  'orange',  'broccoli',  'carrot',  'hot dog',  'pizza',  'donut',  'cake',  'chair',  'couch',  'potted plant',  'bed',  'dining table',  'toilet',  'tv',  'laptop',  'mouse',  'remote',  'keyboard',  'cell phone',  'microwave',  'oven',  'toaster',  'sink',  'refrigerator',  'book',  'clock',  'vase',  'scissors',  'teddy bear',  'hair drier',  'toothbrush')
coco_indexs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.coco_labels = coco_labels_80  # 80 coco labels for detection task
        self.coco_indexs = coco_indexs     # all original coco label index
        self._transforms = transforms

    def prepare(self, image, target):
        w, h = image.size
        # load an image
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        # load an annotation
        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        # bbox target
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # class target
        classes = [self.coco_indexs.index(obj["category_id"]) for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        # filter invalid bbox
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target


def build_coco(args, transform=None, is_train=False):
    root = Path(args.root)
    assert root.exists(), f'provided COCO path {root} does not exist'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / 'instances_train2017.json'),
        "val":   (root / "val2017",   root / "annotations" / 'instances_val2017.json'),
    }

    image_set = "train" if is_train else "val"
    img_folder, ann_file = PATHS[image_set]

    # build transform
    dataset = CocoDetection(img_folder, ann_file, transform)

    return dataset


if __name__ == "__main__":
    import argparse
    import cv2
    import numpy as np
    
    parser = argparse.ArgumentParser(description='COCO-Dataset')

    # opt
    parser.add_argument('--root', default='D:/python_work/dataset/COCO/',
                        help='data root')
    parser.add_argument('--is_train', action="store_true", default=False,
                        help='mixup augmentation.')    
    args = parser.parse_args()

    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(80)]

    # config
    class BaseConfig(object):
        def __init__(self):
            # --------- Data process ---------
            ## input size
            self.train_min_size = [512]   # short edge of image
            self.train_max_size = 736
            self.test_min_size  = [512]
            self.test_max_size  = 736
            ## Pixel mean & std
            self.pixel_mean = [0.485, 0.456, 0.406]
            self.pixel_std  = [0.229, 0.224, 0.225]
            ## Transforms
            self.box_format = 'xyxy'
            self.normalize_coords = False
            self.detr_style = False
            self.trans_config = [
                {'name': 'RandomHFlip'},
                {'name': 'RandomResize'},
                {'name': 'RandomShift', 'max_shift': 32},
            ]
    
    cfg = BaseConfig()
    # build dataset
    transform = build_transform(cfg, is_train=True)
    dataset   = build_coco(args, transform, is_train=False)

    for index, (image, target) in enumerate(dataset):
        print("{} / {}".format(index, len(dataset)))
        # to numpy
        image = image.permute(1, 2, 0).numpy()
        # denormalize
        image = (image * cfg.pixel_std + cfg.pixel_mean) * 255
        image = image.astype(np.uint8)[..., (2, 1, 0)].copy()
        orig_h, orig_w = image.shape[:2]

        tgt_bboxes = target["boxes"]
        tgt_labels = target["labels"]
        for box, label in zip(tgt_bboxes, tgt_labels):
            if cfg.normalize_coords:
                box[..., [0, 2]] *= orig_w
                box[..., [1, 3]] *= orig_h
            if cfg.box_format == 'xywh':
                box_x1y1 = box[..., :2] - box[..., 2:] * 0.5
                box_x2y2 = box[..., :2] + box[..., 2:] * 0.5
                box = torch.cat([box_x1y1, box_x2y2], dim=-1)
            # get box target
            x1, y1, x2, y2 = box.long()
            # get class label
            cls_name = coco_labels_80[label.item()]
            color = class_colors[label.item()]
            # draw bbox
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            # put the test on the bbox
            cv2.putText(image, cls_name, (int(x1), int(y1 - 5)), 0, 0.5, color, 1, lineType=cv2.LINE_AA)

        cv2.imshow("data", image)
        cv2.waitKey(0)

