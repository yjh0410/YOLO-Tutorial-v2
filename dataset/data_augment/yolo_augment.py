import random
import cv2
import math
import numpy as np
import albumentations as albu

import torch
import torchvision.transforms.functional as F


# ------------------------- Basic augmentations -------------------------
## Spatial transform
def random_perspective(image,
                       targets=(),
                       degrees=10,
                       translate=.1,
                       scale=[0.1, 2.0],
                       shear=10,
                       perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = image.shape[0] + border[0] * 2  # shape(h,w,c)
    width = image.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -image.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -image.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(scale[0], scale[1])
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            image = cv2.warpPerspective(image, M, dsize=(width, height), borderValue=(0, 0, 0))
        else:  # affine
            image = cv2.warpAffine(image, M[:2], dsize=(width, height), borderValue=(0, 0, 0))

    # Transform label coordinates
    n = len(targets)
    if n:
        new = np.zeros((n, 4))
        # warp boxes
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        targets[:, 1:5] = new

    return image, targets

## Color transform
def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    return img

## Ablu transform
class Albumentations(object):
    def __init__(self, img_size=640):
        self.img_size = img_size
        self.transform = albu.Compose(
            [albu.Blur(p=0.01),
             albu.MedianBlur(p=0.01),
             albu.ToGray(p=0.01),
             albu.CLAHE(p=0.01),
             ],
             bbox_params=albu.BboxParams(format='pascal_voc', label_fields=['labels'])
        )

    def __call__(self, image, target=None):
        labels = target['labels']
        bboxes = target['boxes']
        if len(labels) > 0:
            new = self.transform(image=image, bboxes=bboxes, labels=labels)
            if len(new["labels"]) > 0:
                image = new['image']
                target['labels'] = np.array(new["labels"], dtype=labels.dtype)
                target['boxes'] = np.array(new["bboxes"], dtype=bboxes.dtype)

        return image, target


# ------------------------- Preprocessers -------------------------
## YOLO-style Transform for Train
class YOLOAugmentation(object):
    def __init__(self,
                 img_size=640,
                 affine_params=None,
                 use_ablu=False,
                 pixel_mean = [0., 0., 0.],
                 pixel_std  = [255., 255., 255.],
                 box_format='xyxy',
                 normalize_coords=False):
        # Basic parameters
        self.img_size   = img_size
        self.pixel_mean = pixel_mean
        self.pixel_std  = pixel_std
        self.box_format = box_format
        self.affine_params = affine_params
        self.normalize_coords = normalize_coords
        self.color_format = 'bgr'
        # Albumentations
        self.ablu_trans = Albumentations(img_size) if use_ablu else None

    def __call__(self, image, target, mosaic=False):
        # --------------- Resize image ---------------
        orig_h, orig_w = image.shape[:2]
        ratio = self.img_size / max(orig_h, orig_w)
        if ratio != 1: 
            new_shape = (int(round(orig_w * ratio)), int(round(orig_h * ratio)))
            image = cv2.resize(image, new_shape)
        img_h, img_w = image.shape[:2]

        # --------------- Filter bad targets ---------------
        tgt_boxes_wh = target["boxes"][..., 2:] - target["boxes"][..., :2]
        min_tgt_size = np.min(tgt_boxes_wh, axis=-1)
        keep = (min_tgt_size > 1)
        target["boxes"]  = target["boxes"][keep]
        target["labels"] = target["labels"][keep]

        # --------------- Albumentations ---------------
        if self.ablu_trans is not None:
            image, target = self.ablu_trans(image, target)

        # --------------- HSV augmentations ---------------
        image = augment_hsv(image,
                            hgain=self.affine_params['hsv_h'], 
                            sgain=self.affine_params['hsv_s'], 
                            vgain=self.affine_params['hsv_v'])
        
        # --------------- Spatial augmentations ---------------
        ## Random perspective
        if not mosaic:
            # rescale bbox
            target["boxes"][..., [0, 2]] = target["boxes"][..., [0, 2]] / orig_w * img_w
            target["boxes"][..., [1, 3]] = target["boxes"][..., [1, 3]] / orig_h * img_h

            # spatial augment
            target_ = np.concatenate((target['labels'][..., None], target['boxes']), axis=-1)
            image, target_ = random_perspective(image, target_,
                                                degrees     = self.affine_params['degrees'],
                                                translate   = self.affine_params['translate'],
                                                scale       = self.affine_params['scale'],
                                                shear       = self.affine_params['shear'],
                                                perspective = self.affine_params['perspective']
                                                )
            target['boxes']  = target_[..., 1:]
            target['labels'] = target_[..., 0]

        ## Random flip
        if random.random() < 0.5:
            w = image.shape[1]
            image = np.fliplr(image).copy()
            boxes = target['boxes'].copy()
            boxes[..., [0, 2]] = w - boxes[..., [2, 0]]
            target["boxes"] = boxes

        # --------------- To torch.Tensor ---------------
        image = F.to_tensor(image) * 255.
        image = F.normalize(image, self.pixel_mean, self.pixel_std)
        if target is not None:
            target["boxes"] = torch.as_tensor(target["boxes"]).float()
            target["labels"] = torch.as_tensor(target["labels"]).long()

            # normalize coords
            if self.normalize_coords:
                target["boxes"][..., [0, 2]] /= img_w
                target["boxes"][..., [1, 3]] /= img_h

            # xyxy -> xywh
            if self.box_format == "xywh":
                box_cxcy = (target["boxes"][..., :2] + target["boxes"][..., 2:]) * 0.5
                box_bwbh =  target["boxes"][..., 2:] - target["boxes"][..., :2]
                target["boxes"] = torch.cat([box_cxcy, box_bwbh], dim=-1)


        # --------------- Pad Image ---------------
        img_h0, img_w0 = image.shape[1:]
        pad_image = torch.zeros([image.size(0), self.img_size, self.img_size]).float()
        pad_image[:, :img_h0, :img_w0] = image

        return pad_image, target, ratio

## YOLO-style Transform for Eval
class YOLOBaseTransform(object):
    def __init__(self,
                 img_size=640,
                 max_stride=32,
                 pixel_mean = [0., 0., 0.],
                 pixel_std  = [255., 255., 255.],
                 box_format='xyxy',
                 normalize_coords=False):
        self.img_size = img_size
        self.max_stride = max_stride
        self.pixel_mean = pixel_mean
        self.pixel_std  = pixel_std
        self.box_format = box_format
        self.normalize_coords = normalize_coords
        self.color_format = 'bgr'

    def __call__(self, image, target=None, mosaic=False):
        # --------------- Resize image ---------------
        orig_h, orig_w = image.shape[:2]
        ratio = self.img_size / max(orig_h, orig_w)
        if ratio != 1: 
            new_shape = (int(round(orig_w * ratio)), int(round(orig_h * ratio)))
            image = cv2.resize(image, new_shape)
        img_h, img_w = image.shape[:2]

        # --------------- Rescale bboxes ---------------
        if target is not None:
            # rescale bbox
            target["boxes"][..., [0, 2]] = target["boxes"][..., [0, 2]] / orig_w * img_w
            target["boxes"][..., [1, 3]] = target["boxes"][..., [1, 3]] / orig_h * img_h

        # --------------- To torch.Tensor ---------------
        image = F.to_tensor(image) * 255.
        image = F.normalize(image, self.pixel_mean, self.pixel_std)
        if target is not None:
            target["boxes"] = torch.as_tensor(target["boxes"]).float()
            target["labels"] = torch.as_tensor(target["labels"]).long()

            # normalize coords
            if self.normalize_coords:
                target["boxes"][..., [0, 2]] /= img_w
                target["boxes"][..., [1, 3]] /= img_h
            
            # xyxy -> xywh
            if self.box_format == "xywh":
                box_cxcy = (target["boxes"][..., :2] + target["boxes"][..., 2:]) * 0.5
                box_bwbh =  target["boxes"][..., 2:] - target["boxes"][..., :2]
                target["boxes"] = torch.cat([box_cxcy, box_bwbh], dim=-1)

        # --------------- Pad image ---------------
        img_h0, img_w0 = image.shape[1:]
        dh = img_h0 % self.max_stride
        dw = img_w0 % self.max_stride
        dh = dh if dh == 0 else self.max_stride - dh
        dw = dw if dw == 0 else self.max_stride - dw
        
        pad_img_h = img_h0 + dh
        pad_img_w = img_w0 + dw
        pad_image = torch.zeros([image.size(0), pad_img_h, pad_img_w]).float()
        pad_image[:, :img_h0, :img_w0] = image

        return pad_image, target, ratio
