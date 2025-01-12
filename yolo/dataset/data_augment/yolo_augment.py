import random
import cv2
import math
import numpy as np

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
            image = cv2.warpPerspective(image, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            image = cv2.warpAffine(image, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

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


# ------------------------- Preprocessers -------------------------
## YOLO-style Transform for Train
class YOLOAugmentation(object):
    def __init__(self,
                 img_size=640,
                 affine_params=None,
                 pixel_mean = [0., 0., 0.],
                 pixel_std  = [255., 255., 255.],
                 ):
        # Basic parameters
        self.img_size   = img_size
        self.pixel_mean = pixel_mean
        self.pixel_std  = pixel_std
        self.affine_params = affine_params
        self.color_format = 'bgr'

    def __call__(self, image, target, mosaic=False):
        # --------------- Resize image ---------------
        orig_h, orig_w = image.shape[:2]
        ratio = self.img_size / max(orig_h, orig_w)
        if ratio != 1: 
            new_shape = (int(round(orig_w * ratio)), int(round(orig_h * ratio)))
            image = cv2.resize(image, new_shape)
        img_h, img_w = image.shape[:2]

        # rescale bbox
        target["boxes"][..., [0, 2]] = target["boxes"][..., [0, 2]] / orig_w * img_w
        target["boxes"][..., [1, 3]] = target["boxes"][..., [1, 3]] / orig_h * img_h

        # --------------- HSV augmentations ---------------
        image = augment_hsv(image,
                            hgain=self.affine_params['hsv_h'], 
                            sgain=self.affine_params['hsv_s'], 
                            vgain=self.affine_params['hsv_v'])
        
        # --------------- Spatial augmentations ---------------
        ## Random perspective
        if not mosaic:
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
        image = torch.as_tensor(image).permute(2, 0, 1).contiguous()
        if target is not None:
            target["boxes"] = torch.as_tensor(target["boxes"]).float()
            target["labels"] = torch.as_tensor(target["labels"]).long()

        # --------------- Pad Image ---------------
        img_h0, img_w0 = image.shape[1:]
        pad_image = torch.ones([image.size(0), self.img_size, self.img_size]).float() * 114.
        pad_image[:, :img_h0, :img_w0] = image

        # --------------- Normalize ---------------
        pad_image = F.normalize(pad_image, self.pixel_mean, self.pixel_std)

        return pad_image, target, ratio

## YOLO-style Transform for Eval
class YOLOBaseTransform(object):
    def __init__(self,
                 img_size=640,
                 max_stride=32,
                 pixel_mean = [0., 0., 0.],
                 pixel_std  = [255., 255., 255.],
                 ):
        self.img_size = img_size
        self.max_stride = max_stride
        self.pixel_mean = pixel_mean
        self.pixel_std  = pixel_std
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
        image = torch.as_tensor(image).permute(2, 0, 1).contiguous()
        if target is not None:
            target["boxes"] = torch.as_tensor(target["boxes"]).float()
            target["labels"] = torch.as_tensor(target["labels"]).long()

        # --------------- Pad image ---------------
        img_h0, img_w0 = image.shape[1:]
        dh = img_h0 % self.max_stride
        dw = img_w0 % self.max_stride
        dh = dh if dh == 0 else self.max_stride - dh
        dw = dw if dw == 0 else self.max_stride - dw
        
        pad_img_h = img_h0 + dh
        pad_img_w = img_w0 + dw
        pad_image = torch.ones([image.size(0), pad_img_h, pad_img_w]).float() * 114.
        pad_image[:, :img_h0, :img_w0] = image

        # --------------- Normalize ---------------
        pad_image = F.normalize(pad_image, self.pixel_mean, self.pixel_std)

        return pad_image, target, ratio


if __name__ == "__main__":
    image_path = "voc_image.jpg"
    is_train = True

    affine_params = {
        'degrees': 0.0,
        'translate': 0.2,
        'scale': [0.1, 2.0],
        'shear': 0.0,
        'perspective': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
    }


    if is_train:
        ssd_augment = YOLOAugmentation(img_size=416,
                                       affine_params=affine_params,
                                       pixel_mean=[0., 0., 0.],
                                       pixel_std=[255., 255., 255.],
                                       )
    else:
        ssd_augment = YOLOBaseTransform(img_size=416,
                                        max_stride=32,
                                        pixel_mean=[0., 0., 0.],
                                        pixel_std=[255., 255., 255.],
                                        )
    
    # 读取图像数据
    orig_image = cv2.imread(image_path)
    target = {
        "boxes": np.array([[86, 96, 256, 425], [132, 71, 243, 282]], dtype=np.float32),
        "labels": np.array([12, 14], dtype=np.int32),
    }

    # 绘制原始数据的边界框
    image_copy = orig_image.copy()
    for box in target["boxes"]:
        x1, y1, x2, y2 = box
        image_copy = cv2.rectangle(image_copy, (int(x1), int(y1)), (int(x2), int(y2)), [0, 0, 255], 2)
    cv2.imshow("original image", image_copy)
    cv2.waitKey(0)

    # 展示预处理后的输入图像数据和标签信息
    image_aug, target_aug, _ = ssd_augment(orig_image, target)
    # [c, h, w] -> [h, w, c]
    image_aug = image_aug.permute(1, 2, 0).contiguous().numpy()
    image_aug = np.clip(image_aug * 255, 0, 255).astype(np.uint8)
    image_aug = image_aug.copy()

    # 绘制处理后的边界框
    for box in target_aug["boxes"]:
        x1, y1, x2, y2 = box
        image_aug = cv2.rectangle(image_aug, (int(x1), int(y1)), (int(x2), int(y2)), [0, 0, 255], 2)
    cv2.imshow("processed image", image_aug)
    cv2.waitKey(0)
