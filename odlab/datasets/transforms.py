# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import PIL
import random

import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F


# ----------------- Basic transform functions -----------------
def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)

def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target

def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target

def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target

def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


# ----------------- Basic transform  -----------------
class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target=None):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)

class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict = None):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target=None):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target

class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)

class RandomShift(object):
    def __init__(self, p=0.5, max_shift=32):
        self.p = p
        self.max_shift = max_shift

    def __call__(self, image, target=None):
        if random.random() < self.p:
            img_h, img_w = image.height, image.width
            shift_x = random.randint(-self.max_shift, self.max_shift)
            shift_y = random.randint(-self.max_shift, self.max_shift)
            shifted_image = F.affine(image, translate=[shift_x, shift_y], angle=0, scale=1.0, shear=0)

            target = target.copy()
            target["boxes"][..., [0, 2]] += shift_x
            target["boxes"][..., [1, 3]] += shift_y
            target["boxes"][..., [0, 2]] = target["boxes"][..., [0, 2]].clip(0, img_w)
            target["boxes"][..., [1, 3]] = target["boxes"][..., [1, 3]].clip(0, img_h)

            return shifted_image, target

        return image, target

class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target=None):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)

class ToTensor(object):
    def __call__(self, img, target=None):
        return F.to_tensor(img), target

class Normalize(object):
    def __init__(self, mean, std, normalize_coords=False):
        self.mean = mean
        self.std = std
        self.normalize_coords = normalize_coords

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        if self.normalize_coords:
            target = target.copy()
            h, w = image.shape[-2:]
            if "boxes" in target:
                boxes = target["boxes"]
                boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
                target["boxes"] = boxes
        return image, target

class RefineBBox(object):
    def __init__(self, min_box_size=1):
        self.min_box_size = min_box_size

    def __call__(self, img, target):
        boxes  = target["boxes"].clone()
        labels = target["labels"].clone()

        tgt_boxes_wh = boxes[..., 2:] - boxes[..., :2]
        min_tgt_size = torch.min(tgt_boxes_wh, dim=-1)[0]

        keep = (min_tgt_size >= self.min_box_size)

        target["boxes"] = boxes[keep]
        target["labels"] = labels[keep]

        return img, target

class ConvertBoxFormat(object):
    def __init__(self, box_format="xyxy"):
        self.box_format = box_format

    def __call__(self, image, target=None):
        # convert box format
        if self.box_format == "xyxy" or target is None:
            pass
        elif self.box_format == "xywh":
            target = target.copy()
            if "boxes" in target:
                boxes_xyxy = target["boxes"]
                boxes_xywh = torch.zeros_like(boxes_xyxy)
                boxes_xywh[..., :2] = (boxes_xyxy[..., :2] + boxes_xyxy[..., 2:]) * 0.5   # cxcy
                boxes_xywh[..., 2:] = boxes_xyxy[..., 2:] - boxes_xyxy[..., :2]           # bwbh
                target["boxes"] = boxes_xywh
        else:
            raise NotImplementedError("Unknown box format: {}".format(self.box_format))

        return image, target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


# build transforms
def build_transform(cfg, is_train=False):
    # ---------------- Transform for Training ----------------
    if is_train:
        transforms = []
        trans_config = cfg.trans_config
        # build transform
        if not cfg.detr_style:
            for t in trans_config:
                if t['name'] == 'RandomHFlip':
                    transforms.append(RandomHorizontalFlip())
                if t['name'] == 'RandomResize':
                    transforms.append(RandomResize(cfg.train_min_size, max_size=cfg.train_max_size))
                if t['name'] == 'RandomSizeCrop':
                    transforms.append(RandomSizeCrop(t['min_crop_size'], max_size=t['max_crop_size']))
                if t['name'] == 'RandomShift':
                    transforms.append(RandomShift(max_shift=t['max_shift']))
                if t['name'] == 'RefineBBox':
                    transforms.append(RefineBBox(min_box_size=t['min_box_size']))
            transforms.extend([
                ToTensor(),
                Normalize(cfg.pixel_mean, cfg.pixel_std, cfg.normalize_coords),
                ConvertBoxFormat(cfg.box_format)
            ])
        # build transform for DETR-style detector
        else:
            transforms = [
                RandomHorizontalFlip(),
                RandomSelect(
                    RandomResize(cfg.train_min_size, max_size=cfg.train_max_size),
                    Compose([
                        RandomResize(cfg.train_min_size2),
                        RandomSizeCrop(*cfg.random_crop_size),
                        RandomResize(cfg.train_min_size, max_size=cfg.train_max_size),
                    ])
                ),
                ToTensor(),
                Normalize(cfg.pixel_mean, cfg.pixel_std, cfg.normalize_coords),
                ConvertBoxFormat(cfg.box_format)
            ]

    # ---------------- Transform for Evaluating ----------------
    else:
        transforms = [
            RandomResize(cfg.test_min_size, max_size=cfg.test_max_size),
            ToTensor(),
            Normalize(cfg.pixel_mean, cfg.pixel_std, cfg.normalize_coords),
            ConvertBoxFormat(cfg.box_format)
        ]
    
    return Compose(transforms)
