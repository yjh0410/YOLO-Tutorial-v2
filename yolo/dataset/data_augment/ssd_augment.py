# ------------------------------------------------------------
# Data preprocessor for Real-time DETR
# ------------------------------------------------------------
import cv2
import numpy as np
from numpy import random

import torch
import torch.nn.functional as F


# ------------------------- Augmentations -------------------------
class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

## Convert color format
class ConvertColorFormat(object):
    def __init__(self, color_format='rgb'):
        self.color_format = color_format

    def __call__(self, image, target=None):
        """
        Input:
            image: (np.array) a OpenCV image with BGR color format.
            target: None
        Output:
            image: (np.array) a OpenCV image with given color format.
            target: None
        """
        # Convert color format
        if self.color_format == 'rgb':
            image = image[..., (2, 1, 0)]    # BGR -> RGB
        elif self.color_format == 'bgr':
            image = image
        else:
            raise NotImplementedError("Unknown color format: <{}>".format(self.color_format))

        return image, target

## Random color jitter
class RandomDistort(object):
    def __init__(self,
                 hue=[-18, 18, 0.5],
                 saturation=[0.5, 1.5, 0.5],
                 contrast=[0.5, 1.5, 0.5],
                 brightness=[0.5, 1.5, 0.5],
                 random_apply=True,
                 count=4,
                 random_channel=False,
                 prob=1.0):
        super(RandomDistort, self).__init__()
        self.hue = hue
        self.saturation = saturation
        self.contrast = contrast
        self.brightness = brightness
        self.random_apply = random_apply
        self.count = count
        self.random_channel = random_channel
        self.prob = prob

    def apply_hue(self, image, target=None):
        if np.random.uniform(0., 1.) < self.prob:
            return image, target

        low, high, prob = self.hue
        image = image.astype(np.float32)
        # it works, but result differ from HSV version
        delta = np.random.uniform(low, high)
        u = np.cos(delta * np.pi)
        w = np.sin(delta * np.pi)
        bt = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]])
        tyiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321],
                         [0.211, -0.523, 0.311]])
        ityiq = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647],
                          [1.0, -1.107, 1.705]])
        t = np.dot(np.dot(ityiq, bt), tyiq).T
        image = np.dot(image, t)

        return image, target

    def apply_saturation(self, image, target=None):
        low, high, prob = self.saturation
        if np.random.uniform(0., 1.) < self.prob:
            return image, target
        delta = np.random.uniform(low, high)
        image = image.astype(np.float32)
        # it works, but result differ from HSV version
        gray = image * np.array([[[0.299, 0.587, 0.114]]], dtype=np.float32)
        gray = gray.sum(axis=2, keepdims=True)
        gray *= (1.0 - delta)
        image *= delta
        image += gray

        return image, target

    def apply_contrast(self, image, target=None):
        if np.random.uniform(0., 1.) < self.prob:
            return image, target
        
        low, high, prob = self.contrast
        delta = np.random.uniform(low, high)
        image = image.astype(np.float32)
        image *= delta

        return image, target

    def apply_brightness(self, image, target=None):
        if np.random.uniform(0., 1.) < self.prob:
            return image, target
        
        low, high, prob = self.brightness
        delta = np.random.uniform(low, high)
        image = image.astype(np.float32)
        image += delta

        return image, target

    def __call__(self, image, target=None):
        if random.random() > self.prob:
            return image, target

        if self.random_apply:
            functions = [
                self.apply_brightness, self.apply_contrast,
                self.apply_saturation, self.apply_hue
            ]
            distortions = np.random.permutation(functions)[:self.count]
            for func in distortions:
                image, target = func(image, target)
                image = np.clip(image, 0.0, 255.)

            return image, target

        image, target = self.apply_brightness(image, target)
        image = np.clip(image, 0.0, 255.)
        mode = np.random.randint(0, 2)

        if mode:
            image, target = self.apply_contrast(image, target)
            image = np.clip(image, 0.0, 255.)

        image, target = self.apply_saturation(image, target)
        image = np.clip(image, 0.0, 255.)
        image, target = self.apply_hue(image, target)
        image = np.clip(image, 0.0, 255.)

        if not mode:
            image, target = self.apply_contrast(image, target)
            image = np.clip(image, 0.0, 255.)

        if self.random_channel:
            if np.random.randint(0, 2):
                image = image[..., np.random.permutation(3)]

        return image, target

## Random scaling
class RandomExpand(object):
    def __init__(self, fill_value) -> None:
        self.fill_value = fill_value

    def __call__(self, image, target=None):
        if random.randint(2):
            return image, target

        height, width, channels = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.ones(
            (int(height*ratio), int(width*ratio), channels),
            dtype=image.dtype) * self.fill_value
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = target['boxes'].copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))
        target['boxes'] = boxes

        return image, target

## Random IoU based Sample Crop
class RandomIoUCrop(object):
    def __init__(self, p=0.5):
        self.p = p
        self.sample_options = (
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            None,
        )

    def intersect(self, box_a, box_b):
        max_xy = np.minimum(box_a[:, 2:], box_b[2:])
        min_xy = np.maximum(box_a[:, :2], box_b[:2])
        inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)

        return inter[:, 0] * inter[:, 1]

    def compute_iou(self, box_a, box_b):
        inter = self.intersect(box_a, box_b)
        area_a = ((box_a[:, 2]-box_a[:, 0]) *
                (box_a[:, 3]-box_a[:, 1]))  # [A,B]
        area_b = ((box_b[2]-box_b[0]) *
                (box_b[3]-box_b[1]))  # [A,B]
        union = area_a + area_b - inter
        return inter / union  # [A,B]

    def __call__(self, image, target=None):
        height, width, _ = image.shape

        # check target
        if len(target["boxes"]) == 0 or random.random() > self.p:
            return image, target

        while True:
            # randomly choose a mode
            sample_id = np.random.randint(len(self.sample_options))
            mode = self.sample_options[sample_id]
            if mode is None:
                return image, target

            boxes = target["boxes"]
            labels = target["labels"]

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = self.compute_iou(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                # update target
                target["boxes"] = current_boxes
                target["labels"] = current_labels

                return current_image, target

## Random JitterCrop
class RandomJitterCrop(object):
    """Jitter and crop the image and box."""
    def __init__(self, fill_value, p=0.5, jitter_ratio=0.3):
        super().__init__()
        self.p = p
        self.jitter_ratio = jitter_ratio
        self.fill_value = fill_value

    def crop(self, image, pleft, pright, ptop, pbot, output_size):
        oh, ow = image.shape[:2]

        swidth, sheight = output_size

        src_rect = [pleft, ptop, swidth + pleft,
                    sheight + ptop]  # x1,y1,x2,y2
        img_rect = [0, 0, ow, oh]
        # rect intersection
        new_src_rect = [max(src_rect[0], img_rect[0]),
                        max(src_rect[1], img_rect[1]),
                        min(src_rect[2], img_rect[2]),
                        min(src_rect[3], img_rect[3])]
        dst_rect = [max(0, -pleft),
                    max(0, -ptop),
                    max(0, -pleft) + new_src_rect[2] - new_src_rect[0],
                    max(0, -ptop) + new_src_rect[3] - new_src_rect[1]]

        # crop the image
        cropped = np.ones([sheight, swidth, 3], dtype=image.dtype) * self.fill_value
        # cropped[:, :, ] = np.mean(image, axis=(0, 1))
        cropped[dst_rect[1]:dst_rect[3], dst_rect[0]:dst_rect[2]] = \
            image[new_src_rect[1]:new_src_rect[3],
            new_src_rect[0]:new_src_rect[2]]

        return cropped

    def __call__(self, image, target=None):
        if random.random() > self.p:
            return image, target
        else:
            oh, ow = image.shape[:2]
            dw = int(ow * self.jitter_ratio)
            dh = int(oh * self.jitter_ratio)
            pleft = np.random.randint(-dw, dw)
            pright = np.random.randint(-dw, dw)
            ptop = np.random.randint(-dh, dh)
            pbot = np.random.randint(-dh, dh)

            swidth = ow - pleft - pright
            sheight = oh - ptop - pbot
            output_size = (swidth, sheight)
            # crop image
            cropped_image = self.crop(image=image,
                                    pleft=pleft, 
                                    pright=pright, 
                                    ptop=ptop, 
                                    pbot=pbot,
                                    output_size=output_size)
            # crop bbox
            if target is not None:
                bboxes = target['boxes'].copy()
                coords_offset = np.array([pleft, ptop], dtype=np.float32)
                bboxes[..., [0, 2]] = bboxes[..., [0, 2]] - coords_offset[0]
                bboxes[..., [1, 3]] = bboxes[..., [1, 3]] - coords_offset[1]
                swidth, sheight = output_size

                bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], 0, swidth - 1)
                bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], 0, sheight - 1)
                target['boxes'] = bboxes

            return cropped_image, target
    
## Random HFlip
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target=None):
        if random.random() < self.p:
            orig_h, orig_w = image.shape[:2]
            image = image[:, ::-1]
            if target is not None:
                if "boxes" in target:
                    boxes = target["boxes"].copy()
                    boxes[..., [0, 2]] = orig_w - boxes[..., [2, 0]]
                    target["boxes"] = boxes

        return image, target

## Resize tensor image
class Resize(object):
    def __init__(self, img_size=640):
        self.img_size = img_size

    def __call__(self, image, target=None):
        orig_h, orig_w = image.shape[:2]

        # resize
        image = cv2.resize(image, (self.img_size, self.img_size)).astype(np.float32)
        img_h, img_w = image.shape[:2]

        # rescale bboxes
        if target is not None:
            boxes = target["boxes"].astype(np.float32)
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / orig_w * img_w
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / orig_h * img_h
            target["boxes"] = boxes

        return image, target

## Normalize tensor image
class Normalize(object):
    def __init__(self, pixel_mean, pixel_std, normalize_coords=False):
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.normalize_coords = normalize_coords

    def __call__(self, image, target=None):
        # normalize image
        image = (image - self.pixel_mean) / self.pixel_std

        # normalize bbox
        if target is not None and self.normalize_coords:
            img_h, img_w = image.shape[:2]
            target["boxes"][..., [0, 2]] = target["boxes"][..., [0, 2]] / float(img_w)
            target["boxes"][..., [1, 3]] = target["boxes"][..., [1, 3]] / float(img_h)

        return image, target

## Convert ndarray to torch.Tensor
class ToTensor(object):
    def __call__(self, image, target=None):        
        # Convert torch.Tensor
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()

        if target is not None:
            target["boxes"] = torch.as_tensor(target["boxes"]).float()
            target["labels"] = torch.as_tensor(target["labels"]).long()

        return image, target

## Convert BBox foramt
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


# ------------------------- Preprocessers -------------------------
## Transform for Train
class SSDAugmentation(object):
    def __init__(self,
                 img_size   = 640,
                 pixel_mean = [123.675, 116.28, 103.53],
                 pixel_std  = [58.395, 57.12, 57.375],
                 box_format = 'xywh',
                 normalize_coords = False):
        # ----------------- Basic parameters -----------------
        self.img_size = img_size
        self.box_format = box_format
        self.pixel_mean = pixel_mean   # RGB format
        self.pixel_std  = pixel_std    # RGB format
        self.normalize_coords = normalize_coords
        self.color_format = 'rgb'
        print("================= Pixel Statistics =================")
        print("Pixel mean: {}".format(self.pixel_mean))
        print("Pixel std:  {}".format(self.pixel_std))

        # ----------------- Transforms -----------------
        self.augment = Compose([
            RandomDistort(prob=0.5),
            RandomExpand(fill_value=self.pixel_mean[::-1]),
            RandomIoUCrop(p=0.8),
            RandomHorizontalFlip(p=0.5),
            Resize(img_size=self.img_size),
            ConvertColorFormat(self.color_format),
            Normalize(self.pixel_mean, self.pixel_std, normalize_coords),
            ToTensor(),
            ConvertBoxFormat(self.box_format),
        ])

    def __call__(self, image, target, mosaic=False):
        orig_h, orig_w = image.shape[:2]
        ratio = [self.img_size / orig_w, self.img_size / orig_h]

        image, target = self.augment(image, target)

        return image, target, ratio

## Transform for Eval
class SSDBaseTransform(object):
    def __init__(self,
                 img_size   = 640,
                 pixel_mean = [123.675, 116.28, 103.53],
                 pixel_std  = [58.395, 57.12, 57.375],
                 box_format = 'xywh',
                 normalize_coords = False):
        # ----------------- Basic parameters -----------------
        self.img_size = img_size
        self.box_format = box_format
        self.pixel_mean = pixel_mean  # RGB format
        self.pixel_std  = pixel_std    # RGB format
        self.normalize_coords = normalize_coords
        self.color_format = 'rgb'
        print("================= Pixel Statistics =================")
        print("Pixel mean: {}".format(self.pixel_mean))
        print("Pixel std:  {}".format(self.pixel_std))

        # ----------------- Transforms -----------------
        self.transform = Compose([
            Resize(img_size=self.img_size),
            ConvertColorFormat(self.color_format),
            Normalize(self.pixel_mean, self.pixel_std, self.normalize_coords),
            ToTensor(),
            ConvertBoxFormat(self.box_format),
        ])


    def __call__(self, image, target=None, mosaic=False):
        orig_h, orig_w = image.shape[:2]
        ratio = [self.img_size / orig_w, self.img_size / orig_h]

        image, target = self.transform(image, target)

        return image, target, ratio


if __name__ == "__main__":
    image_path = "voc_image.jpg"
    is_train = True

    if is_train:
        ssd_augment = SSDAugmentation(img_size=416,
                                      pixel_mean=[0., 0., 0.],
                                      pixel_std=[255., 255., 255.],
                                      box_format="xyxy",
                                      normalize_coords=False,
                                      )
    else:
        ssd_augment = SSDBaseTransform(img_size=416,
                                       pixel_mean=[0., 0., 0.],
                                       pixel_std=[255., 255., 255.],
                                       box_format="xyxy",
                                       normalize_coords=False,
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
    image_aug = image_aug[:, :, (2, 1, 0)]  # 切换为CV2默认的BGR通道顺序
    image_aug = image_aug.copy()

    # 绘制处理后的边界框
    for box in target_aug["boxes"]:
        x1, y1, x2, y2 = box
        image_aug = cv2.rectangle(image_aug, (int(x1), int(y1)), (int(x2), int(y2)), [0, 0, 255], 2)
    cv2.imshow("processed image", image_aug)
    cv2.waitKey(0)
