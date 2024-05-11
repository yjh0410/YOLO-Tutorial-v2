import os
import PIL
import numpy as np
from timm.data import create_transform
import torch.utils.data as data
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


class ImageNet1KDataset(data.Dataset):
    def __init__(self, args, is_train=False, transform=None):
        super().__init__()
        # ----------------- basic parameters -----------------
        self.args = args
        self.is_train   = is_train
        self.pixel_mean = [0.485, 0.456, 0.406]
        self.pixel_std  = [0.229, 0.224, 0.225]
        print("Pixel mean: {}".format(self.pixel_mean))
        print("Pixel std:  {}".format(self.pixel_std))
        self.image_set = 'train' if is_train else 'val'
        self.data_path = os.path.join(args.root, self.image_set)
        # ----------------- dataset & transforms -----------------
        self.transform = transform if transform is not None else self.build_transform(args)
        self.dataset = ImageFolder(root=self.data_path, transform=self.transform)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, target = self.dataset[index]

        return image, target
    
    def pull_image(self, index):
        # laod data
        image, target = self.dataset[index]

        # denormalize image
        image = image.permute(1, 2, 0).numpy()
        image = (image * self.pixel_std + self.pixel_mean) * 255.
        image = image.astype(np.uint8)
        image = image.copy()

        return image, target

    def build_transform(self, args):
        if self.is_train:
            transforms = create_transform(input_size    = args.img_size,
                                          is_training   = True,
                                          color_jitter  = args.color_jitter,
                                          auto_augment  = args.aa,
                                          interpolation = 'bicubic',
                                          re_prob       = args.reprob,
                                          re_mode       = args.remode,
                                          re_count      = args.recount,
                                          mean          = self.pixel_mean,
                                          std           = self.pixel_std,
                                          )
        else:
            t = []
            if args.img_size <= 224:
                crop_pct = 224 / 256
            else:
                crop_pct = 1.0
            size = int(args.img_size / crop_pct)
            t.append(
                T.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(T.CenterCrop(args.img_size))
            t.append(T.ToTensor())
            t.append(T.Normalize(self.pixel_mean, self.pixel_std))
            transforms = T.Compose(t)

        return transforms


if __name__ == "__main__":
    import cv2
    import torch
    import argparse
    
    parser = argparse.ArgumentParser(description='ImageNet-Dataset')

    # opt
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset/imagenet/',
                        help='data root')
    parser.add_argument('--img_size', default=224, type=int,
                        help='input image size.')
    args = parser.parse_args()

    # Transforms
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
  
    # Dataset
    dataset = ImageNet1KDataset(args, is_train=True)  
    print('Dataset size: ', len(dataset))

    for i in range(1000):
        image, target = dataset.pull_image(i)
        # to BGR
        image = image[..., (2, 1, 0)]

        cv2.imshow('image', image)
        cv2.waitKey(0)
