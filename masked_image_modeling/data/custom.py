import os
import PIL
import numpy as np
import torch.utils.data as data
import torchvision.transforms as T
from torchvision.datasets import ImageFolder


class CustomDataset(data.Dataset):
    def __init__(self, args, is_train=False):
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
        self.transform = self.build_transform()
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

    def build_transform(self):
        if self.is_train:
            transforms = T.Compose([
                            T.RandomResizedCrop(224),
                            T.RandomHorizontalFlip(0.5),
                            T.ToTensor(),
                            T.Normalize(self.pixel_mean,
                                        self.pixel_std)])
        else:
            transforms = T.Compose([
                T.Resize(224, interpolation=PIL.Image.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(self.pixel_mean, self.pixel_std),
            ])

        return transforms


if __name__ == "__main__":
    import cv2
    import argparse
    
    parser = argparse.ArgumentParser(description='Custom-Dataset')

    # opt
    parser.add_argument('--root', default='/Users/liuhaoran/Desktop/python_work/classification/dataset/Animals/',
                        help='data root')
    parser.add_argument('--img_size', default=224, type=int,
                        help='input image size.')
    args = parser.parse_args()
  
    # Dataset
    dataset = CustomDataset(args, is_train=True)  
    print('Dataset size: ', len(dataset))

    for i in range(len(dataset)):
        image, target = dataset.pull_image(i)
        # to BGR
        image = image[..., (2, 1, 0)]

        cv2.imshow('image', image)
        cv2.waitKey(0)
