import os
import numpy as np
import torch.utils.data as data
import torchvision.transforms as T
from torchvision.datasets import CIFAR10


class CifarDataset(data.Dataset):
    def __init__(self, is_train=False):
        super().__init__()
        # ----------------- basic parameters -----------------
        self.pixel_mean = [0.5, 0.5, 0.5]
        self.pixel_std =  [0.5, 0.5, 0.5]
        self.is_train  = is_train
        self.image_set = 'train' if is_train else 'val'
        # ----------------- dataset & transforms -----------------
        self.transform = self.build_transform()
        path = os.path.dirname(os.path.abspath(__file__))
        if is_train:
            self.dataset = CIFAR10(os.path.join(path, 'cifar_data/'), train=True, download=True, transform=self.transform)
        else:
            self.dataset = CIFAR10(os.path.join(path, 'cifar_data/'), train=False, download=True, transform=self.transform)

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
            transforms = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])
        else:
            transforms = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])

        return transforms

if __name__ == "__main__":
    import cv2
    
    # dataset
    dataset = CifarDataset(is_train=True)  
    print('Dataset size: ', len(dataset))

    for i in range(len(dataset)):
        image, target = dataset.pull_image(i)
        # to BGR
        image = image[..., (2, 1, 0)]

        cv2.imshow('image', image)
        cv2.waitKey(0)
