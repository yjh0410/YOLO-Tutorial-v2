import os
import torch.utils.data as data
import torchvision.transforms as T
from torchvision.datasets import MNIST


class MnistDataset(data.Dataset):
    def __init__(self, is_train=False):
        super().__init__()
        # ----------------- basic parameters -----------------
        self.is_train   = is_train
        self.pixel_mean = [0.]
        self.pixel_std  = [1.]
        self.image_set  = 'train' if is_train else 'val'
        # ----------------- dataset & transforms -----------------
        self.transform = self.build_transform()
        path = os.path.dirname(os.path.abspath(__file__))
        if is_train:
            self.dataset = MNIST(os.path.join(path, 'mnist_data/'), train=True, download=True, transform=self.transform)
        else:
            self.dataset = MNIST(os.path.join(path, 'mnist_data/'), train=False, download=True, transform=self.transform)

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
        image = image.copy()

        return image, target

    def build_transform(self):
        if self.is_train:
            transforms = T.Compose([T.ToTensor(),])
        else:
            transforms = T.Compose([T.ToTensor(),])

        return transforms

if __name__ == "__main__":
    import cv2
    
    # dataset
    dataset = MnistDataset(is_train=True)  
    print('Dataset size: ', len(dataset))

    for i in range(1000):
        image, target = dataset.pull_image(i)

        cv2.imshow('image', image)
        cv2.waitKey(0)
