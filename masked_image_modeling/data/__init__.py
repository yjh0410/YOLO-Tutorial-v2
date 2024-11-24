import torch

from .cifar import CifarDataset
from .custom import CustomDataset


def build_dataset(args, is_train=False):
    # ----------------- CIFAR dataset -----------------
    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.img_dim = 3
        args.img_size = 32
        args.patch_size = 4
        return CifarDataset(is_train)
        
    # ----------------- Customed dataset -----------------
    elif args.dataset == 'custom':
        assert args.num_classes is not None and isinstance(args.num_classes, int)
        args.img_size = 224
        args.patch_size = 16
        return CustomDataset(args, is_train)
    
    else:
        print("Unknown dataset: {}".format(args.dataset))
    

def build_dataloader(args, dataset, is_train=False):
    if is_train:
        sampler = torch.utils.data.RandomSampler(dataset)
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler, args.batch_size, drop_last=True if is_train else False)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler_train, num_workers=args.num_workers, pin_memory=True)
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return dataloader
