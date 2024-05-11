import torch.utils.data as data

from .cifar import CifarDataset
from .mnist import MnistDataset
from .imagenet import ImageNet1KDataset
from .custom import CustomDataset


def build_dataset(args, transform=None, is_train=False):
    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.img_dim = 3
        return CifarDataset(is_train, transform)
    elif args.dataset == 'mnist':
        args.num_classes = 10
        args.img_dim = 1
        return MnistDataset(is_train, transform)
    elif args.dataset == 'imagenet_1k':
        args.num_classes = 1000
        args.img_dim = 3
        return ImageNet1KDataset(args, is_train, transform)
    elif args.dataset == 'custom':
        assert args.num_classes is not None and isinstance(args.num_classes, int)
        args.img_dim = 3
        return CustomDataset(args, is_train, transform)
    

def build_dataloader(args, dataset, is_train=False):
    if is_train:
        sampler = data.distributed.DistributedSampler(dataset) if args.distributed else data.RandomSampler(dataset)
        batch_sampler_train = data.BatchSampler(sampler, args.batch_size // args.world_size, drop_last=True if is_train else False)
        dataloader = data.DataLoader(dataset, batch_sampler=batch_sampler_train, num_workers=args.num_workers, pin_memory=True)
    else:
        dataloader = data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return dataloader
