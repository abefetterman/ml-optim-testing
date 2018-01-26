import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class Cifar10:
    def __init__(self, datadir, batch_size, augment):
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])

        if augment:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
            ])

        kwargs = {'num_workers': 1, 'pin_memory': True}
        self.train = torch.utils.data.DataLoader(
            datasets.CIFAR10(datadir, train=True, download=True,
                             transform=transform_train),
            batch_size=batch_size, shuffle=True, **kwargs)
        self.test = torch.utils.data.DataLoader(
            datasets.CIFAR10(datadir, train=False, transform=transform_test),
            batch_size=batch_size, shuffle=True, **kwargs)
