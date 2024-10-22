from .TinyImageNet import TinyImageNet
from .Imagenet_R import Imagenet_R
from torchvision.datasets import CIFAR100
from .Core50 import CORe50


__all__ = [
    "TinyImageNet",
    "CIFAR100",
    "Imagenet_R",
    "CORe50",
]


datasets = {
    "cifar100": (CIFAR100, (0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023), 100),
    "tinyimagenet": (TinyImageNet, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 200),
    "imagenet-r": (Imagenet_R, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 200),
    "core50": (CORe50, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 50),
}

def get_dataset(name):
    return datasets[name]
    