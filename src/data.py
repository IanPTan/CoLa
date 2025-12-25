import os
import torch as pt
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

ROOT = os.getcwd() + '/data'
pt.manual_seed(123)

def get_dataloader(batch_size = 32):
    train_set = datasets.FashionMNIST(root=ROOT, 
        train=True, 
        download=True, 
        transform=transforms.ToTensor()
    )

    class_weights = pt.rand(10)
    sample_weights = class_weights[train_set.targets]
    ran_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_set), replacement=True)
    train_dl = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        sampler=ran_sampler
    )
    return train_dl
