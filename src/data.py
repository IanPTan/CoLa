import os
import torch as pt
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
from torchvision import datasets, transforms

ROOT = os.getcwd() + '/data'
pt.manual_seed(123)

def get_dataloader(batch_size = 32):
    # fashion
    f_mnist = datasets.FashionMNIST(root=ROOT, 
        train=True, 
        download=True, 
        transform=transforms.ToTensor()
    )
    # hiragana 
    k_mnist = datasets.KMNIST(root=ROOT, 
        train=True, 
        download=True, 
        transform=transforms.ToTensor()
    )
    # letters 
    e_mnist = datasets.EMNIST(root=ROOT,
        split="letters",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    # big ass concatenated dataset
    train_set = ConcatDataset([f_mnist, k_mnist, e_mnist])

    all_targets = pt.cat((f_mnist.targets, k_mnist.targets, e_mnist.targets))
    class_weights = pt.rand(int(all_targets.max().item()) + 1)
    sample_weights = class_weights[all_targets]
    ran_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_set), replacement=True)

    # actual dataloader
    train_dl = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        sampler=ran_sampler
    )

    # testing datasets for length
    test_dl = DataLoader(
        dataset=e_mnist,
        batch_size=batch_size,
        sampler=ran_sampler
    )

    # train_set: 0-60000=fashion, 60000-120000=hiragana, 120000-244800=letters

    return train_dl, test_dl

if __name__ == "__main__":
    print("Testing dataloader...")
    train_dl, test_dl = get_dataloader()
    dataset = train_dl.dataset
    length = len(dataset)

    test_length = len(test_dl.dataset)
    print(f"Loaded dataset in {ROOT} with length {length}")
    print(f"Loaded test dataset in {ROOT} with length {test_length}")