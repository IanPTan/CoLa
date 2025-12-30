import os
import torch as pt
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
from torchvision import datasets, transforms

ROOT = os.getcwd() + '/data'
pt.manual_seed(123)

class TaskDataset(pt.utils.data.Dataset):
    def __init__(self, dataset, task_id, label_offset=0):
        self.dataset = dataset
        self.task_id = task_id
        self.targets = dataset.targets
        self.label_offset = label_offset
        self.offset_targets = dataset.targets + label_offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y + self.label_offset, self.task_id

def get_dataloader(batch_size = 32):
    # fashion, class labels 0-9
    f_mnist = datasets.FashionMNIST(root=ROOT, 
        train=True, 
        download=True, 
        transform=transforms.ToTensor()
    )
    # hiragana (japanese letters), class labels 0-9, offset by 10
    k_mnist = datasets.KMNIST(root=ROOT, 
        train=True, 
        download=True, 
        transform=transforms.ToTensor()
    )
    # letters, class labels 0-26, offset by 20 
    e_mnist = datasets.EMNIST(root=ROOT,
        split="letters",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    # Wrap datasets to include task IDs
    f_mnist = TaskDataset(f_mnist, task_id=0, label_offset=0)
    k_mnist = TaskDataset(k_mnist, task_id=1, label_offset=10)
    e_mnist = TaskDataset(e_mnist, task_id=2, label_offset=20)

    # 0-60000=fashion, 60000-120000=hiragana, 120000-244800=letters
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

    return train_dl

if __name__ == "__main__":
    print("Testing dataloader...")
    train_dl = get_dataloader()
    dataset = train_dl.dataset
    length = len(dataset)

    print(f"Loaded dataset in {ROOT} with length {length}")