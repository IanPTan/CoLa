import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

class CIFAR10Dataset(Dataset):
    """
    PyTorch Dataset for CIFAR-10 data stored in HDF5 format.
    """
    def __init__(self, h5_path, split='train', transform=None):
        """
        Args:
            h5_path (str): Path to the HDF5 file.
            split (str): 'train' or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.h5_path = h5_path
        self.split = split
        self.transform = transform
        
        # We don't open the file here to ensure it's compatible with 
        # multi-process DataLoader (each worker needs its own file handle).
        with h5py.File(self.h5_path, 'r') as f:
            self.dataset_len = len(f[self.split]['labels'])
        
        self.h5_file = None

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            
        image = self.h5_file[self.split]['images'][index]
        label = self.h5_file[self.split]['labels'][index]
        
        # Convert to torch tensor
        # CIFAR-10 images in our H5 are (3, 32, 32) uint8
        image = torch.from_numpy(image).float() / 255.0  # Normalize to [0, 1]
        label = int(label)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()
