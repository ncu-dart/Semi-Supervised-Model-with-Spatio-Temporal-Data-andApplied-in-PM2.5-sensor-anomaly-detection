# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 18:29:36 2020

@author: Admin
"""

from .base_dataset import BaseADDataset
from torch.utils.data import DataLoader
import torch
import torchvision
from torchvision import transforms, utils


class PM2_5_TorchvisionDataset(BaseADDataset):
    """TorchvisionDataset class for datasets already implemented in torchvision.datasets."""

    def __init__(self, root: str):
        super().__init__(root)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        batch_size = 1
        img_data = torchvision.datasets.ImageFolder('D:\semi_supervised\Deep-SAD-PyTorch-master\Deep-SAD-PyTorch-master\src\pm2_5_data',
                                            transform=transforms.Compose([
                                                transforms.Scale(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor()])
                                            )
        #data_loader = torch.utils.data.DataLoader(img_data, batch_size=20,shuffle=True)
        train_loader = DataLoader(img_data, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(img_data, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, drop_last=False)
        return train_loader, test_loader