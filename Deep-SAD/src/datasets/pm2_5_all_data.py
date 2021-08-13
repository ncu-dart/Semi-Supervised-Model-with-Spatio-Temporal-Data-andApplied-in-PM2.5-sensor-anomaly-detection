from torch.utils.data import Subset
from PIL import Image
#from torchvision.datasets import Dataset
from .preprocessing import create_semisupervised_setting
from .preprocessing import create_semisupervised_setting_unlabeled
import torch
import torchvision.transforms as transforms
import random
from scipy.io import loadmat
from torch.utils.data.dataset import Dataset
from base.torchvision_dataset import TorchvisionDataset
import numpy as np


class PM2_5_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class: int = 0, known_outlier_class: int = 1, n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0):
        super().__init__(root)

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = (0,)
        self.outlier_classes = (1,)

        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        else:
            self.known_outlier_classes = (1,)
        
        print('normal_classes',self.normal_classes)
        print('outlier_classes',self.outlier_classes)

        # PM2_5 preprocessing: feature scaling to [0, 1]
        transform = transforms.ToTensor()
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        # Get train set
        train_set = MyPM2_5(root=self.root, train=True,transform=transform, target_transform=target_transform)
        
        # Create semi-supervised setting
        print('root',root)
        
        self.n_classes = 3
        self.unlabeled_classes = (2,)
        
        idx, _, semi_targets = create_semisupervised_setting_unlabeled(train_set.targets.cpu().data.numpy(), self.normal_classes,
                                                         self.outlier_classes, self.unlabeled_classes, self.known_outlier_classes,
                                                         ratio_known_normal, ratio_known_outlier, ratio_pollution)
        '''
        else:
            idx, _, semi_targets = create_semisupervised_setting(np.array(train_set.targets), self.normal_classes,
                                                                 self.outlier_classes, self.known_outlier_classes,
                                                                 ratio_known_normal, ratio_known_outlier, ratio_pollution)
        '''     
        
        train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels

        # Subset train_set to semi-supervised setup
        self.train_set = Subset(train_set, idx)

        # Get test set
        
        self.test_set = MyPM2_5(root=self.root, train=False,transform=transform, target_transform=target_transform)
        print(self.test_set)
        
class MyPM2_5(Dataset):
    """
    Torchvision MNIST class with additional targets for the semi-supervised setting and patch of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """
    
    def __init__(self,root,transform=None, target_transform=None,train=None):
        super(MyPM2_5,self).__init__()
        imgs = []                      #创建一个名为img的空列表，一会儿用来装东西
        target = []
        if train == True: 
            fh = open('D:/semi_supervised/Deep-SAD-PyTorch-master/Deep-SAD-PyTorch-master/src/pm2_5_data/train.txt','r') 
            n = 0
            for line in fh:
                if n >= 1000:
                    break
                line = line.rstrip()       # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
                words = line.split()       #通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
                imgs.append((words[0],int(words[1]))) #把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定 # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
                target.append(int(words[1]))
                
                if int(words[1]) == 2:
                    n = n + 1
        else:
            fh = open('D:/semi_supervised/Deep-SAD-PyTorch-master/Deep-SAD-PyTorch-master/src/pm2_5_data/test.txt','r')  
        
        
        for line in fh:                #按行循环txt文本中的内容
            line = line.rstrip()       # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()       #通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            imgs.append((words[0],int(words[1]))) #把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定 # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
            target.append(int(words[1]))
          
        
        self.imgs = imgs
        #self.targets = target
        #self.imgs = torch.tensor(imgs, dtype=torch.int64)
        self.targets = torch.tensor(target, dtype=torch.int64)
        self.transform = transform
        self.target_transform = target_transform
        self.semi_targets = torch.zeros(len(target), dtype=torch.int64)
    

    def __getitem__(self, index):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, semi_target, index)
        """
    
        fn, target = self.imgs[index] #fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = Image.open(fn).convert('L')
        semi_target = int(self.semi_targets[index])
        #semi_target = target
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        #if self.target_transform is not None:
        #    target = self.target_transform(target)
        return img, target, semi_target, index
    
    def __len__(self):
        return len(self.imgs)
