import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
# 4 img
data_path = '../DATA5_TEST/'
# 
IMAGE_H = 224
IMAGE_W = 224

data_transform = transforms.Compose([
    transforms.ToTensor()   # 
])
#data_transform = transforms.Compose([
#    transforms.Scale((224, 224), 3),
#    # transforms.RandomHorizontalFlip(),
#    transforms.ToTensor()
#    ])


class MyDataset(Dataset):
    # 
    def __init__(self, txt_path, transform = data_transform, target_transform=None):
        super(MyDataset, self).__init__()
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            words[0] = data_path + words[0]
            words[1] = data_path + words[1]
            words[2] = data_path + words[2]
            words[3] = data_path + words[3]
            # 
            imgs.append((words[0], words[1], words[2], words[3], int(words[4])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn1, fn2, fn3, fn4, label = self.imgs[index]
        img1 = Image.open(fn1)
        img1 = img1.resize((IMAGE_H, IMAGE_W))  # 
        #img1 = np.array(img1)[:, :, :3]  # 
        img2 = Image.open(fn2)
        img2 = img2.resize((IMAGE_H, IMAGE_W))  # 
        img3 = Image.open(fn3)
        img3 = img3.resize((IMAGE_H, IMAGE_W))  # 
        img4 = Image.open(fn4)
        img4 = img4.resize((IMAGE_H, IMAGE_W))  # 
        #img2 = np.array(img2)[:, :, :3]  # 
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            img4 = self.transform(img4)
            # 

        return img1, img2, img3, img4, label

    def __len__(self):
        return len(self.imgs)

