import os
import cv2
import torch

from torchvision.transforms import ToTensor
from torch.utils.data import Dataset


class DatasetPhone(Dataset):

    def __init__(self, data_path="", transform=None):
        self.transform = transform
        with open(f'{data_path}/labels.txt', 'r') as f:
            data = f.readlines()

        self.imgs_path = [os.path.join(
            data_path, i.strip().split()[0]) for i in data]

        self.labels = [tuple(map(float, i.strip().split()[1:])) for i in data]

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs_path[idx], cv2.COLOR_BGR2RGB)
        label = self.labels[idx]

        img, label = self.__scaler(img, label)

        if self.transform:
            img = self.transform(img)

        return img, label

    def __scaler(self, img, label):
        h, w, _ = img.shape
        xscale, yscale = 480/w, 320/h
        img = ToTensor()(cv2.resize(img, (320, 480)))
        label = torch.tensor((label[0]*xscale, label[1]*yscale))
        return img, label
