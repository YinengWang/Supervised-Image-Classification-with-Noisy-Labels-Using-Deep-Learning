# main file for training


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from Custom_dataset import CDONdataset


# Data
print('==> Preparing data..')

transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor()
    ])

rootFolder = "/tmp/pycharm_project_812/Datasets/Test_data"
dataset = CDONdataset("test_data.csv", rootFolder, transform=transform)
# train_set, test_set = torch.utils.data.random_split(dataset, [1, 1])
train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

for train_images, train_labels in train_loader:
    print(train_images.shape)
    print(train_labels)
plt.imshow(train_images.permute( 2, 3, 1, 0)[:,:,:,0])
plt.savefig("Results/test.jpg")
