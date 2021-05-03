# main file for training
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from Custom_dataset import CDONdataset
from conv_block import ConvBlock
from model import ResNet18

def load_dataset():

    # Data
    print('==> Preparing data..')

    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor()
        ])

    rootFolder = "/tmp/pycharm_project_alfred/Datasets/Test_data"
    dataset = CDONdataset("test_data.csv", rootFolder, transform=transform)
    # train_set, test_set = torch.utils.data.random_split(dataset, [1, 1])
    train_loader = DataLoader(dataset=dataset, batch_size=2, shuffle=False)

    #
    # for train_images, train_labels in train_loader:
    #     print(train_images.shape)
    #     print(train_labels)
    #     image = train_images.permute(2, 3, 1, 0)[:, :, :, 0]
    #     plt.imshow(image)
    #     plt.savefig("Results/test.jpg")
    #     plt.show()
    #
    #     image = train_images.permute(2, 3, 1, 0)[:, :, :, 1]
    #     plt.imshow(image)
    #     plt.show()
    #     plt.savefig("Results/test2.jpg")

    return train_loader


def main():
    train_loader = load_dataset()
    layers_in_each_block_list = [2, 2, 2, 2]
    model = ResNet18(layers_in_each_block_list)

    x = np.random.rand(1, 3, 32, 32)

    out = model.forward(torch.tensor(x).float())
    print(out.shape)
    pass


if __name__ == '__main__':
    main()
