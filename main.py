# main file for training
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
import torch.optim as optim
from tqdm import tqdm

from Custom_dataset import CDONdataset
from model import ResNet18

# set global env variable
if torch.cuda.is_available():
    print('GPU is enabled!')
    device = 'cuda'
else:
    print('No GPU!')
    device = 'cpu'


def load_cdon_dataset():

    # Data
    print('==> Preparing CDON data..')

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


def train(model, criterion, optimizer, train_loader):
    loss_batch = []

    # activate train mode
    model.train()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # to(device) copies data from CPU to GPU
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # loss is a Tensor, therefore:
        loss_batch.append(loss.item())

    return np.sum(loss_batch)


def test(model, criterion, test_loader):
    loss_batch = []

    # activate eval mode
    model.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss_batch.append(loss.item())

            # outputs is 100x10 (batch_size x n_classes)
            # max value, axis=1
            # returns max value of each column and the index
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = correct / total
    loss = np.sum(loss_batch)
    return loss, acc


def load_cifar10_dataset():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    train_data.data = train_data.data[:200]
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)

    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False, num_workers=2)

    return train_loader, test_loader


def main():
    # Temp commented for CIFAR-10
    # train_loader = load_cdon_dataset()
    # x = None
    # for train_images, train_labels in train_loader:
    #     print(train_images.shape)
    #     print(train_labels)
    #     x = train_images

    layers_in_each_block_list = [2, 2, 2, 2]
    model = ResNet18(layers_in_each_block_list).to(device)

    """Prepare data"""
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print('==> Preparing data..')
    train_loader, test_loader = load_cifar10_dataset()

    """training for 10 epochs"""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0.001)

    loss_train = []
    loss_test = []
    for _ in tqdm(range(10)):
        loss = train(model, criterion, optimizer, train_loader)
        loss_train.append(loss)

        loss = test(model, criterion, test_loader)
        loss_test.append(loss)

        # anneal learning rate
        scheduler.step()


if __name__ == '__main__':
    main()
