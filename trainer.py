import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from tqdm import tqdm

import model


def CIFAR10_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    trainset = torchvision.datasets.CIFAR10('./Datasets', train=True, transform=transform, download=True)
    testset = torchvision.datasets.CIFAR10('./Datasets', train=False, transform=transform, download=True)
    trainloader = torch.DataLoader(trainset)
    testloader = torch.DataLoader(testset)
    return trainloader, testloader


class Trainer(object):
    def __init__(self, model, criterion, optimizer, train_data_loader, test_data_loader, n_epochs):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.n_epochs = n_epochs

    def train(self):
        self.model.train()

        for _ in tqdm(range(self.n_epochs)):
            running_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(self.train_data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(running_loss)

    def test(self):
        model.eval()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(targets, outputs)
