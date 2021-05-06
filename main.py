# main file for training
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm

from Custom_dataset import CDONdataset
from model import ResNet18
from noise import Noise

# set global env variable
if torch.cuda.is_available():
    print('GPU is enabled!')
    device = 'cuda'
else:
    print('No GPU!')
    device = 'cpu'

SEED = 2021
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


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


def train(model, criterion, optimizer, n_epochs, train_loader, test_loader=None, scheduler=None, noise_rate=0):
    train_noise_generator = Noise(train_loader, noise_rate=noise_rate)
    test_noise_generator = Noise(test_loader, noise_rate=noise_rate) if test_loader is not None else None

    train_loss_per_epoch = []
    test_loss_per_epoch = []
    acc_per_epoch = []
    memorized_per_epoch = []

    for _ in tqdm(range(n_epochs)):
        # activate train mode
        model.train()
        train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            targets_with_noise = train_noise_generator.symmetric_noise(targets, batch_idx)
            # to(device) copies data from CPU to GPU
            inputs, targets = inputs.to(device), targets_with_noise.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss_per_epoch.append(train_loss)

        if test_loader is not None:
            model.eval()
            test_loss = 0
            with torch.no_grad():
                correct, memorized, total = 0, 0, 0
                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(test_loader):
                        original_targets = targets.to(device)
                        targets_with_noise = test_noise_generator.symmetric_noise(targets, batch_idx)
                        inputs, targets = inputs.to(device), targets_with_noise.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(original_targets).sum().item()
                        memorized += ((predicted != original_targets) & (predicted == targets)).sum().item()
                        test_loss += loss.item()

                acc = correct / total
                memorized_rate = memorized / total
                test_loss_per_epoch.append(test_loss)
                acc_per_epoch.append(acc)
                memorized_per_epoch.append(memorized_rate)

        # anneal learning rate
        scheduler.step()

    return train_loss_per_epoch, test_loss_per_epoch, acc_per_epoch, memorized_per_epoch


def test(model, criterion, test_loader, noise_rate=0):
    loss_batch = []

    # activate eval mode
    model.eval()
    noise_generator = Noise(test_loader, noise_rate=noise_rate)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            targets_with_noise = noise_generator.symmetric_noise(targets, batch_idx)
            inputs, targets = inputs.to(device), targets_with_noise.to(device)
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

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)

    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False, num_workers=2)

    return train_loader, test_loader


def load_cifar100_dataset():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_data = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)

    test_data = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False, num_workers=2)

    return train_loader, test_loader


def plot_learning_curve_and_acc(train_cost, test_cost, test_acc, test_memorized):
    # plot learning curve
    plt.plot(train_cost)
    plt.plot(test_cost)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'])
    plt.show()

    plt.plot(test_acc)
    plt.plot(test_memorized)
    plt.xlabel('Epoch')
    plt.ylabel('Ratio')
    plt.legend(['Accuracy', 'Memorized'])
    plt.show()


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
    # train_loader, test_loader = load_cifar10_dataset()
    train_loader, test_loader = load_cifar100_dataset()

    """training for 10 epochs"""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0.001)

    train_loss_per_epoch, test_loss_per_epoch, acc_per_epoch, memorized_per_epoch = train(
        model, criterion, optimizer, n_epochs=100,
        train_loader=train_loader, test_loader=test_loader, scheduler=scheduler,
        noise_rate=0.1)

    """Plot learning curve and accuracy"""
    print(f'acc={acc_per_epoch[-1]}, memorized={memorized_per_epoch[-1]}')
    plot_learning_curve_and_acc(train_loss_per_epoch, test_loss_per_epoch, acc_per_epoch, memorized_per_epoch)
    torch.save(model.state_dict(), './models/ResNet18_sym_noise_10.mdl')


if __name__ == '__main__':
    main()
