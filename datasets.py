import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data._utils import collate

from Custom_dataset import CDONdataset

from math import ceil


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False, device='cpu'):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            # self.indices = torch.randperm(self.dataset_len, device=self.device)    # TODO: a PyTorch bug, upgrade to the newest version
            self.indices = torch.randperm(self.dataset_len).to(self.device)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i:self.i+self.batch_size]
            batch = tuple(torch.index_select(t, 0, indices) for t in self.tensors)
        else:
            batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


def load_cdon_dataset(batch_size=128):
    # Data
    print('==> Preparing CDON data..')

    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor()
        ])

    root_folder = "/tmp/pycharm_project_alfred/Datasets/Test_data"
    dataset = CDONdataset("test_data.csv", root_folder, transform=transform)
    # train_set, test_set = torch.utils.data.random_split(dataset, [1, 1])
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

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


def generate_loader_with_noise(dataset, batch_size, shuffle, noise_rate, is_symmetric_noise, device):
    if noise_rate < 0 or noise_rate >= 1:
        raise ValueError('The rate of noisy labels should be between 0 and 1')
    # load all data into memory
    data = [[inputs, targets, targets] for inputs, targets in dataset]
    if noise_rate != 0.0:
        num_samples = len(dataset.data)
        num_classes = len(dataset.classes)
        num_noisy_labels = ceil(num_samples * noise_rate)
        noisy_label_indices = torch.randperm(num_samples)[:num_noisy_labels]
        if is_symmetric_noise:
            for idx in noisy_label_indices:
                data[idx][1] = np.random.randint(num_classes)
        else:
            raise NotImplementedError()
    inputs, targets, original_targets = collate.default_collate(data)     # concatenate into a single tensor
    inputs, targets, original_targets = inputs.to(device), targets.to(device), original_targets.to(device)
    return FastTensorDataLoader(inputs, targets, original_targets, batch_size=batch_size, shuffle=shuffle, device=device)


def load_cifar10_dataset(batch_size=128, noise_rate=0.0, is_symmetric_noise=True, device='cpu'):
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
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = generate_loader_with_noise(
        train_data, batch_size=batch_size, shuffle=True, noise_rate=noise_rate, is_symmetric_noise=True, device=device)
    test_loader = generate_loader_with_noise(
        test_data, batch_size=100, shuffle=True, noise_rate=noise_rate, is_symmetric_noise=True, device=device)
    return train_loader, test_loader


def load_cifar100_dataset(batch_size=128, noise_rate=0.0, is_symmetric_noise=True, device='cpu'):
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
    test_data = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    train_loader = generate_loader_with_noise(
        train_data, batch_size=batch_size, shuffle=True, noise_rate=noise_rate, is_symmetric_noise=True, device=device)
    test_loader = generate_loader_with_noise(
        test_data, batch_size=100, shuffle=True, noise_rate=noise_rate, is_symmetric_noise=True, device=device)
    return train_loader, test_loader
