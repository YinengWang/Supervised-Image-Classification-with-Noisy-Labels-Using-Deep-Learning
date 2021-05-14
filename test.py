import torchvision
import numpy as np
from torchvision import transforms

import datasets

from math import ceil


def test_noise_correctness(batch_size=128, num_epochs=10, noise_rate=0.2, is_symmetric_noise=True):
    def find(arr, v):
        return [t.all() for t in arr == v].index(True)

    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_data.data = train_data.data[:batch_size*num_epochs]
    train_data.targets = train_data.targets[:batch_size*num_epochs]
    train_loader = datasets.generate_loader_with_noise(
        train_data, batch_size=batch_size, shuffle=True, noise_rate=noise_rate, is_symmetric_noise=is_symmetric_noise)

    def get_inputs_targets_original_targets_one_epoch():
        inputs_epoch, targets_epoch, original_targets_epoch = [], [], []
        for i, t, o in train_loader:
            inputs_epoch.append(i.numpy())
            targets_epoch.append(t.numpy())
            original_targets_epoch.append(o.numpy())

        inputs_epoch = np.vstack(inputs_epoch)
        targets_epoch = np.hstack(targets_epoch)
        original_targets_epoch = np.hstack(original_targets_epoch)
        return inputs_epoch, targets_epoch, original_targets_epoch

    inputs_epoch_0, targets_epoch_0, original_targets_epoch_0 = get_inputs_targets_original_targets_one_epoch()
    inputs_epoch_1, targets_epoch_1, original_targets_epoch_1 = get_inputs_targets_original_targets_one_epoch()

    corresponding_idx_in_epoch_0 = [find(inputs_epoch_0, vec) for vec in inputs_epoch_1]
    assert (targets_epoch_0[corresponding_idx_in_epoch_0] == targets_epoch_1).all()
    assert (original_targets_epoch_0[corresponding_idx_in_epoch_0] == original_targets_epoch_1).all()
    print((targets_epoch_0 != original_targets_epoch_0).sum(),
          ceil(len(targets_epoch_0) * noise_rate * (1 - 1/len(train_data.classes))))
    print("Test noise correctness: passed")


if __name__ == '__main__':
    test_noise_correctness()
