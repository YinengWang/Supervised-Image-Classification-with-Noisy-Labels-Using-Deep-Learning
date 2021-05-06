import numpy as np

from math import ceil


class Noise(object):
    def __init__(self, data_loader, noise_rate=0, seed=None):
        if noise_rate < 0 or noise_rate >= 1:
            raise ValueError('The rate of noisy labels should be between 0 and 1')
        if seed is not None:
            np.random.seed(seed)

        num_samples = len(data_loader.dataset)
        self.num_classes = len(data_loader.dataset.classes)
        batch_size = data_loader.batch_size
        self.noise_rate = noise_rate
        num_batches = ceil(num_samples / batch_size)
        num_noisy_labels = ceil(num_samples * noise_rate)
        noisy_label_idx = np.random.permutation(num_samples)[:num_noisy_labels]
        self.noisy_label_idx_per_batch = [[] for _ in range(num_batches)]
        for idx in noisy_label_idx:
            self.noisy_label_idx_per_batch[idx // batch_size].append(idx % batch_size)

    def symmetric_noise(self, targets, batch_idx):
        if self.noise_rate == 0:
            return targets
        targets_with_noise = targets.clone().detach()
        for idx in self.noisy_label_idx_per_batch[batch_idx]:
            targets_with_noise[idx] = np.random.randint(self.num_classes, dtype=np.int32)
        return targets_with_noise
