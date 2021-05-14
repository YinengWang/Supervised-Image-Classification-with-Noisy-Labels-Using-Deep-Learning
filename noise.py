import numpy as np

from math import ceil


class Noise(object):
    def __init__(self, data_loader, noise_rate=0.0, seed=None):
        if noise_rate < 0 or noise_rate >= 1:
            raise ValueError('The rate of noisy labels should be between 0 and 1')
        if seed is not None:
            np.random.seed(seed)

        num_samples = len(data_loader.dataset)
        self.num_samples = num_samples
        self.num_classes = len(data_loader.dataset.classes)
        batch_size = data_loader.batch_size
        self.noise_rate = noise_rate
        num_batches = ceil(num_samples / batch_size)

        # create list of individual sample indices for each batch
        n_batches_temp = int(np.floor(num_samples / batch_size))
        batch_indices = [[] for _ in range(n_batches_temp)]

        for i, batch_start_idx in enumerate(range(0, n_batches_temp * batch_size, batch_size)):
            batch_indices[i] = list(range(batch_start_idx, batch_start_idx + batch_size, 1))

        if num_samples % batch_size != 0:
            last_batch_size = num_samples - n_batches_temp * batch_size
            assert (last_batch_size > 0)
            last_batch_start_idx = batch_indices[i][-1]
            batch_indices.append(list(range(last_batch_start_idx + 1, num_samples, 1)))
            n_batches_temp += 1
        self.batch_indices = batch_indices

        # randomly sample the indices of labels that will be noisy
        num_noisy_labels = ceil(num_samples * noise_rate)
        noisy_label_idx = np.random.permutation(num_samples)[:num_noisy_labels]
        self.noisy_label_idx_per_batch = [[] for _ in range(num_batches)]
        for idx in noisy_label_idx:
            self.noisy_label_idx_per_batch[idx // batch_size].append(idx % batch_size)

        # change labels randomly in range(num_classes), as specified by self.noisy_label_idx_per_batch
        self.noise_targets_per_batch = [[] for _ in range(num_batches)]
        for idx_batch in range(num_batches):
            for idx in self.noisy_label_idx_per_batch[idx_batch]:
                self.noise_targets_per_batch[idx_batch].append(np.random.randint(self.num_classes, dtype=np.int32))

    def symmetric_noise(self, targets, batch_idx):
        if self.noise_rate == 0.0:
            return targets
        targets_with_noise = targets.clone().detach()
        for counter, idx in enumerate(self.noisy_label_idx_per_batch[batch_idx]):
            targets_with_noise[idx] = self.noise_targets_per_batch[batch_idx][counter]
        return targets_with_noise

    # returns a list of all the indices in a batch
    def get_indices_in_batch(self, batch_idx):
        return self.batch_indices[batch_idx]
