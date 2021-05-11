# main file for training
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

from model import ResNet18
from model import ResNet34
from noise import Noise
import datasets

import os.path
from pathlib import Path
import csv


# set global env variable
if torch.cuda.is_available():
    print('GPU is enabled!')
    print('Using ' + torch.cuda.get_device_name(0))
    device = 'cuda'
else:
    print('No GPU!')
    device = 'cpu'


def train(model, criterion, optimizer, n_epochs, train_loader, test_loader=None, enable_amp=True,
          scheduler=None, noise_rate=0.0, is_symmetric_noise=True):
    train_noise_generator = Noise(train_loader, noise_rate=noise_rate)
    test_noise_generator = Noise(test_loader, noise_rate=noise_rate) if test_loader is not None else None

    train_loss_per_epoch = []
    test_loss_per_epoch = []
    correct_per_epoch = []
    incorrect_per_epoch = []
    memorized_per_epoch = []

    for _ in tqdm(range(n_epochs)):
        # activate train mode
        model.train()
        train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if is_symmetric_noise:
                targets_with_noise = train_noise_generator.symmetric_noise(targets, batch_idx)
            else:
                targets_with_noise = train_noise_generator.asymmetric_noise(targets, batch_idx)
            # to(device) copies data from CPU to GPU
            inputs, targets = inputs.to(device), targets_with_noise.to(device)
            optimizer.zero_grad()

            if enable_amp:
                scaler = GradScaler()
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * targets.size(0)
        train_loss_per_epoch.append(train_loss / len(train_loader.dataset))

        if test_loader is not None:
            model.eval()
            test_loss = 0
            with torch.no_grad():
                correct, incorrect, memorized, total = 0, 0, 0, 0
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    original_targets = targets.to(device)
                    if is_symmetric_noise:
                        targets_with_noise = test_noise_generator.symmetric_noise(targets, batch_idx)
                    else:
                        targets_with_noise = test_noise_generator.asymmetric_noise(targets, batch_idx)
                    inputs, targets = inputs.to(device), targets_with_noise.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct_idx = predicted.eq(original_targets)
                    memorized_idx = ((predicted != original_targets) & (predicted == targets))
                    incorrect_idx = ((predicted != original_targets) & (predicted != targets))
                    correct += correct_idx.sum().item()
                    memorized += memorized_idx.sum().item()
                    incorrect += incorrect_idx.sum().item()
                    test_loss += loss.item() * targets.size(0)

                test_loss_per_epoch.append(test_loss / total)
                correct_per_epoch.append(correct / total)
                memorized_per_epoch.append(memorized / total)
                incorrect_per_epoch.append(incorrect / total)

        # anneal learning rate
        scheduler.step()

    return (train_loss_per_epoch, test_loss_per_epoch,
            correct_per_epoch, memorized_per_epoch, incorrect_per_epoch,)


def plot_learning_curve_and_acc(results, title, path_prefix):
    train_cost, test_cost, test_correct, test_memorized, test_incorrect = results
    epochs = np.arange(1, len(train_cost) + 1)
    # plot learning curve
    plt.plot(epochs, train_cost)
    plt.plot(epochs, test_cost)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'])
    plt.savefig(path_prefix + '_loss.pdf')
    plt.show()

    # plot fraction of correct, incorrect, memorized samples
    plt.plot(epochs, test_correct)
    plt.plot(epochs, test_memorized)
    plt.plot(epochs, test_incorrect)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Fraction of examples')
    plt.legend(['Correct', 'Memorized', 'Incorrect'])
    plt.savefig(path_prefix + '_acc.pdf')
    plt.show()


def record_results(filepath, dataset, noise_rate, is_symmetric_noise, enable_amp, results):
    fieldnames = ['dataset', 'noise_rate', 'is_symmetric_noise', 'enable_amp',
                  'train_loss', 'test_loss', 'correct', 'memorized', 'incorrect']
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            writer = csv.DictWriter(f, fieldnames)
            writer.writeheader()
    with open(filepath, 'a') as f:
        writer = csv.DictWriter(f, fieldnames)
        train_loss, test_loss, correct, memorized, incorrect = results
        writer.writerow({
            'dataset': dataset, 'noise_rate': noise_rate, 'is_symmetric_noise': is_symmetric_noise,
            'enable_amp': enable_amp,
            'train_loss': train_loss[-1], 'test_loss': test_loss[-1],
            'correct': correct[-1], 'memorized': memorized[-1], 'incorrect': incorrect[-1]
        })


def train_CIFAR(CIFAR10=True, n_epochs=100, noise_rate=0.0, is_symmetric_noise=True, trainer_config_custom=None):
    trainer_config = {
        'model': ResNet34, 'enable_amp': True,
        'optimizer': optim.SGD,
        'optimizer_params': {'lr': 0.02, 'momentum': 0.9, 'weight_decay': 1e-3},
        'scheduler': optim.lr_scheduler.CosineAnnealingWarmRestarts,
        'scheduler_params': {'T_0': 10, 'eta_min': 1e-3}
    }
    if trainer_config_custom is not None:
        trainer_config.update(trainer_config_custom)

    output_features = 10 if CIFAR10 else 100
    dataset_name = 'CIFAR10' if CIFAR10 else 'CIFAR100'

    model = trainer_config['model'](output_features).to(device)

    """Prepare data"""
    print('==> Preparing data..')
    if CIFAR10:
        train_loader, test_loader = datasets.load_cifar10_dataset()
    else:
        train_loader, test_loader = datasets.load_cifar100_dataset()

    """training for 10 epochs"""
    print(f'==> Start training {dataset_name} '
          f'with noise level {noise_rate} '
          f'for {n_epochs} epochs')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = trainer_config['optimizer'](model.parameters(), **trainer_config['optimizer_params'])
    scheduler = trainer_config['scheduler'](optimizer, **trainer_config['scheduler_params'])

    train_results = train(
        model, criterion, optimizer, n_epochs=n_epochs,
        train_loader=train_loader, test_loader=test_loader, enable_amp=trainer_config['enable_amp'],
        scheduler=scheduler, noise_rate=noise_rate, is_symmetric_noise=is_symmetric_noise)
    return train_results, model


def main():
    # Create folders to save models and results if not exist
    Path("./models").mkdir(parents=True, exist_ok=True)
    Path("./results").mkdir(parents=True, exist_ok=True)
    is_symmetric_noise = True
    enable_amp = False
    n_epochs = 120
    for noise_rate in [0.2, 0.4, 0.6, 0.8]:
        trainer_config = {
            'enable_amp': enable_amp,
            'scheduler': optim.lr_scheduler.MultiStepLR,
            'scheduler_params': {'milestones': [40, 80], 'gamma': 0.01}
        }
        results, model = train_CIFAR(CIFAR10=True, n_epochs=n_epochs, noise_rate=noise_rate,
                                     is_symmetric_noise=is_symmetric_noise,
                                     trainer_config_custom=trainer_config)

        """Plot learning curve and accuracy. Save results."""
        result_file_prefix = f'CIFAR10_sym_noise_{noise_rate}'
        plot_title = f'CIFAR10, noise_level={noise_rate}'
        plot_learning_curve_and_acc(results, plot_title, path_prefix='./results/' + result_file_prefix)
        torch.save(model.state_dict(), './models/' + result_file_prefix + '.mdl')
        record_results('./results/result.csv', 'CIFAR10', noise_rate, is_symmetric_noise, enable_amp, results)
    # train_CIFAR(CIFAR10=False, n_epochs=150, noise_rate=0, model_path='./models/CIFAR100_noise_level_0.mdl')
    # train_CIFAR(CIFAR10=False, n_epochs=150, noise_rate=0.1, model_path='./models/CIFAR100_noise_level_10.mdl')


if __name__ == '__main__':
    main()
