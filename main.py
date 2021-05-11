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
import wandb


# set global env variable
if torch.cuda.is_available():
    print('GPU is enabled!')
    print('Using ' + torch.cuda.get_device_name(0))
    device = 'cuda'
else:
    print('No GPU!')
    device = 'cpu'


def train(model, criterion, optimizer, n_epochs, train_loader, test_loader=None, scheduler=None, config=None):
    # tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log='all', log_freq=100)

    train_noise_generator = Noise(train_loader, noise_rate=config.noise_rate)
    test_noise_generator = Noise(test_loader, noise_rate=config.noise_rate) if test_loader is not None else None

    train_loss_per_epoch = []
    test_loss_per_epoch = []
    correct_per_epoch = []
    incorrect_per_epoch = []
    memorized_per_epoch = []

    example_ct = 0  # number of examples seen
    batch_ct = 0
    n = len(train_loader.dataset)
    for epoch in tqdm(range(n_epochs)):
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

            if config.enable_amp:
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
            loss_batch = loss.item()

            example_ct += len(inputs)
            batch_ct += 1

            # log loss on WandB every 25 steps
            # if ((batch_ct + 1) % 25) == 0:
            # loss_for_wand_b = float(train_loss/batch_ct)
            wandb.log({"epoch": epoch, "loss": loss_batch}, step=batch_ct)
            # print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss_for_wand_b:.3f}")

        train_loss_per_epoch.append(train_loss / n)

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

                wandb.log({"test_loss": test_loss / total, "test_accuracy": correct / total,
                           "memorized": memorized / total, "incorrect": incorrect / total})
                print(f"Test loss after " + str(example_ct).zfill(5) + f" examples: {test_loss / total:.3f}")
                torch.onnx.export(model, inputs, "model.onnx")
                wandb.save("model.onnx")

        # anneal learning rate
        scheduler.step()

    return train_loss_per_epoch, test_loss_per_epoch, correct_per_epoch, memorized_per_epoch, incorrect_per_epoch


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


def model_pipeline(config, trainer_config, loadExistingWeights=False):
    # Start wandb
    wandb_project = 'dd2424-ResNet-team'
    with wandb.init(project=wandb_project, config=config):
        # access all hyperparameters through wandb.config, so logging matches execution!
        config = wandb.config

        """create the model"""
        model = trainer_config['model'](config['classes']).to(device)

        if loadExistingWeights:
            model.load_state_dict(torch.load(config.model_path))

        """load data"""
        print('==> Preparing data..')
        if config.dataset_name == 'CIFAR10':
            output_features = 10
            train_loader, test_loader = datasets.load_cifar10_dataset(batch_size=config.batch_size)
        elif config.dataset_name == 'CIFAR100':
            output_features = 100
            train_loader, test_loader = datasets.load_cifar100_dataset(batch_size=config.batch_size)
        elif config.dataset_name == 'CDON':
            raise NotImplementedError
        else:
            raise NotImplementedError

        """training algorithm"""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum,
                              weight_decay=config.weight_decay)
        scheduler = trainer_config['scheduler'](optimizer, **trainer_config['scheduler_params'])

        """train model"""
        train_loss_per_epoch, test_loss_per_epoch, correct_per_epoch, memorized_per_epoch, incorrect_per_epoch = train(
            model, criterion, optimizer, n_epochs=config.n_epochs, train_loader=train_loader, test_loader=test_loader,
            scheduler=scheduler, config=config)

        """Plot learning curve and accuracy"""
        print(f'acc={correct_per_epoch[-1]}, memorized={memorized_per_epoch[-1]}')
        plot_title = f'{config.dataset_name}, noise_level={config.noise_rate}'
        plot_learning_curve_and_acc(train_loss_per_epoch, test_loss_per_epoch,
                                    correct_per_epoch, memorized_per_epoch, incorrect_per_epoch, plot_title)
        Path("./models").mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), config.model_path)


def main():
    wandb.login()

    config = dict(
        n_epochs=1,
        batch_size=128,
        classes=10,
        noise_rate=0.2,
        dataset_name='CIFAR10',  # opt: 'CIFAR10', 'CIFAR100', 'CDON' (not implemented)
        model_path='./models/CIFAR10_noise_level_10.mdl',
        learning_rate=0.02,
        momentum=0.9,
        weight_decay=1e-3,
        milestones=[40, 80],
        gamma=0.01,
        enable_amp=True
    )

    trainer_config = {
        'model': ResNet34,
        'optimizer': optim.SGD,
        'optimizer_params': {'lr': config['learning_rate'], 'momentum': config['momentum'],
                             'weight_decay': config['weight_decay']},
        'scheduler': optim.lr_scheduler.MultiStepLR,
        'scheduler_params': {'milestones': config['milestones'], 'gamma': config['gamma']}
    }

    # use_CosAnneal = {
    #     'scheduler': optim.lr_scheduler.CosineAnnealingLR,
    #     'scheduler_params': {'T_max': 200, 'eta_min': 0.001}
    # }
    # trainer_config.update(use_CosAnneal)

    model_pipeline(config, trainer_config, loadExistingWeights=False)


if __name__ == '__main__':
    main()
