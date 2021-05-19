# main file for training
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

from loss import ELR_loss
from model import ResNet18
from model import ResNet34
import datasets

import os.path
from pathlib import Path
import wandb
import csv

# set global env variable
if torch.cuda.is_available():
    print('GPU is enabled!')
    print('Using ' + torch.cuda.get_device_name(0))
    device = 'cuda'
else:
    print('No GPU!')
    device = 'cpu'


def train(model, criterion, optimizer, train_loader, test_loader=None, scheduler=None, config=None):
    # tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log='all', log_freq=100)

    train_loss_per_epoch = []
    test_loss_per_epoch = []
    accuracy_per_epoch = []
    if config.compute_memorization: # keep parameters for memorization
        correct_clean_per_epoch = []
        incorrect_clean_per_epoch = []
        correct_wrong_per_epoch = []
        incorrect_wrong_per_epoch = []
        memorized_wrong_per_epoch = []

    example_ct = 0  # number of examples seen
    batch_ct = 0
    test_criterion = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(config.n_epochs)):
        # activate train mode
        model.train()
        train_loss = 0
        total_clean, total_wrong = 0, 0
        correct_in_clean, incorrect_in_clean = 0, 0
        correct_in_wrong, memorized_in_wrong, incorrect_in_wrong = 0, 0, 0
        for batch_idx, sample in enumerate(train_loader):
            inputs, targets = sample[0].to(device), sample[1].to(device)
            optimizer.zero_grad()
            inds = None

            if config.enable_amp:
                scaler = GradScaler()
                with autocast():
                    outputs = model(inputs)

                    if config['use_ELR']:
                        inds = train_loader.get_indices_in_batch(batch_idx)
                        loss = criterion(inds, outputs, targets)
                    else:
                        loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)

                if config['use_ELR']:
                    inds = train_loader.get_indices_in_batch(batch_idx)
                    loss = criterion(inds, outputs, targets)
                else:
                    loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

            # Compute correct, incorrect, memorized
            if config.compute_memorization:
                original_targets = sample[2].to(device)
                with torch.no_grad():
                    _, predicted = outputs.max(1)

                    is_clean = (targets == original_targets)
                    is_wrong = ~is_clean
                    total_clean += is_clean.sum().item()
                    total_wrong += is_wrong.sum().item()
                    is_correct = (predicted == original_targets)
                    correct_in_clean += (is_clean & is_correct).sum().item()
                    incorrect_in_clean += (is_clean & ~is_correct).sum().item()
                    if config.noise_rate != 0.0:
                        is_memorized = (predicted == targets)
                        correct_in_wrong += (is_wrong & is_correct).sum().item()
                        memorized_in_wrong += (is_wrong & is_memorized).sum().item()
                        incorrect_in_wrong += (is_wrong & ~is_correct & ~is_memorized).sum().item()

            train_loss += loss.item() * targets.size(0)
            loss_batch = loss.item()

            example_ct += len(inputs)
            batch_ct += 1

            # log loss on WandB every 25 steps
            # if ((batch_ct + 1) % 25) == 0:
            # loss_for_wand_b = float(train_loss/batch_ct)
            wandb.log({"epoch": epoch, "loss": loss_batch}, step=batch_ct)
            # print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss_for_wand_b:.3f}")
        train_loss_per_epoch.append(train_loss / len(train_loader.dataset))#len(train_loader))
        if config.compute_memorization:
            correct_in_clean /= total_clean
            incorrect_in_clean /= total_clean
            if config.noise_rate != 0.0:
                correct_in_wrong /= total_wrong
                memorized_in_wrong /= total_wrong
                incorrect_in_wrong /= total_wrong
            correct_clean_per_epoch.append(correct_in_clean)
            incorrect_clean_per_epoch.append(incorrect_in_clean)
            correct_wrong_per_epoch.append(correct_in_wrong)
            memorized_wrong_per_epoch.append(memorized_in_wrong)
            incorrect_wrong_per_epoch.append(incorrect_in_wrong)
            wandb.log({"correct_in_clean_labels": correct_in_clean, "incorrect_in_clean_labels": incorrect_in_clean,
                       "correct_in_wrong_labels": correct_in_wrong, "memorized_in_wrong_labels": memorized_in_wrong,
                       "incorrect_in_wrong_labels": incorrect_in_wrong}, step=batch_ct)

        if test_loader is not None:
            model.eval()
            test_loss = 0
            with torch.no_grad():
                correct, total = 0, 0
                for batch_idx, sample in enumerate(test_loader):
                    inputs, targets = sample[0].to(device), sample[1].to(device)
                    original_targets = sample[2].to(device) if config.compute_memorization else targets
                    outputs = model(inputs)
                    loss = test_criterion(outputs, targets)

                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(original_targets).sum().item()
                    test_loss += loss.item() * targets.size(0)

                test_loss_per_epoch.append(test_loss / total)
                accuracy_per_epoch.append(correct / total)

                wandb.log({"test_loss": test_loss / total, "test_accuracy": correct / total}, step=batch_ct)
                print(f"Test loss after " + str(example_ct).zfill(5) + f" examples: {test_loss / total:.3f}")
                torch.onnx.export(model, inputs, "model.onnx")
                wandb.save("model.onnx")

        # anneal learning rate
        scheduler.step()

    if config.compute_memorization:
        return (train_loss_per_epoch, test_loss_per_epoch, accuracy_per_epoch,
                correct_clean_per_epoch, incorrect_clean_per_epoch,
                correct_wrong_per_epoch, memorized_wrong_per_epoch, incorrect_wrong_per_epoch)
    return (train_loss_per_epoch, test_loss_per_epoch, accuracy_per_epoch)

def plot_learning_curve_and_acc(results, title, path_prefix):
    Path("./results").mkdir(parents=True, exist_ok=True)
    train_cost, test_cost, test_accuracy = results[:3]
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

    # plot accuracy
    plt.plot(epochs, test_accuracy)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('%')
    plt.legend(['Accuracy'])
    plt.savefig(path_prefix + '_acc.pdf')
    plt.show()

    if len(results) > 3:
        # we have more information about noise. Add that
        correct_clean, incorrect_clean, correct_wrong, memorized_wrong, incorrect_wrong = results[-5:]
        # plot fraction of correct, incorrect, memorized samples in wrong labels
        plt.plot(epochs, correct_wrong)
        plt.plot(epochs, memorized_wrong)
        plt.plot(epochs, incorrect_wrong)
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Fraction of examples')
        plt.legend(['Correct', 'Memorized', 'Incorrect'])
        plt.savefig(path_prefix + '_wrong.pdf')
        plt.show()

        # plot fraction of correct and incorrect samples in clean labels
        plt.plot(epochs, correct_clean)
        plt.plot(epochs, incorrect_clean)
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Fraction of examples')
        plt.legend(['Correct', 'Incorrect'])
        plt.savefig(path_prefix + '_clean.pdf')
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
    wandb_project = 'resnet-ce-cdon'
    wandb_entity = 'dd2424-group9'
    with wandb.init(project=wandb_project, entity=wandb_entity, config=config):
        # access all hyperparameters through wandb.config, so logging matches execution!
        config = wandb.config

        """create the model"""
        model = trainer_config['model'](config['classes']).to(device)

        if loadExistingWeights:
            model.load_state_dict(torch.load(config.model_path))

        """load data"""
        print('==> Preparing data..')
        if config.dataset_name.startswith('CIFAR'):
            train_loader, test_loader = datasets.load_cifar_dataset(config.dataset_name,
                                                                    batch_size=config.batch_size,
                                                                    noise_rate=config.noise_rate,
                                                                    fraction=config.fraction)
        elif config.dataset_name == 'CDON':
            train_loader, test_loader = datasets.load_cdon_dataset(config.batch_size)
        else:
            raise NotImplementedError

        """training algorithm"""
        if config['use_ELR']:
            print('--Using ELR--')
            criterion = trainer_config['criterion'](len(train_loader.dataset), n_classes=config['classes'], **trainer_config['criterion_params'])
        else:
            print('--Using CE loss--')
            criterion = trainer_config['criterion'](**trainer_config['criterion_params'])

        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum,
                              weight_decay=config.weight_decay)
        scheduler = trainer_config['scheduler'](optimizer, **trainer_config['scheduler_params'])

        """train model"""
        results = train(
            model, criterion, optimizer, train_loader=train_loader, test_loader=test_loader,
            scheduler=scheduler, config=config)

        """Plot learning curve and accuracy"""
        plot_title = f'{config.dataset_name}, noise_level={config.noise_rate}'
        plot_learning_curve_and_acc(results, plot_title, config.plot_path)
        Path("./models").mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), config.model_path)


def main():
    wandb.login()

    # CIFAR use this
    config = dict(
        n_epochs=120,
        batch_size=128,
        classes=10,
        noise_rate=0.4,
        is_symmetric_noise=True,
        fraction=1.0,
        compute_memorization=True,
        dataset_name='CIFAR10',  # opt: 'CIFAR10', 'CIFAR100', 'CDON' (not implemented)
        model_path='./models/CIFAR10_20.mdl',
        plot_path='./results/CIFAR10_20',
        learning_rate=0.02,
        momentum=0.9,
        weight_decay=1e-3,
        milestones=[40, 80],
        gamma=0.01,
        enable_amp=True,
        use_ELR=True,
        elr_lambda=3.0,
        elr_beta=0.7
    )

    # CDON use this
    config = dict(
        n_epochs=120,
        batch_size=128,
        classes=64, #157 categories for clothing # total subcategories is 3516
        noise_rate=0.0,
        is_symmetric_noise=True,
        fraction=1.0,
        compute_memorization=False,
        dataset_name='CDON',  # opt: 'CIFAR10', 'CIFAR100', 'CDON'
        model_path='./models/CDON_CE.mdl',
        plot_path='./results/CDON_CE',
        learning_rate=0.02,
        momentum=0.9,
        weight_decay=1e-3,
        milestones=[40, 80],
        gamma=0.01,
        enable_amp=True,
        use_ELR=False,
        elr_lambda=3.0,
        elr_beta=0.7
    )

    trainer_config = {
        'model': ResNet34,
        'optimizer': optim.SGD,
        'optimizer_params': {'lr': config['learning_rate'], 'momentum': config['momentum'],
                             'weight_decay': config['weight_decay']},
        'scheduler': optim.lr_scheduler.MultiStepLR,
        'scheduler_params': {'milestones': config['milestones'], 'gamma': config['gamma']},
        'criterion': torch.nn.CrossEntropyLoss,
        'criterion_params': {}
    }

    # use_CosAnneal = {
    #     'scheduler': optim.lr_scheduler.CosineAnnealingWarmRestarts,
    #     'scheduler_params': {"T_0": 10, "eta_min": 0.001},
    #     # 'scheduler_params': {'T_max': 200, 'eta_min': 0.001}
    # }
    # trainer_config.update(use_CosAnneal)

    if config['use_ELR']:
        use_ELR = {
            'criterion': ELR_loss,
            'criterion_params': {'beta': config['elr_beta'], 'lam': config['elr_lambda']}
        }
        trainer_config.update(use_ELR)

    model_pipeline(config, trainer_config, loadExistingWeights=False)


if __name__ == '__main__':
    main()
