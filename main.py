# main file for training
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from tqdm import tqdm

from model import ResNet18
from noise import Noise
import datasets

from pathlib import Path


# set global env variable
if torch.cuda.is_available():
    print('GPU is enabled!')
    device = 'cuda'
else:
    print('No GPU!')
    device = 'cpu'


def train(model, criterion, optimizer, n_epochs, train_loader, test_loader=None, scheduler=None, noise_rate=0.0):
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
                correct, incorrect, memorized, total = 0, 0, 0, 0
                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(test_loader):
                        original_targets = targets.to(device)
                        targets_with_noise = test_noise_generator.symmetric_noise(targets, batch_idx)
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
                        test_loss += loss.item()

                test_loss_per_epoch.append(test_loss)
                correct_per_epoch.append(correct / total)
                memorized_per_epoch.append(memorized / total)
                incorrect_per_epoch.append(incorrect / total)

        # anneal learning rate
        scheduler.step()

    return (train_loss_per_epoch, test_loss_per_epoch,
            correct_per_epoch, memorized_per_epoch, incorrect_per_epoch,)


def plot_learning_curve_and_acc(train_cost, test_cost, test_correct, test_memorized, test_incorrect, title):
    # plot learning curve
    plt.plot(train_cost)
    plt.plot(test_cost)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'])
    plt.show()

    plt.plot(test_correct)
    plt.plot(test_memorized)
    plt.plot(test_incorrect)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Fraction of examples')
    plt.legend(['Accuracy', 'Memorized', 'Incorrect'])
    plt.show()


def train_CIFAR(CIFAR10=True, n_epochs=100, noise_rate=0.0, model_path='./model/CIFAR.mdl'):
    output_features = 10 if CIFAR10 else 100
    dataset_name = 'CIFAR10' if CIFAR10 else 'CIFAR100'
    layers_in_each_block_list = [2, 2, 2, 2]
    model = ResNet18(layers_in_each_block_list, output_features).to(device)

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
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0.001)

    (train_loss_per_epoch, test_loss_per_epoch,
     correct_per_epoch, memorized_per_epoch, incorrect_per_epoch) = train(
        model, criterion, optimizer, n_epochs=n_epochs,
        train_loader=train_loader, test_loader=test_loader, scheduler=scheduler,
        noise_rate=noise_rate)

    """Plot learning curve and accuracy"""
    print(f'acc={correct_per_epoch[-1]}, memorized={memorized_per_epoch[-1]}')
    plot_title = f'{dataset_name}, noise_level={noise_rate}'
    plot_learning_curve_and_acc(train_loss_per_epoch, test_loss_per_epoch,
                                correct_per_epoch, memorized_per_epoch, incorrect_per_epoch, plot_title)
    Path("./models").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)


def main():
    train_CIFAR(CIFAR10=True, n_epochs=100, noise_rate=0, model_path='./models/CIFAR10_noise_level_0.mdl')
    train_CIFAR(CIFAR10=True, n_epochs=100, noise_rate=0.1, model_path='./models/CIFAR10_noise_level_10.mdl')
    # train_CIFAR(CIFAR10=False, n_epochs=50, noise_rate=0, model_path='./models/CIFAR100_noise_level_0.mdl')
    # train_CIFAR(CIFAR10=False, n_epochs=50, noise_rate=0.1, model_path='./models/CIFAR100_noise_level_10.mdl')


if __name__ == '__main__':
    main()
