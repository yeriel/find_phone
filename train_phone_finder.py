import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from utils.trainer import train
from utils.models.model import NET
from utils.dataset import DatasetPhone
from utils.tools import parse_command_line_args, check_folder_contents, count_module_parameters

import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import OneCycleLR


EPOCHS = 20
BATCH_SIZE = 2
VAL_SPLIT = 0.2
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloaders(path):
    data_transform = T.Compose([
        T.RandomApply([
            T.Lambda(lambda x: x + 0.1 * torch.randn_like(x)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ColorJitter(saturation=0.2, hue=0.2),
            T.ElasticTransform(alpha=250.0),
            T.RandomErasing()
        ], p=0.5)
    ])

    dataset = DatasetPhone(path, transform=data_transform)
    dataset_size = len(dataset)

    val_size = int(VAL_SPLIT * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataloader, val_dataloader


def plot(train, val, name):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(train))+1, train, label=f'Train {name}')
    plt.plot(np.arange(len(val))+1, val, label=f'Validation {name}')
    plt.title(f'Training and Validation {name}')
    plt.xlabel('Epochs')
    plt.ylabel(f'{name}')
    plt.legend()
    plt.savefig(f'plots/{name}_plot.png')
    plt.clf()


def main():
    args = parse_command_line_args()

    data_folder_path = args.data_folder

    if check_folder_contents(data_folder_path):

        model = NET(channels_in=3, channels=32, num_classes=2)
        print(f'# Parameters {count_module_parameters(model)}')
        model.to(DEVICE)

        train_dataloader, val_dataloader = get_dataloaders(data_folder_path)

        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = OneCycleLR(optimizer, max_lr=LR,
                               total_steps=len(train_dataloader)*EPOCHS)

        history_train_loss, history_train_mae, history_train_mse, history_val_loss, history_val_mae, history_val_mse = train(EPOCHS,
                                                                                                                             model,
                                                                                                                             train_dataloader,
                                                                                                                             val_dataloader,
                                                                                                                             optimizer,
                                                                                                                             DEVICE,
                                                                                                                             scheduler=scheduler)

        plot(history_train_loss, history_val_loss, name='Loss')
        plot(history_train_mae, history_val_mae, name='MAE')
        plot(history_train_mse, history_val_mse, name='MSE')

    else:
        print('Please provide a valid data_folder that meets the required format for training.')


if __name__ == "__main__":
    main()
