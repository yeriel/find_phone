import torch
import random
import torch.optim as optim
import torchvision.transforms as T

from utils.trainer import train
from utils.models.model import NET
from utils.dataset import DatasetPhone
from utils.tools import parse_command_line_args, check_folder_contents

from torch.utils.data import DataLoader
from torch.utils.data import random_split

EPOCHS = 10
BATCH_SIZE = 2
VAL_SPLIT = 0.2
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_datloaders(path):
    data_transform = T.Compose([
        T.RandomApply([
            T.Lambda(lambda x: x + random.uniform(0.1, 0.4)
                     * torch.randn_like(x)),
            T.ColorJitter(saturation=0.2, hue=0.2),
            T.ColorJitter(brightness=0.2, contrast=0.2)
        ], p=0.5)
    ])

    dataset = DatasetPhone(path, data_transform)
    dataset_size = len(dataset)

    val_size = int(VAL_SPLIT * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataloader, val_dataloader


def main():
    args = parse_command_line_args()

    data_folder_path = args.data_folder

    if check_folder_contents(data_folder_path):

        model = NET(channels_in=3, channels=32, num_classes=2)
        model.to(DEVICE)

        optimizer = optim.Adam(model.parameters(), lr=LR)
        train_dataloader, val_dataloader = get_datloaders(data_folder_path)

        train(EPOCHS,
              model,
              train_dataloader,
              val_dataloader,
              optimizer,
              DEVICE)

    else:
        print('Please provide a valid data_folder that meets the required format for training.')


if __name__ == "__main__":
    main()
