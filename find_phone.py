import cv2
import torch
import torchvision.transforms as transforms

from utils.models.model import NET
from utils.tools import parse_command_line_args


def main():
    args = parse_command_line_args()

    data_folder_path = args.data_folder

    try:
        img = cv2.imread(data_folder_path, cv2.COLOR_BGR2RGB)

        if img is None:
            raise Exception(
                f"Error: Unable to open image at path {data_folder_path}")

        H, W, _ = img.shape
        xscale, yscale = 480/W, 320/H

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((320, 480)),
            transforms.ToTensor(),
        ])

        img = transform(img).unsqueeze(0)

        model = NET(channels_in=3, channels=32, num_classes=2)
        model.load_state_dict(torch.load('weights/best.pth'))
        model.eval()

        with torch.no_grad():
            output = model(img)

        x, y = output.numpy()
        print(f'{round(x/xscale, 5)} {round(y/yscale, 5)}')

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
