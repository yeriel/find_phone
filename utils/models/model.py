import torch
from torch import nn

from .modules import (VggBlock,
                      DoubleResnet,
                      DownSampling)


class NET(nn.Module):
    def __init__(self, channels_in, channels, num_classes):
        super().__init__()

        self.conv1 = VggBlock(channels_in, channels)
        self.drop1 = nn.Dropout2d(0.4)
        self.down_conv1 = DownSampling(channels, 2*channels)

        self.conv2 = DoubleResnet(2*channels)
        self.drop2 = nn.Dropout2d(0.4)
        self.down_conv2 = DownSampling(2*channels, 4*channels)

        self.conv3 = DoubleResnet(4*channels)
        self.drop3 = nn.Dropout2d(0.2)
        self.down_conv3 = DownSampling(4*channels, 8*channels)

        self.conv4 = DoubleResnet(8*channels)
        self.drop4 = nn.Dropout2d(0.2)
        self.down_conv4 = DownSampling(8*channels, 16*channels)

        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(16*channels, num_classes,
                            kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.drop1(x)
        x = self.down_conv1(x)

        x = self.conv2(x)
        x = self.drop2(x)
        x = self.down_conv2(x)

        x = self.conv3(x)
        x = self.drop2(x)
        x = self.down_conv3(x)

        x = self.conv4(x)
        x = self.drop4(x)
        x = self.down_conv4(x)

        x = self.global_avg_pooling(x)
        x = self.fc(x)

        x = torch.sigmoid(x.squeeze())

        return x
