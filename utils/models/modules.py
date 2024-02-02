import torch
from torch import nn


class BnActConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(BnActConv, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        x = self.conv(x)
        return x


class VggBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(VggBlock, self).__init__()

        self.convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.convolution(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x


class ShakeBlock(nn.Module):
    def __init__(self, in_channels, f=2, k=3):
        super(ShakeBlock, self).__init__()
        mid_channels = int(in_channels * f)

        self.conv1 = BnActConv(in_channels, mid_channels,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = BnActConv(mid_channels, mid_channels,
                               kernel_size=k, stride=1, padding=k//2, bias=False)
        self.conv3 = BnActConv(mid_channels, in_channels,
                               kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, alpha):
        x0 = x
        xa = self.conv1(x0)
        xa = self.conv2(xa)
        xa = self.conv3(xa)

        return x0 + alpha * xa


class ResnetV2Shake(nn.Module):
    def __init__(self, in_channels, f=2, k=3):
        super(ResnetV2Shake, self).__init__()

        self.blockA = ShakeBlock(in_channels, f, k)
        self.blockB = ShakeBlock(in_channels, f, k)

    def forward(self, x):
        alpha = 0.5

        x_blockA = self.blockA(x, alpha)
        x_blockB = self.blockB(x, alpha)

        if self.training:
            alpha = torch.rand(()).item()
            return x + alpha * x_blockA + (1 - alpha) * x_blockB
        return x + alpha * (x_blockA + x_blockB)


class DoubleResnet(nn.Module):
    def __init__(self, in_channels):
        super(DoubleResnet, self).__init__()

        self.conv1 = ResnetV2Shake(in_channels)
        self.conv2 = ResnetV2Shake(in_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True, ):
        super(DownSampling, self).__init__()

        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.activation = nn.ReLU(inplace=False)
        self.convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.convolution(x)
        return x
