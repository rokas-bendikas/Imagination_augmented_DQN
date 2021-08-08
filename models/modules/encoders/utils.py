import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
        DoubleConv(in_channels,out_channels),
        nn.MaxPool2d(kernel_size=2,stride=2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
