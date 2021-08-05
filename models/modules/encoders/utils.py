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

"""
class Up(nn.Module):
    #Upscaling then double conv

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels , out_channels, kernel_size=5, stride=2,padding=2,output_padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):

        out = self.up(x)

        return out
"""
