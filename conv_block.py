import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
    # padding_mode='zeros')
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        in_channels_for_batch_norm = out_channels
        self.batch_norm_1 = nn.BatchNorm2d(in_channels_for_batch_norm)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                                          nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv_1(x)
        out = self.batch_norm_1(out)
        out = F.relu(out)

        out = self.conv_2(out)
        out = self.batch_norm_2(out)

        out += self.shortcut(x)

        out = F.relu(out)
        return out
