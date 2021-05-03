import torch
import torch.nn as nn
import torch.nn.functional as F
from conv_block import ConvBlock


class ResNet18(nn.Module):
    def __init__(self, layers_in_each_block_list):
        super().__init__()
        self.in_channels = 64
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=3)
        self.batch_norm_1 = nn.BatchNorm2d(64)
        self.residual_block_1 = self.add_block_layer(out_channels=64,
                                                     n_layers=layers_in_each_block_list[0],
                                                     stride=1)

    def add_block_layer(self, n_layers, stride, out_channels):
        # stride_for_each_layer_list = [1,1]
        stride_for_each_layer_list = [stride] + [1] * (n_layers-1)
        layers = []
        for stride in stride_for_each_layer_list:
            layers.append(ConvBlock(self.in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
            self.in_channels = out_channels

        return nn.Sequential(*layers)
