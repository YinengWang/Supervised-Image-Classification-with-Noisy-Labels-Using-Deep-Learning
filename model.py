import torch
import torch.nn as nn
import torch.nn.functional as F
from conv_block import ConvBlock


class ResNet18(nn.Module):
    def __init__(self, num_classes, layers_in_each_block_list = [2, 2, 2, 2]):
        super().__init__()
        layers_in_each_block_list = layers_in_each_block_list
        self.in_channels = 64
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(64)

        self.residual_block_1 = self.add_block_layer(out_channels=64,
                                                     n_layers=layers_in_each_block_list[0],
                                                     stride=1)
        self.residual_block_2 = self.add_block_layer(out_channels=128,
                                                     n_layers=layers_in_each_block_list[1],
                                                     stride=2)
        self.residual_block_3 = self.add_block_layer(out_channels=256,
                                                     n_layers=layers_in_each_block_list[2],
                                                     stride=2)
        self.residual_block_4 = self.add_block_layer(out_channels=512,
                                                     n_layers=layers_in_each_block_list[3],
                                                     stride=2)
        self.output = nn.Linear(in_features=512, out_features=num_classes)

    def add_block_layer(self, n_layers, stride, out_channels):
        # stride_for_each_layer_list = [stride] concatonated with [1, 1, .....]
        stride_for_each_layer_list = [stride] + [1] * (n_layers-1)
        layers = []
        for stride in stride_for_each_layer_list:
            layers.append(ConvBlock(self.in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        # input shape torch.Size([128, 3, 32, 32])
        # after relu shape torch.Size([128, 64, 32, 32])
        # after residual_block_1 torch.Size([128, 64, 32, 32])
        # after residual_block_2 torch.Size([128, 128, 16, 16])
        # after residual_block_3 torch.Size([128, 256, 8, 8])
        # after residual_block_4 torch.Size([128, 512, 4, 4])
        # after ave pool torch.Size([128, 512, 1, 1])
        # after reshape torch.Size([128, 512])

        out = self.conv_1(x)
        out = self.batch_norm_1(out)
        out = F.relu(out)

        # begin residual blocks 4 * 4
        out = self.residual_block_1(out)
        out = self.residual_block_2(out)
        out = self.residual_block_3(out)
        out = self.residual_block_4(out)

        # ave pool 1*1
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.output(out)

        return out



class ResNet34(ResNet18):
    def __init__(self, num_classes):
        ResNet18.__init__(self, num_classes, layers_in_each_block_list=[3, 4, 6, 3])
