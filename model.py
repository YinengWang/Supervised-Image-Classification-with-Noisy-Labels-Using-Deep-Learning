import torch
import torch.nn as nn
import torch.nn.functional as F
from conv_block import ConvBlock

class ResNet18(nn.Module):
    def __init__(self, block_list):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=3)
        self.batch_norm_1 = nn.BatchNorm2d(64)
