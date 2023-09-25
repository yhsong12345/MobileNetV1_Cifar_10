import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DepthConv(nn.Module):
    def __init__(self, input, output, stride=1, alpha=1):
        super(DepthConv, self).__init__()
        rinput = int(alpha*input)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input, rinput, kernel_size=3, stride=stride, padding=1, groups=rinput, bias=False),
            nn.BatchNorm2d(rinput),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(rinput, output, kernel_size=1, bias=False),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        o = self.conv1(x)
        o = self.conv2(o)
        return o


class Conv(nn.Module):
    def __init__(self, input, output, stride=1):
        super(Conv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input, output, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True)
        )

    
    def forward(self, x):
        return self.conv1(x)