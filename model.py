import torch
import torch.nn as nn
import torch.nn.functional as F
from common import *




def SelectModel(m):
    
    if m == 'Mobilenet':
        return MobileNet()
    elif m == 'Mobilenet0.5':
        return MobileNet(alpha=0.5)
    elif m == 'Mobilenet0.25':
        return MobileNet(alpha=0.25)



class MobileNet(nn.Module):
    def __init__(self, alpha=1, num_classes=10):
        super(MobileNet, self).__init__()
        self.alpha = alpha
        self.conv1 = Conv(3, 32, stride=2)
        self.conv2 = DepthConv(32, 64, alpha=self.alpha)
        self.conv3 = DepthConv(64, 128, stride=2, alpha=self.alpha)
        self.conv4 = DepthConv(128, 128, alpha=self.alpha)
        self.conv5 = DepthConv(128, 256, stride=2, alpha=self.alpha)
        self.conv6 = DepthConv(256, 256, alpha=self.alpha)
        self.conv7 = DepthConv(256, 512, stride=2, alpha=self.alpha)
        self.conv8 = DepthConv(512, 512, alpha=self.alpha)
        self.conv9 = DepthConv(512, 1024, stride=2, alpha=self.alpha)
        self.conv10 = DepthConv(1024, 1024, alpha=self.alpha)
        # self.Avg = nn.AvgPool2d(kernel_size=7, stride=1)
        self.linear = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        o = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        o = self.conv7(self.conv6(self.conv5(o)))
        o = self.conv8(self.conv8(self.conv8(self.conv8(self.conv8(o)))))
        o = self.conv10(self.conv9(o))
        o = F.avg_pool2d(o, o.size()[3])
        # o = self.Avg(o)
        o = o.view(-1, 1024)
        o = self.linear(o)
        return self.softmax(o)
