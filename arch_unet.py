#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, depth=5, wf=48, slope=0.1):
        """
        Args:
            in_channels (int): number of input channels, Default 3
            depth (int): depth of the network, Default 5
            wf (int): number of filters in the first layer, Default 32
        """
        super(UNet, self).__init__()
        self.depth = depth
        self.head = nn.Sequential(
            LR(in_channels, wf, 3, slope), LR(wf, wf, 3, slope))
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(LR(wf, wf, 3, slope))

        self.up_path = nn.ModuleList()
        for i in range(depth):
            if i != depth-1:
                self.up_path.append(UP(wf*2 if i==0 else wf*3, wf*2, slope))
            else:
                self.up_path.append(UP(wf*2+in_channels, wf*2, slope))

        self.last = nn.Sequential(LR(2*wf, 2*wf, 1, slope), 
                    LR(2*wf, 2*wf, 1, slope), conv1x1(2*wf, out_channels, bias=True))

    def forward(self, x):
        blocks = []
        blocks.append(x)
        x = self.head(x)
        for i, down in enumerate(self.down_path):
            x = F.max_pool2d(x, 2)
            if i != len(self.down_path) - 1:
                blocks.append(x)
            x = down(x)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        out = self.last(x)
        return out


class LR(nn.Module):
    def __init__(self, in_size, out_size, ksize=3, slope=0.1):
        super(LR, self).__init__()
        block = []
        block.append(nn.Conv2d(in_size, out_size,
                     kernel_size=ksize, padding=ksize//2, bias=True))
        block.append(nn.LeakyReLU(slope, inplace=True))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UP(nn.Module):
    def __init__(self, in_size, out_size, slope=0.1):
        super(UP, self).__init__()
        self.conv_1 = LR(in_size, out_size)
        self.conv_2 = LR(out_size, out_size)

    def up(self, x):
        s = x.shape
        x = x.reshape(s[0], s[1], s[2], 1, s[3], 1)
        x = x.repeat(1, 1, 1, 2, 1, 2)
        x = x.reshape(s[0], s[1], s[2]*2, s[3]*2)
        return x

    def forward(self, x, pool):
        x = self.up(x)
        x = torch.cat([x, pool], 1)
        x = self.conv_1(x)
        x = self.conv_2(x)

        return x


def conv1x1(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=1,
                      stride=1, padding=0, bias=bias)
    return layer
