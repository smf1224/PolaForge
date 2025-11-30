import torch
from torch import nn
import torch.nn.functional as F

class GroupConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=4):
        super(GroupConv, self).__init__()
        self.group_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding, groups=groups, bias=False)

    def forward(self, x):
        return self.group_conv(x)


class RGBConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=False):
        super(RGBConv, self).__init__()
        self.groups = 4
        self.reduce = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=bias)
        self.group_conv = GroupConv(mid_channels, mid_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding, groups=self.groups)
        self.expand = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.reduce(x)
        x = self.group_conv(x)
        x = self.expand(x)
        return x
