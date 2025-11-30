import torch.nn as nn
from .RGBC import RGBConv


class CARB(nn.Module):
    def __init__(self, in_channels, norm_type='GN') -> None:
        super().__init__()
        self.conv1 = RGBConv(in_channels, in_channels, in_channels//8, 3, 1, 1)
        self.norm1 = nn.InstanceNorm3d(in_channels)
        if(norm_type == 'GN'):
            self.norm1 = nn.GroupNorm(num_channels=in_channels, num_groups=in_channels//16)
        self.act1 = nn.ReLU()

        self.conv2 = RGBConv(in_channels, in_channels, in_channels//8, 1, 1, 0)
        self.norm2 = nn.InstanceNorm3d(in_channels)
        if(norm_type == 'GN'):
            self.norm2 = nn.GroupNorm(num_channels=in_channels, num_groups=in_channels//16)
        self.act2 = nn.ReLU()

        self.conv3 = RGBConv(in_channels, in_channels, in_channels//8, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channels)
        if(norm_type == 'GN'):
            self.norm3 = nn.GroupNorm(num_channels=in_channels, num_groups=16)
        self.act3 = nn.ReLU()

    def forward(self, x):
        x_residual = x
        x1_1 = self.conv1(x)
        x1_1 = self.norm1(x1_1)
        x1_1 = self.act1(x1_1)
        x2_1 = self.conv2(x)
        x2_1 = self.norm2(x2_1)
        x2_1 = self.act2(x2_1)
        x2 = x1_1 + x2_1
        x3_1 = self.conv3(x2)
        x3_1 = self.norm3(x3_1)
        x3_1 = self.act3(x3_1)
        return x3_1 + x_residual