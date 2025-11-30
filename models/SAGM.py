import torch
import torch.nn as nn
from .RGBC import RGBConv


class SAGM(nn.Module):
    def __init__(self, in_channels, mid_channels, reduction=8,
                 after_relu=False, with_channel=True,
                 mid_norm=nn.BatchNorm2d, in_norm=nn.BatchNorm2d):
        super(SAGM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.height = 2
        self.f = nn.Sequential(
            RGBConv(in_channels=in_channels, out_channels=mid_channels, mid_channels=16, kernel_size=1),
            mid_norm(mid_channels)
        )
        if with_channel:
            self.up = nn.Sequential(
                RGBConv(in_channels=mid_channels, out_channels=in_channels * self.height, mid_channels=16, kernel_size=1),
                in_norm(in_channels * self.height)
            )
        d = max(int(in_channels / reduction), 4)
        self.global_mlp = nn.Sequential(
            nn.Conv2d(in_channels, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, in_channels * self.height, 1, bias=False)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        if self.after_relu:
            x = self.relu(x)
            y = self.relu(y)

        B, C, H, W = x.shape
        x_feat = self.f(x)
        y_feat = self.f(y)
        sim = x_feat * y_feat
        if self.with_channel:
            sim_map = torch.sigmoid(self.up(sim))
            sim_map = sim_map.view(B, self.height, C, H, W)
        else:
            sim_map = torch.sigmoid(torch.sum(sim, dim=1, keepdim=True))
            sim_map = sim_map.expand(-1, self.height, -1, H, W)

        feats_sum = x + y
        pooled = self.avg_pool(feats_sum)
        attn = self.global_mlp(pooled)
        attn = attn.view(B, self.height, C, 1, 1)
        attn = self.softmax(attn)

        in_feats = torch.stack([x, y], dim=1)
        fused = torch.sum(in_feats * attn, dim=1)
        return fused * sim_map.sum(dim=1)

