import torch
import torch.nn.functional as F
from torch import nn
from .SAGM import SAGM
from einops.einops import rearrange
from .RGBC import RGBConv


class SACM(nn.Module):
    def __init__(self):
        super(SACM, self).__init__()
        self.reduce_mamba_3 = RGBConv(1024, 512, 64, 3, padding=1)
        self.reduce_feat_3  = RGBConv(1024, 512, 64, 3, padding=1)
        self.reduce_mamba_2 = RGBConv(512, 256, 32, 3, padding=1)
        self.reduce_out1    = RGBConv(512, 256, 32, 3, padding=1)
        self.reduce_feat_1  = RGBConv(256, 128, 32, 3, padding=1)
        self.reduce_out2    = RGBConv(256, 128, 32, 3, padding=1)
        self.reduce_feat_0  = RGBConv(128, 64, 16, 3, padding=1)
        self.reduce_out3    = RGBConv(128, 64, 16, 3, padding=1)
        self.fuse_level_3 = RGBConv(1024, 512, 64, 3, padding=1)
        self.fuse_level_2 = RGBConv(512, 256, 32, 3, padding=1)
        self.fuse_level_1 = RGBConv(256, 128, 32, 3, padding=1)
        self.fuse_level_0 = RGBConv(128, 64, 16, 3, padding=1)
        self.out_conv = RGBConv(64, 1, 32, 1)
        self.gn512 = GN(32, 512)
        self.gn256 = GN(16, 256)
        self.gn128 = GN(8, 128)
        self.gn64  = GN(4, 64)
        self.relu = nn.ReLU(inplace=True)
        self.SAGM_level3 = SAGM(512, 512)
        self.SAGM_level2 = SAGM(256, 256)
        self.SAGM_level1 = SAGM(128, 128)
        self.SAGM_level0 = SAGM(64, 64)

    def forward(self, features, mamba_out):
        out_t = self.reduce_mamba_3(mamba_out[3])
        out_t = F.interpolate(out_t, size=(32, 32), mode="bilinear", align_corners=True)
        feat1 = self.reduce_feat_3(features[3].tensors)
        feat1 = F.interpolate(feat1, size=(32, 32), mode="bilinear", align_corners=True)
        fused1 = self.SAGM_level3(feat1, out_t)
        concat3 = torch.cat([feat1, out_t], dim=1)
        out1 = self.relu(self.gn512(self.fuse_level_3(concat3)))
        out1 = out1 + fused1
        out1 = F.interpolate(out1, size=(64, 64), mode="bilinear")
        feat2 = self.reduce_mamba_2(mamba_out[2])
        feat2 = F.interpolate(feat2, size=(64, 64), mode="bilinear", align_corners=True)
        out1_reduced = self.reduce_out1(out1)
        fused2 = self.SAGM_level2(feat2, out1_reduced)
        concat2 = torch.cat([feat2, out1_reduced], dim=1)
        out2 = self.relu(self.gn256(self.fuse_level_2(concat2)))
        out2 = out2 + fused2
        out2 = F.interpolate(out2, size=(128, 128), mode="bilinear")
        feat3 = self.reduce_feat_1(mamba_out[1])
        feat3 = F.interpolate(feat3, size=(128, 128), mode="bilinear", align_corners=True)
        out2_reduced = self.reduce_out2(out2)
        fused3 = self.SAGM_level1(feat3, out2_reduced)
        concat1 = torch.cat([feat3, out2_reduced], dim=1)
        out3 = self.relu(self.gn128(self.fuse_level_1(concat1)))
        out3 = out3 + fused3
        out3 = F.interpolate(out3, size=(256, 256), mode="bilinear")
        feat4 = self.reduce_feat_0(mamba_out[0])
        feat4 = F.interpolate(feat4, size=(256, 256), mode="bilinear", align_corners=True)
        out3_reduced = self.reduce_out3(out3)
        fused4 = self.SAGM_level0(feat4, out3_reduced)
        concat0 = torch.cat([feat4, out3_reduced], dim=1)
        out4 = self.relu(self.gn64(self.fuse_level_0(concat0)))
        out4 = out4 + fused4
        out4 = F.interpolate(out4, size=(512, 512), mode="bilinear")
        out = self.out_conv(out4)
        return out


class GN(nn.Module):
    def __init__(self, groups: int, channels: int,
                 eps: float = 1e-5, affine: bool = True):
        super(GN, self).__init__()
        assert channels % groups == 0, 'channels should be evenly divisible by groups'
        self.groups = groups
        self.channels = channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.scale = nn.Parameter(torch.ones(channels))
            self.shift = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor):
        x_shape = x.shape
        batch_size = x_shape[0]
        assert self.channels == x.shape[1]
        x = x.view(batch_size, self.groups, -1)
        mean = x.mean(dim=[-1], keepdim=True)
        mean_x2 = (x ** 2).mean(dim=[-1], keepdim=True)
        var = mean_x2 - mean ** 2
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x_norm = x_norm.view(batch_size, self.channels, -1)
            x_norm = self.scale.view(1, -1, 1) * x_norm + self.shift.view(1, -1, 1)
        return x_norm.view(x_shape)
