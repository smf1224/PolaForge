import math
from einops import repeat
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.layernorm import RMSNorm
from .CARB import CARB
from .SAGM import SAGM
from mamba_ssm import Mamba


class Mamba2D(nn.Module):
    def __init__(self, mamba_module, norm_module):
        super().__init__()
        self.mamba = mamba_module
        self.norm = norm_module

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm(x_flat)
        x_out = self.mamba(x_norm)
        return x_out.transpose(1, 2).reshape(B, C, H, W)


class LRFE(nn.Module):
    def __init__(
        self,
        embed_dims,
        use_rms_norm=False,
        with_dwconv=False,
        drop_path_rate=0.1,
        mamba_cfg=None,
    ):
        super(LRFE, self).__init__()
        mamba_cfg = mamba_cfg or {}
        mamba_cfg.update({'d_model': embed_dims})
        norm_layer = RMSNorm(embed_dims) if use_rms_norm else nn.LayerNorm(embed_dims)
        self.mamba2d = Mamba2D(Mamba(**mamba_cfg), norm_layer)
        self.with_dwconv = with_dwconv
        if self.with_dwconv:
            self.dw = nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1, bias=False, groups=embed_dims),
                nn.BatchNorm2d(embed_dims),
                nn.GELU(),
            )
        self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path_rate))
        self.CARB_C = CARB(embed_dims)
        self.SAGM = SAGM(embed_dims, embed_dims // 2)
        self.linear = nn.Linear(embed_dims, embed_dims, bias=True)
        self.GN = nn.GroupNorm(num_channels=embed_dims, num_groups=embed_dims // 16)

    def forward(self, x):
        B, C, H, W = x.shape
        for _ in range(2):
            x = self.CARB_C(x)
        x_mamba = self.drop_path(self.mamba2d(x))
        x = self.SAGM(x, x_mamba)
        x = self.GN(x)
        if self.with_dwconv:
            x = self.CARB_C(x)
        res = self.linear(self.GN(x).flatten(2).transpose(1, 2))
        res = res.transpose(1, 2).reshape(B, C, H, W)
        return x + res
