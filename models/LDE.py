import torch.nn as nn
from .LRFE import LRFE

class LDE(nn.Module):
    def __init__(self, transformer=None, num_queries=None, num_feature_levels=None, args=None):
        super().__init__()
        self.savss1 = LRFE(
            embed_dims=128,
            mamba_cfg={
                "d_state": 16,
                "d_conv": 4,
                "expand": 2,
            }
        )
        self.savss2 = LRFE(
            embed_dims=256,
            mamba_cfg={"d_state": 16, "d_conv": 4, "expand": 2}
        )
        self.savss3 = LRFE(
            embed_dims=512,
            mamba_cfg={"d_state": 16, "d_conv": 4, "expand": 2}
        )
        self.savss4 = LRFE(
            embed_dims=1024,
            mamba_cfg={"d_state": 16, "d_conv": 4, "expand": 2}
        )

    def forward(self, features, pos=None):
        c1 = features[0].tensors
        c2 = features[1].tensors
        c3 = features[2].tensors
        c4 = features[3].tensors
        out1 = self.savss1(c1)
        out2 = self.savss2(c2)
        out3 = self.savss3(c3)
        out4 = self.savss4(c4)
        mamba_out = [out1, out2, out3, out4]
        return features, mamba_out


def build_LDE(args):
    return LDE(
        transformer=None,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        args=args
    )
