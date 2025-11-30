import torch
import torch.nn as nn


# dual
class DGMF(nn.Module):
    def __init__(self, num_channels, scale_factor=1.0):
        super(DGMF, self).__init__()
        self.scale_factor = scale_factor
        self.gate_conv = nn.Sequential(
            nn.Conv2d(num_channels * 2, num_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, 3, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Softmax(dim=1)
        )
    def forward(self, pol1_features, pol2_features):
        fusion_input = torch.cat([pol1_features, pol2_features], dim=1)
        weights = self.gate_conv(fusion_input) * self.scale_factor
        w1, w2, w3 = weights[:, 0:1, :, :], weights[:, 1:2, :, :], weights[:, 2:3, :, :]
        fused_features = w1 * pol1_features + w2 * pol2_features + w3 * pol1_features * pol2_features
        return fused_features


# tri
# class AMFM(nn.Module):
#     def __init__(self, num_channels, scale_factor=1.0):
#         super(AMFM, self).__init__()
#         self.scale_factor = scale_factor
#         self.gate_conv = nn.Sequential(
#             nn.Conv2d(num_channels * 3, num_channels, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(num_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_channels, 4, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, pol1_features, pol2_features, pol3_features):
#         fusion_input = torch.cat([pol1_features, pol2_features, pol3_features], dim=1)
#         weights = self.gate_conv(fusion_input) * self.scale_factor
#         w1, w2, w3, w4 = weights[:, 0:1, :, :], weights[:, 1:2, :, :], weights[:, 2:3, :, :], weights[:, 3:4, :, :]
#         fused_features = w1 * pol1_features + w2 * pol2_features + w3 * pol3_features + w4 * (
#                     pol1_features * pol2_features * pol3_features)
#         return fused_features

