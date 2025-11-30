import torch
import torch.nn.functional as F
from torch import nn
from .MFE.build_MSFE import build_msfe
from .LDE import build_LDE
from util.misc import (NestedTensor, nested_tensor_from_tensor_list)
from .SACM import SACM
from .DGMF import DGMF


# sign
# class Decoder(nn.Module):
#     def __init__(self, backbones, args=None):
#         super().__init__()
#         self.args = args
#         self.backbones = nn.ModuleList(backbones)
#         self.LDE = build_LDE(args)
#         self.SACM = SACM()
#         self.map1 = nn.Identity()
#         self.map2 = nn.Identity()
#         self.map3 = nn.Identity()
#         self.map4 = nn.Identity()
#
#     def forward(self, modal_imgs):
#         if isinstance(modal_imgs, (list, tuple)):
#             assert len(modal_imgs) >= 1, "single-modality expected at least one tensor"
#             modal_img = modal_imgs[0]
#         else:
#             modal_img = modal_imgs
#         backbone = self.backbones[0]
#         backbone_out, pos_out = backbone(modal_img)
#         out1 = self.map1(backbone_out['0'].tensors)
#         out2 = self.map2(backbone_out['1'].tensors)
#         out3 = self.map3(backbone_out['2'].tensors)
#         out4 = self.map4(backbone_out['3'].tensors)
#         pos_outs = [pos_out[0], pos_out[1], pos_out[2], pos_out[3]]
#         out1 = NestedTensor(out1, backbone_out['0'].mask)
#         out2 = NestedTensor(out2, backbone_out['1'].mask)
#         out3 = NestedTensor(out3, backbone_out['2'].mask)
#         out4 = NestedTensor(out4, backbone_out['3'].mask)
#         features, trans_out = self.LDE([out1, out2, out3, out4], pos_outs)
#         out = self.SACM(features, trans_out)
#         return out


# dual
class Decoder(nn.Module):

    def __init__(self, backbones, args=None):
        super().__init__()
        self.args = args
        self.backbones = nn.ModuleList(backbones)
        self.LDE = build_LDE(args)
        self.SACM = SACM()
        self.DGMF_1 = DGMF(num_channels=128)
        self.DGMF_2 = DGMF(num_channels=256)
        self.DGMF_3 = DGMF(num_channels=512)
        self.DGMF_4 = DGMF(num_channels=1024)

    def forward(self, modal_imgs):
        pos_outs = []
        backbone_outs = []
        for i in range(len(self.backbones)):
            backbone_out, pos_out = self.backbones[i](modal_imgs[i])
            backbone_outs.append(backbone_out)
            pos_outs.append(pos_out)
        out1_all = [bo['0'].tensors for bo in backbone_outs]
        out2_all = [bo['1'].tensors for bo in backbone_outs]
        out3_all = [bo['2'].tensors for bo in backbone_outs]
        out4_all = [bo['3'].tensors for bo in backbone_outs]
        def fuse_pol1(feat_list): return torch.stack(feat_list[:-1], dim=0).mean(dim=0)
        def pol2(feat_list): return feat_list[-1]
        out1 = self.DGMF_1(fuse_pol1(out1_all), pol2(out1_all))
        out2 = self.DGMF_2(fuse_pol1(out2_all), pol2(out2_all))
        out3 = self.DGMF_3(fuse_pol1(out3_all), pol2(out3_all))
        out4 = self.DGMF_4(fuse_pol1(out4_all), pol2(out4_all))
        pos_out1 = pos_outs[-1][0]
        pos_out2 = pos_outs[-1][1]
        pos_out3 = pos_outs[-1][2]
        pos_out4 = pos_outs[-1][3]
        out1 = NestedTensor(out1, backbone_outs[-1]['0'].mask)
        out2 = NestedTensor(out2, backbone_outs[-1]['1'].mask)
        out3 = NestedTensor(out3, backbone_outs[-1]['2'].mask)
        out4 = NestedTensor(out4, backbone_outs[-1]['3'].mask)
        out = [out1, out2, out3, out4]
        pos_outs = [pos_out1, pos_out2, pos_out3, pos_out4]
        features, trans_out = self.LDE(out, pos_outs)
        out = self.SACM(features, trans_out)
        return out


# tri
# class Decoder(nn.Module):
#     def __init__(self, backbones, args=None):
#         super().__init__()
#         self.args = args
#         self.backbones = nn.ModuleList(backbones)
#         self.LDE = build_LDE(args)
#         self.SACM = SACM()
#         self.DGMF_1 = DGMF(num_channels=128)
#         self.DGMF_2 = DGMF(num_channels=256)
#         self.DGMF_3 = DGMF(num_channels=512)
#         self.DGMF_4 = DGMF(num_channels=1024)
#     def forward(self, modal_imgs):
#         pos_outs = []
#         backbone_outs = []
#         for i in range(len(self.backbones)):
#             backbone_out, pos_out = self.backbones[i](modal_imgs[i])
#             backbone_outs.append(backbone_out)
#             pos_outs.append(pos_out)
#         angle_backbones = backbone_outs[:-2]
#         aop_backbone = backbone_outs[-2]
#         dop_backbone = backbone_outs[-1]
#         def average_feats(feat_key):
#             return torch.stack([bo[feat_key].tensors for bo in angle_backbones], dim=0).mean(dim=0)
#         out1 = self.DGMF_1(average_feats('0'), aop_backbone['0'].tensors, dop_backbone['0'].tensors)
#         out2 = self.DGMF_2(average_feats('1'), aop_backbone['1'].tensors, dop_backbone['1'].tensors)
#         out3 = self.DGMF_3(average_feats('2'), aop_backbone['2'].tensors, dop_backbone['2'].tensors)
#         out4 = self.DGMF_4(average_feats('3'), aop_backbone['3'].tensors, dop_backbone['3'].tensors)
#         pos_out1, pos_out2, pos_out3, pos_out4 = pos_outs[-1]
#         out1 = NestedTensor(out1, dop_backbone['0'].mask)
#         out2 = NestedTensor(out2, dop_backbone['1'].mask)
#         out3 = NestedTensor(out3, dop_backbone['2'].mask)
#         out4 = NestedTensor(out4, dop_backbone['3'].mask)
#         out = [out1, out2, out3, out4]
#         pos_outs = [pos_out1, pos_out2, pos_out3, pos_out4]
#         features, trans_out = self.LDE(out, pos_outs)
#         out = self.SACM(features, trans_out)
#         return out


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


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.size_average = size_average
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.criterion(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.size_average:
            return F_loss.mean()
        else:
            return F_loss.sum()


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.

    def forward(self, pred, target):
        num = pred.size(0)
        m1 = pred.view(num, -1)
        m2 = target.view(num, -1)
        intersection = (m1 * m2).sum()
        return 1 - (2. * intersection + self.smooth) / (m1.sum() + m2.sum() + self.smooth)


class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        weight = torch.zeros_like(targets)
        weight = torch.fill_(weight, 0.04)
        weight[targets > 0] = 0.96
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean', weight=weight)
        Dice_BCE = BCE*0.8 + dice_loss*0.2
        return Dice_BCE


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        return 1 - dc


class new_bce_dice(nn.Module):
    def __init__(self, args):
        super(new_bce_dice, self).__init__()
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.dice_fn = SoftDiceLoss()
        self.args = args

    def forward(self, y_pred, y_true):
        bce = self.bce_fn(y_pred, y_true)
        dice = self.dice_fn(y_pred.sigmoid(), y_true)
        return self.args.BCELoss_ratio * bce + self.args.DiceLoss_ratio * dice


def build(args):
    device = torch.device(args.device)
    args.device = torch.device(args.device)
    backbone = build_msfe(args)
    model = Decoder(backbone, args)
    # criterion = BinaryFocalLoss()
    # criterion = DiceBCELoss()
    criterion = new_bce_dice(args)
    criterion.to(device)
    return model, criterion


