import torch
from torch import nn
import warnings
import functools
from util.misc import NestedTensor
from typing import Dict, List
import torch.nn.functional as F
from ..position_encoding import build_position_encoding
from .base import BaseNet
import apex


warnings.filterwarnings("ignore")


class Resnet(BaseNet):
    def __init__(self, backbone, aux=True, norm_layer=nn.BatchNorm2d, spm_on=False, **kwargs):
        super(Resnet, self).__init__(backbone, aux, norm_layer=norm_layer, spm_on=spm_on, **kwargs)

    def forward(self, x, y=None):
        _, _, h, w = x.size()
        c1, c2, c3, c4 = self.base_forward(x)
        return c1, c2, c3, c4


class BackboneBase(nn.Module):
    def __init__(self, train_backbone: bool, return_interm_layers: bool, args=None):
        super().__init__()
        backbone = Resnet(backbone='resnet50', norm_layer=nn.BatchNorm2d, spm_on=True)
        if return_interm_layers:
            self.num_channels = [128, 256, 512, 1024]
        else:
            self.num_channels = [1024]
        self.body = backbone
        self.position_embedding = build_position_encoding(args)

    def forward(self, tensor_list: NestedTensor):
        feat1, feat2, feat3, feat4 = self.body(tensor_list.tensors)
        xs: Dict[str, NestedTensor] = {}
        out: Dict[str, NestedTensor] = {}
        pos = []
        xs['0'] = feat1
        xs['1'] = feat2
        xs['2'] = feat3
        xs['3'] = feat4
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        list1 = [out['0'], out['1'], out['2'], out['3']]
        for i in range(len(list1)):
            pos.append(self.position_embedding(list1[i]))
        return out, pos


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def build_msfe(args):
    modals = args.modals
    modal_num = len(modals)
    backbones = []
    for i in range(modal_num):
        backbone = BackboneBase(True, True, args)
        backbones.append(backbone)
    return backbones
