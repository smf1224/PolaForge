import torch.nn as nn
from . import resnet

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

__all__ = ['BaseNet']

class BaseNet(nn.Module):
    def __init__(self, backbone, aux, dilated=True, norm_layer=None, spm_on=False):
        super(BaseNet, self).__init__()
        self.aux = aux
        self.backbone = backbone
        if backbone == 'resnet50':
            self.net = resnet.resnet50(dilated=dilated,  norm_layer=norm_layer, multi_grid=True, spm_on=spm_on)
        elif backbone == 'resnet34':
            self.net = resnet.resnet34(dilated=dilated, norm_layer=norm_layer, multi_grid=True, spm_on=spm_on)
        elif backbone == 'resnet101':
            self.net = resnet.resnet101(dilated=dilated, norm_layer=norm_layer, multi_grid=True, spm_on=spm_on)
        elif backbone == 'resnet152':
            self.net = resnet.resnet152(dilated=dilated, norm_layer=norm_layer, multi_grid=True, spm_on=spm_on)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        self._up_kwargs = up_kwargs

    def base_forward(self, x):
        if self.backbone.startswith('wideresnet'):
            x = self.net.mod1(x)
            x = self.net.pool2(x)
            x = self.net.mod2(x)
            x = self.net.pool3(x)
            x = self.net.mod3(x)
            x = self.net.mod4(x)
            x = self.net.mod5(x)
            c3 = x.clone()
            x = self.net.mod6(x)
            x = self.net.mod7(x)
            x = self.net.bn_out(x)
            return None, None, c3, x
        else:
            x = self.net.conv1(x)
            x = self.net.bn1(x)
            x = self.net.relu(x)
            x = self.net.maxpool(x)

            c1 = self.net.layer1(x)
            c2 = self.net.layer2(c1)
            c3 = self.net.layer3(c2)
            c4 = self.net.layer4(c3)

        return c1, c2, c3, c4
